"""
Neural ODE model for robot dynamics prediction.
Key changes from v2:
1. Linear torque interpolation (differentiable)
2. Per-trajectory integration (no batched ODE)
3. Predicts delta state (not acceleration)
4. Uses odeint (not adjoint)
"""

from typing import Union

import torch
import torch.nn as nn
from torchdiffeq import odeint


class TorqueInterpolatorConstant:
    """Piecewise-constant (zero-order hold) torque interpolation - fully differentiable, batched."""
    
    def __init__(self, torques: torch.Tensor, dt: float):
        """
        Args:
            torques: [batch, seq_len, dof] torque sequence
            dt: timestep between samples
        """
        self.torques = torques  # [batch, seq_len, dof]
        self.dt = dt
        self.seq_len = torques.shape[1]
    
    def __call__(self, t: torch.Tensor) -> torch.Tensor:
        """Return torque at time t using zero-order hold. Returns [batch, dof]."""
        # Compute index (floor to get left/current value)
        idx = torch.floor(t / self.dt).long()
        idx = torch.clamp(idx, 0, self.seq_len - 1)
        return self.torques[:, idx, :]


class TorqueInterpolatorLinear:
    """Linear interpolation of torques - fully differentiable, batched."""
    
    def __init__(self, torques: torch.Tensor, dt: float):
        """
        Args:
            torques: [batch, seq_len, dof] torque sequence
            dt: timestep between samples
        """
        self.torques = torques  # [batch, seq_len, dof]
        self.dt = dt
        self.seq_len = torques.shape[1]
    
    def __call__(self, t: torch.Tensor) -> torch.Tensor:
        """Linear interpolation at time t. Returns [batch, dof]."""
        # Compute fractional index
        idx_float = t / self.dt
        idx_low = torch.floor(idx_float).long()
        idx_high = idx_low + 1
        
        # Clamp indices
        idx_low = torch.clamp(idx_low, 0, self.seq_len - 1)
        idx_high = torch.clamp(idx_high, 0, self.seq_len - 1)
        
        # Interpolation weight
        alpha = idx_float - idx_low.float()
        alpha = torch.clamp(alpha, 0.0, 1.0)
        
        # Linear interpolation: [batch, dof]
        tau_low = self.torques[:, idx_low, :]
        tau_high = self.torques[:, idx_high, :]
        return tau_low + alpha * (tau_high - tau_low)


class ODEFunc(nn.Module):
    """ODE dynamics: predicts acceleration given state and torque."""
    
    def __init__(self, dof: int = 7, hidden_dim: int = 128):
        super().__init__()
        self.dof = dof
        self.state_dim = dof * 2  # angles + velocities
        self.interpolator = None
        
        # Input: state (14) + torque (7) = 21
        # Output: acceleration (7) - only learned component
        self.net = nn.Sequential(
            nn.Linear(self.state_dim + dof, hidden_dim),
            nn.SiLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.SiLU(),
            nn.Linear(hidden_dim, dof),  # Only 7-dim acceleration
        )
        
        # Reasonable init for dynamics learning
        nn.init.normal_(self.net[-1].weight, std=0.01)
        nn.init.zeros_(self.net[-1].bias)
    
    def set_interpolator(self, interpolator: Union[TorqueInterpolatorLinear, TorqueInterpolatorConstant]):
        self.interpolator = interpolator
    
    def forward(self, t: torch.Tensor, state: torch.Tensor) -> torch.Tensor:
        """
        Args:
            t: scalar time
            state: [batch, state_dim] current state (angles, velocities)
        Returns:
            d_state: [batch, state_dim] = [velocities, accelerations]
        """
        torque = self.interpolator(t)  # [batch, dof]
        q_dot = state[:, self.dof:]  # Extract velocities
        x = torch.cat([state, torque], dim=-1)  # [batch, state_dim + dof]
        q_ddot = self.net(x)  # Predict acceleration only
        return torch.cat([q_dot, q_ddot], dim=-1)  # [velocities, accelerations]


class NeuralODE(nn.Module):
    """Neural ODE model with per-trajectory integration."""
    
    def __init__(self, dof: int = 7, hidden_dim: int = 128, dt: float = 0.002):
        super().__init__()
        self.dof = dof
        self.state_dim = dof * 2
        self.dt = dt
        self.ode_func = ODEFunc(dof=dof, hidden_dim=hidden_dim)
    
    def forward(self, initial_state: torch.Tensor, torques: torch.Tensor) -> torch.Tensor:
        """
        Forward pass with batched integration.
        
        Args:
            initial_state: [batch, state_dim]
            torques: [batch, seq_len, dof]
        
        Returns:
            states: [batch, seq_len, state_dim] (excludes initial)
        """
        batch_size, seq_len, _ = torques.shape
        device = initial_state.device
        
        # Time points
        t_span = torch.linspace(0, seq_len * self.dt, seq_len + 1, device=device)
        
        # Set batched interpolator
        self.ode_func.set_interpolator(TorqueInterpolatorConstant(torques, self.dt))
        
        # Batched integration: returns [seq_len+1, batch, state_dim]
        states = odeint(self.ode_func, initial_state, t_span, method='rk4')
        
        # Permute to [batch, seq_len+1, state_dim] and exclude t=0
        return states[1:].permute(1, 0, 2)
