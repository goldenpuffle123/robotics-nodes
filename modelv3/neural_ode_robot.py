from typing import Union, Callable

import torch
import torch.nn as nn
from torchdiffeq import odeint, odeint_adjoint


def velocity_verlet(func: Callable, y0: torch.Tensor, t: torch.Tensor, **kwargs) -> torch.Tensor:
    """
    Batched velocity verlet integrator. Integrates the ODE function for times t given the initial state.
    Args:
        func: ODE function: (t, y) --> dy/dt
        y0: [batch, state_dim] initial state
        t: [num_steps] sequence of time points
    Returns:
        states: [num_steps, batch, state_dim]
    """
    device = y0.device
    dtype = y0.dtype
    batch_size = y0.shape[0]
    state_dim = y0.shape[1]
    dof = state_dim // 2 # degrees of freedom of the arm
    
    num_steps = len(t)
    states = torch.zeros(num_steps, batch_size, state_dim, device=device, dtype=dtype) # initialize tensor of output states
    
    # Initial state
    y = y0.clone()
    states[0] = y
    
    # Extract initial position and velocity components
    q = y[:, :dof]  # positions (angles)
    v = y[:, dof:]  # velocities
    
    # Initial acceleration: predicted using ODE function
    dy = func(t[0], y)
    a = dy[:, dof:]  # acceleration part
    
    for i in range(1, num_steps):
        dt = t[i] - t[i-1]
        
        # Step 1: Update position using current velocity and acceleration
        # q_{n+1} = q_n + v_n * dt + 0.5 * a_n * dt^2
        q_next = q + v * dt + 0.5 * a * dt * dt
        
        # Step 2: Compute new acceleration at predicted state
        # Use predictor velocity: v_pred = v_n + a_n * dt
        v_pred = v + a * dt
        y_pred = torch.cat([q_next, v_pred], dim=-1)
        dy_next = func(t[i], y_pred)
        a_next = dy_next[:, dof:]
        
        # Step 3: Update velocity using average acceleration
        # v_{n+1} = v_n + 0.5 * (a_n + a_{n+1}) * dt
        v_next = v + 0.5 * (a + a_next) * dt
        
        # Store new predicted state
        q = q_next
        v = v_next
        a = a_next
        y = torch.cat([q, v], dim=-1)
        states[i] = y
    
    return states


class TorqueInterpolatorConstant:
    """Piecewise-constant torque interpolator."""
    
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
        """Return torque at time t using constant interpolation. Returns [batch, dof]."""
        # Compute index (floor to get left/current value)
        idx = torch.floor(t / self.dt).long()
        idx = torch.clamp(idx, 0, self.seq_len - 1)
        return self.torques[:, idx, :]


class TorqueInterpolatorLinear:
    """Linear torque interpolator."""
    
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
    """ODE dynamics function: predicts change of state given current state and input torque."""
    
    def __init__(self, dof: int = 7, hidden_dim: int = 128):
        '''
        Args:
            dof: degrees of freedom of the robot arm
            hidden_dim: hidden dimension of the neural network
        '''
        super().__init__()
        self.dof = dof
        self.state_dim = dof * 2  # angles + velocities
        self.interpolator = None
        
        # Input: state (14) + torque (7) = 21
        # Output: acceleration (7)
        self.net = nn.Sequential(
            nn.Linear(self.state_dim + dof, hidden_dim),
            nn.SiLU(), # More stable than tanh or ReLU
            nn.Linear(hidden_dim, hidden_dim),
            nn.SiLU(),
            nn.Linear(hidden_dim, dof),  # Only 7-dim acceleration
        )
        
        # Reasonable initialization for dynamics learning
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
        torque = self.interpolator(t)  # [batch, dof]; get interpolated torque at time t 
        q_dot = state[:, self.dof:]  # Extract velocities: this is the identity v = dq/dt
        x = torch.cat([state, torque], dim=-1)  # [batch, state_dim + dof]; input state
        q_ddot = self.net(x)  # Predict acceleration only
        return torch.cat([q_dot, q_ddot], dim=-1)  # [velocities, accelerations] = dstate/dt


class NeuralODE(nn.Module):
    """Neural ODE model with per-trajectory integration."""
    
    def __init__(self, dof: int = 7, hidden_dim: int = 128, dt: float = 0.002, integrator: str = 'euler'):
        """
        Args:
            dof: degrees of freedom of the robot arm
            hidden_dim: hidden dimension of the neural network
            dt: timestep between torque inputs
            integrator: integration method ('euler', 'rk4', 'dopri5', 'vv')
        """
        super().__init__()
        self.dof = dof
        self.state_dim = dof * 2
        self.dt = dt
        self.integrator = integrator
        self.ode_func = ODEFunc(dof=dof, hidden_dim=hidden_dim) # The ODE function
    
    def forward(self, initial_state: torch.Tensor, torques: torch.Tensor) -> torch.Tensor:
        """
        Forward pass with batched integration. Runs the ODE solver over the torque sequence to get predictions.
        
        Args:
            initial_state: [batch, state_dim]
            torques: [batch, seq_len, dof]
        
        Returns:
            states: [batch, seq_len, state_dim] (excludes initial)
        """
        seq_len = torques.shape[1]
        device = initial_state.device
        
        # Time points for integration using timestep dt
        t_span = torch.linspace(0, seq_len * self.dt, seq_len + 1, device=device)
        
        # Set batched interpolator
        self.ode_func.set_interpolator(TorqueInterpolatorConstant(torques, self.dt))
        
        # Batched integration: returns [seq_len+1, batch, state_dim]
        try:
            if self.integrator == 'vv': # Custom function shown above
                states = velocity_verlet(self.ode_func, initial_state, t_span)
            else: # Use torchdiffeq solvers
                states = odeint(self.ode_func, initial_state, t_span, method=self.integrator)
        except ValueError:
            raise ValueError(f"Unknown integrator: {self.integrator}. Choose from: euler, rk4, dopri5, vv")
        
        # Permute to [batch, seq_len+1, state_dim] and exclude t=0 (initial state)
        return states[1:].permute(1, 0, 2)
