"""
Neural ODE model for robot state prediction.
Given initial state (joint angles + velocities) and torques, predict future states.
Uses torchdiffeq with adjoint sensitivity method for memory-efficient backpropagation.
"""

import torch
import torch.nn as nn
from torchdiffeq import odeint_adjoint as odeint


class RobotDynamicsODE(nn.Module):
    """
    Neural network that models the dynamics: d[q, q_dot]/dt = f(q, q_dot, torque)
    
    State vector: [q (7 joint angles), q_dot (7 joint velocities)] = 14 dims
    Control input: torque (7 dims)
    
    The network learns to predict [q_dot, q_ddot] given [q, q_dot, torque].
    """
    
    def __init__(self, state_dim=14, torque_dim=7, hidden_dim=128, max_accel=100.0):
        super().__init__()
        self.state_dim = state_dim
        self.torque_dim = torque_dim
        self.max_accel = max_accel  # Clamp accelerations to prevent explosion
        
        # Input: state (14) + torque (7) = 21
        input_dim = state_dim + torque_dim
        
        self.net = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.Tanh(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.Tanh(),
            nn.Linear(hidden_dim, state_dim),
        )
        
        # Initialize last layer to SMALL values for stable training
        nn.init.normal_(self.net[-1].weight, mean=0.0, std=0.001)
        nn.init.zeros_(self.net[-1].bias)
        
        # Torque will be set externally for each integration
        self.torque = None
    
    def set_torque(self, torque):
        """Set the control torque for the current integration."""
        self.torque = torque
    
    def forward(self, t, state):
        """
        Compute the time derivative of the state.
        
        Args:
            t: Current time (scalar, not used directly but required by ODE solver)
            state: Current state [batch, state_dim] = [batch, 14]
        
        Returns:
            dstate_dt: Time derivative [batch, state_dim]
        """
        # Concatenate state with torque
        if self.torque is None:
            raise ValueError("Torque must be set before integration")
        
        # state: [batch, 14], torque: [batch, 7]
        x = torch.cat([state, self.torque], dim=-1)
        
        # Network predicts the derivative
        dstate_dt = self.net(x)
        
        # Physics prior: first 7 components of derivative are velocities
        # dq/dt = q_dot (from state)
        q_dot = state[..., 7:]  # velocities from state
        q_ddot = dstate_dt[..., 7:]  # accelerations from network
        
        # CRITICAL: Clamp accelerations to prevent ODE explosion
        q_ddot = torch.clamp(q_ddot, -self.max_accel, self.max_accel)
        
        # Combine: d[q, q_dot]/dt = [q_dot, q_ddot]
        dstate_dt = torch.cat([q_dot, q_ddot], dim=-1)
        
        return dstate_dt


class NeuralODERobot(nn.Module):
    """
    Full Neural ODE model for robot state prediction.
    
    Given initial state and torques, integrates the ODE to predict future states.
    """
    
    def __init__(self, state_dim=14, torque_dim=7, hidden_dim=128):
        super().__init__()
        self.dynamics = RobotDynamicsODE(state_dim, torque_dim, hidden_dim)
        self.state_dim = state_dim
        self.torque_dim = torque_dim
    
    def forward(self, initial_state, torque, t_span, method='rk4', rtol=1e-5, atol=1e-6):
        """
        Integrate the Neural ODE from initial_state over t_span.
        
        Args:
            initial_state: Initial state [batch, state_dim]
            torque: Control torque (assumed constant over integration) [batch, torque_dim]
            t_span: Time points to return solutions at [num_times]
            method: ODE solver method (default: 'rk4')
            rtol, atol: Tolerances for adaptive solver
        
        Returns:
            states: Predicted states at each time point [num_times, batch, state_dim]
        """
        self.dynamics.set_torque(torque)
        
        # Use adjoint method for memory-efficient backpropagation
        states = odeint(
            self.dynamics,
            initial_state,
            t_span,
            method=method,
            rtol=rtol,
            atol=atol,
        )
        
        return states


class NeuralODERobotSequence(nn.Module):
    """
    Neural ODE model that handles sequences with varying torques.
    
    For each timestep, integrates from current state to next state using current torque.
    This is more accurate for trajectories where torque varies over time.
    
    Args:
        dt: Time step between samples in seconds. For 500 Hz data, dt = 1/500 = 0.002.
    """
    
    def __init__(self, state_dim=14, torque_dim=7, hidden_dim=128, dt=0.002, max_accel=50.0):
        super().__init__()
        self.dynamics = RobotDynamicsODE(state_dim, torque_dim, hidden_dim, max_accel)
        self.state_dim = state_dim
        self.torque_dim = torque_dim
        self.dt = dt  # For 500 Hz: dt = 1/500 = 0.002 seconds
    
    def forward(self, initial_state, torques, num_steps=None, method='rk4'):
        """
        Integrate the Neural ODE step by step with varying torques.
        
        Args:
            initial_state: Initial state [batch, state_dim]
            torques: Torques at each timestep [batch, seq_len, torque_dim]
            num_steps: Number of steps to predict (default: seq_len from torques)
            method: ODE solver method (default: 'rk4' for accuracy)
        
        Returns:
            states: Predicted states [batch, num_steps+1, state_dim]
        """
        batch_size = initial_state.shape[0]
        device = initial_state.device
        
        if num_steps is None:
            num_steps = torques.shape[1]
        
        # Store predicted states
        states = [initial_state]
        current_state = initial_state
        
        t_span = torch.tensor([0.0, self.dt], device=device)
        
        for i in range(num_steps):
            # Get torque for this step
            torque = torques[:, min(i, torques.shape[1]-1), :]
            self.dynamics.set_torque(torque)
            
            # Integrate one step
            next_states = odeint(
                self.dynamics,
                current_state,
                t_span,
                method=method,
            )
            
            # Get final state (index 1 is at t=dt)
            current_state = next_states[1]
            states.append(current_state)
        
        # Stack: [batch, num_steps+1, state_dim]
        states = torch.stack(states, dim=1)
        
        return states


def create_model(model_type='sequence', **kwargs):
    """
    Factory function to create a Neural ODE model.
    
    Args:
        model_type: 'single' for constant torque, 'sequence' for varying torques
        **kwargs: Additional arguments for the model
    
    Returns:
        model: Neural ODE model
    """
    if model_type == 'single':
        return NeuralODERobot(**kwargs)
    elif model_type == 'sequence':
        return NeuralODERobotSequence(**kwargs)
    else:
        raise ValueError(f"Unknown model type: {model_type}")


if __name__ == "__main__":
    # Quick test
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")
    
    # Create model with dt=0.002 for 500 Hz sampling rate
    model = NeuralODERobotSequence(hidden_dim=64, dt=0.002).to(device)
    
    # Test forward pass
    batch_size = 4
    seq_len = 10
    
    initial_state = torch.randn(batch_size, 14, device=device)
    torques = torch.randn(batch_size, seq_len, 7, device=device)
    
    print(f"Initial state shape: {initial_state.shape}")
    print(f"Torques shape: {torques.shape}")
    
    states = model(initial_state, torques)
    print(f"Output states shape: {states.shape}")
    
    # Test backward pass
    loss = states.sum()
    loss.backward()
    print("Backward pass successful!")
