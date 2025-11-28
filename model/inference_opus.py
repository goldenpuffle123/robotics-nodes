"""
Inference utilities for Neural ODE robot model.
Provides easy-to-use functions for making predictions.
"""

import numpy as np
import torch
from pathlib import Path
from typing import Union, Optional

from neural_ode_robot_opus import create_model
from data_processing_opus import Normalizer, STATE_COLS, TORQUE_COLS


class NeuralODEPredictor:
    """
    Easy-to-use predictor class for robot state prediction.
    
    Example usage:
        predictor = NeuralODEPredictor.from_checkpoint('checkpoints/best_model.pt')
        
        # Single prediction
        initial_state = np.array([...])  # 14 values: 7 angles + 7 velocities
        torques = np.array([...])  # [seq_len, 7]
        predicted_states = predictor.predict(initial_state, torques)
        
        # Just predict angles
        predicted_angles = predictor.predict_angles(initial_state, torques)
    """
    
    def __init__(
        self,
        model,
        state_normalizer: Normalizer,
        torque_normalizer: Normalizer,
        device: torch.device,
        config: dict,
    ):
        self.model = model
        self.state_normalizer = state_normalizer
        self.torque_normalizer = torque_normalizer
        self.device = device
        self.config = config
        self.model.eval()
    
    @classmethod
    def from_checkpoint(cls, checkpoint_path: str, device: str = None):
        """Load predictor from a checkpoint file."""
        if device is None:
            device = 'cuda' if torch.cuda.is_available() else 'cpu'
        device = torch.device(device)
        
        checkpoint = torch.load(checkpoint_path, map_location=device)
        config = checkpoint['config']
        
        model = create_model(
            model_type='sequence',
            hidden_dim=config['hidden_dim'],
            dt=config['dt'],
        ).to(device)
        model.load_state_dict(checkpoint['model_state_dict'])
        
        state_normalizer = Normalizer.from_dict(checkpoint['state_normalizer'])
        torque_normalizer = Normalizer.from_dict(checkpoint['torque_normalizer'])
        
        return cls(model, state_normalizer, torque_normalizer, device, config)
    
    @torch.no_grad()
    def predict(
        self,
        initial_state: np.ndarray,
        torques: np.ndarray,
        return_all: bool = True,
    ) -> np.ndarray:
        """
        Predict future states given initial state and torques.
        
        Args:
            initial_state: Initial state [14] (7 angles + 7 velocities)
            torques: Torques for each timestep [seq_len, 7]
            return_all: If True, return all states including initial.
                        If False, return only predicted states.
        
        Returns:
            states: Predicted states [seq_len+1, 14] or [seq_len, 14]
        """
        # Ensure numpy arrays
        initial_state = np.asarray(initial_state, dtype=np.float32)
        torques = np.asarray(torques, dtype=np.float32)
        
        # Validate shapes
        if initial_state.shape != (14,):
            raise ValueError(f"initial_state must be shape (14,), got {initial_state.shape}")
        if torques.ndim != 2 or torques.shape[1] != 7:
            raise ValueError(f"torques must be shape (seq_len, 7), got {torques.shape}")
        
        # Normalize
        initial_state_norm = self.state_normalizer.transform(initial_state.reshape(1, -1))
        torques_norm = self.torque_normalizer.transform(torques)
        
        # Convert to tensors
        initial_state_t = torch.from_numpy(initial_state_norm).float().to(self.device)
        torques_t = torch.from_numpy(torques_norm).float().unsqueeze(0).to(self.device)
        
        # Predict
        pred_states = self.model(initial_state_t, torques_t)
        
        # Denormalize
        pred_states_np = pred_states[0].cpu().numpy()
        pred_states_denorm = self.state_normalizer.inverse_transform(pred_states_np)
        
        if return_all:
            return pred_states_denorm
        else:
            return pred_states_denorm[1:]  # Skip initial state
    
    def predict_angles(
        self,
        initial_state: np.ndarray,
        torques: np.ndarray,
        return_all: bool = True,
    ) -> np.ndarray:
        """
        Predict future joint angles given initial state and torques.
        
        Returns:
            angles: Predicted angles [seq_len+1, 7] or [seq_len, 7]
        """
        states = self.predict(initial_state, torques, return_all)
        return states[..., :7]
    
    def predict_velocities(
        self,
        initial_state: np.ndarray,
        torques: np.ndarray,
        return_all: bool = True,
    ) -> np.ndarray:
        """
        Predict future joint velocities given initial state and torques.
        
        Returns:
            velocities: Predicted velocities [seq_len+1, 7] or [seq_len, 7]
        """
        states = self.predict(initial_state, torques, return_all)
        return states[..., 7:]
    
    @torch.no_grad()
    def predict_batch(
        self,
        initial_states: np.ndarray,
        torques: np.ndarray,
    ) -> np.ndarray:
        """
        Predict for a batch of initial states and torques.
        
        Args:
            initial_states: [batch, 14]
            torques: [batch, seq_len, 7]
        
        Returns:
            states: [batch, seq_len+1, 14]
        """
        # Normalize
        batch_size = initial_states.shape[0]
        initial_states_norm = self.state_normalizer.transform(initial_states)
        torques_flat = torques.reshape(-1, 7)
        torques_norm = self.torque_normalizer.transform(torques_flat).reshape(torques.shape)
        
        # Convert to tensors
        initial_states_t = torch.from_numpy(initial_states_norm).float().to(self.device)
        torques_t = torch.from_numpy(torques_norm).float().to(self.device)
        
        # Predict
        pred_states = self.model(initial_states_t, torques_t)
        
        # Denormalize
        pred_states_np = pred_states.cpu().numpy()
        pred_states_flat = pred_states_np.reshape(-1, 14)
        pred_states_denorm = self.state_normalizer.inverse_transform(pred_states_flat)
        pred_states_denorm = pred_states_denorm.reshape(pred_states_np.shape)
        
        return pred_states_denorm
    
    def rollout(
        self,
        initial_state: np.ndarray,
        torque_fn,
        num_steps: int,
    ) -> np.ndarray:
        """
        Rollout predictions using a torque function.
        
        This is useful for closed-loop simulation where torques
        depend on the current state.
        
        Args:
            initial_state: Initial state [14]
            torque_fn: Function that takes state and returns torque
                       torque_fn(state: np.ndarray) -> np.ndarray
            num_steps: Number of steps to rollout
        
        Returns:
            states: [num_steps+1, 14]
        """
        states = [initial_state.copy()]
        current_state = initial_state.copy()
        
        for _ in range(num_steps):
            # Get torque from function
            torque = torque_fn(current_state)
            
            # Predict one step
            pred = self.predict(current_state, torque.reshape(1, -1), return_all=False)
            current_state = pred[0]
            states.append(current_state)
        
        return np.array(states)


def load_predictor(checkpoint_path: str, device: str = None) -> NeuralODEPredictor:
    """
    Convenience function to load a predictor.
    
    Args:
        checkpoint_path: Path to checkpoint file
        device: Device to use ('cuda', 'cpu', or None for auto)
    
    Returns:
        predictor: NeuralODEPredictor instance
    """
    return NeuralODEPredictor.from_checkpoint(checkpoint_path, device)


if __name__ == "__main__":
    import pandas as pd
    
    # Example usage
    print("Neural ODE Robot Predictor Example")
    print("=" * 50)
    
    # Check if checkpoint exists
    checkpoint_path = "checkpoints/neural_ode_opus/best_model.pt"
    if not Path(checkpoint_path).exists():
        print(f"Checkpoint not found at {checkpoint_path}")
        print("Run train_opus.py first to train a model.")
        exit(1)
    
    # Load predictor
    print(f"Loading predictor from {checkpoint_path}...")
    predictor = load_predictor(checkpoint_path)
    print(f"Model loaded with config: {predictor.config}")
    
    # Load some test data
    test_file = "datasets/baxter/left_circle_p-15_t105.csv"
    df = pd.read_csv(test_file, sep=" ")
    
    # Get initial state and torques
    states = df[STATE_COLS].values.astype(np.float32)
    torques = df[TORQUE_COLS].values.astype(np.float32)
    
    initial_state = states[0]
    torque_seq = torques[:50]
    
    print(f"\nInitial state: {initial_state[:3]}... (first 3 of 14)")
    print(f"Torque sequence shape: {torque_seq.shape}")
    
    # Predict
    print("\nPredicting...")
    predicted_states = predictor.predict(initial_state, torque_seq)
    print(f"Predicted states shape: {predicted_states.shape}")
    
    # Compare first few steps
    print("\nComparison (first 5 steps, first 3 angles):")
    print("Ground truth vs Predicted:")
    for i in range(5):
        gt = states[i, :3]
        pred = predicted_states[i, :3]
        print(f"  Step {i}: GT={gt}, Pred={pred}")
