"""
Data processing utilities for Baxter robot Neural ODE training.
Handles loading, normalization, and batching of robot trajectory data.
"""

import os
import glob
import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset, DataLoader
from typing import List, Tuple, Optional, Dict


# Column names for the Baxter robot dataset
ANGLE_COLS = ['ang_s0', 'ang_s1', 'ang_e0', 'ang_e1', 'ang_w0', 'ang_w1', 'ang_w2']
VELOCITY_COLS = ['vel_s0', 'vel_s1', 'vel_e0', 'vel_e1', 'vel_w0', 'vel_w1', 'vel_w2']
TORQUE_COLS = ['torq_s0', 'torq_s1', 'torq_e0', 'torq_e1', 'torq_w0', 'torq_w1', 'torq_w2']

STATE_COLS = ANGLE_COLS + VELOCITY_COLS  # 14 dimensions


class Normalizer:
    """Normalizes and denormalizes data using mean and std."""
    
    def __init__(self):
        self.mean = None
        self.std = None
        self.is_fitted = False
    
    def fit(self, data: np.ndarray):
        """Compute mean and std from data."""
        self.mean = data.mean(axis=0)
        self.std = data.std(axis=0)
        # Avoid division by zero
        self.std[self.std < 1e-8] = 1.0
        self.is_fitted = True
        return self
    
    def transform(self, data: np.ndarray) -> np.ndarray:
        """Normalize data."""
        if not self.is_fitted:
            raise RuntimeError("Normalizer must be fitted before transform")
        return (data - self.mean) / self.std
    
    def inverse_transform(self, data: np.ndarray) -> np.ndarray:
        """Denormalize data."""
        if not self.is_fitted:
            raise RuntimeError("Normalizer must be fitted before inverse_transform")
        return data * self.std + self.mean
    
    def fit_transform(self, data: np.ndarray) -> np.ndarray:
        """Fit and transform data."""
        self.fit(data)
        return self.transform(data)
    
    def to_dict(self) -> Dict:
        """Convert to dictionary for saving."""
        return {
            'mean': self.mean.tolist() if self.mean is not None else None,
            'std': self.std.tolist() if self.std is not None else None,
            'is_fitted': self.is_fitted
        }
    
    @classmethod
    def from_dict(cls, d: Dict) -> 'Normalizer':
        """Load from dictionary."""
        norm = cls()
        norm.mean = np.array(d['mean']) if d['mean'] is not None else None
        norm.std = np.array(d['std']) if d['std'] is not None else None
        norm.is_fitted = d['is_fitted']
        return norm


class RobotTrajectoryDataset(Dataset):
    """
    Dataset for robot trajectory sequences.
    
    Each sample is a sequence of (state, torque) pairs.
    The model predicts future states given initial state and torques.
    """
    
    def __init__(
        self,
        data_dir: str,
        seq_len: int = 100,
        stride: int = 1,
        files: Optional[List[str]] = None,
        state_normalizer: Optional[Normalizer] = None,
        torque_normalizer: Optional[Normalizer] = None,
        fit_normalizers: bool = False,
    ):
        """
        Args:
            data_dir: Directory containing CSV files
            seq_len: Length of each sequence
            stride: Stride between sequences (for data augmentation)
            files: Optional list of specific files to use
            state_normalizer: Normalizer for states (will be created if None and fit_normalizers=True)
            torque_normalizer: Normalizer for torques
            fit_normalizers: Whether to fit normalizers on this data
        """
        self.seq_len = seq_len
        self.stride = stride
        
        # Load all data
        if files is None:
            files = glob.glob(os.path.join(data_dir, "*.csv"))
        
        all_states = []
        all_torques = []
        self.file_boundaries = [0]  # Track where each file's data starts
        
        for f in files:
            df = pd.read_csv(f, sep=" ")
            
            # Extract states and torques
            states = df[STATE_COLS].values.astype(np.float32)
            torques = df[TORQUE_COLS].values.astype(np.float32)
            
            all_states.append(states)
            all_torques.append(torques)
            self.file_boundaries.append(self.file_boundaries[-1] + len(states))
        
        self.states = np.concatenate(all_states, axis=0)
        self.torques = np.concatenate(all_torques, axis=0)
        
        # Setup normalizers
        self.state_normalizer = state_normalizer or Normalizer()
        self.torque_normalizer = torque_normalizer or Normalizer()
        
        if fit_normalizers:
            self.state_normalizer.fit(self.states)
            self.torque_normalizer.fit(self.torques)
        
        # Normalize data
        if self.state_normalizer.is_fitted:
            self.states_norm = self.state_normalizer.transform(self.states)
        else:
            self.states_norm = self.states
        
        if self.torque_normalizer.is_fitted:
            self.torques_norm = self.torque_normalizer.transform(self.torques)
        else:
            self.torques_norm = self.torques
        
        # Create valid sequence indices (don't cross file boundaries)
        self.valid_indices = []
        for i in range(len(self.file_boundaries) - 1):
            start = self.file_boundaries[i]
            end = self.file_boundaries[i + 1]
            # Sequences that fit within this file
            for idx in range(start, end - seq_len, stride):
                self.valid_indices.append(idx)
        
        print(f"Loaded {len(files)} files, {len(self.states)} samples, {len(self.valid_indices)} sequences")
    
    def __len__(self):
        return len(self.valid_indices)
    
    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Returns:
            initial_state: [state_dim]
            torques: [seq_len-1, torque_dim]
            target_states: [seq_len-1, state_dim]
        """
        start_idx = self.valid_indices[idx]
        
        # Initial state
        initial_state = torch.from_numpy(self.states_norm[start_idx])
        
        # Torques for each step (seq_len-1 steps to predict seq_len-1 future states)
        torques = torch.from_numpy(self.torques_norm[start_idx:start_idx + self.seq_len - 1])
        
        # Target states (excluding initial state)
        target_states = torch.from_numpy(self.states_norm[start_idx + 1:start_idx + self.seq_len])
        
        return initial_state, torques, target_states


class RobotPredictionDataset(Dataset):
    """
    Dataset for single-step prediction.
    Predicts next state given current state and torque.
    """
    
    def __init__(
        self,
        data_dir: str,
        files: Optional[List[str]] = None,
        state_normalizer: Optional[Normalizer] = None,
        torque_normalizer: Optional[Normalizer] = None,
        fit_normalizers: bool = False,
    ):
        if files is None:
            files = glob.glob(os.path.join(data_dir, "*.csv"))
        
        all_states = []
        all_torques = []
        
        for f in files:
            df = pd.read_csv(f, sep=" ")
            states = df[STATE_COLS].values.astype(np.float32)
            torques = df[TORQUE_COLS].values.astype(np.float32)
            all_states.append(states)
            all_torques.append(torques)
        
        self.states = np.concatenate(all_states, axis=0)
        self.torques = np.concatenate(all_torques, axis=0)
        
        # Setup normalizers
        self.state_normalizer = state_normalizer or Normalizer()
        self.torque_normalizer = torque_normalizer or Normalizer()
        
        if fit_normalizers:
            self.state_normalizer.fit(self.states)
            self.torque_normalizer.fit(self.torques)
        
        # Normalize
        if self.state_normalizer.is_fitted:
            self.states_norm = self.state_normalizer.transform(self.states)
        else:
            self.states_norm = self.states
        
        if self.torque_normalizer.is_fitted:
            self.torques_norm = self.torque_normalizer.transform(self.torques)
        else:
            self.torques_norm = self.torques
        
        print(f"Loaded {len(files)} files, {len(self.states) - 1} prediction samples")
    
    def __len__(self):
        return len(self.states) - 1
    
    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Returns:
            current_state: [state_dim]
            torque: [torque_dim]
            next_state: [state_dim]
        """
        current_state = torch.from_numpy(self.states_norm[idx])
        torque = torch.from_numpy(self.torques_norm[idx])
        next_state = torch.from_numpy(self.states_norm[idx + 1])
        
        return current_state, torque, next_state


def create_dataloaders(
    data_dir: str,
    batch_size: int = 32,
    seq_len: int = 100,
    stride: int = 10,
    train_split: float = 0.8,
    num_workers: int = 0,
    seed: int = 42,
) -> Tuple[DataLoader, DataLoader, Normalizer, Normalizer]:
    """
    Create train and validation dataloaders.
    
    Args:
        data_dir: Directory containing CSV files
        batch_size: Batch size
        seq_len: Sequence length
        stride: Stride between sequences
        train_split: Fraction of files to use for training
        num_workers: Number of dataloader workers
        seed: Random seed for reproducibility
    
    Returns:
        train_loader, val_loader, state_normalizer, torque_normalizer
    """
    # Get all files
    files = sorted(glob.glob(os.path.join(data_dir, "*.csv")))
    
    # Split files into train/val
    np.random.seed(seed)
    np.random.shuffle(files)
    
    n_train = int(len(files) * train_split)
    train_files = files[:n_train]
    val_files = files[n_train:]
    
    if len(val_files) == 0:
        val_files = train_files  # Use same files if only one available
    
    print(f"Train files: {len(train_files)}, Val files: {len(val_files)}")
    
    # Create train dataset (fit normalizers on training data)
    train_dataset = RobotTrajectoryDataset(
        data_dir=data_dir,
        seq_len=seq_len,
        stride=stride,
        files=train_files,
        fit_normalizers=True,
    )
    
    # Create val dataset (use train normalizers)
    val_dataset = RobotTrajectoryDataset(
        data_dir=data_dir,
        seq_len=seq_len,
        stride=stride,
        files=val_files,
        state_normalizer=train_dataset.state_normalizer,
        torque_normalizer=train_dataset.torque_normalizer,
        fit_normalizers=False,
    )
    
    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=True,
    )
    
    val_loader = DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=True,
    )
    
    return (
        train_loader,
        val_loader,
        train_dataset.state_normalizer,
        train_dataset.torque_normalizer,
    )


if __name__ == "__main__":
    # Test data loading
    data_dir = "datasets/baxter"
    
    train_loader, val_loader, state_norm, torque_norm = create_dataloaders(
        data_dir,
        batch_size=8,
        seq_len=50,
        stride=10,
    )
    
    print(f"\nTrain batches: {len(train_loader)}")
    print(f"Val batches: {len(val_loader)}")
    
    # Check one batch
    for initial_state, torques, target_states in train_loader:
        print(f"\nBatch shapes:")
        print(f"  Initial state: {initial_state.shape}")
        print(f"  Torques: {torques.shape}")
        print(f"  Target states: {target_states.shape}")
        break
    
    print(f"\nState normalizer mean: {state_norm.mean[:3]}...")
    print(f"Torque normalizer mean: {torque_norm.mean[:3]}...")
