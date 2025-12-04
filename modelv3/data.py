"""Data loading utilities for Neural ODE training."""

import os
import glob
import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset, DataLoader
from typing import Tuple, Dict


ANGLE_COLS = ['ang_s0', 'ang_s1', 'ang_e0', 'ang_e1', 'ang_w0', 'ang_w1', 'ang_w2']
VELOCITY_COLS = ['vel_s0', 'vel_s1', 'vel_e0', 'vel_e1', 'vel_w0', 'vel_w1', 'vel_w2']
TORQUE_COLS = ['torq_s0', 'torq_s1', 'torq_e0', 'torq_e1', 'torq_w0', 'torq_w1', 'torq_w2']
STATE_COLS = ANGLE_COLS + VELOCITY_COLS


class Normalizer:
    """Mean/std normalizer."""
    
    def __init__(self):
        self.mean = None
        self.std = None
        self.is_fitted = False
    
    def fit(self, data: np.ndarray):
        self.mean = data.mean(axis=0)
        self.std = data.std(axis=0)
        self.std[self.std < 1e-8] = 1.0
        self.is_fitted = True
        return self
    
    def norm(self, data: np.ndarray) -> np.ndarray:
        return (data - self.mean) / self.std
    
    def denorm(self, data: np.ndarray) -> np.ndarray:
        return data * self.std + self.mean
    
    def to_dict(self) -> Dict:
        return {
            'mean': self.mean.tolist() if self.mean is not None else None,
            'std': self.std.tolist() if self.std is not None else None,
        }
    
    @classmethod
    def from_dict(cls, d: Dict) -> 'Normalizer':
        norm = cls()
        norm.mean = np.array(d['mean']) if d['mean'] else None
        norm.std = np.array(d['std']) if d['std'] else None
        norm.is_fitted = norm.mean is not None
        return norm


class TrajectoryDataset(Dataset):
    """Dataset for trajectory sequences."""
    
    def __init__(self, states: np.ndarray, torques: np.ndarray, indices: list, seq_len: int):
        self.states = states
        self.torques = torques
        self.indices = indices
        self.seq_len = seq_len
    
    def __len__(self):
        return len(self.indices)
    
    def __getitem__(self, idx):
        start = self.indices[idx]
        initial_state = torch.from_numpy(self.states[start].copy())
        torques = torch.from_numpy(self.torques[start:start + self.seq_len - 1].copy())
        target_states = torch.from_numpy(self.states[start + 1:start + self.seq_len].copy())
        return initial_state, torques, target_states


def create_dataloaders(
    data_dir: str,
    batch_size: int = 32,
    seq_len: int = 100,
    stride: int = 10,
    train_split: float = 0.7,
    num_workers: int = 4,
    seed: int = 42,
) -> Tuple[DataLoader, DataLoader, Normalizer, Normalizer]:
    """Create train and validation dataloaders."""
    
    files = sorted(glob.glob(os.path.join(data_dir, "left_*.csv")))
    print(f"Loading {len(files)} files")
    
    all_states, all_torques = [], []
    boundaries = [0]
    
    for f in files:
        df = pd.read_csv(f, sep=" ")
        all_states.append(df[STATE_COLS].values.astype(np.float32))
        all_torques.append(df[TORQUE_COLS].values.astype(np.float32))
        boundaries.append(boundaries[-1] + len(df))
    
    all_states = np.concatenate(all_states)
    all_torques = np.concatenate(all_torques)
    
    # Only normalize torques, NOT states (states stay in physical units for ODE)
    state_norm = Normalizer().fit(all_states)  # Still fit for reference/denorm
    torque_norm = Normalizer().fit(all_torques)
    
    # States: RAW (physical units) - required for correct ODE physics
    # Torques: normalized (helps network training)
    torques_n = torque_norm.norm(all_torques)
    
    # Generate indices (respect file boundaries)
    indices = []
    for i in range(len(boundaries) - 1):
        for idx in range(boundaries[i], boundaries[i + 1] - seq_len, stride):
            indices.append(idx)
    
    # Shuffle and split
    np.random.seed(seed)
    np.random.shuffle(indices)
    n_train = int(len(indices) * train_split)
    
    print(f"Total: {len(indices)}, Train: {n_train}, Val: {len(indices) - n_train}")
    
    # Use raw states (all_states), normalized torques (torques_n)
    train_ds = TrajectoryDataset(all_states, torques_n, indices[:n_train], seq_len)
    val_ds = TrajectoryDataset(all_states, torques_n, indices[n_train:], seq_len)
    
    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True, num_workers=num_workers, pin_memory=True)
    val_loader = DataLoader(val_ds, batch_size=batch_size, shuffle=False, num_workers=num_workers, pin_memory=True)
    
    return train_loader, val_loader, state_norm, torque_norm
