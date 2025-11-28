"""
Training script for Neural ODE robot model.
Uses torchdiffeq with adjoint sensitivity for memory-efficient backpropagation.
"""

import os
import argparse
import json
import time
from datetime import datetime
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
from torch.optim import AdamW
from torch.optim.lr_scheduler import CosineAnnealingLR, ReduceLROnPlateau
from tqdm import tqdm

from neural_ode_robot_opus import NeuralODERobotSequence, create_model
from data_processing_opus import create_dataloaders, Normalizer


def train_epoch(
    model: nn.Module,
    train_loader,
    optimizer,
    device,
    epoch: int,
    gradient_clip: float = 1.0,
) -> dict:
    """Train for one epoch."""
    model.train()
    total_loss = 0.0
    total_angle_loss = 0.0
    total_vel_loss = 0.0
    num_batches = 0
    
    pbar = tqdm(train_loader, desc=f"Epoch {epoch} [Train]")
    for initial_state, torques, target_states in pbar:
        initial_state = initial_state.to(device)
        torques = torques.to(device)
        target_states = target_states.to(device)
        
        optimizer.zero_grad()
        
        # Forward pass: predict states given initial state and torques
        # Model returns [batch, seq_len+1, state_dim], we skip initial state
        pred_states = model(initial_state, torques)[:, 1:, :]
        
        # Compute loss (MSE on both angles and velocities)
        # Weight angles and velocities separately
        angle_loss = nn.functional.mse_loss(pred_states[..., :7], target_states[..., :7])
        vel_loss = nn.functional.mse_loss(pred_states[..., 7:], target_states[..., 7:])
        loss = angle_loss + 0.5 * vel_loss  # Weight velocities less
        
        # Check for NaN loss (skip batch if NaN)
        if torch.isnan(loss) or torch.isinf(loss):
            print(f"Warning: NaN/Inf loss detected, skipping batch")
            optimizer.zero_grad()
            continue
        
        # Backward pass (adjoint method handles this efficiently)
        loss.backward()
        
        # Check for NaN gradients
        has_nan_grad = False
        for param in model.parameters():
            if param.grad is not None and (torch.isnan(param.grad).any() or torch.isinf(param.grad).any()):
                has_nan_grad = True
                break
        
        if has_nan_grad:
            print(f"Warning: NaN/Inf gradient detected, skipping batch")
            optimizer.zero_grad()
            continue
        
        # Gradient clipping
        if gradient_clip > 0:
            torch.nn.utils.clip_grad_norm_(model.parameters(), gradient_clip)
        
        optimizer.step()
        
        total_loss += loss.item()
        total_angle_loss += angle_loss.item()
        total_vel_loss += vel_loss.item()
        num_batches += 1
        
        pbar.set_postfix({
            'loss': f'{loss.item():.4f}',
            'ang': f'{angle_loss.item():.4f}',
            'vel': f'{vel_loss.item():.4f}'
        })
    
    return {
        'loss': total_loss / num_batches,
        'angle_loss': total_angle_loss / num_batches,
        'vel_loss': total_vel_loss / num_batches,
    }


@torch.no_grad()
def validate(model: nn.Module, val_loader, device) -> dict:
    """Validate the model."""
    model.eval()
    total_loss = 0.0
    total_angle_loss = 0.0
    total_vel_loss = 0.0
    num_batches = 0
    
    for initial_state, torques, target_states in val_loader:
        initial_state = initial_state.to(device)
        torques = torques.to(device)
        target_states = target_states.to(device)
        
        pred_states = model(initial_state, torques)[:, 1:, :]
        
        angle_loss = nn.functional.mse_loss(pred_states[..., :7], target_states[..., :7])
        vel_loss = nn.functional.mse_loss(pred_states[..., 7:], target_states[..., 7:])
        loss = angle_loss + 0.5 * vel_loss
        
        total_loss += loss.item()
        total_angle_loss += angle_loss.item()
        total_vel_loss += vel_loss.item()
        num_batches += 1
    
    return {
        'loss': total_loss / num_batches,
        'angle_loss': total_angle_loss / num_batches,
        'vel_loss': total_vel_loss / num_batches,
    }


def save_checkpoint(
    model: nn.Module,
    optimizer,
    scheduler,
    epoch: int,
    train_metrics: dict,
    val_metrics: dict,
    state_normalizer: Normalizer,
    torque_normalizer: Normalizer,
    config: dict,
    save_path: str,
):
    """Save model checkpoint."""
    checkpoint = {
        'epoch': epoch,
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'scheduler_state_dict': scheduler.state_dict() if scheduler else None,
        'train_metrics': train_metrics,
        'val_metrics': val_metrics,
        'state_normalizer': state_normalizer.to_dict(),
        'torque_normalizer': torque_normalizer.to_dict(),
        'config': config,
    }
    torch.save(checkpoint, save_path)


def load_checkpoint(checkpoint_path: str, device):
    """Load model checkpoint."""
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
    
    return model, state_normalizer, torque_normalizer, checkpoint


def train(
    data_dir: str,
    output_dir: str,
    epochs: int = 100,
    batch_size: int = 32,
    seq_len: int = 50,
    stride: int = 10,
    hidden_dim: int = 128,
    dt: float = 0.002,  # 500 Hz sampling rate: dt = 1/500 = 0.002 seconds
    lr: float = 1e-3,
    weight_decay: float = 1e-4,
    gradient_clip: float = 1.0,
    scheduler_type: str = 'cosine',
    seed: int = 42,
    device: str = None,
):
    """Main training function."""
    
    # Setup
    if device is None:
        device = 'cuda' if torch.cuda.is_available() else 'cpu'
    device = torch.device(device)
    print(f"Using device: {device}")
    
    torch.manual_seed(seed)
    np.random.seed(seed)
    
    # Create output directory
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Save config
    config = {
        'data_dir': data_dir,
        'epochs': epochs,
        'batch_size': batch_size,
        'seq_len': seq_len,
        'stride': stride,
        'hidden_dim': hidden_dim,
        'dt': dt,
        'lr': lr,
        'weight_decay': weight_decay,
        'gradient_clip': gradient_clip,
        'scheduler_type': scheduler_type,
        'seed': seed,
    }
    
    with open(output_dir / 'config.json', 'w') as f:
        json.dump(config, f, indent=2)
    
    # Create dataloaders
    print(f"\nLoading data from {data_dir}...")
    train_loader, val_loader, state_normalizer, torque_normalizer = create_dataloaders(
        data_dir=data_dir,
        batch_size=batch_size,
        seq_len=seq_len,
        stride=stride,
        seed=seed,
    )
    
    # Create model
    print(f"\nCreating model with hidden_dim={hidden_dim}, dt={dt}...")
    model = create_model(
        model_type='sequence',
        hidden_dim=hidden_dim,
        dt=dt,
    ).to(device)
    
    num_params = sum(p.numel() for p in model.parameters())
    print(f"Model parameters: {num_params:,}")
    
    # Optimizer and scheduler
    optimizer = AdamW(model.parameters(), lr=lr, weight_decay=weight_decay)
    
    if scheduler_type == 'cosine':
        scheduler = CosineAnnealingLR(optimizer, T_max=epochs, eta_min=lr * 0.01)
    elif scheduler_type == 'plateau':
        scheduler = ReduceLROnPlateau(optimizer, mode='min', factor=0.5, patience=10)
    else:
        scheduler = None
    
    # Training loop
    best_val_loss = float('inf')
    history = {'train': [], 'val': []}
    
    print(f"\nStarting training for {epochs} epochs...")
    start_time = time.time()
    
    for epoch in range(1, epochs + 1):
        # Train
        train_metrics = train_epoch(
            model, train_loader, optimizer, device, epoch, gradient_clip
        )
        
        # Validate
        val_metrics = validate(model, val_loader, device)
        
        # Update scheduler
        if scheduler is not None:
            if scheduler_type == 'plateau':
                scheduler.step(val_metrics['loss'])
            else:
                scheduler.step()
        
        # Log
        history['train'].append(train_metrics)
        history['val'].append(val_metrics)
        
        current_lr = optimizer.param_groups[0]['lr']
        print(f"Epoch {epoch:3d} | Train Loss: {train_metrics['loss']:.4f} | "
              f"Val Loss: {val_metrics['loss']:.4f} | LR: {current_lr:.2e}")
        
        # Save best model
        if val_metrics['loss'] < best_val_loss:
            best_val_loss = val_metrics['loss']
            save_checkpoint(
                model, optimizer, scheduler, epoch,
                train_metrics, val_metrics,
                state_normalizer, torque_normalizer,
                config,
                output_dir / 'best_model.pt',
            )
            print(f"  -> New best model saved!")
        
        # Save periodic checkpoint
        if epoch % 10 == 0:
            save_checkpoint(
                model, optimizer, scheduler, epoch,
                train_metrics, val_metrics,
                state_normalizer, torque_normalizer,
                config,
                output_dir / f'checkpoint_epoch{epoch}.pt',
            )
    
    # Save final model
    save_checkpoint(
        model, optimizer, scheduler, epochs,
        train_metrics, val_metrics,
        state_normalizer, torque_normalizer,
        config,
        output_dir / 'final_model.pt',
    )
    
    # Save history
    with open(output_dir / 'history.json', 'w') as f:
        json.dump(history, f, indent=2)
    
    elapsed = time.time() - start_time
    print(f"\nTraining complete in {elapsed/60:.1f} minutes")
    print(f"Best validation loss: {best_val_loss:.4f}")
    print(f"Models saved to {output_dir}")
    
    return model, history


def main():
    parser = argparse.ArgumentParser(description='Train Neural ODE robot model')
    parser.add_argument('--data_dir', type=str, default='datasets/baxter',
                        help='Directory containing CSV data files')
    parser.add_argument('--output_dir', type=str, default='checkpoints/neural_ode_opus',
                        help='Directory to save checkpoints')
    parser.add_argument('--epochs', type=int, default=100,
                        help='Number of training epochs')
    parser.add_argument('--batch_size', type=int, default=32,
                        help='Batch size')
    parser.add_argument('--seq_len', type=int, default=50,
                        help='Sequence length')
    parser.add_argument('--stride', type=int, default=10,
                        help='Stride between sequences')
    parser.add_argument('--hidden_dim', type=int, default=128,
                        help='Hidden dimension of the network')
    parser.add_argument('--dt', type=float, default=0.002,
                        help='Time step for ODE integration (1/sampling_rate). For 500 Hz: dt=0.002')
    parser.add_argument('--lr', type=float, default=1e-3,
                        help='Learning rate')
    parser.add_argument('--weight_decay', type=float, default=1e-4,
                        help='Weight decay')
    parser.add_argument('--gradient_clip', type=float, default=1.0,
                        help='Gradient clipping norm')
    parser.add_argument('--scheduler', type=str, default='cosine',
                        choices=['cosine', 'plateau', 'none'],
                        help='Learning rate scheduler')
    parser.add_argument('--seed', type=int, default=42,
                        help='Random seed')
    parser.add_argument('--device', type=str, default=None,
                        help='Device to use (cuda/cpu)')
    
    args = parser.parse_args()
    
    train(
        data_dir=args.data_dir,
        output_dir=args.output_dir,
        epochs=args.epochs,
        batch_size=args.batch_size,
        seq_len=args.seq_len,
        stride=args.stride,
        hidden_dim=args.hidden_dim,
        dt=args.dt,
        lr=args.lr,
        weight_decay=args.weight_decay,
        gradient_clip=args.gradient_clip,
        scheduler_type=args.scheduler,
        seed=args.seed,
        device=args.device,
    )


if __name__ == '__main__':
    main()
