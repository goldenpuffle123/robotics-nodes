"""Training script for Neural ODE model."""

import os
import argparse
import json
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
from torch.optim import AdamW
from torch.optim.lr_scheduler import CosineAnnealingLR
from tqdm import tqdm

from neural_ode_robot import NeuralODE
from data import create_dataloaders, Normalizer


def train_epoch(model, loader, optimizer, device, epoch, grad_clip=1.0):
    model.train()
    total_loss, total_ang, total_vel, n = 0, 0, 0, 0
    
    pbar = tqdm(loader, desc=f"Epoch {epoch} [Train]")
    for init, torques, targets in pbar:
        init, torques, targets = init.to(device), torques.to(device), targets.to(device)
        
        optimizer.zero_grad()
        pred = model(init, torques)
        
        ang_loss = nn.functional.mse_loss(pred[..., :7], targets[..., :7])
        vel_loss = nn.functional.mse_loss(pred[..., 7:], targets[..., 7:])
        loss = ang_loss + 0.5 * vel_loss
        
        if torch.isnan(loss):
            continue
        
        loss.backward()
        
        if grad_clip > 0:
            nn.utils.clip_grad_norm_(model.parameters(), grad_clip)
        
        optimizer.step()
        
        total_loss += loss.item()
        total_ang += ang_loss.item()
        total_vel += vel_loss.item()
        n += 1
        
        pbar.set_postfix(loss=f'{loss.item():.4f}')
    
    return {'loss': total_loss/n, 'ang': total_ang/n, 'vel': total_vel/n}


@torch.no_grad()
def validate(model, loader, device):
    model.eval()
    total_loss, total_ang, total_vel, n = 0, 0, 0, 0
    
    for init, torques, targets in loader:
        init, torques, targets = init.to(device), torques.to(device), targets.to(device)
        pred = model(init, torques)
        
        ang_loss = nn.functional.mse_loss(pred[..., :7], targets[..., :7])
        vel_loss = nn.functional.mse_loss(pred[..., 7:], targets[..., 7:])
        loss = ang_loss + 0.5 * vel_loss
        
        total_loss += loss.item()
        total_ang += ang_loss.item()
        total_vel += vel_loss.item()
        n += 1
    
    return {'loss': total_loss/n, 'ang': total_ang/n, 'vel': total_vel/n}


def save_checkpoint(model, optimizer, scheduler, epoch, metrics, state_norm, torque_norm, config, path):
    torch.save({
        'epoch': epoch,
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'scheduler_state_dict': scheduler.state_dict() if scheduler else None,
        'metrics': metrics,
        'state_normalizer': state_norm.to_dict(),
        'torque_normalizer': torque_norm.to_dict(),
        'config': config,
    }, path)


def load_checkpoint(checkpoint_path, model, optimizer=None, scheduler=None, device='cpu'):
    """Load model and training state from checkpoint."""
    checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=False)
    model.load_state_dict(checkpoint['model_state_dict'])
    
    start_epoch = checkpoint['epoch']
    
    if optimizer is not None and 'optimizer_state_dict' in checkpoint:
        optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
    
    if scheduler is not None and checkpoint.get('scheduler_state_dict') is not None:
        scheduler.load_state_dict(checkpoint['scheduler_state_dict'])
    
    best_val = checkpoint.get('metrics', {}).get('loss', float('inf'))
    
    print(f"Loaded checkpoint from epoch {start_epoch}, best val loss: {best_val:.6f}")
    return start_epoch, best_val


def train(
    data_dir='datasets/baxter',
    output_dir='checkpoints/modelv3',
    epochs=100,
    batch_size=32,
    seq_len=50,
    stride=10,
    hidden_dim=128,
    dt=0.002,
    lr=1e-3,
    weight_decay=1e-4,
    grad_clip=1.0,
    seed=42,
    device=None,
    resume=None,
):
    device = device or ('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Device: {device}")
    
    torch.manual_seed(seed)
    np.random.seed(seed)
    
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    config = dict(
        hidden_dim=hidden_dim, dt=dt, seq_len=seq_len, batch_size=batch_size,
        lr=lr, weight_decay=weight_decay, epochs=epochs,
    )
    
    with open(output_dir / 'config.json', 'w') as f:
        json.dump(config, f, indent=2)
    
    train_loader, val_loader, state_norm, torque_norm = create_dataloaders(
        data_dir, batch_size, seq_len, stride, seed=seed
    )
    
    model = NeuralODE(dof=7, hidden_dim=hidden_dim, dt=dt).to(device)
    print(f"Parameters: {sum(p.numel() for p in model.parameters()):,}")
    
    optimizer = AdamW(model.parameters(), lr=lr, weight_decay=weight_decay)
    scheduler = CosineAnnealingLR(optimizer, T_max=epochs, eta_min=lr * 0.01)
    
    # Resume from checkpoint if specified
    start_epoch = 1
    best_val = float('inf')
    history = {'train': [], 'val': []}
    
    if resume:
        start_epoch, best_val = load_checkpoint(resume, model, optimizer, scheduler, device)
        start_epoch += 1  # Start from next epoch
        # Load history if exists
        history_path = output_dir / 'history.json'
        if history_path.exists():
            with open(history_path) as f:
                history = json.load(f)
    
    for epoch in range(start_epoch, epochs + 1):
        train_m = train_epoch(model, train_loader, optimizer, device, epoch, grad_clip)
        val_m = validate(model, val_loader, device)
        scheduler.step()
        
        history['train'].append(train_m)
        history['val'].append(val_m)
        
        print(f"Epoch {epoch:3d} | Train: {train_m['loss']:.4f} | Val: {val_m['loss']:.4f}")
        
        if val_m['loss'] < best_val:
            best_val = val_m['loss']
            save_checkpoint(model, optimizer, scheduler, epoch, val_m, state_norm, torque_norm, config, output_dir / 'best_model.pt')
            print("  -> Saved best model")
        
        if epoch % 10 == 0:
            save_checkpoint(model, optimizer, scheduler, epoch, val_m, state_norm, torque_norm, config, output_dir / f'checkpoint_{epoch}.pt')
    
    save_checkpoint(model, optimizer, scheduler, epochs, val_m, state_norm, torque_norm, config, output_dir / 'final_model.pt')
    
    with open(output_dir / 'history.json', 'w') as f:
        json.dump(history, f, indent=2)
    
    print(f"Done. Best val: {best_val:.4f}")
    return model, history


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--data_dir', default='datasets/baxter')
    parser.add_argument('--output_dir', default='checkpoints/modelv3')
    parser.add_argument('--epochs', type=int, default=100)
    parser.add_argument('--batch_size', type=int, default=32)
    parser.add_argument('--seq_len', type=int, default=50)
    parser.add_argument('--stride', type=int, default=10)
    parser.add_argument('--hidden_dim', type=int, default=128)
    parser.add_argument('--dt', type=float, default=0.002)
    parser.add_argument('--lr', type=float, default=1e-3)
    parser.add_argument('--device', default=None)
    parser.add_argument('--resume', type=str, default=None, help='Path to checkpoint to resume from')
    args = parser.parse_args()
    
    train(**vars(args))


if __name__ == '__main__':
    main()
