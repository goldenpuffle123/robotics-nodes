"""Test script for visualizing predicted vs ground-truth EE trajectories."""

import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import torch
import numpy as np
import matplotlib.pyplot as plt
import argparse

from neural_ode_robot import NeuralODE
from data import create_dataloaders, Normalizer
from vis.fk import calculate_fk


def load_model(path, device='cpu'):
    ckpt = torch.load(path, map_location=device, weights_only=False)
    config = ckpt.get('config', {})
    
    model = NeuralODE(
        dof=7,
        hidden_dim=config.get('hidden_dim', 128),
        dt=config.get('dt', 0.002),
    )
    model.load_state_dict(ckpt['model_state_dict'])
    model.to(device)
    model.eval()
    
    return model, config


def angles_to_ee(angles):
    """Convert joint angles to EE positions."""
    return np.array([calculate_fk(a)[:3, 3] for a in angles])


def plot_ee_trajectories(pred_ang, gt_ang, title="EE Trajectory", save_path=None):
    pred_ee = angles_to_ee(pred_ang)
    gt_ee = angles_to_ee(gt_ang)
    
    fig = plt.figure(figsize=(14, 5))
    
    # 3D plot
    ax1 = fig.add_subplot(131, projection='3d')
    ax1.plot(gt_ee[:, 0], gt_ee[:, 1], gt_ee[:, 2], 'b-', lw=2, label='Ground Truth')
    ax1.plot(pred_ee[:, 0], pred_ee[:, 1], pred_ee[:, 2], 'r--', lw=2, label='Predicted')
    ax1.scatter(*gt_ee[0], c='g', s=100, marker='o', label='Start')
    ax1.scatter(*gt_ee[-1], c='b', s=100, marker='x', label='End (GT)')
    ax1.scatter(*pred_ee[-1], c='r', s=100, marker='^', label='End (Pred)')
    ax1.set_xlabel('X'); ax1.set_ylabel('Y'); ax1.set_zlabel('Z')
    ax1.legend(fontsize=8)
    ax1.set_title('3D Trajectory')
    
    # XYZ over time
    t = np.arange(len(gt_ee))
    ax2 = fig.add_subplot(132)
    for i, (c, l) in enumerate(zip(['b', 'g', 'r'], ['X', 'Y', 'Z'])):
        ax2.plot(t, gt_ee[:, i], f'{c}-', label=f'GT {l}')
        ax2.plot(t, pred_ee[:, i], f'{c}--', label=f'Pred {l}')
    ax2.set_xlabel('Timestep'); ax2.set_ylabel('Position (m)')
    ax2.legend(fontsize=8); ax2.grid(True, alpha=0.3)
    ax2.set_title('Position Components')
    
    # Error
    ax3 = fig.add_subplot(133)
    err = np.linalg.norm(pred_ee - gt_ee, axis=1) * 1000
    ax3.plot(t, err, 'k-', lw=2)
    ax3.fill_between(t, 0, err, alpha=0.3)
    ax3.set_xlabel('Timestep'); ax3.set_ylabel('Error (mm)')
    ax3.set_title(f'Position Error\nMean: {err.mean():.2f}mm, Max: {err.max():.2f}mm')
    ax3.grid(True, alpha=0.3)
    
    plt.suptitle(title, fontweight='bold')
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.show()


def test(checkpoint, data_dir='datasets/baxter', seq_len=50, num_seq=5, device='cpu'):
    model, _ = load_model(checkpoint, device)
    _, val_loader, _, _ = create_dataloaders(data_dir, batch_size=8, seq_len=seq_len, stride=seq_len, seed=44)
    
    for i, (init, torques, targets) in enumerate(val_loader):
        if i >= num_seq:
            break
        
        init_d = init[0:1].to(device)
        torq_d = torques[0:1].to(device)
        
        with torch.no_grad():
            pred = model(init_d, torq_d)[0].cpu().numpy()
        
        # States are already in physical units (no denormalization needed)
        init_np = init[0].numpy()
        target_np = targets[0].numpy()
        
        # Full trajectories
        pred_full = np.vstack([init_np[None], pred])
        gt_full = np.vstack([init_np[None], target_np])
        
        plot_ee_trajectories(pred_full[:, :7], gt_full[:, :7], title=f'Sequence {i}')


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--checkpoint', required=True)
    parser.add_argument('--data_dir', default='datasets/baxter')
    parser.add_argument('--seq_len', type=int, default=50)
    parser.add_argument('--num_sequences', type=int, default=5)
    parser.add_argument('--device', default='cuda' if torch.cuda.is_available() else 'cpu')
    args = parser.parse_args()
    
    test(args.checkpoint, args.data_dir, args.seq_len, args.num_sequences, args.device)


if __name__ == '__main__':
    main()
