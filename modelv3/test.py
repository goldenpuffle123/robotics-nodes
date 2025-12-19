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
from fk import calculate_fk



def load_model(path, device='cpu'): # Load model from config in checkpoint
    ckpt = torch.load(path, map_location=device, weights_only=False)
    config = ckpt.get('config', {})
    
    model = NeuralODE(
        dof=7,
        hidden_dim=config.get('hidden_dim', 128),
        dt=config.get('dt', 0.002),
        integrator=config.get('integrator', 'euler'),
    )
    model.load_state_dict(ckpt['model_state_dict'])
    model.to(device)
    model.eval()
    
    return model, config


def angles_to_ee(angles):
    """Convert joint angles to end effector positions."""
    return np.array([calculate_fk(a)[:3, 3] for a in angles])


def plot_ee_trajectories(pred_ang, gt_ang, title="EE Trajectory", save_path=None):
    """Plot predicted vs ground truth end effector trajectories for one model."""
    pred_ee = angles_to_ee(pred_ang) # Predicted (from model)
    gt_ee = angles_to_ee(gt_ang) # Ground truth
    
    fig = plt.figure(figsize=(14, 5))
    
    # 3D plot of predicted vs ground truth
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


def plot_multi_model_trajectories(model_predictions, gt_ang, model_names, title="EE Trajectory Comparison", save_path=None):
    """
    Plot multiple model predictions against ground truth on the same graph.
    
    Args:
        model_predictions: List of predicted angle arrays, each [seq_len, 7]
        gt_ang: Ground truth angles [seq_len, 7]
        model_names: List of names for each model (e.g., ['Model v1', 'Model v2'])
        title: Plot title
        save_path: Optional path to save the figure
    """
    # Convert all to EE positions
    gt_ee = angles_to_ee(gt_ang)
    pred_ees = [angles_to_ee(pred) for pred in model_predictions]
    
    # Colors for different models (up to 10 models)
    colors = plt.cm.tab10(np.linspace(0, 1, len(model_predictions)))
    markers = ['^', 's', 'D', 'v', 'p', '*', 'h', 'X', 'P', 'o']
    
    fig = plt.figure(figsize=(10, 5))
    
    # 3D plot
    ax1 = fig.add_subplot(121, projection='3d')
    ax1.plot(gt_ee[:, 0], gt_ee[:, 1], gt_ee[:, 2], 'b-', lw=2.5, label='Ground Truth')
    ax1.scatter(*gt_ee[0], c='g', s=100, marker='o', label='Start')
    ax1.scatter(*gt_ee[-1], c='b', s=100, marker='x', label='End (GT)')
    
    for i, (pred_ee, name, color) in enumerate(zip(pred_ees, model_names, colors)):
        ax1.plot(pred_ee[:, 0], pred_ee[:, 1], pred_ee[:, 2], '--', color=color, lw=1.5, label=name)
        ax1.scatter(*pred_ee[-1], c=[color], s=80, marker=markers[i % len(markers)])
    
    ax1.set_xlabel('X'); ax1.set_ylabel('Y'); ax1.set_zlabel('Z')
    ax1.legend(fontsize=7, loc='upper left')
    ax1.set_title('3D Trajectory')
    
    # XYZ over time (show only one axis at a time for clarity, or all GT + one model)
    t = np.arange(len(gt_ee))
    """ ax2 = fig.add_subplot(132)
    
    # Plot GT components
    for j, (c, label) in enumerate(zip(['b', 'g', 'r'], ['X', 'Y', 'Z'])):
        ax2.plot(t, gt_ee[:, j], f'{c}-', lw=2, label=f'GT {label}')
    
    # Plot each model's components with dashed lines
    for i, (pred_ee, name, color) in enumerate(zip(pred_ees, model_names, colors)):
        for j, label in enumerate(['X', 'Y', 'Z']):
            linestyle = ['--', '-.', ':'][j]
            ax2.plot(t, pred_ee[:, j], linestyle=linestyle, color=color, lw=1, 
                     label=f'{name} {label}' if j == 0 else None, alpha=0.7)
    
    ax2.set_xlabel('Timestep'); ax2.set_ylabel('Position (m)')
    ax2.legend(fontsize=6, ncol=2); ax2.grid(True, alpha=0.3)
    ax2.set_title('Position Components') """
    
    # Error comparison
    ax3 = fig.add_subplot(122)
    
    error_stats = []
    for i, (pred_ee, name, color) in enumerate(zip(pred_ees, model_names, colors)):
        err = np.linalg.norm(pred_ee - gt_ee, axis=1) * 1000  # mm
        ax3.plot(t, err, '-', color=color, lw=1.5, label=f'{name} (mean: {err.mean():.1f}mm)')
        error_stats.append((name, err.mean(), err.max()))
    
    ax3.set_xlabel('Timestep'); ax3.set_ylabel('Error (mm)')
    ax3.legend(fontsize=7)
    ax3.grid(True, alpha=0.3)
    ax3.set_title('Position Error Comparison')
    
    plt.suptitle(title, fontweight='bold')
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.show()
    
    # Print error summary
    print("\n" + "="*50)
    print("Error Summary (mm)")
    print("="*50)
    print(f"{'Model':<20} {'Mean':>10} {'Max':>10}")
    print("-"*50)
    for name, mean_err, max_err in error_stats:
        print(f"{name:<20} {mean_err:>10.2f} {max_err:>10.2f}")
    print("="*50)


import time

def test(checkpoint, data_dir='datasets/baxter', seq_len=50, num_seq=5, device='cpu'):
    """Test a single model and visualize predictions."""
    model, _ = load_model(checkpoint, device)
    _, _, test_loader, _, _ = create_dataloaders(data_dir, batch_size=8, seq_len=seq_len, stride=seq_len, seed=42)
    
    total_time = 0
    total_steps = 0
    
    for i, (init, torques, targets) in enumerate(test_loader):
        if i >= num_seq:
            break
        
        init_d = init[0:1].to(device)
        torq_d = torques[0:1].to(device)
        
        # Warmup
        if i == 0:
            with torch.no_grad():
                _ = model(init_d, torq_d)
        
        start_time = time.perf_counter()
        with torch.no_grad():
            pred = model(init_d, torq_d)[0].cpu().numpy()
        end_time = time.perf_counter()
        
        total_time += (end_time - start_time)
        total_steps += seq_len
        
        # States are already in physical units (no denormalization needed)
        init_np = init[0].numpy()
        target_np = targets[0].numpy()
        
        # Full trajectories
        pred_full = np.vstack([init_np[None], pred])
        gt_full = np.vstack([init_np[None], target_np])
        
        plot_ee_trajectories(pred_full[:, :7], gt_full[:, :7], title=f'Sequence {i}')
        
    avg_time_per_step = (total_time / total_steps) * 1000 if total_steps > 0 else 0
    print(f"\nInference Speed: {avg_time_per_step:.4f} ms/step (Total: {total_time:.4f}s for {total_steps} steps)")


def compare_models(checkpoints, model_names=None, data_dir='datasets/baxter', 
                   seq_len=50, num_seq=5, device='cpu', save_dir=None):
    """
    Compare multiple models on the same sequences.
    
    Args:
        checkpoints: List of checkpoint paths
        model_names: List of names for each model (optional, uses filenames if not provided)
        data_dir: Path to dataset
        seq_len: Sequence length
        num_seq: Number of sequences to visualize
        device: Device to run on
        save_dir: Optional directory to save figures
    """
    # Load all models
    models = []
    for ckpt_path in checkpoints:
        model, _ = load_model(ckpt_path, device)
        models.append(model)
    
    # Generate model names if not provided
    if model_names is None:
        model_names = []
        for p in checkpoints:
            # Extract meaningful name from path
            parts = p.replace('\\', '/').split('/')
            if 'checkpoint' in parts[-1].lower() or 'best' in parts[-1].lower():
                # Use parent folder + filename
                name = f"{parts[-2]}/{parts[-1]}" if len(parts) > 1 else parts[-1]
            else:
                name = parts[-1]
            # Remove .pt extension
            name = name.replace('.pt', '')
            model_names.append(name)
    
    # Create save directory if needed
    if save_dir:
        os.makedirs(save_dir, exist_ok=True)
    
    # Load data (use test split - unseen during training/validation)
    _, _, test_loader, _, _ = create_dataloaders(data_dir, batch_size=8, seq_len=seq_len, stride=seq_len, seed=42)
    
    # Timing stats
    model_times = {name: 0.0 for name in model_names}
    total_steps = 0
    
    for i, (init, torques, targets) in enumerate(test_loader):
        if i >= num_seq:
            break
        
        init_d = init[0:1].to(device)
        torq_d = torques[0:1].to(device)
        init_np = init[0].numpy()
        target_np = targets[0].numpy()
        
        # Warmup on first iteration
        if i == 0:
            for model in models:
                with torch.no_grad():
                    _ = model(init_d, torq_d)
        
        # Get predictions from all models
        predictions = []
        for model, name in zip(models, model_names):
            start_time = time.perf_counter()
            with torch.no_grad():
                pred = model(init_d, torq_d)[0].cpu().numpy()
            end_time = time.perf_counter()
            
            model_times[name] += (end_time - start_time)
            
            # Full trajectory including initial state
            pred_full = np.vstack([init_np[None], pred])
            predictions.append(pred_full[:, :7])  # Only angles
            
        total_steps += seq_len
        
        # Ground truth full trajectory
        gt_full = np.vstack([init_np[None], target_np])
        
        # Plot comparison
        save_path = os.path.join(save_dir, f'comparison_seq{i}.pdf') if save_dir else None
        plot_multi_model_trajectories(
            predictions, 
            gt_full[:, :7],
            model_names,
            title=f'Model Comparison - Sequence {i}',
            save_path=save_path
        )
        
    # Print timing summary
    print("\n" + "="*60)
    print("Inference Speed Comparison")
    print("="*60)
    print(f"{'Model':<25} {'Time/Step (ms)':>15} {'Total (s)':>15}")
    print("-"*60)
    for name in model_names:
        avg_ms = (model_times[name] / total_steps) * 1000 if total_steps > 0 else 0
        print(f"{name:<25} {avg_ms:>15.4f} {model_times[name]:>15.4f}")
    print("="*60)



def main():
    parser = argparse.ArgumentParser(description='Test Neural ODE models on EE trajectory prediction')
    subparsers = parser.add_subparsers(dest='command', help='Commands')
    
    # Single model test arguments
    single_parser = subparsers.add_parser('single', help='Test a single model')
    single_parser.add_argument('--checkpoint', required=True, help='Path to model checkpoint')
    single_parser.add_argument('--data_dir', default='datasets/baxter')
    single_parser.add_argument('--seq_len', type=int, default=50)
    single_parser.add_argument('--num_sequences', type=int, default=5)
    single_parser.add_argument('--device', default='cuda' if torch.cuda.is_available() else 'cpu')
    
    # Multi-model comparison arguments
    compare_parser = subparsers.add_parser('compare', help='Compare multiple models')
    compare_parser.add_argument('--checkpoints', nargs='+', required=True, 
                                help='Paths to model checkpoints (space-separated)')
    compare_parser.add_argument('--names', nargs='+', default=None,
                                help='Names for each model (optional, space-separated)')
    compare_parser.add_argument('--data_dir', default='datasets/baxter')
    compare_parser.add_argument('--seq_len', type=int, default=50)
    compare_parser.add_argument('--num_sequences', type=int, default=5)
    compare_parser.add_argument('--device', default='cuda' if torch.cuda.is_available() else 'cpu')
    compare_parser.add_argument('--save_dir', default=None, help='Directory to save figures')
    
    args = parser.parse_args()
    
    if args.command == 'single':
        test(args.checkpoint, args.data_dir, args.seq_len, args.num_sequences, args.device)
    elif args.command == 'compare':
        compare_models(args.checkpoints, args.names, args.data_dir, 
                       args.seq_len, args.num_sequences, args.device, args.save_dir)
    else:
        raise ValueError("Please specify a command: 'single' or 'compare'")


if __name__ == '__main__':
    main()
