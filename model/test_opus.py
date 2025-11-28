"""
Testing and evaluation script for Neural ODE robot model.
Computes metrics and generates visualizations of predictions.
"""

import os
import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import matplotlib.pyplot as plt
from tqdm import tqdm

from neural_ode_robot_opus import create_model
from data_processing_opus import (
    RobotTrajectoryDataset, Normalizer,
    STATE_COLS, ANGLE_COLS, VELOCITY_COLS, TORQUE_COLS
)
from train_opus import load_checkpoint


@torch.no_grad()
def evaluate_model(
    model,
    test_loader,
    state_normalizer: Normalizer,
    device,
) -> dict:
    """
    Evaluate model on test data.
    
    Returns:
        metrics: Dictionary of evaluation metrics
    """
    model.eval()
    
    all_pred_states = []
    all_target_states = []
    all_initial_states = []
    
    for initial_state, torques, target_states in tqdm(test_loader, desc="Evaluating"):
        initial_state = initial_state.to(device)
        torques = torques.to(device)
        target_states = target_states.to(device)
        
        # Predict
        pred_states = model(initial_state, torques)[:, 1:, :]
        
        all_pred_states.append(pred_states.cpu().numpy())
        all_target_states.append(target_states.cpu().numpy())
        all_initial_states.append(initial_state.cpu().numpy())
    
    # Concatenate all batches
    pred_states = np.concatenate(all_pred_states, axis=0)
    target_states = np.concatenate(all_target_states, axis=0)
    initial_states = np.concatenate(all_initial_states, axis=0)
    
    # Denormalize for interpretable metrics
    pred_states_denorm = state_normalizer.inverse_transform(
        pred_states.reshape(-1, 14)
    ).reshape(pred_states.shape)
    target_states_denorm = state_normalizer.inverse_transform(
        target_states.reshape(-1, 14)
    ).reshape(target_states.shape)
    
    # Compute metrics
    metrics = {}
    
    # MSE (normalized)
    metrics['mse_normalized'] = float(np.mean((pred_states - target_states) ** 2))
    
    # MSE (denormalized, interpretable)
    metrics['mse_denormalized'] = float(np.mean((pred_states_denorm - target_states_denorm) ** 2))
    
    # Separate angle and velocity metrics
    angle_mse = np.mean((pred_states_denorm[..., :7] - target_states_denorm[..., :7]) ** 2)
    vel_mse = np.mean((pred_states_denorm[..., 7:] - target_states_denorm[..., 7:]) ** 2)
    metrics['angle_mse'] = float(angle_mse)
    metrics['velocity_mse'] = float(vel_mse)
    
    # RMSE
    metrics['angle_rmse'] = float(np.sqrt(angle_mse))
    metrics['velocity_rmse'] = float(np.sqrt(vel_mse))
    
    # MAE
    metrics['angle_mae'] = float(np.mean(np.abs(pred_states_denorm[..., :7] - target_states_denorm[..., :7])))
    metrics['velocity_mae'] = float(np.mean(np.abs(pred_states_denorm[..., 7:] - target_states_denorm[..., 7:])))
    
    # Per-step error (how error accumulates over sequence)
    step_errors = np.sqrt(np.mean((pred_states_denorm - target_states_denorm) ** 2, axis=(0, 2)))
    metrics['step_errors'] = step_errors.tolist()
    
    # Per-joint angle RMSE
    joint_angle_rmse = np.sqrt(np.mean((pred_states_denorm[..., :7] - target_states_denorm[..., :7]) ** 2, axis=(0, 1)))
    for i, name in enumerate(ANGLE_COLS):
        metrics[f'{name}_rmse'] = float(joint_angle_rmse[i])
    
    return metrics, pred_states_denorm, target_states_denorm


def plot_trajectory_comparison(
    pred_states: np.ndarray,
    target_states: np.ndarray,
    sample_idx: int = 0,
    save_path: str = None,
):
    """
    Plot predicted vs actual trajectory for a single sample.
    
    Args:
        pred_states: [batch, seq_len, state_dim]
        target_states: [batch, seq_len, state_dim]
        sample_idx: Which sample to plot
        save_path: Path to save figure
    """
    pred = pred_states[sample_idx]
    target = target_states[sample_idx]
    seq_len = pred.shape[0]
    t = np.arange(seq_len)
    
    fig, axes = plt.subplots(2, 4, figsize=(16, 8))
    
    # Plot joint angles
    for i in range(7):
        ax = axes[0, i] if i < 4 else axes[1, i - 4]
        ax.plot(t, target[:, i], 'b-', label='Ground Truth', linewidth=2)
        ax.plot(t, pred[:, i], 'r--', label='Predicted', linewidth=2)
        ax.set_xlabel('Time Step')
        ax.set_ylabel(f'{ANGLE_COLS[i]} (rad)')
        ax.set_title(ANGLE_COLS[i])
        ax.legend()
        ax.grid(True, alpha=0.3)
    
    # Hide the last subplot
    axes[1, 3].axis('off')
    
    plt.suptitle('Joint Angle Trajectories: Predicted vs Ground Truth', fontsize=14)
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        print(f"Saved trajectory plot to {save_path}")
    
    plt.show()


def plot_error_over_time(
    step_errors: list,
    save_path: str = None,
):
    """Plot how prediction error grows over time."""
    fig, ax = plt.subplots(figsize=(10, 6))
    
    t = np.arange(len(step_errors))
    ax.plot(t, step_errors, 'b-', linewidth=2)
    ax.fill_between(t, 0, step_errors, alpha=0.3)
    
    ax.set_xlabel('Time Step', fontsize=12)
    ax.set_ylabel('RMSE (denormalized)', fontsize=12)
    ax.set_title('Prediction Error Accumulation Over Time', fontsize=14)
    ax.grid(True, alpha=0.3)
    
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        print(f"Saved error plot to {save_path}")
    
    plt.show()


def plot_joint_errors(
    metrics: dict,
    save_path: str = None,
):
    """Plot per-joint RMSE."""
    joint_rmse = [metrics[f'{name}_rmse'] for name in ANGLE_COLS]
    
    fig, ax = plt.subplots(figsize=(10, 6))
    
    bars = ax.bar(ANGLE_COLS, joint_rmse, color='steelblue', edgecolor='black')
    
    ax.set_xlabel('Joint', fontsize=12)
    ax.set_ylabel('RMSE (rad)', fontsize=12)
    ax.set_title('Per-Joint Angle Prediction RMSE', fontsize=14)
    ax.grid(True, alpha=0.3, axis='y')
    
    # Add value labels on bars
    for bar, val in zip(bars, joint_rmse):
        ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.001,
                f'{val:.4f}', ha='center', va='bottom', fontsize=10)
    
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        print(f"Saved joint error plot to {save_path}")
    
    plt.show()


def run_single_prediction(
    model,
    initial_state: np.ndarray,
    torques: np.ndarray,
    state_normalizer: Normalizer,
    torque_normalizer: Normalizer,
    device,
) -> np.ndarray:
    """
    Run a single prediction from initial state with given torques.
    
    Args:
        model: Trained model
        initial_state: [14] array of initial state (denormalized)
        torques: [seq_len, 7] array of torques (denormalized)
        state_normalizer: State normalizer
        torque_normalizer: Torque normalizer
        device: Device
    
    Returns:
        pred_states: [seq_len+1, 14] predicted states (denormalized)
    """
    model.eval()
    
    # Normalize inputs
    initial_state_norm = state_normalizer.transform(initial_state.reshape(1, -1))
    torques_norm = torque_normalizer.transform(torques)
    
    # Convert to tensors
    initial_state_t = torch.from_numpy(initial_state_norm).float().to(device)
    torques_t = torch.from_numpy(torques_norm).float().unsqueeze(0).to(device)
    
    # Predict
    with torch.no_grad():
        pred_states = model(initial_state_t, torques_t)
    
    # Denormalize
    pred_states_np = pred_states[0].cpu().numpy()
    pred_states_denorm = state_normalizer.inverse_transform(pred_states_np)
    
    return pred_states_denorm


def test_on_csv(
    model,
    csv_path: str,
    state_normalizer: Normalizer,
    torque_normalizer: Normalizer,
    device,
    seq_len: int = 100,
    start_idx: int = 0,
) -> dict:
    """
    Test model on a specific CSV file.
    
    Returns:
        results: Dictionary with predictions and ground truth
    """
    df = pd.read_csv(csv_path, sep=" ")
    
    # Extract data
    states = df[STATE_COLS].values.astype(np.float32)
    torques = df[TORQUE_COLS].values.astype(np.float32)
    
    # Get initial state and torques for prediction
    initial_state = states[start_idx]
    torque_seq = torques[start_idx:start_idx + seq_len - 1]
    target_states = states[start_idx:start_idx + seq_len]
    
    # Predict
    pred_states = run_single_prediction(
        model, initial_state, torque_seq,
        state_normalizer, torque_normalizer, device
    )
    
    return {
        'predictions': pred_states,
        'ground_truth': target_states,
        'torques': torque_seq,
        'initial_state': initial_state,
    }


def main():
    parser = argparse.ArgumentParser(description='Test Neural ODE robot model')
    parser.add_argument('--checkpoint', type=str, required=True,
                        help='Path to model checkpoint')
    parser.add_argument('--data_dir', type=str, default='datasets/baxter',
                        help='Directory containing CSV data files')
    parser.add_argument('--output_dir', type=str, default='results/test_opus',
                        help='Directory to save results')
    parser.add_argument('--seq_len', type=int, default=50,
                        help='Sequence length for testing')
    parser.add_argument('--batch_size', type=int, default=32,
                        help='Batch size')
    parser.add_argument('--device', type=str, default=None,
                        help='Device to use')
    parser.add_argument('--num_plot_samples', type=int, default=3,
                        help='Number of samples to plot')
    
    args = parser.parse_args()
    
    # Setup
    if args.device is None:
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    else:
        device = torch.device(args.device)
    print(f"Using device: {device}")
    
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Load model
    print(f"Loading checkpoint from {args.checkpoint}...")
    model, state_normalizer, torque_normalizer, checkpoint = load_checkpoint(
        args.checkpoint, device
    )
    print(f"Loaded model from epoch {checkpoint['epoch']}")
    
    # Create test dataloader
    print(f"\nLoading test data from {args.data_dir}...")
    test_dataset = RobotTrajectoryDataset(
        data_dir=args.data_dir,
        seq_len=args.seq_len,
        stride=args.seq_len,  # No overlap for testing
        state_normalizer=state_normalizer,
        torque_normalizer=torque_normalizer,
        fit_normalizers=False,
    )
    
    test_loader = torch.utils.data.DataLoader(
        test_dataset,
        batch_size=args.batch_size,
        shuffle=False,
    )
    
    # Evaluate
    print("\nEvaluating model...")
    metrics, pred_states, target_states = evaluate_model(
        model, test_loader, state_normalizer, device
    )
    
    # Print metrics
    print("\n" + "=" * 50)
    print("EVALUATION RESULTS")
    print("=" * 50)
    print(f"Angle RMSE: {metrics['angle_rmse']:.6f} rad ({np.rad2deg(metrics['angle_rmse']):.4f} deg)")
    print(f"Velocity RMSE: {metrics['velocity_rmse']:.6f} rad/s")
    print(f"Angle MAE: {metrics['angle_mae']:.6f} rad ({np.rad2deg(metrics['angle_mae']):.4f} deg)")
    print(f"Velocity MAE: {metrics['velocity_mae']:.6f} rad/s")
    print("\nPer-joint angle RMSE:")
    for name in ANGLE_COLS:
        print(f"  {name}: {metrics[f'{name}_rmse']:.6f} rad")
    
    # Save metrics
    with open(output_dir / 'metrics.json', 'w') as f:
        # Convert numpy types to native Python types
        metrics_save = {k: v if not isinstance(v, np.ndarray) else v.tolist()
                       for k, v in metrics.items()}
        json.dump(metrics_save, f, indent=2)
    print(f"\nMetrics saved to {output_dir / 'metrics.json'}")
    
    # Generate plots
    print("\nGenerating plots...")
    
    # Plot trajectory comparisons
    for i in range(min(args.num_plot_samples, len(pred_states))):
        plot_trajectory_comparison(
            pred_states, target_states, sample_idx=i,
            save_path=output_dir / f'trajectory_sample_{i}.png'
        )
    
    # Plot error over time
    plot_error_over_time(
        metrics['step_errors'],
        save_path=output_dir / 'error_over_time.png'
    )
    
    # Plot per-joint errors
    plot_joint_errors(
        metrics,
        save_path=output_dir / 'joint_errors.png'
    )
    
    print(f"\nResults saved to {output_dir}")


if __name__ == '__main__':
    main()
