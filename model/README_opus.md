# Neural ODE Robot Dynamics Model (`_opus` files)

This directory contains a complete implementation of a **Neural ODE model for predicting Baxter robot dynamics**. Given an initial joint state (angles + velocities) and torque commands, the model predicts future robot states using neural ordinary differential equations (ODEs) with adjoint sensitivity for memory-efficient training.

## Overview

### What is a Neural ODE?

A Neural ODE is a continuous-time dynamical system where the derivative is parameterized by a neural network:

$$\frac{d\mathbf{x}}{dt} = f_\theta(\mathbf{x}, \mathbf{u}, t)$$

For robot dynamics:
- **State**: $\mathbf{x} = [q_1, \ldots, q_7, \dot{q}_1, \ldots, \dot{q}_7]$ (14 dims: 7 joint angles + 7 velocities)
- **Control**: $\mathbf{u} = [\tau_1, \ldots, \tau_7]$ (7 joint torques)
- **Model learns**: $[\dot{q}, \ddot{q}] = f_\theta(q, \dot{q}, \tau)$

### Why Neural ODE?

1. **Memory-efficient**: Adjoint method avoids storing intermediate activations
2. **Continuous time**: No fixed timestep in the model; ODE solver adapts
3. **Flexible horizon**: Can predict to any future time without retraining
4. **Physics-informed**: Learns smooth, differentiable state transitions

## Files

### Core Model (`neural_ode_robot_opus.py`)

**Classes:**
- `RobotDynamicsODE`: Neural network that models $d[q, \dot{q}]/dt = f(q, \dot{q}, \tau)$
  - 3-layer MLP with SiLU activations
  - Input: state (14) + torque (7) = 21 dims
  - Output: state derivatives (14 dims)
  - Physics prior: $dq/dt = \dot{q}$ (enforced)

- `NeuralODERobot`: Single forward pass with constant torque
  - Integrates ODE from initial state using `odeint_adjoint`
  - Memory-efficient backpropagation through ODE solver

- `NeuralODERobotSequence`: Step-by-step prediction with varying torques
  - Predicts one timestep at a time
  - Torque can vary at each step
  - Better for realistic trajectories

**Key Features:**
```python
# Create model
model = NeuralODERobotSequence(hidden_dim=128, dt=0.01)

# Predict trajectory
initial_state = torch.randn(batch_size, 14)
torques = torch.randn(batch_size, seq_len, 7)
predicted_states = model(initial_state, torques)  # [batch, seq_len+1, 14]
```

### Data Processing (`data_processing_opus.py`)

**Classes:**
- `Normalizer`: Fit/transform data using mean and std
  - Handles denormalization for interpretable metrics
  - Saves/loads from dict for checkpointing

- `RobotTrajectoryDataset`: Sequences of (state, torque, target) pairs
  - Loads CSV files from Baxter dataset
  - Creates fixed-length sequences with stride
  - Normalizes states and torques independently
  - Respects file boundaries (no sequences cross files)

- `RobotPredictionDataset`: Single-step (s, u, s') pairs
  - Simpler alternative for one-step predictions

**Key Functions:**
```python
# Create train/val loaders with automatic normalization
train_loader, val_loader, state_norm, torque_norm = create_dataloaders(
    data_dir='datasets/baxter',
    batch_size=32,
    seq_len=50,
    stride=10,
    train_split=0.8,
)

# Batch structure
initial_state: [batch, 14]          # Initial state
torques: [batch, seq_len-1, 7]      # Torques for seq_len-1 steps
target_states: [batch, seq_len-1, 14]  # Ground truth future states
```

### Training (`train_opus.py`)

**Main Training Loop:**
```python
python model/train_opus.py \
    --data_dir datasets/baxter \
    --epochs 100 \
    --batch_size 32 \
    --seq_len 50 \
    --hidden_dim 128 \
    --lr 1e-3 \
    --output_dir checkpoints/neural_ode_opus
```

**Key Features:**
- **Adjoint sensitivity**: Memory-efficient backprop through ODE solver
- **Loss function**: $L = MSE(\hat{q}) + 0.5 \cdot MSE(\hat{\dot{q}})$
  - Angle MSE weighted higher than velocity MSE
- **Gradient clipping**: Norm clipping to 1.0 (important for ODE gradients)
- **Learning rate scheduling**: Cosine annealing or ReduceLROnPlateau
- **Checkpointing**: Saves best model, periodic checkpoints, config, normalizers

**Arguments:**
```
--data_dir          Directory with CSV files (default: datasets/baxter)
--output_dir        Where to save checkpoints (default: checkpoints/neural_ode_opus)
--epochs            Number of training epochs (default: 100)
--batch_size        Batch size (default: 32)
--seq_len           Sequence length (default: 50)
--stride            Stride between sequences (default: 10)
--hidden_dim        Hidden dimension of ODE network (default: 128)
--dt                Timestep for ODE integration (default: 0.01)
--lr                Learning rate (default: 1e-3)
--weight_decay      L2 regularization (default: 1e-4)
--gradient_clip     Gradient clipping norm (default: 1.0)
--scheduler         LR scheduler: cosine|plateau|none (default: cosine)
--device            cuda|cpu (default: auto-detect)
```

**Outputs:**
```
checkpoints/neural_ode_opus/
├── best_model.pt           # Best model on validation set
├── final_model.pt          # Final model after all epochs
├── checkpoint_epoch10.pt   # Periodic checkpoints
├── config.json             # Hyperparameter config
└── history.json            # Train/val losses per epoch
```

### Testing & Evaluation (`test_opus.py`)

**Evaluate on test set:**
```python
python model/test_opus.py \
    --checkpoint checkpoints/neural_ode_opus/best_model.pt \
    --data_dir datasets/baxter \
    --seq_len 50 \
    --batch_size 32 \
    --num_plot_samples 3
```

**Metrics Computed:**
- **Angle RMSE**: Root mean squared error in joint angles (rad)
- **Velocity RMSE**: Root mean squared error in joint velocities (rad/s)
- **Per-joint RMSE**: Error breakdown by each joint
- **Step errors**: How prediction error grows over time horizon
- **Per-step analysis**: Identifies when model starts diverging

**Outputs:**
```
results/test_opus/
├── metrics.json                 # Summary metrics
├── trajectory_sample_0.png      # Predicted vs ground truth trajectories
├── trajectory_sample_1.png
├── error_over_time.png          # Error accumulation plot
└── joint_errors.png             # Per-joint error bars
```

**Key Functions:**
```python
# Evaluate on full test dataset
metrics, pred_states, target_states = evaluate_model(model, test_loader, state_norm, device)

# Test on specific CSV file
results = test_on_csv(model, 'datasets/baxter/left_circle_p-15_t105.csv', state_norm, torque_norm, device)
```

### Inference (`inference_opus.py`)

**Easy-to-use predictor class for deployment:**

```python
from model.inference_opus import load_predictor
import numpy as np

# Load model
predictor = load_predictor('checkpoints/neural_ode_opus/best_model.pt', device='cuda')

# Single prediction
initial_state = np.array([...])  # [14] - 7 angles + 7 velocities
torques = np.array([...])        # [seq_len, 7]
predicted_states = predictor.predict(initial_state, torques)  # [seq_len+1, 14]

# Just angles
predicted_angles = predictor.predict_angles(initial_state, torques)  # [seq_len+1, 7]

# Batch prediction
initial_states = np.random.randn(32, 14)
torques_batch = np.random.randn(32, 50, 7)
batch_predictions = predictor.predict_batch(initial_states, torques_batch)  # [32, 51, 14]

# Closed-loop rollout (torque depends on state)
def torque_fn(state):
    # Custom control law
    return np.zeros(7)

states = predictor.rollout(initial_state, torque_fn, num_steps=100)  # [101, 14]
```

## Workflow

### 1. Data Preparation
Your data should be in CSV format with columns:
```
ang_s0 ang_s1 ang_e0 ang_e1 ang_w0 ang_w1 ang_w2 
vel_s0 vel_s1 vel_e0 vel_e1 vel_w0 vel_w1 vel_w2 
torq_s0 torq_s1 torq_e0 torq_e1 torq_w0 torq_w1 torq_w2
```

Place CSV files in `datasets/baxter/` (or any directory you specify).

### 2. Train Model
```bash
python model/train_opus.py \
    --data_dir datasets/baxter \
    --epochs 50 \
    --batch_size 32 \
    --seq_len 50
```

**Expected output:**
- Epoch logs with train/val losses
- Best model saved when validation loss improves
- Periodic checkpoints every 10 epochs
- Training takes ~5-30 minutes depending on data size and hardware

### 3. Evaluate Model
```bash
python model/test_opus.py \
    --checkpoint checkpoints/neural_ode_opus/best_model.pt \
    --data_dir datasets/baxter \
    --num_plot_samples 5
```

**Generates:**
- Summary metrics (angle RMSE, velocity RMSE, etc.)
- Trajectory comparison plots
- Error accumulation curves
- Per-joint error analysis

### 4. Use for Inference
```python
from model.inference_opus import load_predictor

predictor = load_predictor('checkpoints/neural_ode_opus/best_model.pt')

# Make predictions
pred = predictor.predict(initial_state, torques)
```

## Model Architecture

### Dynamics Network

```
Input: [q, qdot, torque] (14 + 7 = 21 dims)
  ↓
Linear (21 → 128) + SiLU
  ↓
Linear (128 → 128) + SiLU
  ↓
Linear (128 → 128) + SiLU
  ↓
Linear (128 → 14)  ← small initialization
  ↓
Output: [dq/dt, dqdot/dt] (14 dims)
```

### Physics Prior

The model enforces a physics constraint:
```
dq/dt = qdot  (velocities are derivatives of angles)
dqdot/dt = network_output[7:]  (accelerations from network)
```

This reduces the network's burden and improves generalization.

## Key Hyperparameters

| Parameter | Default | Notes |
|-----------|---------|-------|
| `hidden_dim` | 128 | Network capacity; 64-256 typical |
| `seq_len` | 50 | Sequence length for training; longer = harder but more info |
| `batch_size` | 32 | Larger = faster but higher memory |
| `lr` | 1e-3 | Learning rate; may need tuning |
| `dt` | 0.01 | Integration timestep for ODE solver |
| `gradient_clip` | 1.0 | Important for ODE gradients |
| `weight_decay` | 1e-4 | L2 regularization |

## Understanding the Output

### Checkpoint File Structure

```python
checkpoint = {
    'epoch': 50,
    'model_state_dict': {...},              # Model weights
    'optimizer_state_dict': {...},          # Optimizer state
    'scheduler_state_dict': {...},          # LR scheduler state
    'train_metrics': {'loss': 0.001, ...},  # Final training metrics
    'val_metrics': {'loss': 0.0015, ...},   # Best validation metrics
    'state_normalizer': {...},              # Saved normalizer params
    'torque_normalizer': {...},
    'config': {...},                        # Full config used in training
}
```

### Metrics Explained

- **MSE (normalized)**: Loss on normalized data (what the model directly optimizes)
- **MSE (denormalized)**: Loss on original units (radians) — more interpretable
- **RMSE**: Square root of MSE (same units as predictions)
- **MAE**: Mean absolute error (robust to outliers)
- **Step errors**: RMSE at each timestep; shows if error accumulates linearly or exponentially

## Troubleshooting

### High Training Loss
- **Issue**: Loss not decreasing
- **Solutions**:
  - Lower learning rate (try 5e-4 or 1e-4)
  - Reduce sequence length (try 30 instead of 50)
  - Check data normalization (normalizers should fit to training set only)

### Out of Memory
- **Issue**: CUDA out of memory
- **Solutions**:
  - Reduce batch size (try 16 or 8)
  - Reduce sequence length
  - Use `--device cpu` for CPU-only training

### Poor Generalization
- **Issue**: Model performs well on train but poorly on val
- **Solutions**:
  - Increase weight decay (try 1e-3 or 1e-2)
  - Use dropout (modify `neural_ode_robot_opus.py`)
  - Collect more diverse training data
  - Check data normalization (verify train/val use same normalizer)

### Unstable Gradients
- **Issue**: Loss goes NaN or oscillates wildly
- **Solutions**:
  - Increase gradient clipping (try 10.0)
  - Lower learning rate
  - Check for data anomalies (outliers, NaNs)

## Advanced Usage

### Custom Training Loop

```python
from model.neural_ode_robot_opus import NeuralODERobotSequence
from model.data_processing_opus import create_dataloaders
import torch.optim as optim

model = NeuralODERobotSequence(hidden_dim=128, dt=0.01)
train_loader, val_loader, state_norm, torque_norm = create_dataloaders(...)

optimizer = optim.AdamW(model.parameters(), lr=1e-3)

for epoch in range(100):
    for initial_state, torques, target_states in train_loader:
        pred = model(initial_state, torques)[:, 1:, :]
        loss = ((pred - target_states) ** 2).mean()
        loss.backward()
        optimizer.step()
        optimizer.zero_grad()
```

### Fine-tuning on New Robot

1. Load pretrained weights:
```python
checkpoint = torch.load('checkpoints/neural_ode_opus/best_model.pt')
model.load_state_dict(checkpoint['model_state_dict'])
```

2. Retrain on new data with lower learning rate:
```bash
python model/train_opus.py --lr 1e-4 --epochs 20 ...
```

### Different ODE Solver

Modify `train_opus.py` and `inference_opus.py`:
```python
odeint(..., method='dopri5')  # Default; good accuracy
odeint(..., method='adams')   # Higher-order; slower
odeint(..., method='euler')   # Faster; lower accuracy
```

## References

- Chen et al., "Neural Ordinary Differential Equations" (NeurIPS 2018)
- Dupont et al., "Augmented Neural ODEs" (NeurIPS 2019)
- torchdiffeq documentation: https://github.com/rtqichen/torchdiffeq

## File Dependencies

```
neural_ode_robot_opus.py
├── torch, torchdiffeq

train_opus.py
├── neural_ode_robot_opus.py
├── data_processing_opus.py
└── torch, tqdm

test_opus.py
├── neural_ode_robot_opus.py
├── data_processing_opus.py
├── train_opus.py (for load_checkpoint)
└── matplotlib, numpy, pandas

inference_opus.py
├── neural_ode_robot_opus.py
├── data_processing_opus.py
└── torch, numpy

data_processing_opus.py
└── torch, pandas, numpy
```

---

**Last updated:** November 27, 2025  
**Status:** Production-ready
