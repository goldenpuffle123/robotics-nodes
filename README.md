# Continuous Intelligence: Neural ODEs for Modelling Robot Dynamics

Learning robot dynamics using Neural ODE models trained on Baxter arm end-effector trajectories. The model predicts joint accelerations given current state and applied torques, with integration handled by a select choice of numerical solvers.

## Quick Start

### Prerequisites
- Python 3.10+
- CUDA 11.8+ (for GPU acceleration, optional)
- Git LFS (for checkpoint files)

### Installation

```bash
# Clone repository
git clone https://github.com/goldenpuffle123/robotics-nodes.git
cd robotics-nodes
```
```bash
# 1. using uv (recommended)
pip install uv

# 2. without uv
python -m venv .venv
# on windows
.venv\Scripts\activate
# on mac
source .venv/bin/activate
```

#### Train with GPU Acceleration (Optional)
- If you have a compatible NVIDIA GPU, ensure CUDA is installed
- Go to `cmd/powershell` and run `nvcc --version` to check CUDA version
- Go to [PyTorch](https://pytorch.org/get-started/locally/) and follow instructions to install PyTorch with CUDA support (note: replace `pip3` with `uv pip` for uv)
- For example:
```bash
# For uv and CUDA 12.8
uv pip install torch --index-url https://download.pytorch.org/whl/cu128
```

#### Packages
```bash
# 1. using uv (recommended)
uv sync
# skip this step if torch with CUDA is already installed
uv add torch

# 2. without uv
pip install -e
# skip this step if torch with CUDA is already installed
pip install torch
```
#### Git LFS
- Go to [Git LFS](https://git-lfs.com/) and follow installation instructions for your OS
```
# Setup Git LFS
git lfs install
git lfs pull
```

### Dataset

The dataset contains 24 CSV files with Baxter left-arm trajectories:
- **Spiral**: 60.05% (1,079,336 samples)
- **Random**: 30.01% (539,322 samples)  
- **Circle**: 4.97% (89,379 samples)
- **Squared**: 4.97% (89,306 samples)

Files are located in `datasets/baxter/` with format: `left_<movement>_p<params>_t<time>.csv`

## Training (Optional)

### Run Training

```bash
cd modelv3

# Basic training (100 epochs, batch size 32)
python train.py --epochs 100 --batch_size 32

# Custom configuration
python train.py \
    --epochs 200 \
    --batch_size 64 \
    --seq_len 50 \
    --lr 1e-3 \
    --hidden_dim 128 \
    --output_dir checkpoints/modelv3_v2

# Resume from checkpoint
python train.py --resume checkpoints/modelv3/best_model.pt --epochs 100
```

**Key Options:**
- `--epochs`: Number of training epochs (default: 100)
- `--batch_size`: Batch size (default: 32)
- `--seq_len`: Sequence length in timesteps (default: 50)
- `--lr`: Learning rate (default: 1e-3)
- `--hidden_dim`: Hidden layer dimension (default: 128)
- `--output_dir`: Checkpoint directory (default: checkpoints/modelv3)
- `--resume`: Resume from checkpoint (optional)
- `--device`: cuda/mps/cpu (default: auto-detect)

**Outputs:**
- `config.json` - Training configuration
- `best_model.pt` - Best model checkpoint
- `history.json` - Training/validation metrics
- `checkpoint_*.pt` - Periodic checkpoints

### Monitor Training

Training logs are printed to console with loss values. Training takes ~10 minutes on GPU for 10 epochs.

## Pretrained Models

Pretrained models (from project) are available in `checkpoints/modelv3/`:
- Interpolation tests:
    - `interp_constant.pt`: Neural ODE with piecewise-constant torque interpolation
    - `interp_linear.pt`: Linear torque interpolation
- Integrator tests:
    - `euler.pt`: Euler method
    - `rk4.pt`: Runge-Kutta 4th order
    - `velocity_verlet.pt`: Velocity Verlet integrator
- Prediction frame tests:
    - `frame_100.pt`: trained on 100 frames = 0.2 sec of data
    - `frame_500.pt`: 500 frames = 1.0 sec
    - `frame_1000.pt`: 1000 frames = 2.0 sec of data

## Testing & Evaluation

### Single Model Test

Visualize predictions vs ground truth for a single model:

```bash
cd modelv3

python test.py single --checkpoint checkpoints/modelv3/best_model.pt --num_sequences 5
```

### Compare Multiple Models

Compare predictions from different models on the same sequences:

```bash
cd modelv3

python test.py compare \
    --checkpoints checkpoints/modelv3/best_model.pt checkpoints/modelv3/best_model-2.pt \
    --names "Model v1" "Model v2" \
    --num_sequences 10 \
    --save_dir results/comparison
```

**Options:**
- `--seq_len`: Sequence length (default: 50)
- `--num_sequences`: Number of sequences to visualize (default: 5)
- `--save_dir`: Save figures to directory (optional)
- `--device`: cuda/mps/cpu (default: auto-detect)

**Outputs:**
- 3D trajectory plots comparing predictions
- Position error vs time graphs
- Error summary table (mean/max error in mm)

## Model Architecture

### Neural ODE Model (`modelv3/neural_ode_robot.py`)

**Components:**
- `TorqueInterpolatorLinear`: Differentiable linear torque interpolation between timesteps
- `TorqueInterpolatorConstant`: Piecewise-constant (zero-order hold) interpolation
- `ODEFunc`: Neural network predicting 7-DOF accelerations given state + torques
- `NeuralODE`: Main model using Velocity Verlet integrator for stable integration
- ODE Solver:
```python
# For euler:
states = odeint(self.ode_func, initial_state, t_span, method='euler')
# For rk4:
states = odeint_adjoint(self.ode_func, initial_state, t_span, method='rk4')
# For velocity_verlet:
states = velocity_verlet(self.ode_func, initial_state, t_span)
```

**Key Design:**
- **Input**: Initial state [batch, 14] (7 angles + 7 velocities) + torques [batch, seq_len, 7]
- **Output**: Predicted states [batch, seq_len, 14]
- **Integration**: Symplectic Velocity Verlet (2x faster than RK4, energy-conserving)
- **Physics Constraint**: `dq/dt = v` enforced in ODE structure
- **State Format**: Raw physical units (no normalization), only torques normalized

### Data Format (`modelv3/data.py`)

**Dataset Split:**
- Train: 70%
- Val: 20%
- Test: 10%

**Normalization:**
- States: Raw (radians, rad/s) - required for correct ODE physics
- Torques: Normalized to mean=0, std=1

**Per-Sample:**
- Initial state: [14,] (7 angles + 7 velocities)
- Input torques: [seq_len-1, 7] normalized torques
- Target states: [seq_len-1, 14] next states

## Utilities

### Forward Kinematics
```bash
cd vis
python -c "from fk import calculate_fk; import numpy as np; q = np.zeros(7); T = calculate_fk(q); print(T)"
```
Compute end-effector position from joint angles.

## References

- **Neural ODEs**: Chen et al. (2018) - "Neural Ordinary Differential Equations"
- **Velocity Verlet**: Swope et al. (1982) - Symplectic integrator for molecular dynamics
- **Baxter Robot**: Rethink Robotics - 7-DOF dual-arm collaborative robot