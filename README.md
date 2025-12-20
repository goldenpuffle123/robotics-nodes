# Continuous Intelligence: Neural ODEs for Modelling Robot Dynamics

Learning robot dynamics using Neural ODE models trained on Baxter arm end-effector trajectories. The model predicts joint accelerations given current state and applied torques, with integration handled by a select choice of numerical solvers.

## Quick Start

### Demo Video:
https://drive.google.com/file/d/1PkbBpfY2uqX7I86Z1kTtsSEnHsJAeBlD/view?usp=sharing

### Prerequisites
- Python 3.10+
- CUDA 11.8+ (for GPU acceleration, optional)
- Git LFS (for checkpoint files)

### Installation

#### Git LFS
- Go to [Git LFS](https://git-lfs.com/) and follow installation instructions for your OS
```
# Setup Git LFS
git lfs install
```
#### Clone Repository
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

#### GPU Acceleration (Optional)
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
pip install -r requirements.txt
pip install torch
```

### Generating Results from Paper

See `COMMANDS.md` for commands to reproduce figures from the paper, including training, testing, and model comparisons.

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
    --stride 10 \
    --lr 1e-3 \
    --hidden_dim 128 \
    --dt 0.002 \
    --integrator euler \
    --output_dir checkpoints/modelv3

# Resume from checkpoint
python train.py --resume checkpoints/modelv3/best_model.pt --epochs 100
```

**Key Options:**
- `--epochs`: Number of training epochs (default: 100)
- `--batch_size`: Batch size (default: 32)
- `--seq_len`: Sequence length in timesteps (default: 50)
- `--stride`: Stride for sampling sequences (default: 10)
- `--lr`: Learning rate (default: 1e-3)
- `--hidden_dim`: Hidden layer dimension (default: 128)
- `--dt`: Timestep size (default: 0.002)
- `--integrator`: ODE solver method - `euler`, `rk4`, `dopri5`, or `vv` (default: euler)
- `--output_dir`: Checkpoint directory (default: checkpoints/modelv3)
- `--resume`: Resume from checkpoint (optional)
- `--device`: cuda/mps/cpu (default: auto-detect)

**Outputs:**
- `config.json` - Training configuration
- `best_model.pt` - Best model checkpoint
- `history.json` - Training/validation metrics
- `checkpoint_*.pt` - Periodic checkpoints

## Pretrained Models

Pretrained models (from project) are available in `models_trained/`:
- Interpolators:
    - `interp_constant.pt`: Piecewise-constant torque interpolation
    - `interp_linear.pt`: Linear torque interpolation
- Integrators:
    - `vv_long.pt`: Velocity Verlet integrator
    - `rk4_long.pt`: Runge-Kutta 4th order integrator
    - `euler_long.pt`: Euler integrator

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

**Single Model Options:**
- `--checkpoint`: Path to model checkpoint (required)
- `--seq_len`: Sequence length (default: 50)
- `--num_sequences`: Number of sequences to visualize (default: 5)
- `--device`: cuda/cpu (default: auto-detect)

**Compare Models Options:**
- `--checkpoints`: Paths to model checkpoints (required, space-separated)
- `--names`: Names for each model (optional, space-separated)
- `--seq_len`: Sequence length (default: 50)
- `--num_sequences`: Number of sequences to visualize (default: 5)
- `--save_dir`: Directory to save figures (optional)
- `--device`: cuda/cpu (default: auto-detect)

## Model Architecture

### Neural ODE Model (`modelv3/neural_ode_robot.py`)

**Components:**
- `TorqueInterpolatorLinear`: Differentiable linear torque interpolation between timesteps
- `TorqueInterpolatorConstant`: Piecewise-constant (zero-order hold) interpolation
- `ODEFunc`: Neural network predicting 7-DOF accelerations given state + torques
- `NeuralODE`: Main model supporting multiple ODE solvers

**Supported Integrators:**
- `euler`: Explicit Euler method
- `rk4`: Runge-Kutta 4th order
- `dopri5`: Dormand-Prince 5th order (adaptive)
- `vv`: Velocity Verlet (symplectic, energy-conserving)

**Key Design:**
- **Input**: Initial state [batch, 14] (7 angles + 7 velocities) + torques [batch, seq_len, 7]
- **Output**: Predicted states [batch, seq_len, 14]
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

### Forward Kinematics (`modelv3/fk.py`)
```bash
cd vis
python -c "from fk import calculate_fk; import numpy as np; q = np.zeros(7); T = calculate_fk(q); print(T)"
```
Compute end-effector position from joint angles.

## References

- **Neural ODEs**: Chen et al. (2018) - "Neural Ordinary Differential Equations"
- **Velocity Verlet**: Swope et al. (1982) - Symplectic integrator for molecular dynamics
- **Baxter Robot**: Rethink Robotics - 7-DOF dual-arm collaborative robot
