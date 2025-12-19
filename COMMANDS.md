### Torque Interpolator Comparison
```bash
# uv
uv run modelv3/test.py compare --checkpoints models_trained\interp_const.pt models_trained\interp_linear.pt --names "Constant" "Linear" --num_sequences 5 --seq_len 300
# venv
python modelv3/test.py compare --checkpoints models_trained\interp_const.pt models_trained\interp_linear.pt --names "Constant" "Linear" --num_sequences 5 --seq_len 300
```
### Integrator Comparison
```bash
# uv
uv run modelv3/test.py compare --checkpoints models_trained\vv_long.pt models_trained\rk4_long.pt models_trained\euler_long.pt --names "VV" "RK4" "Euler" --num_sequences 5 --seq_len 1000
# venv
python modelv3/test.py compare --checkpoints models_trained\vv_long.pt models_trained\rk4_long.pt models_trained\euler_long.pt --names "VV" "RK4" "Euler" --num_sequences 5 --seq_len 1000
```