# Run Series TensorBoard Auto-Start Design

## Goal
Enable automatic TensorBoard startup for run-series scripts that *emit* TensorBoard logs, without changing training behavior or log layout.

## Scope
- `pipelines/run_mlx.sh`: auto-start TensorBoard when `TF_DIR` is non-empty and at least one training stage will run.
- `pipelines/run_mlx_distill_ollama.sh`: auto-start only when the user passes `--tensorboard_dir` to training.
- `pipelines/run_mlx_hf_qwen3.sh` and `pipelines/run_mlx_rl_gsm8k.sh`: unchanged (no TB logs by default).
- `pipelines/run.sh`: already auto-starts; no change.

## Behavior
- New/consistent env vars: `TB_AUTO` (default 1), `TB_HOST` (default `127.0.0.1`), `TB_PORT` (default `6006`).
- Use the same Python executable as training (the MLX venv `PY`) to run TensorBoard.
- If the requested port is already in use, bind a free ephemeral port.
- Log the URL and PID when TensorBoard starts; warn when TensorBoard is unavailable.
- Only start when the script is actually emitting TensorBoard logs.

## Data Flow
1. Resolve `TF_DIR` (or detect `--tensorboard_dir` in distill).
2. Decide whether training stages will run (for `run_mlx.sh`).
3. If `TB_AUTO != 0` and logging is enabled, start TensorBoard in background:
   - check `python -m tensorboard --version`
   - check port availability and select a free one if needed
   - run `python -m tensorboard --logdir <dir> --host <host> --port <port>`

## Error Handling
- If TensorBoard is not installed, print a warning and continue.
- If a preferred port is busy, pick an open port automatically.
- No change to training behavior or outputs.

## Testing
- Lightweight static test to assert TB auto-start hooks exist in the scripts.
- Manual smoke check (optional): run `scripts/run_mlx.sh --smoke-test` and verify the printed TensorBoard URL.
