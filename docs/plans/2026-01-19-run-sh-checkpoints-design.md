# Run.sh TensorBoard + Checkpoint Resume Design

## Goal
Add auto TensorBoard launch, MLX-style step checkpoints with optimizer state and resume,
and align `pipelines/run.sh` stage outputs with `pipelines/run_mlx.sh` while keeping
legacy `out/*.pth` files for compatibility.

## Current State
- `pipelines/run.sh` saves weight-only checkpoints in `out/*_{hidden}.pth`.
- No optimizer/scaler/state saves, so training cannot resume from a step.
- TensorBoard logs are written to `TF_DIR`, but TensorBoard is not auto-started.
- Checkpoint selection for later stages is weight-only and lives in the output root.

## Proposed Changes
1. **MLX-style step checkpoints (Torch):**
   - Stage outputs become `out/{pretrain,sft,dpo,r1}`.
   - Step checkpoints live under `out/<stage>/checkpoints/step_00000123/`.
   - Each step directory contains:
     - `model.pth` (model state)
     - `optimizer.pt` (optimizer + scaler state)
     - `state.json` (epoch, step_in_epoch, global_step, args, metadata)
     - `rng_state.pt` (Python + Torch RNG states)
   - Keep last N via `KEEP_LAST` (default 3), like MLX.
   - Continue writing legacy `out/*_{hidden}.pth` at each save for compatibility.

2. **Training script resume support:**
   - Add `--resume`, `--ckpt_dir`, and `--keep_last_checkpoints` flags to:
     - `trainer/train_pretrain.py`
     - `trainer/train_full_sft.py`
     - `trainer/train_dpo.py`
     - `trainer/train_distill_reason.py`
   - Resume restores model, optimizer, scaler, RNG, and training state.
   - Save interval and `max_steps` are based on optimizer update steps.
   - First epoch resumes at stored `epoch` and `step_in_epoch` (skip batches).

3. **Pipeline alignment (`pipelines/run.sh`):**
   - Stage output dirs mirror MLX: `PRETRAIN_OUT`, `SFT_OUT`, `DPO_OUT`, `R1_OUT`.
   - Add `latest_ckpt` + `is_valid_ckpt` helpers for step checkpoints.
   - Auto-resume when a step checkpoint exists; else fall back to pretrained weight.
   - Pass `SAVE_INTERVAL` and `KEEP_LAST` to training scripts.

4. **Auto-start TensorBoard:**
   - New env vars: `TB_AUTO` (default 1), `TB_PORT` (default 6006), `TB_HOST`.
   - When enabled and `TF_DIR` is non-empty, start TensorBoard in background.
   - Log PID and URL; warn if TensorBoard is unavailable.

## Non-Goals
- No changes to dataset discovery or training algorithms.
- No new CLI flags beyond checkpointing and TensorBoard control.

## Compatibility Notes
- Existing workflows expecting `out/*_{hidden}.pth` keep working.
- `--skip-pretrain` behavior remains, but now prefers step checkpoints when available.

## Risks and Mitigations
- **Different save cadence:** save interval now tracks optimizer updates.
  Mitigation: keep defaults similar to MLX and document change.
- **DDP resume mismatch:** mismatched world size may break optimizer state.
  Mitigation: log warning and require matching settings.
