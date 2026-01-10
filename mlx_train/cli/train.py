from __future__ import annotations

import argparse
import json
import math
import os
import re
import shutil
import time
from typing import Any, Callable, Dict, Optional, Tuple

import mlx.core as mx
import mlx.nn as nn
import mlx.optimizers as optim
import mlx.utils as mlx_utils

from ..config import MiniLLMConfig, minillm_200mb
from ..data import make_microbatch_iterator, pretokenize_jsonl, resolve_jsonl_paths
from ..download import resolve_data_path_spec
from ..models import MiniLLMForCausalLM, count_parameters, parameters_bytes
from ..optim import make_optimizer
from ..ops.loss import chunked_ce_loss, sparse_ce_loss


def set_seed(seed: int) -> None:
    mx.random.seed(seed)


def cosine_lr(
    step: int,
    total_steps: int,
    base_lr: float,
    *,
    warmup_steps: int = 0,
    min_lr_ratio: float = 0.1,
) -> float:
    if total_steps <= 0:
        return base_lr
    if warmup_steps > 0 and step < warmup_steps:
        return base_lr * (step + 1) / warmup_steps
    progress = (step - warmup_steps) / max(1, total_steps - warmup_steps)
    cosine = 0.5 * (1.0 + math.cos(math.pi * progress))
    return base_lr * (min_lr_ratio + (1.0 - min_lr_ratio) * cosine)


def save_optimizer_state(optimizer: optim.Optimizer, path: str) -> None:
    flat: Dict[str, Any] = {}
    mlx_utils.tree_flatten(optimizer.state, destination=flat)
    mx.savez(path, **flat)


def load_optimizer_state(optimizer: optim.Optimizer, path: str) -> None:
    flat = dict(mx.load(path))
    optimizer.state = mlx_utils.tree_unflatten(flat)


def restore_tree_in_place(dst: Any, src: Any) -> Any:
    if isinstance(dst, dict) and isinstance(src, dict):
        for k, v in src.items():
            if k in dst:
                dst[k] = restore_tree_in_place(dst[k], v)
            else:
                dst[k] = v
        return dst
    if isinstance(dst, list) and isinstance(src, list):
        n = min(len(dst), len(src))
        for i in range(n):
            dst[i] = restore_tree_in_place(dst[i], src[i])
        if len(src) > len(dst):
            dst.extend(src[len(dst) :])
        return dst
    return src


def compile_value_and_grad(
    fn: Callable[..., Tuple[mx.array, Any]],
    *,
    model: nn.Module,
    batch_size: int,
    seq_len: int,
    label_len: int,
    sparse_loss: bool,
) -> Callable[..., Tuple[mx.array, Any]]:
    compiled = mx.compile(
        fn,
        inputs={"model": model, "rng": mx.random.state},
        outputs={"rng": mx.random.state},
    )

    # Warm up compilation and then restore parameters. `nn.value_and_grad` uses
    # `model.update()` with tracer arrays during compilation; without restoring,
    # the model would be left in a non-evaluable state for the Python training loop.
    params_backup = model.parameters()
    x0 = mx.zeros((batch_size, seq_len), dtype=mx.int32)
    y0 = mx.zeros((batch_size, seq_len), dtype=mx.int32)
    m0 = mx.ones((batch_size, seq_len), dtype=mx.float32)
    try:
        if sparse_loss:
            p0 = mx.zeros((batch_size, label_len), dtype=mx.int32)
            pm0 = mx.ones((batch_size, label_len), dtype=mx.float32)
            loss0, grads0 = compiled(x0, y0, m0, p0, pm0)
        else:
            loss0, grads0 = compiled(x0, y0, m0)
        mx.eval(loss0, grads0)
    finally:
        model.update(params_backup)

    return compiled


def compile_optimizer_step(
    fn: Callable[..., mx.array],
    *,
    model: nn.Module,
    optimizer: optim.Optimizer,
) -> Callable[..., mx.array]:
    optimizer.init(model.trainable_parameters())
    compiled = mx.compile(
        fn,
        inputs={"model": model, "opt": optimizer.state},
        outputs={"model": model, "opt": optimizer.state},
    )

    params_backup = model.parameters()
    opt_backup = mlx_utils.tree_map(lambda x: x, optimizer.state)
    dummy_grads = mlx_utils.tree_map(
        lambda p: mx.zeros_like(p), model.trainable_parameters()
    )
    try:
        out0 = compiled(dummy_grads)
        mx.eval(out0)
    finally:
        model.update(params_backup)
        restore_tree_in_place(optimizer.state, opt_backup)
        optimizer.init(model.trainable_parameters())

    return compiled


def compile_train_step(
    *,
    model: nn.Module,
    optimizer: optim.Optimizer,
    value_and_grad: Callable[..., Tuple[mx.array, Any]],
    batch_size: int,
    seq_len: int,
    label_len: int,
    accum_steps: int,
    grad_clip: float,
    sparse_loss: bool,
) -> Callable[..., Tuple[mx.array, mx.array]]:
    if accum_steps <= 0:
        raise ValueError("accum_steps must be > 0")

    optimizer.init(model.trainable_parameters())

    if sparse_loss:
        def train_step(
            xs: mx.array,
            ys: mx.array,
            ms: mx.array,
            ps: mx.array,
            pms: mx.array,
            micro_batches: mx.array,
        ) -> Tuple[mx.array, mx.array]:
            loss_sum = mx.array(0.0, dtype=mx.float32)
            grad_accum: Any = mlx_utils.tree_map(lambda p: mx.zeros_like(p), model.trainable_parameters())
            for i in range(accum_steps):
                loss, grads = value_and_grad(xs[i], ys[i], ms[i], ps[i], pms[i])
                loss_sum = loss_sum + loss.astype(mx.float32)
                grad_accum = mlx_utils.tree_map(lambda a, b: a + b, grad_accum, grads)

            denom = mx.maximum(micro_batches.astype(mx.float32), mx.array(1.0, dtype=mx.float32))
            grad_accum = mlx_utils.tree_map(lambda g: g / denom, grad_accum)

            if grad_clip > 0:
                grad_accum, grad_norm = optim.clip_grad_norm(grad_accum, max_norm=grad_clip)
            else:
                grad_norm = mx.array(0.0, dtype=mx.float32)

            optimizer.update(model, grad_accum)
            return loss_sum, grad_norm
    else:
        def train_step(
            xs: mx.array, ys: mx.array, ms: mx.array, micro_batches: mx.array
        ) -> Tuple[mx.array, mx.array]:
            loss_sum = mx.array(0.0, dtype=mx.float32)
            grad_accum: Any = mlx_utils.tree_map(lambda p: mx.zeros_like(p), model.trainable_parameters())
            for i in range(accum_steps):
                loss, grads = value_and_grad(xs[i], ys[i], ms[i])
                loss_sum = loss_sum + loss.astype(mx.float32)
                grad_accum = mlx_utils.tree_map(lambda a, b: a + b, grad_accum, grads)

            denom = mx.maximum(micro_batches.astype(mx.float32), mx.array(1.0, dtype=mx.float32))
            grad_accum = mlx_utils.tree_map(lambda g: g / denom, grad_accum)

            if grad_clip > 0:
                grad_accum, grad_norm = optim.clip_grad_norm(grad_accum, max_norm=grad_clip)
            else:
                grad_norm = mx.array(0.0, dtype=mx.float32)

            optimizer.update(model, grad_accum)
            return loss_sum, grad_norm

    compiled = mx.compile(
        train_step,
        inputs={"model": model, "opt": optimizer.state, "rng": mx.random.state},
        outputs={"model": model, "opt": optimizer.state, "rng": mx.random.state},
    )

    params_backup = model.parameters()
    opt_backup = mlx_utils.tree_map(lambda x: x, optimizer.state)
    x0 = mx.zeros((accum_steps, batch_size, seq_len), dtype=mx.int32)
    y0 = mx.zeros((accum_steps, batch_size, seq_len), dtype=mx.int32)
    m0 = mx.ones((accum_steps, batch_size, seq_len), dtype=mx.float32)
    micro0 = mx.array(float(accum_steps), dtype=mx.float32)
    try:
        if sparse_loss:
            p0 = mx.zeros((accum_steps, batch_size, label_len), dtype=mx.int32)
            pm0 = mx.ones((accum_steps, batch_size, label_len), dtype=mx.float32)
            loss0, grad_norm0 = compiled(x0, y0, m0, p0, pm0, micro0)
        else:
            loss0, grad_norm0 = compiled(x0, y0, m0, micro0)
        mx.eval(loss0, grad_norm0)
    finally:
        model.update(params_backup)
        restore_tree_in_place(optimizer.state, opt_backup)
        optimizer.init(model.trainable_parameters())

    return compiled


def loss_fn(
    model: MiniLLMForCausalLM,
    x: mx.array,
    y: mx.array,
    loss_mask: mx.array,
    *,
    logits_chunk_size: int,
) -> mx.array:
    hidden = model.model(x)  # [B, T, H]
    return chunked_ce_loss(
        hidden=hidden,
        lm_head_weight=model.model.embed_tokens.weight,
        labels=y,
        loss_mask=loss_mask,
        chunk_size=int(logits_chunk_size),
    )


def sparse_loss_fn(
    model: MiniLLMForCausalLM,
    x: mx.array,
    y: mx.array,
    loss_mask: mx.array,
    label_positions: mx.array,
    label_pos_mask: mx.array,
    *,
    logits_chunk_size: int,
) -> mx.array:
    hidden = model.model(x)  # [B, T, H]
    return sparse_ce_loss(
        hidden=hidden,
        lm_head_weight=model.model.embed_tokens.weight,
        labels=y,
        label_positions=label_positions,
        label_pos_mask=label_pos_mask,
        chunk_size=int(logits_chunk_size),
    )


def make_config(args, tokenizer) -> MiniLLMConfig:
    if args.preset == "200mb":
        cfg = minillm_200mb()
    elif args.preset == "tiny":
        cfg = MiniLLMConfig(
            hidden_size=256,
            num_hidden_layers=4,
            num_attention_heads=4,
            num_key_value_heads=2,
            vocab_size=tokenizer.vocab_size,
            dropout=args.dropout,
            rope_theta=args.rope_theta,
            max_position_embeddings=args.max_position_embeddings,
            use_moe=False,
        ).finalize()
    elif args.preset == "custom":
        cfg = MiniLLMConfig(
            hidden_size=args.hidden_size,
            num_hidden_layers=args.num_hidden_layers,
            num_attention_heads=args.num_attention_heads,
            num_key_value_heads=args.num_key_value_heads,
            vocab_size=tokenizer.vocab_size
            if args.vocab_size is None
            else args.vocab_size,
            dropout=args.dropout,
            rope_theta=args.rope_theta,
            max_position_embeddings=args.max_position_embeddings,
            use_moe=False,
        ).finalize()
    else:
        raise ValueError(f"Unknown preset: {args.preset}")

    cfg.lora_r = int(args.lora_r)
    cfg.lora_alpha = float(args.lora_alpha)
    cfg.lora_dropout = float(args.lora_dropout)
    cfg.lora_targets = str(args.lora_targets)

    if args.attn_gate is not None:
        cfg.use_attn_gate = bool(args.attn_gate)
    if bool(cfg.use_attn_gate):
        cfg.attn_gate_init = float(args.attn_gate_init)

    return cfg


def prune_checkpoints(ckpt_dir: str, *, keep_last: int) -> None:
    if keep_last <= 0:
        return

    pat = re.compile(r"^step_(\d+)$")
    ckpts: list[tuple[int, str]] = []
    for name in os.listdir(ckpt_dir):
        m = pat.match(name)
        if not m:
            continue
        try:
            step = int(m.group(1))
        except ValueError:
            continue
        ckpts.append((step, os.path.join(ckpt_dir, name)))

    ckpts.sort(key=lambda x: x[0])
    for _, path in ckpts[:-keep_last]:
        shutil.rmtree(path, ignore_errors=True)


def load_config_from_checkpoint(checkpoint_dir: str) -> MiniLLMConfig:
    path = os.path.join(checkpoint_dir, "config.json")
    if not os.path.isfile(path):
        raise FileNotFoundError(path)
    with open(path, "r", encoding="utf-8") as f:
        data = json.load(f)
    if not isinstance(data, dict):
        raise ValueError(f"config.json must be an object, got: {type(data).__name__}")
    return MiniLLMConfig.from_dict(data)


def _remap_base_weights_for_lora(weights: Dict[str, mx.array], *, lora_targets: str) -> Dict[str, mx.array]:
    targets = {t.strip() for t in str(lora_targets).split(",") if t.strip()}
    if not targets:
        return dict(weights)

    out: Dict[str, mx.array] = {}
    for name, arr in weights.items():
        parts = name.split(".")
        if len(parts) >= 2 and parts[-1] in {"weight", "bias"} and parts[-2] in targets:
            # e.g. `...q_proj.weight` -> `...q_proj.base.weight`
            new_name = ".".join(parts[:-1] + ["base", parts[-1]])
            out[new_name] = arr
        else:
            out[name] = arr
    return out


def main() -> None:
    parser = argparse.ArgumentParser(description="MiniLLM (MLX) training")
    parser.add_argument("--tokenizer_path", type=str, default="./model")
    parser.add_argument(
        "--data_path",
        type=str,
        required=True,
        help="JSONL file/dir/glob; can be comma-separated.",
    )
    parser.add_argument(
        "--data_dir",
        type=str,
        default="./dataset",
        help="Download/cache directory when `data_path` contains URLs or `minimind:*` specs.",
    )
    parser.add_argument(
        "--hf_dataset_repo",
        type=str,
        default="jingyaogong/minimind_dataset",
        help="HuggingFace dataset repo used by `minimind:*` specs.",
    )
    parser.add_argument(
        "--hf_endpoint",
        type=str,
        default=None,
        help="Optional HuggingFace endpoint/mirror.",
    )
    parser.add_argument(
        "--data_source",
        type=str,
        default=os.environ.get("MINIMIND_DATA_SOURCE", "modelscope"),
        help="Dataset source for `minimind:*` specs: modelscope|hf (default: MINIMIND_DATA_SOURCE or modelscope).",
    )
    parser.add_argument(
        "--ms_dataset_repo",
        type=str,
        default=os.environ.get("MINIMIND_DATA_REPO", "gongjy/minimind_dataset"),
        help="ModelScope dataset repo used by `minimind:*` specs (default: MINIMIND_DATA_REPO or gongjy/minimind_dataset).",
    )
    parser.add_argument(
        "--ms_cache_dir",
        type=str,
        default=os.environ.get("MINIMIND_MS_CACHE"),
        help="ModelScope cache directory (default: MINIMIND_MS_CACHE).",
    )
    parser.add_argument(
        "--ms_revision",
        type=str,
        default=None,
        help="Optional ModelScope revision/tag.",
    )
    parser.add_argument(
        "--force_download",
        action="store_true",
        help="Re-download remote datasets even if present.",
    )
    parser.add_argument(
        "--max_download_mb",
        type=int,
        default=2048,
        help="Safety guard for remote dataset downloads (MB); set 0 to disable.",
    )
    parser.add_argument(
        "--task", type=str, choices=["pretrain", "sft"], default="pretrain"
    )

    parser.add_argument(
        "--preset", type=str, choices=["200mb", "tiny", "custom"], default="200mb"
    )
    parser.add_argument("--hidden_size", type=int, default=512)
    parser.add_argument("--num_hidden_layers", type=int, default=8)
    parser.add_argument("--num_attention_heads", type=int, default=8)
    parser.add_argument("--num_key_value_heads", type=int, default=2)
    parser.add_argument("--vocab_size", type=int, default=None)
    parser.add_argument("--max_position_embeddings", type=int, default=32768)
    parser.add_argument("--rope_theta", type=float, default=1_000_000.0)
    parser.add_argument("--dropout", type=float, default=0.0)
    parser.add_argument(
        "--attn_gate",
        action=argparse.BooleanOptionalAction,
        default=None,
        help="Enable a learnable scalar gate on the attention residual branch (default: preset-dependent).",
    )
    parser.add_argument(
        "--attn_gate_init",
        type=float,
        default=4.0,
        help="Initial logit for attention gate when enabled (sigmoid(init) is the initial multiplier).",
    )

    parser.add_argument("--seq_len", type=int, default=512)
    parser.add_argument("--batch_size", type=int, default=4)
    parser.add_argument("--accum_steps", type=int, default=1)
    parser.add_argument("--learning_rate", type=float, default=3e-4)
    parser.add_argument("--weight_decay", type=float, default=0.1)
    parser.add_argument(
        "--optimizer",
        type=str,
        choices=["adamw", "adafactor", "lion"],
        default="adamw",
        help="Optimizer for full-parameter training.",
    )
    parser.add_argument(
        "--optim_state_dtype",
        type=str,
        choices=["float32", "param"],
        default="float32",
        help="AdamW optimizer state dtype: float32 (stable, memory-heavy) or param (uses parameter dtype).",
    )
    parser.add_argument("--grad_clip", type=float, default=1.0)
    parser.add_argument("--warmup_steps", type=int, default=0)
    parser.add_argument("--epochs", type=int, default=1)
    parser.add_argument("--max_steps", type=int, default=None)
    parser.add_argument("--shuffle_buffer", type=int, default=2048)
    parser.add_argument(
        "--cache_tokenized",
        action=argparse.BooleanOptionalAction,
        default=False,
        help="Pre-tokenize the dataset once and reuse token IDs across epochs (trades RAM for speed).",
    )
    parser.add_argument(
        "--keep_last_checkpoints",
        type=int,
        default=3,
        help="Keep only the latest N checkpoints under out_dir/checkpoints (0 to disable pruning).",
    )

    parser.add_argument(
        "--dtype",
        type=str,
        choices=["float16", "bfloat16", "float32"],
        default="bfloat16",
    )
    parser.add_argument(
        "--metal_kernels",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Use custom Metal fused kernels (RMSNorm / SiLU*mul).",
    )
    parser.add_argument("--out_dir", type=str, default="./out/mlx")
    parser.add_argument("--log_interval", type=int, default=10)
    parser.add_argument("--save_interval", type=int, default=200)
    parser.add_argument("--seed", type=int, default=1337)
    parser.add_argument(
        "--logits_chunk_size",
        type=int,
        default=0,
        help="Compute CE loss in sequence chunks to avoid materializing full [B,T,V] logits (0 = disable).",
    )
    parser.add_argument(
        "--sparse_loss",
        action=argparse.BooleanOptionalAction,
        default=False,
        help=(
            "Compute CE loss only on loss_mask==1 positions (gathered/sparse). "
            "Useful for SFT where the loss mask is sparse."
        ),
    )
    parser.add_argument(
        "--label_bucket_sizes",
        type=str,
        default=None,
        help=(
            "Optional comma-separated buckets for number of loss tokens when --sparse_loss is enabled. "
            "Defaults to --bucket_sizes; if both unset, uses powers-of-2 buckets up to seq_len."
        ),
    )
    parser.add_argument("--lora_r", type=int, default=0, help="Enable LoRA with rank r (0 = disable).")
    parser.add_argument("--lora_alpha", type=float, default=16.0)
    parser.add_argument("--lora_dropout", type=float, default=0.0)
    parser.add_argument(
        "--lora_targets",
        type=str,
        default="q_proj,k_proj,v_proj,o_proj,gate_proj,up_proj,down_proj",
        help="Comma-separated Linear module names to LoRA-ize (matched by leaf name).",
    )
    parser.add_argument(
        "--checkpoint_every_n",
        type=int,
        default=0,
        help="Gradient checkpoint every N transformer blocks (0 = disable).",
    )
    parser.add_argument(
        "--bucket_sizes",
        type=str,
        default=None,
        help=(
            "Optional comma-separated bucketing sizes <= seq_len (e.g. 256,512,1024). "
            "If set, batches are padded to the smallest bucket that fits the sample to reduce padding."
        ),
    )
    parser.add_argument(
        "--profile_timing",
        action="store_true",
        help="Print per-step timing breakdown (adds extra synchronization; slower).",
    )
    parser.add_argument(
        "--profile_warmup_steps",
        type=int,
        default=2,
        help="Ignore the first N steps for timing stats when --profile_timing is enabled.",
    )
    parser.add_argument(
        "--compile",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Use mx.compile to speed up forward+backward (recommended).",
    )
    parser.add_argument(
        "--compile_optimizer",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Use mx.compile to fuse grad clipping + optimizer update (recommended).",
    )
    parser.add_argument(
        "--metal_capture",
        type=str,
        default=None,
        help="Optional path to write a Metal capture (.gputrace).",
    )
    parser.add_argument(
        "--metal_capture_steps",
        type=int,
        default=1,
        help="How many optimizer steps to capture when --metal_capture is set.",
    )
    parser.add_argument(
        "--metal_capture_start_step",
        type=int,
        default=None,
        help="Which global step to start capture (default: first step after resume).",
    )

    parser.add_argument(
        "--init_from",
        type=str,
        default=None,
        help="Initialise weights from a checkpoint dir (containing model.safetensors) or a .safetensors file.",
    )
    parser.add_argument(
        "--resume",
        type=str,
        default=None,
        help="Checkpoint directory produced by this script.",
    )

    args = parser.parse_args()

    if args.init_from and args.resume:
        raise ValueError("`--init_from` and `--resume` are mutually exclusive.")

    os.makedirs(args.out_dir, exist_ok=True)
    ckpt_dir = os.path.join(args.out_dir, "checkpoints")
    os.makedirs(ckpt_dir, exist_ok=True)

    set_seed(args.seed)

    try:
        from transformers import AutoTokenizer
    except ImportError as e:
        raise ImportError(
            "Failed to import `transformers`. Install MLX training deps via "
            "`python3 -m pip install -r mlx_train/requirements.txt`."
        ) from e

    tokenizer = AutoTokenizer.from_pretrained(args.tokenizer_path)
    if tokenizer.pad_token_id is None:
        raise ValueError("Tokenizer must define pad_token_id.")

    data_spec = resolve_data_path_spec(
        args.data_path,
        task=args.task,
        data_dir=args.data_dir,
        hf_repo_id=args.hf_dataset_repo,
        hf_endpoint=args.hf_endpoint,
        force_download=args.force_download,
        max_download_mb=args.max_download_mb,
        data_source=args.data_source,
        ms_repo_id=args.ms_dataset_repo or args.hf_dataset_repo,
        ms_cache_dir=args.ms_cache_dir,
        ms_revision=args.ms_revision,
    )
    paths = resolve_jsonl_paths(data_spec)
    cached_dataset = None
    if args.cache_tokenized:
        cached_dataset = pretokenize_jsonl(paths=paths, tokenizer=tokenizer, task=args.task)
        print(f"[data] cached {len(cached_dataset)} tokenized samples for reuse")

    cfg = (
        load_config_from_checkpoint(args.resume)
        if args.resume
        else make_config(args, tokenizer)
    )
    cfg.use_metal_kernels = bool(args.metal_kernels)
    if cfg.vocab_size != tokenizer.vocab_size:
        raise ValueError(
            f"Config vocab_size={cfg.vocab_size} != tokenizer.vocab_size={tokenizer.vocab_size}"
        )
    model = MiniLLMForCausalLM(cfg)
    if int(args.checkpoint_every_n) > 0:
        model.model.checkpoint_every_n = int(args.checkpoint_every_n)

    start_step = 0
    resume_optimizer_path = None
    seen_tokens = 0
    resume_args: Optional[Dict[str, Any]] = None
    if args.resume:
        model_path = os.path.join(args.resume, "model.safetensors")
        opt_path = os.path.join(args.resume, "optimizer.npz")
        state_path = os.path.join(args.resume, "state.json")
        if not os.path.isfile(model_path):
            raise FileNotFoundError(model_path)
        model_size = os.path.getsize(model_path)
        if model_size < 8:
            raise RuntimeError(
                f"Checkpoint appears incomplete/corrupted: {model_path} ({model_size} bytes). "
                "Try resuming from the previous checkpoint and/or delete this directory."
            )
        try:
            model.load_weights(model_path)
        except Exception as e:
            raise RuntimeError(
                f"Failed to load weights from checkpoint: {model_path}. "
                "The checkpoint may be corrupted; try resuming from an earlier step."
            ) from e
        if os.path.isfile(state_path):
            with open(state_path, "r", encoding="utf-8") as f:
                state = json.load(f)
            start_step = int(state.get("step", 0))
            seen_tokens = int(state.get("seen_tokens", 0))
            maybe_args = state.get("args")
            if isinstance(maybe_args, dict):
                resume_args = maybe_args
        if os.path.isfile(opt_path) and os.path.getsize(opt_path) > 0:
            resume_optimizer_path = opt_path
        print(f"[resume] step={start_step} from {args.resume}")
    elif args.init_from:
        init_path = args.init_from
        if os.path.isdir(init_path):
            init_path = os.path.join(init_path, "model.safetensors")
        if not os.path.isfile(init_path):
            raise FileNotFoundError(init_path)
        try:
            model.load_weights(init_path)
        except Exception as e:
            if int(cfg.lora_r) > 0:
                # Base checkpoints don't have LoRA wrappers, so keys like `q_proj.weight`
                # need to map into `q_proj.base.weight` before loading.
                print(f"[init] strict load failed (LoRA enabled), trying base->LoRA key remap: {e}")
                weights = dict(mx.load(init_path))
                weights = _remap_base_weights_for_lora(weights, lora_targets=cfg.lora_targets)
                model.load_weights(list(weights.items()), strict=False)
            else:
                raise
        print(f"[init] loaded weights from {init_path}")

    # Dtype casting for memory/throughput.
    dtype_map = {"float16": mx.float16, "bfloat16": mx.bfloat16, "float32": mx.float32}
    model.apply(lambda p: p.astype(dtype_map[args.dtype]))

    params = model.parameters()
    trainable = model.trainable_parameters()
    n_params = count_parameters(params)
    n_bytes = parameters_bytes(params)
    n_trainable = count_parameters(trainable)
    print(
        f"[model] params={n_params / 1e6:.2f}M (trainable={n_trainable / 1e6:.2f}M) "
        f"| approx_size={n_bytes / 1024 / 1024:.1f} MiB | dtype={args.dtype}"
    )

    if resume_args is not None:
        prev_opt = str(resume_args.get("optimizer", "adamw"))
        prev_state_dtype = str(resume_args.get("optim_state_dtype", "float32"))
        if str(args.optimizer) != prev_opt or str(args.optim_state_dtype) != prev_state_dtype:
            raise ValueError(
                "Optimizer mismatch for --resume checkpoint.\n"
                f"- checkpoint: optimizer={prev_opt} optim_state_dtype={prev_state_dtype}\n"
                f"- current:    optimizer={args.optimizer} optim_state_dtype={args.optim_state_dtype}\n"
                "Use matching flags, or resume from a checkpoint created with the desired optimizer."
            )

    optimizer = make_optimizer(
        name=str(args.optimizer),
        learning_rate=float(args.learning_rate),
        weight_decay=float(args.weight_decay),
        state_dtype=str(args.optim_state_dtype),
    )
    if resume_optimizer_path is not None:
        load_optimizer_state(optimizer, resume_optimizer_path)

    # Estimate total steps for lr scheduling (best-effort).
    total_steps = (
        args.max_steps if args.max_steps is not None else (args.epochs * 10**9)
    )

    model.train()
    use_sparse_loss = bool(args.sparse_loss)
    if use_sparse_loss:
        loss_fn_wrapped = lambda x, y, m, p, pm: sparse_loss_fn(
            model,
            x,
            y,
            m,
            p,
            pm,
            logits_chunk_size=int(args.logits_chunk_size),
        )
    else:
        loss_fn_wrapped = lambda x, y, m: loss_fn(
            model, x, y, m, logits_chunk_size=int(args.logits_chunk_size)
        )
    raw_value_and_grad = nn.value_and_grad(model, loss_fn_wrapped)

    compiled_train_steps: Dict[Tuple[int, int], Callable[..., Tuple[mx.array, mx.array]]] = {}
    compiled_value_and_grads: Dict[Tuple[int, int], Callable[..., Tuple[mx.array, Any]]] = {}
    compiled_opt_step: Optional[Callable[..., mx.array]] = None

    def get_compiled_train_step(seq_len: int, label_len: int) -> Callable[..., Tuple[mx.array, mx.array]]:
        key = (int(seq_len), int(label_len) if use_sparse_loss else 0)
        if key not in compiled_train_steps:
            compiled_train_steps[key] = compile_train_step(
                model=model,
                optimizer=optimizer,
                value_and_grad=raw_value_and_grad,
                batch_size=int(args.batch_size),
                seq_len=int(seq_len),
                label_len=int(label_len),
                accum_steps=int(args.accum_steps),
                grad_clip=float(args.grad_clip),
                sparse_loss=bool(use_sparse_loss),
            )
        return compiled_train_steps[key]

    def get_value_and_grad(seq_len: int, label_len: int) -> Callable[..., Tuple[mx.array, Any]]:
        key = (int(seq_len), int(label_len) if use_sparse_loss else 0)
        if key not in compiled_value_and_grads:
            compiled_value_and_grads[key] = compile_value_and_grad(
                raw_value_and_grad,
                model=model,
                batch_size=int(args.batch_size),
                seq_len=int(seq_len),
                label_len=int(label_len),
                sparse_loss=bool(use_sparse_loss),
            )
        return compiled_value_and_grads[key]

    if args.compile_optimizer:
        def opt_step(grads: Any) -> mx.array:
            if args.grad_clip > 0:
                grads, grad_norm = optim.clip_grad_norm(grads, max_norm=args.grad_clip)
            else:
                grad_norm = mx.array(0.0, dtype=mx.float32)
            optimizer.update(model, grads)
            return grad_norm

        compiled_opt_step = compile_optimizer_step(
            opt_step,
            model=model,
            optimizer=optimizer,
        )

    bucket_sizes = None
    if args.bucket_sizes is not None:
        bucket_sizes = [int(s) for s in str(args.bucket_sizes).split(",") if s.strip()]

    label_bucket_sizes = None
    if args.label_bucket_sizes is not None:
        label_bucket_sizes = [
            int(s) for s in str(args.label_bucket_sizes).split(",") if s.strip()
        ]

    global_step = start_step
    t0 = time.time()
    profile_timing = bool(args.profile_timing)
    timing_window_steps = 0
    timing_data_s = 0.0
    timing_to_mx_s = 0.0
    timing_fwd_bwd_s = 0.0
    timing_clip_s = 0.0
    timing_opt_s = 0.0
    timing_total_s = 0.0

    metal_capture_path: Optional[str] = args.metal_capture
    metal_capture_start_step = (
        start_step
        if args.metal_capture_start_step is None
        else int(args.metal_capture_start_step)
    )
    metal_capture_end_step: Optional[int] = None
    metal_capturing = False
    if metal_capture_path is not None and not metal_capture_path.endswith(".gputrace"):
        raise ValueError("--metal_capture must end with .gputrace")

    if args.max_steps is not None and start_step >= int(args.max_steps):
        print(
            f"[warn] start_step={start_step} >= max_steps={args.max_steps}; "
            "no training steps will run (use a fresh --out_dir or increase --max_steps)."
        )

    def save_checkpoint(step: int) -> str:
        path = os.path.join(ckpt_dir, f"step_{step:08d}")
        os.makedirs(path, exist_ok=True)

        model_path = os.path.join(path, "model.safetensors")
        model_tmp = model_path + ".tmp.safetensors"
        model.save_weights(model_tmp)
        os.replace(model_tmp, model_path)

        opt_path = os.path.join(path, "optimizer.npz")
        opt_tmp = opt_path + ".tmp.npz"
        save_optimizer_state(optimizer, opt_tmp)
        os.replace(opt_tmp, opt_path)

        config_path = os.path.join(path, "config.json")
        config_tmp = config_path + ".tmp"
        with open(config_tmp, "w", encoding="utf-8") as f:
            json.dump(cfg.to_dict(), f, ensure_ascii=False, indent=2)
        os.replace(config_tmp, config_path)

        state_path = os.path.join(path, "state.json")
        state_tmp = state_path + ".tmp"
        with open(state_tmp, "w", encoding="utf-8") as f:
            json.dump(
                {"step": step, "seen_tokens": seen_tokens, "args": vars(args)},
                f,
                ensure_ascii=False,
                indent=2,
            )
        os.replace(state_tmp, state_path)

        prune_checkpoints(ckpt_dir, keep_last=args.keep_last_checkpoints)
        return path

    try:
        for epoch in range(args.epochs):
            micro_iter = iter(
                make_microbatch_iterator(
                    paths=paths,
                    tokenizer=tokenizer,
                    task=args.task,
                    seq_len=int(args.seq_len),
                    batch_size=int(args.batch_size),
                    accum_steps=int(args.accum_steps),
                    shuffle_buffer=int(args.shuffle_buffer),
                    seed=int(args.seed) + int(epoch),
                    bucket_sizes=bucket_sizes,
                    return_label_positions=bool(use_sparse_loss),
                    label_bucket_sizes=label_bucket_sizes,
                    pretokenized=cached_dataset,
                )
            )

            while True:
                if args.max_steps is not None and global_step >= args.max_steps:
                    break

                if (
                    metal_capture_path is not None
                    and not metal_capturing
                    and global_step == metal_capture_start_step
                ):
                    try:
                        mx.metal.start_capture(metal_capture_path)
                    except Exception as e:
                        print(f"[metal] capture failed to start: {e}")
                        metal_capture_path = None
                        metal_capturing = False
                        metal_capture_end_step = None
                    else:
                        metal_capturing = True
                        metal_capture_end_step = global_step + int(
                            args.metal_capture_steps
                        )
                        print(
                            f"[metal] capture started: {metal_capture_path} (steps={args.metal_capture_steps})"
                        )

                step_t0 = time.perf_counter() if profile_timing else 0.0
                try:
                    data_t0 = time.perf_counter() if profile_timing else 0.0
                    group = next(micro_iter)
                    if profile_timing:
                        timing_data_s += time.perf_counter() - data_t0
                except StopIteration:
                    break  # finished this epoch

                micro_batches = int(group.micro_batches)
                step_seq_len = int(group.seq_len)
                xs = list(group.x)
                ys = list(group.y)
                ms = list(group.loss_mask)
                ps = list(group.label_pos) if use_sparse_loss and group.label_pos is not None else []
                pms = list(group.label_pos_mask) if use_sparse_loss and group.label_pos_mask is not None else []
                step_label_len = int(group.label_len) if use_sparse_loss else 0

                grad_norm = mx.array(0.0, dtype=mx.float32)

                lr = cosine_lr(
                    global_step,
                    total_steps,
                    args.learning_rate,
                    warmup_steps=args.warmup_steps,
                )
                optimizer.learning_rate = lr
                to_mx_t0 = time.perf_counter() if profile_timing else 0.0
                if micro_batches < int(args.accum_steps) and micro_batches > 0:
                    pad_m = [[0] * int(step_seq_len) for _ in range(int(args.batch_size))]
                    last_x = xs[micro_batches - 1]
                    last_y = ys[micro_batches - 1]
                    while len(xs) < int(args.accum_steps):
                        xs.append(last_x)
                        ys.append(last_y)
                        ms.append(pad_m)

                    if use_sparse_loss:
                        pad_p = [[0] * int(step_label_len) for _ in range(int(args.batch_size))]
                        pad_pm = [[0] * int(step_label_len) for _ in range(int(args.batch_size))]
                        last_p = ps[micro_batches - 1] if ps else pad_p
                        while len(ps) < int(args.accum_steps):
                            ps.append(last_p)
                            pms.append(pad_pm)

                x = mx.array(xs, dtype=mx.int32)
                y = mx.array(ys, dtype=mx.int32)
                m = mx.array(ms, dtype=mx.float32)
                if use_sparse_loss:
                    p = mx.array(ps, dtype=mx.int32)
                    pm = mx.array(pms, dtype=mx.float32)
                micro = mx.array(float(micro_batches), dtype=mx.float32)
                if profile_timing:
                    if use_sparse_loss:
                        mx.eval(x, y, m, p, pm, micro)
                    else:
                        mx.eval(x, y, m, micro)
                    timing_to_mx_s += time.perf_counter() - to_mx_t0

                opt_t0 = time.perf_counter() if profile_timing else 0.0
                if args.compile and args.compile_optimizer:
                    if use_sparse_loss:
                        loss_accum, grad_norm = get_compiled_train_step(int(step_seq_len), int(step_label_len))(
                            x, y, m, p, pm, micro
                        )
                    else:
                        loss_accum, grad_norm = get_compiled_train_step(int(step_seq_len), 0)(x, y, m, micro)
                else:
                    # Fallback: run `value_and_grad` micro-batches in Python.
                    grad_accum = None
                    loss_accum = mx.array(0.0, dtype=mx.float32)
                    for i in range(micro_batches):
                        fwd_bwd_t0 = time.perf_counter() if profile_timing else 0.0
                        if args.compile and not args.compile_optimizer:
                            if use_sparse_loss:
                                loss, grads = get_value_and_grad(int(step_seq_len), int(step_label_len))(
                                    x[i], y[i], m[i], p[i], pm[i]
                                )
                            else:
                                loss, grads = get_value_and_grad(int(step_seq_len), 0)(x[i], y[i], m[i])
                        else:
                            if use_sparse_loss:
                                loss, grads = raw_value_and_grad(x[i], y[i], m[i], p[i], pm[i])
                            else:
                                loss, grads = raw_value_and_grad(x[i], y[i], m[i])
                        if profile_timing:
                            mx.eval(loss, grads)
                            timing_fwd_bwd_s += time.perf_counter() - fwd_bwd_t0
                        loss_accum = loss_accum + loss.astype(mx.float32)
                        if grad_accum is None:
                            grad_accum = grads
                        else:
                            grad_accum = mlx_utils.tree_map(lambda a, b: a + b, grad_accum, grads)

                    assert grad_accum is not None
                    grad_accum = mlx_utils.tree_map(lambda g: g / micro_batches, grad_accum)

                    if compiled_opt_step is None and args.grad_clip > 0:
                        clip_t0 = time.perf_counter() if profile_timing else 0.0
                        grad_accum, grad_norm = optim.clip_grad_norm(
                            grad_accum, max_norm=args.grad_clip
                        )
                        if profile_timing:
                            mx.eval(grad_accum, grad_norm)
                            timing_clip_s += time.perf_counter() - clip_t0

                    if compiled_opt_step is None:
                        optimizer.update(model, grad_accum)
                    else:
                        grad_norm = compiled_opt_step(grad_accum)
                mx.eval(model.parameters(), optimizer.state, loss_accum, grad_norm)
                if profile_timing:
                    timing_opt_s += time.perf_counter() - opt_t0

                global_step += 1
                seen_tokens += int(args.batch_size) * int(step_seq_len) * micro_batches
                if profile_timing:
                    timing_total_s += time.perf_counter() - step_t0
                    if global_step > start_step + max(0, int(args.profile_warmup_steps)):
                        timing_window_steps += 1
                    else:
                        timing_data_s = 0.0
                        timing_to_mx_s = 0.0
                        timing_fwd_bwd_s = 0.0
                        timing_clip_s = 0.0
                        timing_opt_s = 0.0
                        timing_total_s = 0.0
                        timing_window_steps = 0

                if metal_capturing and metal_capture_end_step is not None:
                    if global_step >= metal_capture_end_step:
                        mx.metal.stop_capture()
                        metal_capturing = False
                        print(f"[metal] capture saved: {metal_capture_path}")
                        metal_capture_path = None

                if global_step % args.log_interval == 0:
                    dt = time.time() - t0
                    tok_s = seen_tokens / max(dt, 1e-6)
                    avg_loss = float(loss_accum.item()) / micro_batches
                    timing_msg = ""
                    if profile_timing and timing_window_steps > 0:
                        to_ms = 1000.0 / timing_window_steps
                        timing_msg = (
                            f" | step_ms={(timing_total_s * to_ms):.1f}"
                            f" data_ms={(timing_data_s * to_ms):.1f}"
                            f" to_mx_ms={(timing_to_mx_s * to_ms):.1f}"
                            f" fwd_bwd_ms={(timing_fwd_bwd_s * to_ms):.1f}"
                            f" clip_ms={(timing_clip_s * to_ms):.1f}"
                            f" opt_ms={(timing_opt_s * to_ms):.1f}"
                        )
                    print(
                        f"[train] step={global_step} epoch={epoch + 1}/{args.epochs} "
                        f"loss={avg_loss:.4f} lr={lr:.2e} tok/s={tok_s:.0f}{timing_msg}"
                    )

                if global_step % args.save_interval == 0:
                    path = save_checkpoint(global_step)
                    print(f"[ckpt] saved {path}")

            if args.max_steps is not None and global_step >= args.max_steps:
                break

    except KeyboardInterrupt:
        print("\n[train] interrupted, saving last checkpoint...")
    finally:
        path = save_checkpoint(global_step)
        print(f"[ckpt] saved {path}")


if __name__ == "__main__":
    main()
