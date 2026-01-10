from __future__ import annotations

import argparse
import statistics
import time
from dataclasses import dataclass
from typing import Any, Callable, Iterable, Optional, Tuple

import mlx.core as mx
import mlx.nn as nn
import mlx.optimizers as optim
import mlx.utils as mlx_utils

from ..config import MiniLLMConfig, minillm_200mb
from ..models import MiniLLMForCausalLM
from ..nn.lora import merge_lora
from ..optim import make_optimizer
from ..ops.loss import chunked_ce_loss, sparse_ce_loss


@dataclass(frozen=True)
class BenchResult:
    name: str
    steps: int
    seq_len: int
    batch_size: int
    ms_per_step: float
    tokens_per_s: float
    active_mem_mib: float
    peak_mem_mib: float


def _to_dtype(dtype: str) -> mx.Dtype:
    m = {"float16": mx.float16, "bfloat16": mx.bfloat16, "float32": mx.float32}
    if dtype not in m:
        raise ValueError(f"Unknown dtype: {dtype}")
    return m[dtype]


def _bytes_to_mib(x: int) -> float:
    return float(x) / 1024.0 / 1024.0


def _clear_cache() -> None:
    mx.clear_cache()


def _reset_peak_memory() -> None:
    mx.reset_peak_memory()


def _get_peak_memory() -> int:
    return int(mx.get_peak_memory())


def _get_active_memory() -> int:
    return int(mx.get_active_memory())


def _restore_tree_in_place(dst: Any, src: Any) -> Any:
    """
    Restore a Python tree `dst` from `src` without replacing container objects.

    MLX compiled functions capture stateful trees (e.g. optimizer.state) by
    reference; replacing the root container can break compiled caches.
    """
    if isinstance(dst, dict) and isinstance(src, dict):
        for k, v in src.items():
            if k in dst:
                dst[k] = _restore_tree_in_place(dst[k], v)
            else:
                dst[k] = v
        return dst
    if isinstance(dst, list) and isinstance(src, list):
        n = min(len(dst), len(src))
        for i in range(n):
            dst[i] = _restore_tree_in_place(dst[i], src[i])
        if len(src) > len(dst):
            dst.extend(src[len(dst) :])
        return dst
    return src


def _loss_fn(
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


def _sparse_loss_fn(
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


def _compile_train_step(
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

    # Warm up compilation.
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
        _restore_tree_in_place(optimizer.state, opt_backup)
        optimizer.init(model.trainable_parameters())

    return compiled


def bench_train(
    *,
    name: str,
    cfg: MiniLLMConfig,
    seq_len: int,
    batch_size: int,
    steps: int,
    warmup_steps: int,
    accum_steps: int,
    dtype: str,
    logits_chunk_size: int,
    checkpoint_every_n: int,
    sparse_loss: bool,
    mask_density: float,
    optimizer_name: str,
    optim_state_dtype: str,
    learning_rate: float,
    weight_decay: float,
    compile: bool,
    compile_optimizer: bool,
) -> BenchResult:
    _clear_cache()
    _reset_peak_memory()

    model = MiniLLMForCausalLM(cfg)
    model.apply(lambda p: p.astype(_to_dtype(dtype)))
    model.model.checkpoint_every_n = int(checkpoint_every_n)
    model.train()

    use_sparse_loss = bool(sparse_loss)

    if use_sparse_loss:
        def loss_wrapped(x: mx.array, y: mx.array, m: mx.array, p: mx.array, pm: mx.array) -> mx.array:
            return _sparse_loss_fn(
                model, x, y, m, p, pm, logits_chunk_size=int(logits_chunk_size)
            )
    else:
        def loss_wrapped(x: mx.array, y: mx.array, m: mx.array) -> mx.array:
            return _loss_fn(model, x, y, m, logits_chunk_size=int(logits_chunk_size))

    value_and_grad = nn.value_and_grad(model, loss_wrapped)

    optimizer = make_optimizer(
        name=str(optimizer_name),
        learning_rate=float(learning_rate),
        weight_decay=float(weight_decay),
        state_dtype=str(optim_state_dtype),
    )
    optimizer.init(model.trainable_parameters())

    density = float(mask_density)
    if not (0.0 <= density <= 1.0):
        raise ValueError("--mask_density must be in [0, 1]")
    label_len = max(1, int(round(int(seq_len) * density)))
    prefix = int(seq_len) - int(label_len)

    compiled_step: Optional[Callable[..., Tuple[mx.array, mx.array]]] = None
    if compile and compile_optimizer:
        compiled_step = _compile_train_step(
            model=model,
            optimizer=optimizer,
            value_and_grad=value_and_grad,
            batch_size=int(batch_size),
            seq_len=int(seq_len),
            label_len=int(label_len),
            accum_steps=int(accum_steps),
            grad_clip=1.0,
            sparse_loss=bool(use_sparse_loss),
        )

    def run_one_step(
        x: mx.array, y: mx.array, m: mx.array, micro: mx.array, p: Optional[mx.array], pm: Optional[mx.array]
    ) -> None:
        if compiled_step is not None:
            if use_sparse_loss:
                assert p is not None and pm is not None
                loss_sum, grad_norm = compiled_step(x, y, m, p, pm, micro)
            else:
                loss_sum, grad_norm = compiled_step(x, y, m, micro)
            mx.eval(loss_sum, grad_norm)
            return
        loss_sum = mx.array(0.0, dtype=mx.float32)
        grads_acc: Any = None
        for i in range(int(micro.item())):
            if use_sparse_loss:
                assert p is not None and pm is not None
                loss, grads = value_and_grad(x[i], y[i], m[i], p[i], pm[i])
            else:
                loss, grads = value_and_grad(x[i], y[i], m[i])
            loss_sum = loss_sum + loss.astype(mx.float32)
            grads_acc = grads if grads_acc is None else mlx_utils.tree_map(lambda a, b: a + b, grads_acc, grads)
        assert grads_acc is not None
        grads_acc = mlx_utils.tree_map(lambda g: g / mx.maximum(micro, mx.array(1.0, dtype=mx.float32)), grads_acc)
        optimizer.update(model, grads_acc)
        mx.eval(loss_sum, optimizer.state, model.parameters())

    # Synthetic batch (fixed shapes; match compiled path).
    xs = mx.random.randint(0, int(cfg.vocab_size), (int(accum_steps), int(batch_size), int(seq_len)), dtype=mx.int32)
    ys = mx.random.randint(0, int(cfg.vocab_size), (int(accum_steps), int(batch_size), int(seq_len)), dtype=mx.int32)
    ms = mx.concatenate(
        [
            mx.zeros((int(accum_steps), int(batch_size), int(prefix)), dtype=mx.float32),
            mx.ones((int(accum_steps), int(batch_size), int(label_len)), dtype=mx.float32),
        ],
        axis=-1,
    )
    micro = mx.array(float(accum_steps), dtype=mx.float32)
    p = None
    pm = None
    if use_sparse_loss:
        pos = mx.arange(int(prefix), int(seq_len), dtype=mx.int32)
        p = mx.broadcast_to(pos, (int(accum_steps), int(batch_size), int(label_len)))
        pm = mx.ones((int(accum_steps), int(batch_size), int(label_len)), dtype=mx.float32)
        mx.eval(xs, ys, ms, p, pm, micro)
    else:
        mx.eval(xs, ys, ms, micro)

    for _ in range(int(warmup_steps)):
        run_one_step(xs, ys, ms, micro, p, pm)

    mx.eval(model.parameters(), optimizer.state)
    active_before = _get_active_memory()

    _reset_peak_memory()
    t0 = time.perf_counter()
    for _ in range(int(steps)):
        run_one_step(xs, ys, ms, micro, p, pm)
    mx.eval(model.parameters(), optimizer.state)
    dt = time.perf_counter() - t0

    tokens = float(batch_size * seq_len * steps * max(accum_steps, 1))
    ms_per_step = (dt / max(steps, 1)) * 1000.0
    tok_s = tokens / max(dt, 1e-9)
    peak = max(_get_peak_memory(), active_before)
    return BenchResult(
        name=name,
        steps=int(steps),
        seq_len=int(seq_len),
        batch_size=int(batch_size),
        ms_per_step=float(ms_per_step),
        tokens_per_s=float(tok_s),
        active_mem_mib=_bytes_to_mib(active_before),
        peak_mem_mib=_bytes_to_mib(peak),
    )


def bench_infer(
    *,
    name: str,
    cfg: MiniLLMConfig,
    prompt_len: int,
    new_tokens: int,
    dtype: str,
) -> BenchResult:
    _clear_cache()
    _reset_peak_memory()

    model = MiniLLMForCausalLM(cfg)
    model.apply(lambda p: p.astype(_to_dtype(dtype)))
    model.eval()
    if int(cfg.lora_r) > 0:
        merge_lora(model)

    prompt = mx.random.randint(0, int(cfg.vocab_size), (1, int(prompt_len)), dtype=mx.int32)
    cache = model.allocate_kv_cache(batch_size=1, max_seq_len=int(prompt_len + new_tokens + 1))
    mx.eval(*(c.k for c in cache), *(c.v for c in cache))

    # Prefill (populate KV cache once)
    mx.eval(prompt)
    logits, cache = model.forward_with_cache(prompt, start_pos=0, cache=cache)
    mx.eval(logits)

    # Decode (fixed token id to reduce Python overhead; measures pure model throughput)
    token = mx.zeros((1, 1), dtype=mx.int32)
    mx.eval(token)

    if int(new_tokens) > 0:
        logits_w, _ = model.forward_with_cache(token, start_pos=int(prompt_len), cache=cache)
        mx.eval(logits_w)

    mx.eval(model.parameters(), *(c.k for c in cache), *(c.v for c in cache))
    active_before = _get_active_memory()

    _reset_peak_memory()
    t1 = time.perf_counter()
    for i in range(int(new_tokens)):
        pos = int(prompt_len) + i
        logits, cache = model.forward_with_cache(token, start_pos=pos, cache=cache)
        mx.eval(logits)
    t_decode = time.perf_counter() - t1

    # Report decode throughput as "tokens/s"
    tokens = float(new_tokens)
    tok_s = tokens / max(t_decode, 1e-9)
    peak = max(_get_peak_memory(), active_before)
    return BenchResult(
        name=name,
        steps=int(new_tokens),
        seq_len=int(prompt_len),
        batch_size=1,
        ms_per_step=(t_decode / max(new_tokens, 1)) * 1000.0,
        tokens_per_s=tok_s,
        active_mem_mib=_bytes_to_mib(active_before),
        peak_mem_mib=_bytes_to_mib(peak),
    )


def _print_result(r: BenchResult) -> None:
    print(
        f"[{r.name}] steps={r.steps} bs={r.batch_size} seq={r.seq_len} "
        f"step_ms={r.ms_per_step:.2f} tok/s={r.tokens_per_s:.0f} "
        f"active_mem={r.active_mem_mib:.1f} MiB peak_mem={r.peak_mem_mib:.1f} MiB"
    )


def _summarize_results(name: str, results: Iterable[BenchResult]) -> Optional[BenchResult]:
    rs = list(results)
    if not rs:
        return None
    return BenchResult(
        name=f"{name}_mean",
        steps=int(statistics.mean(r.steps for r in rs)),
        seq_len=int(statistics.mean(r.seq_len for r in rs)),
        batch_size=int(statistics.mean(r.batch_size for r in rs)),
        ms_per_step=float(statistics.mean(r.ms_per_step for r in rs)),
        tokens_per_s=float(statistics.mean(r.tokens_per_s for r in rs)),
        active_mem_mib=float(statistics.mean(r.active_mem_mib for r in rs)),
        peak_mem_mib=float(statistics.mean(r.peak_mem_mib for r in rs)),
    )


def main() -> None:
    parser = argparse.ArgumentParser(description="MiniLLM (MLX) speed benchmark (train + infer)")
    parser.add_argument("--preset", type=str, choices=["200mb", "tiny", "custom"], default="200mb")
    parser.add_argument("--hidden_size", type=int, default=768)
    parser.add_argument("--num_hidden_layers", type=int, default=15)
    parser.add_argument("--num_attention_heads", type=int, default=12)
    parser.add_argument("--num_key_value_heads", type=int, default=3)
    parser.add_argument("--vocab_size", type=int, default=6400)
    parser.add_argument("--rope_theta", type=float, default=1_000_000.0)
    parser.add_argument("--dropout", type=float, default=0.0)

    parser.add_argument("--seq_len", type=int, default=1024)
    parser.add_argument("--batch_size", type=int, default=1)
    parser.add_argument("--steps", type=int, default=5)
    parser.add_argument("--warmup_steps", type=int, default=1)
    parser.add_argument("--accum_steps", type=int, default=1)
    parser.add_argument("--dtype", type=str, choices=["float16", "bfloat16", "float32"], default="float16")
    parser.add_argument("--runs", type=int, default=1, help="Repeat the benchmark N times and report the mean.")

    parser.add_argument("--logits_chunk_size", type=int, default=0)
    parser.add_argument(
        "--sparse_loss",
        action=argparse.BooleanOptionalAction,
        default=False,
        help="Benchmark sparse gathered loss (only meaningful when mask_density < 1).",
    )
    parser.add_argument(
        "--mask_density",
        type=float,
        default=1.0,
        help="Fraction of tokens with loss_mask=1 (synthetic); used by --sparse_loss and to mimic SFT masks.",
    )
    parser.add_argument("--checkpoint_every_n", type=int, default=0)
    parser.add_argument(
        "--metal_kernels",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Use custom Metal fused kernels (RMSNorm / SiLU*mul).",
    )

    parser.add_argument("--learning_rate", type=float, default=3e-4)
    parser.add_argument("--weight_decay", type=float, default=0.1)
    parser.add_argument("--optimizer", type=str, choices=["adamw", "adafactor", "lion"], default="adamw")
    parser.add_argument(
        "--optim_state_dtype",
        type=str,
        choices=["float32", "param"],
        default="float32",
        help="AdamW state dtype: float32 (stable, memory-heavy) or param (uses parameter dtype).",
    )

    parser.add_argument("--lora_r", type=int, default=0)
    parser.add_argument("--lora_alpha", type=float, default=16.0)
    parser.add_argument("--lora_dropout", type=float, default=0.0)
    parser.add_argument(
        "--lora_targets",
        type=str,
        default="q_proj,k_proj,v_proj,o_proj,gate_proj,up_proj,down_proj",
    )

    parser.add_argument("--compile", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--compile_optimizer", action=argparse.BooleanOptionalAction, default=True)

    parser.add_argument("--infer_prompt_len", type=int, default=1024)
    parser.add_argument("--infer_new_tokens", type=int, default=256)

    args = parser.parse_args()

    if args.preset == "200mb":
        cfg = minillm_200mb()
    elif args.preset == "tiny":
        cfg = MiniLLMConfig(
            hidden_size=256,
            num_hidden_layers=4,
            num_attention_heads=4,
            num_key_value_heads=2,
            vocab_size=int(args.vocab_size),
            dropout=float(args.dropout),
            rope_theta=float(args.rope_theta),
            max_position_embeddings=32768,
            use_moe=False,
        ).finalize()
    else:
        cfg = MiniLLMConfig(
            hidden_size=int(args.hidden_size),
            num_hidden_layers=int(args.num_hidden_layers),
            num_attention_heads=int(args.num_attention_heads),
            num_key_value_heads=int(args.num_key_value_heads),
            vocab_size=int(args.vocab_size),
            dropout=float(args.dropout),
            rope_theta=float(args.rope_theta),
            max_position_embeddings=32768,
            use_moe=False,
        ).finalize()

    cfg.lora_r = int(args.lora_r)
    cfg.lora_alpha = float(args.lora_alpha)
    cfg.lora_dropout = float(args.lora_dropout)
    cfg.lora_targets = str(args.lora_targets)
    cfg.use_metal_kernels = bool(args.metal_kernels)

    head_dim = int(cfg.hidden_size) // int(cfg.num_attention_heads)
    print(
        f"[cfg] head_dim={head_dim} seq_len={int(args.seq_len)} preset={args.preset} "
        f"dtype={args.dtype} metal_kernels={bool(args.metal_kernels)} "
        f"sparse_loss={bool(args.sparse_loss)} mask_density={float(args.mask_density):.2f} "
        f"optimizer={args.optimizer} optim_state_dtype={args.optim_state_dtype}"
    )

    runs = max(1, int(args.runs))
    train_results: list[BenchResult] = []
    infer_results: list[BenchResult] = []
    for i in range(runs):
        suffix = f"run{i + 1}" if runs > 1 else ""
        r_train = bench_train(
            name=f"train_{suffix}" if suffix else "train",
            cfg=cfg,
            seq_len=int(args.seq_len),
            batch_size=int(args.batch_size),
            steps=int(args.steps),
            warmup_steps=int(args.warmup_steps),
            accum_steps=int(args.accum_steps),
            dtype=str(args.dtype),
            logits_chunk_size=int(args.logits_chunk_size),
            checkpoint_every_n=int(args.checkpoint_every_n),
            sparse_loss=bool(args.sparse_loss),
            mask_density=float(args.mask_density),
            optimizer_name=str(args.optimizer),
            optim_state_dtype=str(args.optim_state_dtype),
            learning_rate=float(args.learning_rate),
            weight_decay=float(args.weight_decay),
            compile=bool(args.compile),
            compile_optimizer=bool(args.compile_optimizer),
        )
        train_results.append(r_train)
        _print_result(r_train)

        r_infer = bench_infer(
            name=f"infer_decode_{suffix}" if suffix else "infer_decode",
            cfg=cfg,
            prompt_len=int(args.infer_prompt_len),
            new_tokens=int(args.infer_new_tokens),
            dtype=str(args.dtype),
        )
        infer_results.append(r_infer)
        _print_result(r_infer)

    if runs > 1:
        train_mean = _summarize_results("train", train_results)
        infer_mean = _summarize_results("infer_decode", infer_results)
        if train_mean is not None:
            _print_result(train_mean)
        if infer_mean is not None:
            _print_result(infer_mean)


if __name__ == "__main__":
    main()
