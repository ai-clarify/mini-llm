from __future__ import annotations

import argparse
import json
import math
import os
import re
import shutil
import subprocess
import time
from queue import Queue
from pathlib import Path
from threading import Thread
from typing import Any, Callable, Dict, Iterator, List, Optional, Sequence, Tuple

import mlx.core as mx
import mlx.nn as nn
import mlx.optimizers as optim
import mlx.utils as mlx_utils

from ..config import MiniLLMConfig, minillm_200mb
from ..data import (
    Bin2DDataset,
    BinDataset,
    detect_bin_format,
    looks_like_bin_path,
    make_dpo_microbatch_iterator,
    make_microbatch_iterator,
    pretokenize_dpo_jsonl,
    pretokenize_jsonl,
    resolve_bin_prefix,
    resolve_jsonl_paths,
)
from ..download import resolve_data_path_spec
from ..models import MiniLLMForCausalLM, count_parameters, parameters_bytes
from ..optim import make_optimizer
from ..ops.loss import (
    chunked_ce_loss,
    chunked_ce_loss_sum_and_tokens,
    dpo_loss,
    sequence_logprobs,
    sparse_ce_loss,
)
from ..trace import TimingTracer, write_timing_trace


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


def count_jsonl_samples(paths: Sequence[str]) -> int:
    """
    Count non-empty JSONL lines across paths.

    Complexity: O(N) time, O(1) space for N lines (best/avg/worst).
    """
    if not paths:
        raise ValueError("paths must be non-empty to count JSONL samples.")
    total = 0
    for path in paths:
        with open(path, "r", encoding="utf-8") as f:
            for line in f:
                if line.strip():
                    total += 1
    return total


def _run_cmd_first(cmds: Sequence[Sequence[str]]) -> Optional[str]:
    for cmd in cmds:
        try:
            return subprocess.check_output(list(cmd), text=True).strip()
        except Exception:
            continue
    return None


def _read_sysctl_int(name: str) -> Optional[int]:
    out = _run_cmd_first(
        [
            ["/usr/sbin/sysctl", "-n", name],
            ["/usr/bin/sysctl", "-n", name],
            ["sysctl", "-n", name],
        ]
    )
    if out is None:
        return None
    try:
        return int(out)
    except ValueError:
        return None


def _read_vm_stat() -> Optional[str]:
    return _run_cmd_first(
        [
            ["/usr/bin/vm_stat"],
            ["/usr/sbin/vm_stat"],
            ["vm_stat"],
        ]
    )


def _parse_vm_stat() -> Tuple[Optional[int], Dict[str, int]]:
    out = _read_vm_stat()
    if not out:
        return None, {}
    page_size = None
    pages: Dict[str, int] = {}
    for line in out.splitlines():
        line = line.strip()
        if not line:
            continue
        if "page size of" in line and "bytes" in line:
            m = re.search(r"page size of (\d+) bytes", line)
            if m:
                page_size = int(m.group(1))
            continue
        if ":" not in line:
            continue
        key, val = line.split(":", 1)
        key = key.strip().lower()
        val = val.strip().strip(".")
        try:
            pages[key] = int(val)
        except ValueError:
            continue
    return page_size, pages


def _estimate_available_mem_bytes(*, mode: str = "available") -> Optional[int]:
    page_size, pages = _parse_vm_stat()
    if page_size is None or not pages:
        return None
    mode = str(mode).lower().strip()
    if mode == "free":
        keys = ("pages free", "pages speculative")
    else:
        # "available" includes reclaimable caches (inactive/purgeable).
        keys = ("pages free", "pages inactive", "pages speculative", "pages purgeable")
    avail_pages = 0
    for key in keys:
        if key in pages:
            avail_pages += pages[key]
    if avail_pages <= 0:
        return None
    avail_bytes = int(avail_pages) * int(page_size)
    total_bytes = _read_sysctl_int("hw.memsize")
    if total_bytes is not None and avail_bytes > total_bytes:
        avail_bytes = total_bytes
    return avail_bytes


def _bytes_per_dtype(dtype: str) -> int:
    dt = str(dtype).lower().strip()
    if dt in ("float16", "bfloat16"):
        return 2
    if dt == "float32":
        return 4
    return 4


def _estimate_bin_bytes(path: str) -> Optional[int]:
    prefix = resolve_bin_prefix(path)
    meta_path = Path(prefix + ".meta.json")
    files: List[Path] = []
    if meta_path.exists():
        try:
            with meta_path.open("r", encoding="utf-8") as f:
                meta = json.load(f)
            base_dir = meta_path.parent
            ids = meta.get("ids", {})
            ids_bin = ids.get("bin")
            if ids_bin:
                files.append(base_dir / str(ids_bin))
            labels = meta.get("labels", {})
            lbl_bin = labels.get("bin")
            if lbl_bin:
                files.append(base_dir / str(lbl_bin))
        except Exception:
            files = []
    if not files:
        ids_bin = Path(prefix + ".ids.bin")
        if ids_bin.exists():
            files.append(ids_bin)
        lbl_bin = Path(prefix + ".lbl.bin")
        if lbl_bin.exists():
            files.append(lbl_bin)
    if not files:
        return None
    total = 0
    for p in files:
        try:
            total += int(p.stat().st_size)
        except Exception:
            continue
    return total or None


def _choose_bin_cache(
    path: str,
    cache: str,
    *,
    avail_bytes: Optional[int],
    reserve_mb: int = 2048,
    ratio: float = 0.35,
) -> str:
    cache = str(cache).lower().strip()
    if cache != "auto":
        return cache
    bin_bytes = _estimate_bin_bytes(path)
    if bin_bytes is None:
        return "mmap"
    if avail_bytes is None:
        return "mmap"
    reserve = int(reserve_mb) * 1024 * 1024
    budget = max(0, int(avail_bytes) - reserve)
    if budget <= 0:
        return "mmap"
    if bin_bytes <= int(budget * float(ratio)):
        return "memory"
    return "mmap"


def _auto_logits_chunk_size(
    *,
    batch_size: int,
    seq_len: int,
    vocab_size: int,
    dtype: str,
    mem_limit_bytes: int,
    sparse_loss: bool,
    label_bucket_sizes: Optional[Sequence[int]],
    budget_ratio: float = 0.25,
) -> int:
    if mem_limit_bytes <= 0:
        return 0
    label_len = int(seq_len)
    if sparse_loss and label_bucket_sizes:
        label_len = int(max(label_bucket_sizes))
    bytes_per = _bytes_per_dtype(dtype)
    full_bytes = int(batch_size) * int(label_len) * int(vocab_size) * int(bytes_per)
    budget = int(float(mem_limit_bytes) * float(budget_ratio))
    if budget <= 0 or full_bytes <= budget:
        return 0
    per_step = int(batch_size) * int(vocab_size) * int(bytes_per)
    if per_step <= 0:
        return 0
    chunk = max(1, min(int(seq_len), int(budget // per_step)))
    return int(chunk)


def prefetch_iterator(items: Iterator[Any], *, prefetch: int) -> Iterator[Any]:
    if int(prefetch) <= 0:
        yield from items
        return

    queue: Queue[Any] = Queue(maxsize=int(prefetch))
    sentinel = object()
    errors: List[BaseException] = []

    def _worker() -> None:
        try:
            for item in items:
                queue.put(item)
        except BaseException as exc:
            errors.append(exc)
        finally:
            queue.put(sentinel)

    Thread(target=_worker, daemon=True).start()
    while True:
        item = queue.get()
        if item is sentinel:
            if errors:
                raise errors[0]
            break
        yield item


def estimate_steps_per_epoch(num_samples: int, batch_size: int, accum_steps: int) -> int:
    """
    Estimate optimizer steps per epoch from sample count.

    Complexity: O(1) time, O(1) space (best/avg/worst).
    """
    if num_samples <= 0:
        return 0
    if batch_size <= 0:
        raise ValueError("batch_size must be > 0")
    if accum_steps <= 0:
        raise ValueError("accum_steps must be > 0")
    per_step = int(batch_size) * int(accum_steps)
    steps = num_samples // per_step
    if num_samples % per_step >= int(batch_size):
        steps += 1
    return int(steps)


def _create_tensorboard_writer(log_dir: str) -> Optional[Any]:
    try:
        from torch.utils.tensorboard import SummaryWriter  # type: ignore[import]

        return SummaryWriter(log_dir=log_dir)
    except Exception:
        pass

    try:
        from tensorboard.compat.proto.event_pb2 import Event  # type: ignore[import]
        from tensorboard.compat.proto.summary_pb2 import Summary  # type: ignore[import]
        from tensorboard.summary.writer.event_file_writer import (  # type: ignore[import]
            EventFileWriter,
        )
    except Exception:
        print(
            "[warn] TensorBoard logging requested but tensorboard is not installed. "
            "Install via `pip install tensorboard` (or install torch) to enable it."
        )
        return None

    class _SimpleTBWriter:
        def __init__(self, path: str) -> None:
            self._writer = EventFileWriter(path)

        def add_scalar(self, tag: str, value: float, step: int) -> None:
            summary = Summary(value=[Summary.Value(tag=tag, simple_value=float(value))])
            event = Event(wall_time=time.time(), step=int(step), summary=summary)
            self._writer.add_event(event)

        def flush(self) -> None:
            self._writer.flush()

        def close(self) -> None:
            self._writer.close()

    return _SimpleTBWriter(log_dir)


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


def parse_int_schedule(spec: Optional[str]) -> List[Tuple[int, int]]:
    if not spec:
        return []
    items: List[Tuple[int, int]] = []
    for part in str(spec).split(","):
        part = part.strip()
        if not part:
            continue
        if ":" in part:
            step_s, val_s = part.split(":", 1)
        elif "=" in part:
            step_s, val_s = part.split("=", 1)
        else:
            continue
        try:
            step = int(step_s.strip())
            val = int(val_s.strip())
        except ValueError:
            continue
        items.append((step, val))
    items.sort(key=lambda x: x[0])
    return items


def schedule_value(step: int, schedule: List[Tuple[int, int]], default: int) -> int:
    if not schedule:
        return int(default)
    cur = int(default)
    for s, v in schedule:
        if int(step) >= int(s):
            cur = int(v)
        else:
            break
    return int(cur)


def parse_skip_connections(spec: Optional[str]) -> Optional[List[List[int]]]:
    if not spec:
        return None
    pairs: List[List[int]] = []
    for part in str(spec).split(","):
        part = part.strip()
        if not part:
            continue
        if "->" in part:
            left, right = part.split("->", 1)
        elif ":" in part:
            left, right = part.split(":", 1)
        elif "-" in part:
            left, right = part.split("-", 1)
        else:
            continue
        try:
            src = int(left.strip())
            tgt = int(right.strip())
        except ValueError:
            continue
        pairs.append([src, tgt])
    return pairs or None


def build_slow_param_mask(model: MiniLLMForCausalLM) -> Any:
    targets = {id(model.model.embed_tokens.weight), id(model.lm_head_weight())}

    def _mask(p: Any) -> Any:
        if isinstance(p, mx.array):
            return mx.array(1.0 if id(p) in targets else 0.0, dtype=p.dtype)
        if isinstance(p, dict):
            return {k: _mask(v) for k, v in p.items()}
        if isinstance(p, list):
            return [_mask(v) for v in p]
        if isinstance(p, tuple):
            return tuple(_mask(v) for v in p)
        return p

    return _mask(model.trainable_parameters())


def loss_fn(
    model: MiniLLMForCausalLM,
    x: mx.array,
    y: mx.array,
    loss_mask: mx.array,
    *,
    logits_chunk_size: int,
) -> mx.array:
    hidden, mtp_hidden, aux_loss = model.forward_with_mtp_hidden(x)
    logit_softcap = float(model.config.logit_softcap)
    loss = chunked_ce_loss(
        hidden=hidden,
        lm_head_weight=model.lm_head_weight(),
        labels=y,
        loss_mask=loss_mask,
        chunk_size=int(logits_chunk_size),
        logit_softcap=logit_softcap,
    )
    if mtp_hidden and float(model.config.mtp_loss_weight) > 0.0:
        mtp_loss = mx.array(0.0, dtype=mx.float32)
        count = 0
        for idx, mtp_h in enumerate(mtp_hidden):
            offset = idx + 2
            if int(y.shape[1]) <= offset:
                continue
            mtp_loss = mtp_loss + chunked_ce_loss(
                hidden=mtp_h[:, :-offset, :],
                lm_head_weight=model.lm_head_weight(),
                labels=y[:, offset:],
                loss_mask=loss_mask[:, offset:],
                chunk_size=int(logits_chunk_size),
                logit_softcap=logit_softcap,
            )
            count += 1
        if count > 0:
            loss = loss + (mtp_loss / float(count)) * float(model.config.mtp_loss_weight)
    return loss + aux_loss


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
    hidden, _, aux_loss = model.forward_with_mtp_hidden(x)
    logit_softcap = float(model.config.logit_softcap)
    loss = sparse_ce_loss(
        hidden=hidden,
        lm_head_weight=model.lm_head_weight(),
        labels=y,
        label_positions=label_positions,
        label_pos_mask=label_pos_mask,
        chunk_size=int(logits_chunk_size),
        logit_softcap=logit_softcap,
    )
    return loss + aux_loss


def _r1_weighted_mask(
    labels: mx.array,
    loss_mask: mx.array,
    *,
    special_ids: Optional[mx.array],
    token_weight: float,
) -> Tuple[mx.array, mx.array]:
    base_mask = loss_mask.astype(mx.float32)
    if special_ids is None or int(special_ids.size) == 0 or float(token_weight) == 1.0:
        return base_mask, base_mask
    matches = mx.equal(labels[..., None], special_ids)
    special_mask = mx.any(matches, axis=-1).astype(mx.float32)
    weighted = base_mask + (float(token_weight) - 1.0) * base_mask * special_mask
    return base_mask, weighted


def r1_loss_fn(
    model: MiniLLMForCausalLM,
    x: mx.array,
    y: mx.array,
    loss_mask: mx.array,
    *,
    logits_chunk_size: int,
    r1_special_ids: Optional[mx.array],
    r1_token_weight: float,
) -> mx.array:
    hidden, mtp_hidden, aux_loss = model.forward_with_mtp_hidden(x)
    logit_softcap = float(model.config.logit_softcap)
    base_mask, weighted_mask = _r1_weighted_mask(
        y, loss_mask, special_ids=r1_special_ids, token_weight=r1_token_weight
    )
    loss_sum, _ = chunked_ce_loss_sum_and_tokens(
        hidden=hidden,
        lm_head_weight=model.lm_head_weight(),
        labels=y,
        loss_mask=weighted_mask,
        chunk_size=int(logits_chunk_size),
        logit_softcap=logit_softcap,
    )
    base_tokens = mx.sum(base_mask)
    denom = mx.maximum(base_tokens, mx.array(1.0, dtype=mx.float32))
    loss = loss_sum / denom
    if mtp_hidden and float(model.config.mtp_loss_weight) > 0.0:
        mtp_loss = mx.array(0.0, dtype=mx.float32)
        count = 0
        for idx, mtp_h in enumerate(mtp_hidden):
            offset = idx + 2
            if int(y.shape[1]) <= offset:
                continue
            mtp_loss = mtp_loss + chunked_ce_loss(
                hidden=mtp_h[:, :-offset, :],
                lm_head_weight=model.lm_head_weight(),
                labels=y[:, offset:],
                loss_mask=weighted_mask[:, offset:],
                chunk_size=int(logits_chunk_size),
                logit_softcap=logit_softcap,
            )
            count += 1
        if count > 0:
            loss = loss + (mtp_loss / float(count)) * float(model.config.mtp_loss_weight)
    return loss + aux_loss


def dpo_loss_fn(
    model: MiniLLMForCausalLM,
    ref_model: MiniLLMForCausalLM,
    x: mx.array,
    y: mx.array,
    loss_mask: mx.array,
    *,
    logits_chunk_size: int,
    beta: float,
) -> mx.array:
    hidden, _, _ = model.forward_with_mtp_hidden(x)
    policy_logp = sequence_logprobs(
        hidden=hidden,
        lm_head_weight=model.lm_head_weight(),
        labels=y,
        loss_mask=loss_mask,
        chunk_size=int(logits_chunk_size),
        logit_softcap=float(model.config.logit_softcap),
    )
    ref_hidden, _, _ = ref_model.forward_with_mtp_hidden(x)
    ref_hidden = mx.stop_gradient(ref_hidden)
    ref_logp = sequence_logprobs(
        hidden=ref_hidden,
        lm_head_weight=ref_model.lm_head_weight(),
        labels=y,
        loss_mask=loss_mask,
        chunk_size=int(logits_chunk_size),
        logit_softcap=float(ref_model.config.logit_softcap),
    )
    return dpo_loss(policy_logp=policy_logp, ref_logp=ref_logp, beta=float(beta))


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
    cfg.num_nextn_predict_layers = int(args.mtp_layers)
    cfg.mtp_loss_weight = float(args.mtp_loss_weight)
    cfg.hidden_act = str(args.hidden_act)
    cfg.qk_norm = bool(args.qk_norm)
    cfg.qk_norm_eps = float(args.qk_norm_eps)
    cfg.logit_softcap = float(args.logit_softcap)
    cfg.value_mix = float(args.value_mix)
    cfg.residual_scale = float(args.residual_scale)
    cfg.zero_init_residual = bool(args.zero_init_residual)
    cfg.residual_decay = float(args.residual_decay)
    cfg.tie_word_embeddings = bool(args.tie_word_embeddings)
    cfg.untie_lm_head_at_ratio = float(args.untie_lm_head_at_ratio)
    cfg.embed_skip_scale = float(args.embed_skip_scale)
    cfg.embed_skip_gate = bool(args.embed_skip_gate)
    cfg.skip_connections = parse_skip_connections(args.skip_connections)
    cfg.skip_scale = float(args.skip_scale)
    cfg.skip_gate = bool(args.skip_gate)
    cfg.value_embed_count = int(args.value_embed_count)
    cfg.value_embed_scale = float(args.value_embed_scale)
    cfg.value_embed_gate = bool(args.value_embed_gate)
    cfg.value_embed_repeat_ends = bool(args.value_embed_repeat_ends)
    cfg.smear = bool(args.smear)
    cfg.smear_scale = float(args.smear_scale)
    cfg.bigram_hash_size = int(args.bigram_hash_size)
    cfg.bigram_hash_scale = float(args.bigram_hash_scale)
    cfg.bigram_hash_base = int(args.bigram_hash_base)
    cfg.partial_key_offset = int(args.partial_key_offset)
    cfg.paired_heads = bool(args.paired_heads)
    cfg.attn_window = int(args.attn_window)
    cfg.attn_global_tokens = int(args.attn_global_tokens)
    cfg.sparse_attn_gate = bool(args.sparse_attn_gate)
    cfg.sparse_attn_gate_topk = int(args.sparse_attn_gate_topk)
    cfg.back_out_ratio = float(args.back_out_ratio)
    cfg.back_out_scale = float(args.back_out_scale)

    moe_args = [
        args.n_routed_experts,
        args.num_experts_per_tok,
        args.n_shared_experts,
        args.n_group,
        args.topk_group,
        args.scoring_func,
        args.norm_topk_prob,
        args.routed_scaling_factor,
        args.aux_loss_alpha,
        args.seq_aux,
        args.moe_layer_freq,
        args.first_k_dense_replace,
        args.moe_intermediate_size,
    ]
    if args.use_moe is not None:
        cfg.use_moe = bool(args.use_moe)
    elif any(v is not None for v in moe_args):
        cfg.use_moe = True

    if args.n_routed_experts is not None:
        cfg.n_routed_experts = int(args.n_routed_experts)
    if args.num_experts_per_tok is not None:
        cfg.num_experts_per_tok = int(args.num_experts_per_tok)
    if args.n_shared_experts is not None:
        cfg.n_shared_experts = int(args.n_shared_experts)
    if args.n_group is not None:
        cfg.n_group = int(args.n_group)
    if args.topk_group is not None:
        cfg.topk_group = int(args.topk_group)
    if args.scoring_func is not None:
        cfg.scoring_func = str(args.scoring_func)
    if args.norm_topk_prob is not None:
        cfg.norm_topk_prob = bool(args.norm_topk_prob)
    if args.routed_scaling_factor is not None:
        cfg.routed_scaling_factor = float(args.routed_scaling_factor)
    if args.aux_loss_alpha is not None:
        cfg.aux_loss_alpha = float(args.aux_loss_alpha)
    if args.seq_aux is not None:
        cfg.seq_aux = bool(args.seq_aux)
    if args.moe_layer_freq is not None:
        cfg.moe_layer_freq = int(args.moe_layer_freq)
    if args.first_k_dense_replace is not None:
        cfg.first_k_dense_replace = int(args.first_k_dense_replace)
    if args.moe_intermediate_size is not None:
        cfg.moe_intermediate_size = int(args.moe_intermediate_size)

    cfg.index_n_heads = int(args.index_n_heads)
    cfg.index_head_dim = int(args.index_head_dim)
    cfg.index_topk = int(args.index_topk)

    if args.attn_gate is not None:
        cfg.use_attn_gate = bool(args.attn_gate)
    if bool(cfg.use_attn_gate):
        cfg.attn_gate_init = float(args.attn_gate_init)

    return cfg.finalize()


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
        "--tokenizer_type",
        type=str,
        choices=["auto", "hf", "rustbpe"],
        default="auto",
        help="Tokenizer backend: auto (prefer RustBPE), hf, or rustbpe.",
    )
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
        "--data_format",
        type=str,
        choices=["auto", "jsonl", "bin", "bin2d"],
        default="auto",
        help="Dataset format: jsonl, bin (mlx bin format), bin2d (fixed-length), or auto-detect.",
    )
    parser.add_argument(
        "--bin_cache",
        type=str,
        choices=["mmap", "memory", "auto"],
        default="mmap",
        help="Binary dataset cache mode: mmap, memory, or auto (choose based on free RAM).",
    )
    parser.add_argument(
        "--task", type=str, choices=["pretrain", "sft", "r1", "dpo"], default="pretrain"
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
        "--hidden_act",
        type=str,
        default="silu",
        choices=["silu", "relu2"],
        help="MLP activation (silu or relu2).",
    )
    parser.add_argument(
        "--tie_word_embeddings",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Tie input embedding and lm_head weights (default: True).",
    )
    parser.add_argument(
        "--untie_lm_head_at_ratio",
        type=float,
        default=0.0,
        help="Untie embed/lm_head at this fraction of total steps (0 = disable).",
    )
    parser.add_argument(
        "--embed_skip_scale",
        type=float,
        default=0.0,
        help="Scale for embedding skip added into each block (0 = disable).",
    )
    parser.add_argument(
        "--embed_skip_gate",
        action=argparse.BooleanOptionalAction,
        default=False,
        help="Learn a gate for the embedding skip connection.",
    )
    parser.add_argument(
        "--skip_connections",
        type=str,
        default=None,
        help="Comma-separated skip pairs like '3->6,5->7' (0-indexed).",
    )
    parser.add_argument(
        "--skip_scale",
        type=float,
        default=1.0,
        help="Scale factor for cross-layer skip connections.",
    )
    parser.add_argument(
        "--skip_gate",
        action=argparse.BooleanOptionalAction,
        default=False,
        help="Learn gates for cross-layer skip connections.",
    )
    parser.add_argument(
        "--value_embed_count",
        type=int,
        default=0,
        help="Number of extra value embedding tables (0 = disable).",
    )
    parser.add_argument(
        "--value_embed_scale",
        type=float,
        default=0.0,
        help="Scale for extra value embeddings mixed into attention values.",
    )
    parser.add_argument(
        "--value_embed_gate",
        action=argparse.BooleanOptionalAction,
        default=False,
        help="Learn a gate for value embedding mix.",
    )
    parser.add_argument(
        "--value_embed_repeat_ends",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Repeat value embeddings on first/last layers.",
    )
    parser.add_argument(
        "--smear",
        action=argparse.BooleanOptionalAction,
        default=False,
        help="Enable smear module (1-token lookback).",
    )
    parser.add_argument(
        "--smear_scale",
        type=float,
        default=0.0,
        help="Scale for smear module output.",
    )
    parser.add_argument(
        "--bigram_hash_size",
        type=int,
        default=0,
        help="Bigram hash embedding table size (0 = disable).",
    )
    parser.add_argument(
        "--bigram_hash_scale",
        type=float,
        default=0.0,
        help="Scale for bigram hash embeddings.",
    )
    parser.add_argument(
        "--bigram_hash_base",
        type=int,
        default=1000003,
        help="Hash base for bigram embeddings.",
    )
    parser.add_argument(
        "--qk_norm",
        action=argparse.BooleanOptionalAction,
        default=False,
        help="Apply RMS normalization to Q/K vectors before attention.",
    )
    parser.add_argument(
        "--qk_norm_eps",
        type=float,
        default=1e-6,
        help="Epsilon for Q/K RMS normalization.",
    )
    parser.add_argument(
        "--logit_softcap",
        type=float,
        default=0.0,
        help="If >0, apply tanh softcap to logits (cap * tanh(logits/cap)).",
    )
    parser.add_argument(
        "--value_mix",
        type=float,
        default=0.0,
        help="Mix input projection into attention values (0 = disable).",
    )
    parser.add_argument(
        "--partial_key_offset",
        type=int,
        default=0,
        help="Shift rotary positions for keys by this offset.",
    )
    parser.add_argument(
        "--paired_heads",
        action=argparse.BooleanOptionalAction,
        default=False,
        help="Pair adjacent attention heads (averaged).",
    )
    parser.add_argument(
        "--attn_window",
        type=int,
        default=0,
        help="Sliding window size for attention (0 = full causal).",
    )
    parser.add_argument(
        "--attn_global_tokens",
        type=int,
        default=0,
        help="Number of global tokens always visible in window attention.",
    )
    parser.add_argument(
        "--sparse_attn_gate",
        action=argparse.BooleanOptionalAction,
        default=False,
        help="Enable per-head attention gating with optional top-k sparsity.",
    )
    parser.add_argument(
        "--sparse_attn_gate_topk",
        type=int,
        default=0,
        help="Keep top-k attention head gates when sparse gating is enabled (0 = keep all).",
    )
    parser.add_argument(
        "--residual_scale",
        type=float,
        default=1.0,
        help="Scale factor for residual branches (1.0 = default).",
    )
    parser.add_argument(
        "--residual_decay",
        type=float,
        default=0.0,
        help="Exponential decay rate for residual scale across layers.",
    )
    parser.add_argument(
        "--zero_init_residual",
        action=argparse.BooleanOptionalAction,
        default=False,
        help="Zero-initialize attention/MLP output projections.",
    )
    parser.add_argument(
        "--back_out_ratio",
        type=float,
        default=0.0,
        help="Subtract hidden state at this fraction of layers before final norm (0 = disable).",
    )
    parser.add_argument(
        "--back_out_scale",
        type=float,
        default=1.0,
        help="Scale for back-out hidden subtraction.",
    )
    parser.add_argument("--mtp_layers", type=int, default=1, help="Number of MTP predictor layers.")
    parser.add_argument("--mtp_loss_weight", type=float, default=0.1, help="Weight for MTP auxiliary loss.")
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
    parser.add_argument(
        "--use_moe",
        action=argparse.BooleanOptionalAction,
        default=None,
        help="Enable MoE MLP blocks (default: preset).",
    )
    parser.add_argument("--n_routed_experts", type=int, default=None, help="Number of routed MoE experts.")
    parser.add_argument("--num_experts_per_tok", type=int, default=None, help="Top-k experts per token.")
    parser.add_argument("--n_shared_experts", type=int, default=None, help="Number of shared experts.")
    parser.add_argument("--n_group", type=int, default=None, help="Number of expert groups for routing.")
    parser.add_argument("--topk_group", type=int, default=None, help="Top-k groups to consider for routing.")
    parser.add_argument(
        "--scoring_func",
        type=str,
        default=None,
        choices=["softmax", "sigmoid"],
        help="MoE gate scoring function.",
    )
    parser.add_argument(
        "--norm_topk_prob",
        action=argparse.BooleanOptionalAction,
        default=None,
        help="Normalize top-k routing probabilities.",
    )
    parser.add_argument("--routed_scaling_factor", type=float, default=None, help="MoE routed scaling factor.")
    parser.add_argument("--aux_loss_alpha", type=float, default=None, help="MoE auxiliary loss weight.")
    parser.add_argument(
        "--seq_aux",
        action=argparse.BooleanOptionalAction,
        default=None,
        help="Use sequence-level MoE auxiliary loss (else token-level).",
    )
    parser.add_argument("--moe_layer_freq", type=int, default=None, help="Use MoE every N layers.")
    parser.add_argument(
        "--first_k_dense_replace",
        type=int,
        default=None,
        help="Keep the first K layers dense before enabling MoE.",
    )
    parser.add_argument(
        "--moe_intermediate_size",
        type=int,
        default=None,
        help="Intermediate size for MoE expert MLP.",
    )
    parser.add_argument(
        "--index_n_heads",
        type=int,
        default=0,
        help="Enable indexer with N heads (0 = disable).",
    )
    parser.add_argument(
        "--index_head_dim",
        type=int,
        default=32,
        help="Indexer head dimension.",
    )
    parser.add_argument(
        "--index_topk",
        type=int,
        default=0,
        help="Indexer top-k keys per query (0 = disable).",
    )

    parser.add_argument("--seq_len", type=int, default=512)
    parser.add_argument("--batch_size", type=int, default=4)
    parser.add_argument("--accum_steps", type=int, default=1)
    parser.add_argument(
        "--batch_size_schedule",
        type=str,
        default=None,
        help="Optional batch size schedule like '0:24,1000:32' (applied at epoch boundaries).",
    )
    parser.add_argument(
        "--accum_schedule",
        type=str,
        default=None,
        help="Optional accum_steps schedule like '0:2,1000:1' (applied at epoch boundaries).",
    )
    parser.add_argument("--learning_rate", type=float, default=3e-4)
    parser.add_argument("--weight_decay", type=float, default=0.1)
    parser.add_argument(
        "--weight_decay_schedule",
        type=str,
        default="none",
        choices=["none", "lr"],
        help="Weight decay schedule: none or lr (scale by lr/base_lr).",
    )
    parser.add_argument(
        "--optimizer",
        type=str,
        choices=["adamw", "adafactor", "lion", "muon"],
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
    parser.add_argument("--muon_momentum", type=float, default=0.9)
    parser.add_argument("--muon_ns_steps", type=int, default=5)
    parser.add_argument("--muon_eps", type=float, default=1e-7)
    parser.add_argument("--muon_adam_beta1", type=float, default=0.9)
    parser.add_argument("--muon_adam_beta2", type=float, default=0.999)
    parser.add_argument("--muon_adam_eps", type=float, default=1e-8)
    parser.add_argument(
        "--muon_adam_for_1d",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Use AdamW-style update for 1D parameters when optimizer=muon.",
    )
    parser.add_argument(
        "--muon_variant",
        type=str,
        choices=["default", "polar", "norm"],
        default="default",
        help="Muon variant: default | polar (fewer NS steps) | norm (normalize update).",
    )
    parser.add_argument(
        "--muon_normalize_update",
        action=argparse.BooleanOptionalAction,
        default=False,
        help="Normalize Muon update to unit Frobenius norm.",
    )
    parser.add_argument("--grad_clip", type=float, default=1.0)
    parser.add_argument(
        "--slow_embed_head_steps",
        type=int,
        default=1,
        help="Update embedding/lm_head every N steps (1 = normal).",
    )
    parser.add_argument("--warmup_steps", type=int, default=0)
    parser.add_argument("--epochs", type=int, default=1)
    parser.add_argument("--max_steps", type=int, default=None)
    parser.add_argument("--shuffle_buffer", type=int, default=2048)
    parser.add_argument(
        "--pack_pretrain",
        action=argparse.BooleanOptionalAction,
        default=False,
        help="Pack pretrain documents into full sequences to reduce padding (pretrain only).",
    )
    parser.add_argument(
        "--pack_eos",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="When packing, append EOS tokens between documents.",
    )
    parser.add_argument(
        "--pack_no_doc_split",
        action=argparse.BooleanOptionalAction,
        default=False,
        help="When packing, do not split a document across sequences (align batch starts with EOS).",
    )
    parser.add_argument(
        "--max_doc_tokens",
        type=int,
        default=0,
        help="Optional per-document token cap before packing/truncation (0 = no cap).",
    )
    parser.add_argument(
        "--drop_long",
        action=argparse.BooleanOptionalAction,
        default=False,
        help="Drop documents longer than --max_doc_tokens instead of truncating.",
    )
    parser.add_argument(
        "--cache_tokenized",
        action=argparse.BooleanOptionalAction,
        default=False,
        help="Pre-tokenize the dataset once and reuse token IDs across epochs (trades RAM for speed).",
    )
    parser.add_argument(
        "--prefetch_batches",
        type=int,
        default=0,
        help="Prefetch N micro-batch groups in a background thread to overlap CPU data prep with GPU compute.",
    )
    parser.add_argument(
        "--log_memory",
        action=argparse.BooleanOptionalAction,
        default=False,
        help="Log MLX Metal memory stats (active/cache/peak) alongside training metrics.",
    )
    parser.add_argument(
        "--memory_limit_mb",
        type=int,
        default=0,
        help="Optional MLX Metal memory limit in MB (0 = no limit).",
    )
    parser.add_argument(
        "--memory_limit_auto_ratio",
        type=float,
        default=0.0,
        help="Auto-set memory limit as ratio of estimated free memory (0 = disable).",
    )
    parser.add_argument(
        "--memory_limit_auto_mode",
        type=str,
        choices=["available", "free"],
        default="free",
        help="Memory estimation mode for auto limit (free uses free+speculative only).",
    )
    parser.add_argument(
        "--memory_limit_auto_reserve_mb",
        type=int,
        default=0,
        help="Reserve MiB from auto-estimated memory to avoid OS pressure (0 = no reserve).",
    )
    parser.add_argument(
        "--memory_limit_auto_min_mb",
        type=int,
        default=0,
        help="Minimum MiB when auto memory limit is enabled (0 = no minimum).",
    )
    parser.add_argument(
        "--memory_limit_auto_max_mb",
        type=int,
        default=0,
        help="Maximum MiB when auto memory limit is enabled (0 = no maximum).",
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
    parser.add_argument(
        "--save_interval",
        type=int,
        default=200,
        help="Save a checkpoint every N steps (0 to disable).",
    )
    parser.add_argument(
        "--tensorboard_dir",
        type=str,
        default=None,
        help="Optional TensorBoard log dir (requires tensorboard or torch installed).",
    )
    parser.add_argument("--seed", type=int, default=1337)
    parser.add_argument(
        "--logits_chunk_size",
        type=int,
        default=0,
        help=(
            "Compute CE loss in sequence chunks to avoid materializing full [B,T,V] logits "
            "(0 = disable; may auto-tune when memory limit is enabled)."
        ),
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
    parser.add_argument(
        "--dpo_beta",
        type=float,
        default=0.1,
        help="DPO temperature beta (only used when --task dpo).",
    )
    parser.add_argument(
        "--dpo_ref_from",
        type=str,
        default=None,
        help="Reference checkpoint dir or model.safetensors for DPO (defaults to --init_from).",
    )
    parser.add_argument(
        "--r1_token_weight",
        type=float,
        default=10.0,
        help="Loss weight multiplier for reasoning special tokens (only used when --task r1).",
    )
    parser.add_argument(
        "--r1_tokens",
        type=str,
        default="<think>,</think>,<answer>,</answer>",
        help="Comma-separated tokens to upweight for R1 training.",
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
        "--trace_timing_out",
        type=str,
        default=None,
        help="Write a fine-grained timing trace JSON to a file or directory (disables compile for traced steps).",
    )
    parser.add_argument(
        "--trace_timing_steps",
        type=int,
        default=1,
        help="Number of optimizer steps to trace when --trace_timing_out is set.",
    )
    parser.add_argument(
        "--trace_timing_start_step",
        type=int,
        default=None,
        help="Which global step to start timing trace (default: first step after resume).",
    )
    parser.add_argument(
        "--trace_timing_memory",
        action="store_true",
        help="Include active/peak memory snapshots in timing trace events.",
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

    batch_schedule = parse_int_schedule(args.batch_size_schedule)
    accum_schedule = parse_int_schedule(args.accum_schedule)

    if int(args.slow_embed_head_steps) > 1 and (bool(args.compile) or bool(args.compile_optimizer)):
        print("[warn] slow_embed_head_steps requires Python update path; disabling --compile/--compile_optimizer")
        args.compile = False
        args.compile_optimizer = False

    if (
        args.preset in ("200mb", "tiny")
        and int(args.memory_limit_mb) == 0
        and float(args.memory_limit_auto_ratio) == 0.0
        and int(args.memory_limit_auto_reserve_mb) == 0
        and int(args.memory_limit_auto_max_mb) == 0
    ):
        args.memory_limit_auto_ratio = 0.9
        args.memory_limit_auto_mode = "available"
        args.memory_limit_auto_reserve_mb = 2048
        args.memory_limit_auto_max_mb = 0
        print(
            f"[metal] default auto memory limit for preset={args.preset}: "
            "mode=available ratio=0.90 reserve=2048MiB max=0MiB"
        )

    if args.init_from and args.resume:
        raise ValueError("`--init_from` and `--resume` are mutually exclusive.")

    is_dpo = str(args.task) == "dpo"
    is_r1 = str(args.task) == "r1"
    if is_dpo and bool(args.sparse_loss):
        raise ValueError("--sparse_loss is not supported for DPO.")
    if is_r1 and bool(args.sparse_loss):
        raise ValueError("--sparse_loss is not supported for R1.")
    if is_dpo and float(args.dpo_beta) <= 0.0:
        raise ValueError("--dpo_beta must be > 0 for DPO training.")
    if is_r1 and float(args.r1_token_weight) <= 0.0:
        raise ValueError("--r1_token_weight must be > 0 for R1 training.")

    os.makedirs(args.out_dir, exist_ok=True)
    ckpt_dir = os.path.join(args.out_dir, "checkpoints")
    os.makedirs(ckpt_dir, exist_ok=True)

    avail_mem_bytes: Optional[int] = None
    mem_limit_mb = int(args.memory_limit_mb)
    if mem_limit_mb <= 0 and float(args.memory_limit_auto_ratio) > 0.0:
        avail = _estimate_available_mem_bytes(mode=str(args.memory_limit_auto_mode))
        if avail is not None:
            avail_mem_bytes = int(avail)
            ratio = float(args.memory_limit_auto_ratio)
            reserve_mb = int(args.memory_limit_auto_reserve_mb)
            mem_limit_mb = int((avail / 1024 / 1024) * ratio) - reserve_mb
            min_mb = int(args.memory_limit_auto_min_mb)
            max_mb = int(args.memory_limit_auto_max_mb)
            if mem_limit_mb < 0:
                mem_limit_mb = 0
            if min_mb > 0:
                mem_limit_mb = max(mem_limit_mb, min_mb)
            if max_mb > 0:
                mem_limit_mb = min(mem_limit_mb, max_mb)
            print(
                f"[metal] auto memory limit: avail={int(avail / 1024 / 1024)}MiB "
                f"mode={args.memory_limit_auto_mode} ratio={ratio:.2f} "
                f"reserve={reserve_mb}MiB -> limit={mem_limit_mb}MiB"
            )
        else:
            print("[warn] auto memory limit requested but could not estimate free memory")

    if mem_limit_mb > 0:
        limit_bytes = int(mem_limit_mb) * 1024 * 1024
        try:
            if hasattr(mx, "set_memory_limit"):
                mx.set_memory_limit(limit_bytes)
            else:
                mx.metal.set_memory_limit(limit_bytes)
            print(f"[metal] memory_limit_mb={int(mem_limit_mb)}")
        except Exception as exc:
            print(f"[warn] failed to set memory limit: {exc}")

    set_seed(args.seed)

    from .tokenizer_utils import load_tokenizer

    tokenizer = load_tokenizer(
        args.tokenizer_path,
        tokenizer_type=str(args.tokenizer_type),
    )
    if tokenizer.pad_token_id is None:
        raise ValueError("Tokenizer must define pad_token_id.")

    r1_special_ids: Optional[mx.array] = None
    if is_r1:
        token_specs = [t.strip() for t in str(args.r1_tokens).split(",") if t.strip()]
        token_ids: List[int] = []
        for tok in token_specs:
            token_ids.extend(int(t) for t in tokenizer.encode(tok, add_special_tokens=False))
        token_ids = sorted({int(t) for t in token_ids})
        if token_ids:
            r1_special_ids = mx.array(token_ids, dtype=mx.int32)
            print(
                f"[r1] token_weight={float(args.r1_token_weight):.2f} "
                f"special_ids={len(token_ids)} tokens={','.join(token_specs)}"
            )
        else:
            print("[r1] No special token ids found; using base loss mask only.")

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
    bin_dataset: Optional[BinDataset] = None
    bin2d_dataset: Optional[Bin2DDataset] = None
    paths: Sequence[str] = []
    fmt = str(args.data_format)
    if fmt == "auto":
        detected = detect_bin_format(str(data_spec))
        fmt = detected if detected is not None else "jsonl"
    if fmt == "bin2d":
        if is_dpo:
            raise ValueError("Binary datasets are not supported for DPO.")
        avail_for_cache = avail_mem_bytes or _estimate_available_mem_bytes(mode="available")
        bin_cache = _choose_bin_cache(
            str(data_spec), str(args.bin_cache), avail_bytes=avail_for_cache
        )
        if str(args.bin_cache) == "auto":
            print(f"[data] bin_cache=auto resolved to {bin_cache}")
        bin2d_dataset = Bin2DDataset(str(data_spec), cache=str(bin_cache))
        print(f"[data] using bin2d dataset: {data_spec}")
    elif fmt == "bin":
        if is_dpo:
            raise ValueError("Binary datasets are not supported for DPO.")
        avail_for_cache = avail_mem_bytes or _estimate_available_mem_bytes(mode="available")
        bin_cache = _choose_bin_cache(
            str(data_spec), str(args.bin_cache), avail_bytes=avail_for_cache
        )
        if str(args.bin_cache) == "auto":
            print(f"[data] bin_cache=auto resolved to {bin_cache}")
        bin_dataset = BinDataset(str(data_spec), cache=str(bin_cache))
        print(f"[data] using bin dataset: {data_spec}")
    else:
        paths = resolve_jsonl_paths(data_spec)
    cached_dataset = None
    if args.cache_tokenized:
        if bin_dataset is not None or bin2d_dataset is not None:
            print("[warn] --cache_tokenized ignored for binary datasets")
        else:
            if is_dpo:
                cached_dataset = pretokenize_dpo_jsonl(paths=paths, tokenizer=tokenizer)
            else:
                cached_dataset = pretokenize_jsonl(paths=paths, tokenizer=tokenizer, task=args.task)
            print(f"[data] cached {len(cached_dataset)} tokenized samples for reuse")

    estimated_samples: Optional[int] = None
    estimated_steps_per_epoch: Optional[int] = None
    if args.max_steps is None:
        if bin2d_dataset is not None:
            estimated_samples = len(bin2d_dataset)
        elif bin_dataset is not None:
            estimated_samples = len(bin_dataset)
        elif cached_dataset is not None:
            estimated_samples = len(cached_dataset)
        else:
            estimated_samples = count_jsonl_samples(paths)
        estimated_steps_per_epoch = estimate_steps_per_epoch(
            num_samples=int(estimated_samples),
            batch_size=int(args.batch_size),
            accum_steps=int(args.accum_steps),
        )

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

    ref_model: Optional[MiniLLMForCausalLM] = None
    dpo_ref_from: Optional[str] = None
    if is_dpo:
        dpo_ref_from = args.dpo_ref_from
        if resume_args is not None:
            prev_ref = resume_args.get("dpo_ref_from") or resume_args.get("init_from")
            if dpo_ref_from is None:
                dpo_ref_from = prev_ref
            elif prev_ref and str(dpo_ref_from) != str(prev_ref):
                raise ValueError(
                    "DPO reference mismatch for --resume checkpoint.\n"
                    f"- checkpoint: dpo_ref_from={prev_ref}\n"
                    f"- current:    dpo_ref_from={dpo_ref_from}\n"
                    "Use the same --dpo_ref_from as the original run."
                )
        if dpo_ref_from is None:
            dpo_ref_from = args.init_from
        if dpo_ref_from is None:
            raise ValueError(
                "DPO training requires --dpo_ref_from (or --init_from) to load a reference model."
            )
        args.dpo_ref_from = dpo_ref_from
        ref_path = dpo_ref_from
        if os.path.isdir(ref_path):
            ref_path = os.path.join(ref_path, "model.safetensors")
        if not os.path.isfile(ref_path):
            raise FileNotFoundError(ref_path)
        ref_model = MiniLLMForCausalLM(cfg)
        try:
            ref_model.load_weights(ref_path)
        except Exception as e:
            if int(cfg.lora_r) > 0:
                print(f"[dpo] strict load failed (LoRA enabled), trying base->LoRA key remap: {e}")
                weights = dict(mx.load(ref_path))
                weights = _remap_base_weights_for_lora(weights, lora_targets=cfg.lora_targets)
                ref_model.load_weights(list(weights.items()), strict=False)
            else:
                raise
        ref_model.eval()
        print(f"[dpo] reference loaded from {ref_path}")

    # Dtype casting for memory/throughput.
    dtype_map = {"float16": mx.float16, "bfloat16": mx.bfloat16, "float32": mx.float32}
    model.apply(lambda p: p.astype(dtype_map[args.dtype]))
    if ref_model is not None:
        ref_model.apply(lambda p: p.astype(dtype_map[args.dtype]))

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
        muon_momentum=float(args.muon_momentum),
        muon_ns_steps=int(args.muon_ns_steps),
        muon_eps=float(args.muon_eps),
        muon_adam_beta1=float(args.muon_adam_beta1),
        muon_adam_beta2=float(args.muon_adam_beta2),
        muon_adam_eps=float(args.muon_adam_eps),
        muon_adam_for_1d=bool(args.muon_adam_for_1d),
        muon_variant=str(args.muon_variant),
        muon_normalize_update=bool(args.muon_normalize_update),
    )
    if resume_optimizer_path is not None:
        load_optimizer_state(optimizer, resume_optimizer_path)

    slow_steps = max(1, int(args.slow_embed_head_steps))
    slow_param_mask = build_slow_param_mask(model) if slow_steps > 1 else None
    slow_grad_accum: Optional[Any] = None

    # Estimate total steps for lr scheduling (best-effort).
    if args.max_steps is not None:
        total_steps = int(args.max_steps)
    elif estimated_steps_per_epoch is not None:
        total_steps = int(estimated_steps_per_epoch) * int(args.epochs)
        if estimated_samples is not None:
            print(
                f"[data] samples={estimated_samples} estimated_steps_per_epoch={estimated_steps_per_epoch} "
                f"total_steps={total_steps}"
            )
    else:
        total_steps = 50_000

    untie_step: Optional[int] = None
    if float(args.untie_lm_head_at_ratio) > 0.0 and total_steps > 0:
        untie_step = int(float(args.untie_lm_head_at_ratio) * float(total_steps))
        untie_step = max(0, untie_step)

    tb_writer = None
    if args.tensorboard_dir:
        os.makedirs(args.tensorboard_dir, exist_ok=True)
        tb_writer = _create_tensorboard_writer(args.tensorboard_dir)
        if tb_writer is not None:
            print(f"[tb] logging to {args.tensorboard_dir}")

    model.train()
    cur_batch_size = schedule_value(start_step, batch_schedule, int(args.batch_size))
    cur_accum_steps = schedule_value(start_step, accum_schedule, int(args.accum_steps))
    effective_batch_size = int(cur_batch_size) * (2 if is_dpo else 1)
    use_sparse_loss = bool(args.sparse_loss) if not (is_dpo or is_r1) else False

    bucket_sizes = None
    if args.bucket_sizes is not None:
        bucket_sizes = [int(s) for s in str(args.bucket_sizes).split(",") if s.strip()]

    label_bucket_sizes = None
    if args.label_bucket_sizes is not None:
        label_bucket_sizes = [int(s) for s in str(args.label_bucket_sizes).split(",") if s.strip()]

    auto_logits_chunk = int(args.logits_chunk_size) == 0 and int(mem_limit_mb) > 0
    if auto_logits_chunk:
        label_len = int(args.seq_len)
        if use_sparse_loss and label_bucket_sizes:
            label_len = int(max(label_bucket_sizes))
        bytes_per = _bytes_per_dtype(str(args.dtype))
        full_logits_bytes = (
            int(effective_batch_size) * int(label_len) * int(model.config.vocab_size) * int(bytes_per)
        )
        auto_chunk = _auto_logits_chunk_size(
            batch_size=int(effective_batch_size),
            seq_len=int(args.seq_len),
            vocab_size=int(model.config.vocab_size),
            dtype=str(args.dtype),
            mem_limit_bytes=int(mem_limit_mb) * 1024 * 1024,
            sparse_loss=bool(use_sparse_loss),
            label_bucket_sizes=label_bucket_sizes,
        )
        if auto_chunk > 0:
            args.logits_chunk_size = int(auto_chunk)
            print(
                "[mem] auto logits_chunk_size="
                f"{int(args.logits_chunk_size)} (batch={effective_batch_size} "
                f"full_logits={full_logits_bytes / 1024 / 1024:.1f}MiB)"
            )
    if is_dpo:
        if ref_model is None:
            raise RuntimeError("DPO reference model was not initialized.")
        loss_fn_wrapped = lambda x, y, m: dpo_loss_fn(
            model,
            ref_model,
            x,
            y,
            m,
            logits_chunk_size=int(args.logits_chunk_size),
            beta=float(args.dpo_beta),
        )
    elif is_r1:
        loss_fn_wrapped = lambda x, y, m: r1_loss_fn(
            model,
            x,
            y,
            m,
            logits_chunk_size=int(args.logits_chunk_size),
            r1_special_ids=r1_special_ids,
            r1_token_weight=float(args.r1_token_weight),
        )
    elif use_sparse_loss:
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

    def get_compiled_train_step(
        seq_len: int, label_len: int, *, eff_batch: int, accum_steps: int
    ) -> Callable[..., Tuple[mx.array, mx.array]]:
        key = (int(seq_len), int(label_len) if use_sparse_loss else 0, int(eff_batch), int(accum_steps))
        if key not in compiled_train_steps:
            compiled_train_steps[key] = compile_train_step(
                model=model,
                optimizer=optimizer,
                value_and_grad=raw_value_and_grad,
                batch_size=int(eff_batch),
                seq_len=int(seq_len),
                label_len=int(label_len),
                accum_steps=int(accum_steps),
                grad_clip=float(args.grad_clip),
                sparse_loss=bool(use_sparse_loss),
            )
        return compiled_train_steps[key]

    def get_value_and_grad(
        seq_len: int, label_len: int, *, eff_batch: int, accum_steps: int
    ) -> Callable[..., Tuple[mx.array, Any]]:
        key = (int(seq_len), int(label_len) if use_sparse_loss else 0, int(eff_batch), int(accum_steps))
        if key not in compiled_value_and_grads:
            compiled_value_and_grads[key] = compile_value_and_grad(
                raw_value_and_grad,
                model=model,
                batch_size=int(eff_batch),
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

    trace_timing_out: Optional[str] = args.trace_timing_out
    trace_timing_steps = 0
    trace_timing_start_step = start_step
    if trace_timing_out is not None:
        trace_timing_steps = max(1, int(args.trace_timing_steps))
        trace_timing_start_step = (
            start_step
            if args.trace_timing_start_step is None
            else int(args.trace_timing_start_step)
        )
    trace_timing_end_step = trace_timing_start_step + trace_timing_steps
    timing_traces: List[Dict[str, Any]] = []

    stop_steps = int(total_steps) if int(total_steps) > 0 else None
    if stop_steps is not None and start_step >= stop_steps:
        print(
            f"[warn] start_step={start_step} >= total_steps={stop_steps}; "
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
            scheduled_batch = schedule_value(global_step, batch_schedule, int(args.batch_size))
            scheduled_accum = schedule_value(global_step, accum_schedule, int(args.accum_steps))
            if scheduled_batch != cur_batch_size or scheduled_accum != cur_accum_steps:
                cur_batch_size = int(scheduled_batch)
                cur_accum_steps = int(scheduled_accum)
                effective_batch_size = int(cur_batch_size) * (2 if is_dpo else 1)
                if auto_logits_chunk:
                    new_chunk = _auto_logits_chunk_size(
                        batch_size=int(effective_batch_size),
                        seq_len=int(args.seq_len),
                        vocab_size=int(model.config.vocab_size),
                        dtype=str(args.dtype),
                        mem_limit_bytes=int(mem_limit_mb) * 1024 * 1024,
                        sparse_loss=bool(use_sparse_loss),
                        label_bucket_sizes=label_bucket_sizes,
                    )
                    if new_chunk > 0 and int(args.logits_chunk_size) != int(new_chunk):
                        args.logits_chunk_size = int(new_chunk)
                        print(
                            f"[mem] auto logits_chunk_size={int(args.logits_chunk_size)} "
                            f"(batch={effective_batch_size})"
                        )
                compiled_train_steps.clear()
                compiled_value_and_grads.clear()
                print(
                    f"[schedule] epoch={epoch + 1} batch_size={cur_batch_size} accum_steps={cur_accum_steps}"
                )
            if is_dpo:
                micro_iter = iter(
                    make_dpo_microbatch_iterator(
                        paths=paths,
                        tokenizer=tokenizer,
                        seq_len=int(args.seq_len),
                        batch_size=int(cur_batch_size),
                        accum_steps=int(cur_accum_steps),
                        shuffle_buffer=int(args.shuffle_buffer),
                        seed=int(args.seed) + int(epoch),
                        bucket_sizes=bucket_sizes,
                        pretokenized=cached_dataset,
                    )
                )
            else:
                micro_iter = iter(
                    make_microbatch_iterator(
                        paths=paths,
                        bin_dataset=bin_dataset,
                        bin2d_dataset=bin2d_dataset,
                        tokenizer=tokenizer,
                        task=args.task,
                        seq_len=int(args.seq_len),
                        batch_size=int(cur_batch_size),
                        accum_steps=int(cur_accum_steps),
                        shuffle_buffer=int(args.shuffle_buffer),
                        seed=int(args.seed) + int(epoch),
                        bucket_sizes=bucket_sizes,
                        return_label_positions=bool(use_sparse_loss),
                        label_bucket_sizes=label_bucket_sizes,
                        pretokenized=cached_dataset,
                        pack_sequences=bool(args.pack_pretrain),
                        pack_eos=bool(args.pack_eos),
                        pack_no_doc_split=bool(args.pack_no_doc_split),
                        max_doc_tokens=int(args.max_doc_tokens),
                        drop_long=bool(args.drop_long),
                    )
                )

            if int(args.prefetch_batches) > 0:
                micro_iter = iter(
                    prefetch_iterator(
                        micro_iter,
                        prefetch=int(args.prefetch_batches),
                    )
                )

            while True:
                if stop_steps is not None and global_step >= stop_steps:
                    break

                trace_this_step = (
                    trace_timing_out is not None
                    and trace_timing_steps > 0
                    and trace_timing_start_step <= global_step < trace_timing_end_step
                )
                timing_tracer: Optional[TimingTracer] = None
                trace_step_t0 = 0.0
                ckpt_restore: Optional[int] = None
                if trace_this_step:
                    timing_tracer = TimingTracer(record_memory=bool(args.trace_timing_memory))
                    model.model.timing_tracer = timing_tracer
                    ckpt_restore = int(model.model.checkpoint_every_n)
                    model.model.checkpoint_every_n = 0
                    timing_tracer.set_prefix(None)
                    trace_step_t0 = timing_tracer.start()

                if untie_step is not None and global_step >= untie_step and bool(model.tie_word_embeddings):
                    model.untie_lm_head()
                    try:
                        optimizer.init(model.trainable_parameters())
                        compiled_train_steps.clear()
                        compiled_value_and_grads.clear()
                        compiled_opt_step = None
                    except Exception:
                        pass
                    if slow_steps > 1:
                        slow_param_mask = build_slow_param_mask(model)
                        slow_grad_accum = None
                    print(f"[untie] step={global_step} untied lm_head (optimizer state reset)")

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
                    data_t0 = time.perf_counter() if (profile_timing or timing_tracer is not None) else 0.0
                    group = next(micro_iter)
                    if profile_timing:
                        timing_data_s += time.perf_counter() - data_t0
                    if timing_tracer is not None:
                        timing_tracer.end("data", data_t0)
                except StopIteration:
                    if timing_tracer is not None:
                        model.model.timing_tracer = None
                        if ckpt_restore is not None:
                            model.model.checkpoint_every_n = ckpt_restore
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
                if str(args.weight_decay_schedule) == "lr":
                    try:
                        optimizer.weight_decay = float(args.weight_decay) * float(lr) / float(args.learning_rate)
                    except Exception:
                        pass
                to_mx_t0 = time.perf_counter() if (profile_timing or timing_tracer is not None) else 0.0
                if micro_batches < int(cur_accum_steps) and micro_batches > 0:
                    pad_m = [[0] * int(step_seq_len) for _ in range(int(effective_batch_size))]
                    last_x = xs[micro_batches - 1]
                    last_y = ys[micro_batches - 1]
                    while len(xs) < int(cur_accum_steps):
                        xs.append(last_x)
                        ys.append(last_y)
                        ms.append(pad_m)

                    if use_sparse_loss:
                        pad_p = [[0] * int(step_label_len) for _ in range(int(effective_batch_size))]
                        pad_pm = [[0] * int(step_label_len) for _ in range(int(effective_batch_size))]
                        last_p = ps[micro_batches - 1] if ps else pad_p
                        while len(ps) < int(cur_accum_steps):
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
                if timing_tracer is not None:
                    if use_sparse_loss:
                        timing_tracer.end("to_mx", to_mx_t0, arrays=[x, y, m, p, pm, micro], depth=1)
                    else:
                        timing_tracer.end("to_mx", to_mx_t0, arrays=[x, y, m, micro], depth=1)

                opt_t0 = time.perf_counter() if (profile_timing or timing_tracer is not None) else 0.0
                use_compile = bool(args.compile) and not trace_this_step
                use_compile_opt = bool(args.compile_optimizer) and not trace_this_step
                opt_step_fn = compiled_opt_step if use_compile_opt else None
                if use_compile and use_compile_opt:
                    if use_sparse_loss:
                        loss_accum, grad_norm = get_compiled_train_step(
                            int(step_seq_len), int(step_label_len), eff_batch=int(effective_batch_size), accum_steps=int(cur_accum_steps)
                        )(x, y, m, p, pm, micro)
                    else:
                        loss_accum, grad_norm = get_compiled_train_step(
                            int(step_seq_len), 0, eff_batch=int(effective_batch_size), accum_steps=int(cur_accum_steps)
                        )(x, y, m, micro)
                else:
                    # Fallback: run `value_and_grad` micro-batches in Python.
                    grad_accum = None
                    loss_accum = mx.array(0.0, dtype=mx.float32)
                    for i in range(micro_batches):
                        if timing_tracer is not None:
                            timing_tracer.set_prefix(f"mb{i}")
                        fwd_bwd_t0 = time.perf_counter() if (profile_timing or timing_tracer is not None) else 0.0
                        if use_compile and not use_compile_opt:
                            if use_sparse_loss:
                                loss, grads = get_value_and_grad(
                                    int(step_seq_len), int(step_label_len), eff_batch=int(effective_batch_size), accum_steps=int(cur_accum_steps)
                                )(
                                    x[i], y[i], m[i], p[i], pm[i]
                                )
                            else:
                                loss, grads = get_value_and_grad(
                                    int(step_seq_len), 0, eff_batch=int(effective_batch_size), accum_steps=int(cur_accum_steps)
                                )(x[i], y[i], m[i])
                        else:
                            if use_sparse_loss:
                                loss, grads = raw_value_and_grad(x[i], y[i], m[i], p[i], pm[i])
                            else:
                                loss, grads = raw_value_and_grad(x[i], y[i], m[i])
                        if profile_timing:
                            mx.eval(loss, grads)
                            timing_fwd_bwd_s += time.perf_counter() - fwd_bwd_t0
                        if timing_tracer is not None:
                            timing_tracer.end(
                                "fwd_bwd",
                                fwd_bwd_t0,
                                arrays=[loss, grads],
                                depth=1,
                            )
                        loss_accum = loss_accum + loss.astype(mx.float32)
                        if grad_accum is None:
                            grad_accum = grads
                        else:
                            grad_accum = mlx_utils.tree_map(lambda a, b: a + b, grad_accum, grads)

                    assert grad_accum is not None
                    grad_accum = mlx_utils.tree_map(lambda g: g / micro_batches, grad_accum)

                    if slow_steps > 1 and slow_param_mask is not None:
                        slow_part = mlx_utils.tree_map(lambda g, m: g * m, grad_accum, slow_param_mask)
                        fast_part = mlx_utils.tree_map(lambda g, m: g * (1.0 - m), grad_accum, slow_param_mask)
                        if slow_grad_accum is None:
                            slow_grad_accum = mlx_utils.tree_map(lambda g: mx.zeros_like(g), slow_part)
                        slow_grad_accum = mlx_utils.tree_map(lambda a, b: a + b, slow_grad_accum, slow_part)
                        if (global_step + 1) % int(slow_steps) == 0:
                            slow_update = mlx_utils.tree_map(
                                lambda g: g / float(slow_steps), slow_grad_accum
                            )
                            grad_accum = mlx_utils.tree_map(lambda f, s: f + s, fast_part, slow_update)
                            slow_grad_accum = mlx_utils.tree_map(lambda g: mx.zeros_like(g), slow_grad_accum)
                        else:
                            grad_accum = fast_part

                    if opt_step_fn is None and args.grad_clip > 0:
                        clip_t0 = time.perf_counter() if (profile_timing or timing_tracer is not None) else 0.0
                        grad_accum, grad_norm = optim.clip_grad_norm(
                            grad_accum, max_norm=args.grad_clip
                        )
                        if profile_timing:
                            mx.eval(grad_accum, grad_norm)
                            timing_clip_s += time.perf_counter() - clip_t0
                        if timing_tracer is not None:
                            timing_tracer.end("clip", clip_t0, arrays=[grad_accum, grad_norm], depth=1)

                    if opt_step_fn is None:
                        optimizer.update(model, grad_accum)
                    else:
                        grad_norm = opt_step_fn(grad_accum)
                mx.eval(model.parameters(), optimizer.state, loss_accum, grad_norm)
                if profile_timing:
                    timing_opt_s += time.perf_counter() - opt_t0
                if timing_tracer is not None:
                    timing_tracer.set_prefix(None)
                    timing_tracer.end(
                        "opt",
                        opt_t0,
                        arrays=[model.parameters(), optimizer.state, loss_accum, grad_norm],
                        depth=1,
                    )

                global_step += 1
                seen_tokens += int(effective_batch_size) * int(step_seq_len) * micro_batches
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
                    mem_msg = ""
                    if bool(args.log_memory):
                        try:
                            if hasattr(mx, "get_active_memory"):
                                active = mx.get_active_memory() / (1024 * 1024)
                                cache = mx.get_cache_memory() / (1024 * 1024)
                                peak = mx.get_peak_memory() / (1024 * 1024)
                            else:
                                active = mx.metal.get_active_memory() / (1024 * 1024)
                                cache = mx.metal.get_cache_memory() / (1024 * 1024)
                                peak = mx.metal.get_peak_memory() / (1024 * 1024)
                            mem_msg = (
                                f" mem_active={active:.0f}MiB"
                                f" mem_cache={cache:.0f}MiB"
                                f" mem_peak={peak:.0f}MiB"
                            )
                        except Exception:
                            mem_msg = ""
                    print(
                        f"[train] step={global_step} epoch={epoch + 1}/{args.epochs} "
                        f"loss={avg_loss:.4f} lr={lr:.2e} tok/s={tok_s:.0f}{timing_msg}{mem_msg}"
                    )
                    if tb_writer is not None:
                        tb_writer.add_scalar("train/loss", avg_loss, global_step)
                        tb_writer.add_scalar("train/lr", lr, global_step)
                        tb_writer.add_scalar("train/tok_s", tok_s, global_step)
                        tb_writer.add_scalar("train/epoch", epoch + 1, global_step)
                        tb_writer.add_scalar("train/seen_tokens", seen_tokens, global_step)
                        tb_writer.add_scalar("train/grad_norm", float(grad_norm.item()), global_step)
                        tb_writer.flush()

                if timing_tracer is not None:
                    timing_tracer.end(
                        "step",
                        trace_step_t0,
                        arrays=[model.parameters(), optimizer.state, loss_accum, grad_norm],
                        depth=0,
                    )
                    model.model.timing_tracer = None
                    if ckpt_restore is not None:
                        model.model.checkpoint_every_n = ckpt_restore
                    step_meta = {
                        "step": int(global_step - 1),
                        "epoch": int(epoch + 1),
                        "seq_len": int(step_seq_len),
                        "batch_size": int(effective_batch_size),
                        "micro_batches": int(micro_batches),
                        "accum_steps": int(cur_accum_steps),
                    }
                    timing_traces.append(timing_tracer.to_dict(meta=step_meta))
                    json_path = write_timing_trace(
                        out_path=str(trace_timing_out),
                        trace={"steps": list(timing_traces)},
                    )
                    summary = timing_tracer.summary(top_n=8)
                    if summary:
                        top_str = ", ".join([f"{name}={ms:.1f}ms" for name, ms in summary])
                    else:
                        top_str = "no events"
                    print(f"[trace] wrote {json_path} ({top_str})")

                if int(args.save_interval) > 0 and global_step > 0 and global_step % args.save_interval == 0:
                    path = save_checkpoint(global_step)
                    print(f"[ckpt] saved {path}")


            if stop_steps is not None and global_step >= stop_steps:
                break

    except KeyboardInterrupt:
        if int(args.save_interval) != 0:
            print("\n[train] interrupted, saving last checkpoint...")
        else:
            print("\n[train] interrupted, skipping checkpoint save (--save_interval 0).")
    finally:
        if int(args.save_interval) != 0:
            path = save_checkpoint(global_step)
            print(f"[ckpt] saved {path}")
        if tb_writer is not None:
            tb_writer.close()


if __name__ == "__main__":
    main()
