from __future__ import annotations

import glob
import json
import os
import random
from dataclasses import dataclass
from typing import Any, Dict, Iterable, Iterator, List, Optional, Sequence, Tuple, cast


def resolve_jsonl_paths(data_path: str) -> List[str]:
    parts: List[str] = []
    for piece in (p.strip() for p in data_path.split(",")):
        if not piece:
            continue
        if os.path.isdir(piece):
            parts.extend(sorted(glob.glob(os.path.join(piece, "*.jsonl"))))
        else:
            expanded = sorted(glob.glob(piece))
            parts.extend(expanded if expanded else [piece])

    paths = [p for p in parts if os.path.isfile(p)]
    if not paths:
        raise FileNotFoundError(f"No JSONL files found from: {data_path}")
    return paths


def iter_jsonl(paths: Sequence[str]) -> Iterator[Dict[str, Any]]:
    for path in paths:
        with open(path, "r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                obj = json.loads(line)
                if not isinstance(obj, dict):
                    raise ValueError(f"Expected a JSON object per line in {path}, got: {type(obj).__name__}")
                yield cast(Dict[str, Any], obj)


def shuffle_stream(
    items: Iterable[Dict[str, Any]], *, buffer_size: int, seed: int
) -> Iterator[Dict[str, Any]]:
    if buffer_size <= 0:
        yield from items
        return

    rng = random.Random(seed)
    buf: List[Dict[str, Any]] = []
    for item in items:
        if len(buf) < buffer_size:
            buf.append(item)
            continue
        idx = rng.randrange(len(buf))
        yield buf[idx]
        buf[idx] = item
    rng.shuffle(buf)
    yield from buf


def _encode_sample(obj: Dict[str, Any], *, tokenizer, task: str) -> List[int]:
    if "ids" in obj:
        return [int(t) for t in cast(Sequence[Any], obj["ids"])]

    if task == "pretrain":
        if "text" in obj:
            text = str(obj["text"])
        elif "conversations" in obj:
            conversations = cast(Sequence[Dict[str, Any]], obj["conversations"])
            text = tokenizer.apply_chat_template(conversations, tokenize=False, add_generation_prompt=False)
        else:
            text = json.dumps(obj, ensure_ascii=False)
        return tokenizer.encode(text, add_special_tokens=False)

    if task == "sft":
        if "conversations" not in obj:
            raise ValueError("SFT task expects JSONL lines with a `conversations` field.")
        prompt = tokenizer.apply_chat_template(
            cast(Sequence[Dict[str, Any]], obj["conversations"]),
            tokenize=False,
            add_generation_prompt=False,
        )
        return tokenizer.encode(prompt, add_special_tokens=False)

    raise ValueError(f"Unknown task: {task}")


def _pad_or_truncate(ids: List[int], *, length: int, pad_id: int) -> List[int]:
    if len(ids) >= length:
        return ids[:length]
    return ids + [pad_id] * (length - len(ids))


def _generate_sft_loss_mask(input_ids: Sequence[int], bos_id: Sequence[int], eos_id: Sequence[int], max_length: int) -> List[int]:
    loss_mask = [0] * len(input_ids)
    i = 0
    while i < len(input_ids):
        if list(input_ids[i : i + len(bos_id)]) == list(bos_id):
            start = i + len(bos_id)
            end = start
            while end < len(input_ids):
                if list(input_ids[end : end + len(eos_id)]) == list(eos_id):
                    break
                end += 1
            for j in range(start + 1, min(end + len(eos_id) + 1, max_length)):
                loss_mask[j] = 1
            i = end + len(eos_id) if end < len(input_ids) else len(input_ids)
        else:
            i += 1
    return loss_mask


@dataclass(frozen=True)
class TokenizedBatch:
    x: List[List[int]]
    y: List[List[int]]
    loss_mask: List[List[int]]


@dataclass(frozen=True)
class MicroBatchGroup:
    seq_len: int
    micro_batches: int
    x: List[List[List[int]]]
    y: List[List[List[int]]]
    loss_mask: List[List[List[int]]]
    label_len: int = 0
    label_pos: Optional[List[List[List[int]]]] = None
    label_pos_mask: Optional[List[List[List[int]]]] = None


def tokenize_pretrain_sample(
    *,
    tokenizer,
    text: str,
    seq_len: int,
    pad_id: int,
    add_special_tokens: bool = False,
) -> Tuple[List[int], List[int], List[int]]:
    ids: List[int] = tokenizer.encode(text, add_special_tokens=add_special_tokens)
    ids = _pad_or_truncate(ids, length=seq_len + 1, pad_id=pad_id)
    x = ids[:-1]
    y = ids[1:]
    mask = [1 if t != pad_id else 0 for t in y]
    return x, y, mask


def tokenize_pretrain_from_ids(
    *,
    ids: List[int],
    seq_len: int,
    pad_id: int,
) -> Tuple[List[int], List[int], List[int]]:
    ids = _pad_or_truncate(list(ids), length=seq_len + 1, pad_id=pad_id)
    x = ids[:-1]
    y = ids[1:]
    mask = [1 if t != pad_id else 0 for t in y]
    return x, y, mask


def tokenize_sft_sample(
    *,
    tokenizer,
    conversations: Sequence[Dict[str, Any]],
    seq_len: int,
    pad_id: int,
    bos_id: Sequence[int],
    eos_id: Sequence[int],
) -> Tuple[List[int], List[int], List[int]]:
    prompt = tokenizer.apply_chat_template(conversations, tokenize=False, add_generation_prompt=False)
    ids: List[int] = tokenizer.encode(prompt, add_special_tokens=False)
    ids = _pad_or_truncate(ids, length=seq_len + 1, pad_id=pad_id)

    loss_mask = _generate_sft_loss_mask(ids, bos_id, eos_id, seq_len + 1)
    x = ids[:-1]
    y = ids[1:]
    mask = loss_mask[1:]
    return x, y, mask


def tokenize_sft_from_ids(
    *,
    ids: List[int],
    seq_len: int,
    pad_id: int,
    bos_id: Sequence[int],
    eos_id: Sequence[int],
) -> Tuple[List[int], List[int], List[int]]:
    ids = _pad_or_truncate(list(ids), length=seq_len + 1, pad_id=pad_id)
    loss_mask = _generate_sft_loss_mask(ids, bos_id, eos_id, seq_len + 1)
    x = ids[:-1]
    y = ids[1:]
    mask = loss_mask[1:]
    return x, y, mask


def _pick_bucket(seq_len: int, buckets: Sequence[int]) -> int:
    if seq_len <= 0:
        return int(buckets[0])
    for b in buckets:
        if int(seq_len) <= int(b):
            return int(b)
    return int(buckets[-1])


def make_batch_iterator(
    *,
    paths: Sequence[str],
    pretokenized: Optional[Sequence[Dict[str, Any]]] = None,
    tokenizer,
    task: str,
    seq_len: int,
    batch_size: int,
    shuffle_buffer: int,
    seed: int,
) -> Iterator[TokenizedBatch]:
    pad_id = tokenizer.pad_token_id
    if pad_id is None:
        raise ValueError("Tokenizer must have pad_token_id set.")

    bos_id: Optional[List[int]] = None
    eos_id: Optional[List[int]] = None
    if task == "sft":
        bos_id = tokenizer.encode(f"{tokenizer.bos_token}assistant", add_special_tokens=False)
        eos_id = tokenizer.encode(f"{tokenizer.eos_token}", add_special_tokens=False)

    stream: Iterable[Dict[str, Any]] = pretokenized if pretokenized is not None else iter_jsonl(paths)
    stream = shuffle_stream(stream, buffer_size=shuffle_buffer, seed=seed)

    cur_x: List[List[int]] = []
    cur_y: List[List[int]] = []
    cur_m: List[List[int]] = []

    for obj in stream:
        ids = _encode_sample(obj, tokenizer=tokenizer, task=task)
        if task == "pretrain":
            x, y, m = tokenize_pretrain_from_ids(ids=ids, seq_len=seq_len, pad_id=pad_id)
        elif task == "sft":
            x, y, m = tokenize_sft_from_ids(
                ids=ids,
                seq_len=seq_len,
                pad_id=pad_id,
                bos_id=bos_id or [],
                eos_id=eos_id or [],
            )
        else:
            raise ValueError(f"Unknown task: {task}")

        cur_x.append(x)
        cur_y.append(y)
        cur_m.append(m)

        if len(cur_x) >= batch_size:
            yield TokenizedBatch(x=cur_x, y=cur_y, loss_mask=cur_m)
            cur_x, cur_y, cur_m = [], [], []


def make_microbatch_iterator(
    *,
    paths: Sequence[str],
    tokenizer,
    task: str,
    seq_len: int,
    batch_size: int,
    accum_steps: int,
    shuffle_buffer: int,
    seed: int,
    bucket_sizes: Optional[Sequence[int]] = None,
    return_label_positions: bool = False,
    label_bucket_sizes: Optional[Sequence[int]] = None,
    pretokenized: Optional[Sequence[Dict[str, Any]]] = None,
) -> Iterator[MicroBatchGroup]:
    """
    Yield groups of `accum_steps` micro-batches with the same padded `seq_len`.

    If `bucket_sizes` is provided, each sample is assigned to the smallest bucket
    that can fit its (clipped) length, reducing padding and compute.

    If `return_label_positions` is True, the iterator also groups by the number of
    loss tokens (bucketed via `label_bucket_sizes`) and returns per-sample padded
    label positions + masks to enable sparse masked-loss computation.
    """
    if accum_steps <= 0:
        raise ValueError("accum_steps must be > 0")
    if batch_size <= 0:
        raise ValueError("batch_size must be > 0")

    max_seq_len = int(seq_len)
    if max_seq_len <= 0:
        raise ValueError("seq_len must be > 0")

    buckets = sorted({int(b) for b in (bucket_sizes or [max_seq_len]) if int(b) > 0})
    if not buckets:
        buckets = [max_seq_len]
    if buckets[-1] > max_seq_len:
        raise ValueError(f"bucket_sizes max {buckets[-1]} exceeds seq_len {max_seq_len}")
    if buckets[-1] != max_seq_len:
        buckets.append(max_seq_len)

    label_buckets: Optional[List[int]] = None
    if return_label_positions:
        if label_bucket_sizes is None and bucket_sizes is None:
            # Default label buckets (powers of 2) to make sparse loss useful even
            # when seq_len bucketing is disabled.
            label_buckets = []
            b = 32
            while b < max_seq_len:
                label_buckets.append(int(b))
                b *= 2
            label_buckets.append(max_seq_len)
        else:
            base = bucket_sizes if label_bucket_sizes is None else label_bucket_sizes
            label_buckets = sorted({int(b) for b in (base or [max_seq_len]) if int(b) > 0})
            if not label_buckets:
                label_buckets = [max_seq_len]
            if label_buckets[-1] > max_seq_len:
                raise ValueError(
                    f"label_bucket_sizes max {label_buckets[-1]} exceeds seq_len {max_seq_len}"
                )
            if label_buckets[-1] != max_seq_len:
                label_buckets.append(max_seq_len)

    pad_id = tokenizer.pad_token_id
    if pad_id is None:
        raise ValueError("Tokenizer must have pad_token_id set.")

    bos_id: Optional[List[int]] = None
    eos_id: Optional[List[int]] = None
    if task == "sft":
        bos_id = tokenizer.encode(f"{tokenizer.bos_token}assistant", add_special_tokens=False)
        eos_id = tokenizer.encode(f"{tokenizer.eos_token}", add_special_tokens=False)

    stream: Iterable[Dict[str, Any]] = pretokenized if pretokenized is not None else iter_jsonl(paths)
    stream = shuffle_stream(stream, buffer_size=shuffle_buffer, seed=seed)

    buffers: Dict[int, List[Tuple[List[int], List[int], List[int]]]] = {b: [] for b in buckets}
    buffers2: Dict[Tuple[int, int], List[Tuple[List[int], List[int], List[int], List[int], List[int]]]] = {}
    need = int(batch_size) * int(accum_steps)

    def maybe_yield_from_bucket(b: int) -> Optional[MicroBatchGroup]:
        buf = buffers[b]
        if len(buf) < need:
            return None
        items = buf[:need]
        del buf[:need]
        xs: List[List[List[int]]] = []
        ys: List[List[List[int]]] = []
        ms: List[List[List[int]]] = []
        for i in range(int(accum_steps)):
            chunk = items[i * int(batch_size) : (i + 1) * int(batch_size)]
            xs.append([t[0] for t in chunk])
            ys.append([t[1] for t in chunk])
            ms.append([t[2] for t in chunk])
        return MicroBatchGroup(seq_len=int(b), micro_batches=int(accum_steps), x=xs, y=ys, loss_mask=ms)

    def maybe_yield_from_bucket2(b: int, l: int) -> Optional[MicroBatchGroup]:
        key = (int(b), int(l))
        buf = buffers2.setdefault(key, [])
        if len(buf) < need:
            return None
        items = buf[:need]
        del buf[:need]
        xs: List[List[List[int]]] = []
        ys: List[List[List[int]]] = []
        ms: List[List[List[int]]] = []
        ps: List[List[List[int]]] = []
        pms: List[List[List[int]]] = []
        for i in range(int(accum_steps)):
            chunk = items[i * int(batch_size) : (i + 1) * int(batch_size)]
            xs.append([t[0] for t in chunk])
            ys.append([t[1] for t in chunk])
            ms.append([t[2] for t in chunk])
            ps.append([t[3] for t in chunk])
            pms.append([t[4] for t in chunk])
        return MicroBatchGroup(
            seq_len=int(b),
            micro_batches=int(accum_steps),
            x=xs,
            y=ys,
            loss_mask=ms,
            label_len=int(l),
            label_pos=ps,
            label_pos_mask=pms,
        )

    for obj in stream:
        ids = _encode_sample(obj, tokenizer=tokenizer, task=task)
        ids = ids[: max_seq_len + 1]
        b = _pick_bucket(max(1, len(ids) - 1), buckets)

        if task == "pretrain":
            x, y, m = tokenize_pretrain_from_ids(ids=ids, seq_len=int(b), pad_id=pad_id)
        elif task == "sft":
            x, y, m = tokenize_sft_from_ids(
                ids=ids,
                seq_len=int(b),
                pad_id=pad_id,
                bos_id=bos_id or [],
                eos_id=eos_id or [],
            )
        else:
            raise ValueError(f"Unknown task: {task}")

        if return_label_positions:
            assert label_buckets is not None
            pos = [i for i, v in enumerate(m) if int(v) != 0]
            n = len(pos)
            l = _pick_bucket(max(1, n), label_buckets)
            if n > int(l):
                raise RuntimeError(f"label bucket {l} < label tokens {n}; buckets={label_buckets}")
            pos_p = pos + [0] * (int(l) - n)
            m_p = ([1] * n) + ([0] * (int(l) - n))
            buffers2.setdefault((int(b), int(l)), []).append((x, y, m, pos_p, m_p))
            out = maybe_yield_from_bucket2(int(b), int(l))
            if out is not None:
                yield out
        else:
            buffers[int(b)].append((x, y, m))
            out = maybe_yield_from_bucket(int(b))
            if out is not None:
                yield out

    # Flush leftovers (drop < batch_size samples).
    if return_label_positions:
        for (b, l), buf in sorted(buffers2.items()):
            full = len(buf) // int(batch_size)
            if full <= 0:
                continue
            micro_batches = min(int(accum_steps), int(full))
            take = micro_batches * int(batch_size)
            items = buf[:take]
            xs: List[List[List[int]]] = []
            ys: List[List[List[int]]] = []
            ms: List[List[List[int]]] = []
            ps: List[List[List[int]]] = []
            pms: List[List[List[int]]] = []
            for i in range(int(micro_batches)):
                chunk = items[i * int(batch_size) : (i + 1) * int(batch_size)]
                xs.append([t[0] for t in chunk])
                ys.append([t[1] for t in chunk])
                ms.append([t[2] for t in chunk])
                ps.append([t[3] for t in chunk])
                pms.append([t[4] for t in chunk])
            yield MicroBatchGroup(
                seq_len=int(b),
                micro_batches=int(micro_batches),
                x=xs,
                y=ys,
                loss_mask=ms,
                label_len=int(l),
                label_pos=ps,
                label_pos_mask=pms,
            )
    else:
        for b in buckets:
            buf = buffers[int(b)]
            full = len(buf) // int(batch_size)
            if full <= 0:
                continue
            micro_batches = min(int(accum_steps), int(full))
            take = micro_batches * int(batch_size)
            items = buf[:take]
            xs: List[List[List[int]]] = []
            ys: List[List[List[int]]] = []
            ms: List[List[List[int]]] = []
            for i in range(int(micro_batches)):
                chunk = items[i * int(batch_size) : (i + 1) * int(batch_size)]
                xs.append([t[0] for t in chunk])
                ys.append([t[1] for t in chunk])
                ms.append([t[2] for t in chunk])
            yield MicroBatchGroup(seq_len=int(b), micro_batches=int(micro_batches), x=xs, y=ys, loss_mask=ms)


def pretokenize_jsonl(
    *,
    paths: Sequence[str],
    tokenizer,
    task: str,
) -> List[Dict[str, Any]]:
    """
    Pre-tokenize a JSONL dataset once to avoid repeated text->ids work across epochs.

    Returns a list of {"ids": [...]} items compatible with `make_microbatch_iterator`.
    """
    cache: List[Dict[str, Any]] = []
    for obj in iter_jsonl(paths):
        ids = _encode_sample(obj, tokenizer=tokenizer, task=task)
        cache.append({"ids": ids})
    return cache
