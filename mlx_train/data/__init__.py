from __future__ import annotations

import glob
import json
import os
import random
from dataclasses import dataclass
from typing import Any, Dict, Iterable, Iterator, List, Optional, Sequence, Tuple, cast
import mmap
import sys
from array import array
from pathlib import Path


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


def detect_bin_format(path: str) -> Optional[str]:
    p = str(path)
    if p.endswith(".ids2d.bin") or p.endswith(".ids2d.idx"):
        return "bin2d"
    if p.endswith(".ids.bin") or p.endswith(".ids.idx"):
        return "bin"
    if p.endswith(".meta.json"):
        try:
            with open(p, "r", encoding="utf-8") as f:
                meta = json.load(f)
            fmt = str(meta.get("format", "bin")).lower()
            if fmt in ("bin", "bin2d"):
                return fmt
        except Exception:
            return "bin"
    return None


def looks_like_bin_path(path: str) -> bool:
    return detect_bin_format(path) is not None


def resolve_bin_prefix(path: str) -> str:
    p = str(path)
    if p.endswith(".meta.json"):
        return p[: -len(".meta.json")]
    if p.endswith(".ids.bin"):
        return p[: -len(".ids.bin")]
    if p.endswith(".ids.idx"):
        return p[: -len(".ids.idx")]
    return p


class BinDataset:
    def __init__(self, prefix: str, *, cache: str = "mmap") -> None:
        if cache not in ("mmap", "memory"):
            raise ValueError(f"cache must be 'mmap' or 'memory', got {cache}")

        prefix = resolve_bin_prefix(prefix)
        meta_path = Path(prefix + ".meta.json")
        if meta_path.exists():
            with meta_path.open("r", encoding="utf-8") as f:
                meta = json.load(f)
            base_dir = meta_path.parent
            ids_bin = base_dir / str(meta["ids"]["bin"])
            ids_idx = base_dir / str(meta["ids"]["idx"])
            labels = meta.get("labels")
            lbl_bin = base_dir / str(labels["bin"]) if labels and labels.get("bin") else None
            lbl_idx = base_dir / str(labels["idx"]) if labels and labels.get("idx") else None
            byteorder = str(meta.get("byteorder", "little"))
        else:
            ids_bin = Path(prefix + ".ids.bin")
            ids_idx = Path(prefix + ".ids.idx")
            lbl_bin = Path(prefix + ".lbl.bin")
            lbl_idx = Path(prefix + ".lbl.idx")
            byteorder = "little"

        if not ids_bin.exists() or not ids_idx.exists():
            raise FileNotFoundError(f"Missing bin dataset files for prefix={prefix}")

        self.byteorder = byteorder
        self.ids_bin_path = ids_bin
        self.ids_idx_path = ids_idx
        self.lbl_bin_path = lbl_bin if lbl_bin and lbl_bin.exists() else None
        self.lbl_idx_path = lbl_idx if lbl_idx and lbl_idx.exists() else None

        self._ids_idx = self._load_idx(self.ids_idx_path)
        self._ids_buf = self._open_bin(self.ids_bin_path, cache=cache)

        self._lbl_idx = None
        self._lbl_buf = None
        if self.lbl_bin_path and self.lbl_idx_path:
            self._lbl_idx = self._load_idx(self.lbl_idx_path)
            self._lbl_buf = self._open_bin(self.lbl_bin_path, cache=cache)

    def __len__(self) -> int:
        return len(self._ids_idx) // 2

    def _load_idx(self, path: Path) -> array:
        data = array("Q")
        with path.open("rb") as f:
            data.fromfile(f, path.stat().st_size // data.itemsize)
        if self.byteorder != sys.byteorder:
            data.byteswap()
        return data

    def _open_bin(self, path: Path, *, cache: str) -> memoryview:
        if cache == "memory":
            raw = path.read_bytes()
            return memoryview(raw)
        f = path.open("rb")
        mm = mmap.mmap(f.fileno(), 0, access=mmap.ACCESS_READ)
        return memoryview(mm)

    def get_ids(self, idx: int) -> Sequence[int]:
        off = int(self._ids_idx[2 * idx])
        length = int(self._ids_idx[2 * idx + 1])
        start = off * 4
        end = (off + length) * 4
        view = self._ids_buf[start:end].cast("I")
        if self.byteorder != sys.byteorder:
            tmp = array("I", view)
            tmp.byteswap()
            return tmp
        return view

    def get_label_pos(self, idx: int) -> Optional[List[int]]:
        if self._lbl_idx is None or self._lbl_buf is None:
            return None
        off = int(self._lbl_idx[2 * idx])
        length = int(self._lbl_idx[2 * idx + 1])
        start = off * 4
        end = (off + length) * 4
        view = self._lbl_buf[start:end].cast("I")
        if self.byteorder != sys.byteorder:
            tmp = array("I", view)
            tmp.byteswap()
            return list(tmp)
        return list(view)


class Bin2DDataset:
    def __init__(self, prefix: str, *, cache: str = "mmap") -> None:
        if cache not in ("mmap", "memory"):
            raise ValueError(f"cache must be 'mmap' or 'memory', got {cache}")
        prefix = resolve_bin_prefix(prefix)
        meta_path = Path(prefix + ".meta.json")
        if not meta_path.exists():
            raise FileNotFoundError(f"Missing meta.json for bin2d dataset: {meta_path}")
        with meta_path.open("r", encoding="utf-8") as f:
            meta = json.load(f)
        fmt = str(meta.get("format", "bin2d")).lower()
        if fmt != "bin2d":
            raise ValueError(f"meta format={fmt} is not bin2d")
        base_dir = meta_path.parent
        ids_bin = base_dir / str(meta["ids"]["bin"])
        lbl = meta.get("labels")
        lbl_bin = base_dir / str(lbl["bin"]) if lbl and lbl.get("bin") else None
        lbl_idx = base_dir / str(lbl["idx"]) if lbl and lbl.get("idx") else None
        if not ids_bin.exists():
            raise FileNotFoundError(f"Missing ids bin: {ids_bin}")

        self.ids_bin_path = ids_bin
        self.lbl_bin_path = lbl_bin if lbl_bin and lbl_bin.exists() else None
        self.lbl_idx_path = lbl_idx if lbl_idx and lbl_idx.exists() else None
        self.byteorder = str(meta.get("byteorder", "little"))
        self.seq_len = int(meta["seq_len"])
        self.stride = int(meta.get("stride", self.seq_len + 1))
        self.num_samples = int(meta["num_samples"])

        self._ids_buf = self._open_bin(self.ids_bin_path, cache=cache)
        if self._ids_buf.nbytes < self.num_samples * self.stride * 4:
            raise ValueError("bin2d ids.bin is smaller than expected from meta")

        self._lbl_idx = None
        self._lbl_buf = None
        if self.lbl_bin_path and self.lbl_idx_path:
            self._lbl_idx = self._load_idx(self.lbl_idx_path)
            self._lbl_buf = self._open_bin(self.lbl_bin_path, cache=cache)

    def _load_idx(self, path: Path) -> array:
        data = array("Q")
        with path.open("rb") as f:
            data.fromfile(f, path.stat().st_size // data.itemsize)
        if self.byteorder != sys.byteorder:
            data.byteswap()
        return data

    def _open_bin(self, path: Path, *, cache: str) -> memoryview:
        if cache == "memory":
            raw = path.read_bytes()
            return memoryview(raw)
        f = path.open("rb")
        mm = mmap.mmap(f.fileno(), 0, access=mmap.ACCESS_READ)
        return memoryview(mm)

    def __len__(self) -> int:
        return self.num_samples

    def get_ids(self, idx: int) -> Sequence[int]:
        start = int(idx) * int(self.stride) * 4
        end = start + int(self.stride) * 4
        view = self._ids_buf[start:end].cast("I")
        if self.byteorder != sys.byteorder:
            tmp = array("I", view)
            tmp.byteswap()
            return tmp
        return view

    def get_label_pos(self, idx: int) -> Optional[List[int]]:
        if self._lbl_idx is None or self._lbl_buf is None:
            return None
        off = int(self._lbl_idx[2 * idx])
        length = int(self._lbl_idx[2 * idx + 1])
        start = off * 4
        end = (off + length) * 4
        view = self._lbl_buf[start:end].cast("I")
        if self.byteorder != sys.byteorder:
            tmp = array("I", view)
            tmp.byteswap()
            return list(tmp)
        return list(view)


def iter_bin_dataset(dataset: BinDataset) -> Iterator[Dict[str, Any]]:
    for i in range(len(dataset)):
        ids = dataset.get_ids(i)
        item: Dict[str, Any] = {"ids": ids}
        pos = dataset.get_label_pos(i)
        if pos is not None:
            item["label_pos"] = pos
        yield item


def iter_bin2d_dataset(dataset: Bin2DDataset) -> Iterator[Dict[str, Any]]:
    for i in range(len(dataset)):
        ids = dataset.get_ids(i)
        item: Dict[str, Any] = {"ids": ids}
        pos = dataset.get_label_pos(i)
        if pos is not None:
            item["label_pos"] = pos
        yield item


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

    if task in ("sft", "r1"):
        if "conversations" not in obj:
            raise ValueError("SFT task expects JSONL lines with a `conversations` field.")
        prompt = tokenizer.apply_chat_template(
            cast(Sequence[Dict[str, Any]], obj["conversations"]),
            tokenize=False,
            add_generation_prompt=False,
        )
        return tokenizer.encode(prompt, add_special_tokens=False)

    raise ValueError(f"Unknown task: {task}")


def _encode_dpo_pair(obj: Dict[str, Any], *, tokenizer) -> Tuple[List[int], List[int]]:
    if "chosen_ids" in obj and "rejected_ids" in obj:
        chosen_ids = [int(t) for t in cast(Sequence[Any], obj["chosen_ids"])]
        rejected_ids = [int(t) for t in cast(Sequence[Any], obj["rejected_ids"])]
        return chosen_ids, rejected_ids

    if "chosen" not in obj or "rejected" not in obj:
        raise ValueError("DPO task expects JSONL lines with `chosen` and `rejected` fields.")
    chosen = cast(Sequence[Dict[str, Any]], obj["chosen"])
    rejected = cast(Sequence[Dict[str, Any]], obj["rejected"])
    chosen_prompt = tokenizer.apply_chat_template(
        chosen, tokenize=False, add_generation_prompt=False
    )
    rejected_prompt = tokenizer.apply_chat_template(
        rejected, tokenize=False, add_generation_prompt=False
    )
    chosen_ids = tokenizer.encode(chosen_prompt, add_special_tokens=False)
    rejected_ids = tokenizer.encode(rejected_prompt, add_special_tokens=False)
    return chosen_ids, rejected_ids


def _pad_or_truncate(ids: List[int], *, length: int, pad_id: int) -> List[int]:
    if len(ids) >= length:
        return ids[:length]
    return ids + [pad_id] * (length - len(ids))


def pack_pretrain_ids(
    ids_iter: Iterable[List[int]],
    *,
    seq_len: int,
    pad_id: int,
    eos_id: Optional[int],
    add_eos: bool,
    allow_doc_split: bool,
) -> Iterator[List[int]]:
    """
    Pack variable-length pretrain token IDs into fixed-size sequences.

    Complexity: O(N) time, O(seq_len) space for total tokens N.
    """
    max_len = int(seq_len) + 1
    if max_len <= 1:
        raise ValueError("seq_len must be > 0 for packing")
    cur: List[int] = []
    for ids in ids_iter:
        if not ids:
            continue
        if add_eos and eos_id is not None:
            if ids[-1] != int(eos_id):
                ids = list(ids) + [int(eos_id)]
        if not bool(allow_doc_split):
            if len(ids) > max_len:
                ids = ids[:max_len]
            if len(cur) + len(ids) > max_len:
                if cur:
                    yield _pad_or_truncate(cur, length=max_len, pad_id=pad_id)
                cur = []
            cur.extend(ids)
            if len(cur) >= max_len:
                yield cur[:max_len]
                cur = []
        else:
            while ids:
                space = max_len - len(cur)
                if space <= 0:
                    yield cur[:max_len]
                    cur = []
                    space = max_len
                take = ids[:space]
                cur.extend(take)
                ids = ids[space:]
                if len(cur) >= max_len:
                    yield cur[:max_len]
                    cur = []
    if cur:
        yield _pad_or_truncate(cur, length=max_len, pad_id=pad_id)


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


def tokenize_sft_from_ids_fast(
    *,
    ids: Sequence[int],
    seq_len: int,
    pad_id: int,
) -> Tuple[List[int], List[int]]:
    ids_list = _pad_or_truncate(list(ids), length=seq_len + 1, pad_id=pad_id)
    x = ids_list[:-1]
    y = ids_list[1:]
    return x, y


def tokenize_sft_from_ids_fixed(
    *,
    ids: Sequence[int],
) -> Tuple[List[int], List[int]]:
    if len(ids) < 2:
        raise ValueError("ids must contain at least 2 tokens for fixed-length slicing.")
    if isinstance(ids, list):
        ids_list = ids
    else:
        ids_list = list(ids)
    return ids_list[:-1], ids_list[1:]


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
    bin_dataset: Optional[BinDataset] = None,
    bin2d_dataset: Optional[Bin2DDataset] = None,
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
    if task in ("sft", "r1"):
        bos_id = tokenizer.encode(f"{tokenizer.bos_token}assistant", add_special_tokens=False)
        eos_id = tokenizer.encode(f"{tokenizer.eos_token}", add_special_tokens=False)

    if bin2d_dataset is not None:
        stream = iter_bin2d_dataset(bin2d_dataset)
    elif bin_dataset is not None:
        stream = iter_bin_dataset(bin_dataset)
    else:
        stream = pretokenized if pretokenized is not None else iter_jsonl(paths)
    stream = shuffle_stream(stream, buffer_size=shuffle_buffer, seed=seed)

    cur_x: List[List[int]] = []
    cur_y: List[List[int]] = []
    cur_m: List[List[int]] = []

    for obj in stream:
        ids = _encode_sample(obj, tokenizer=tokenizer, task=task)
        if task == "pretrain":
            x, y, m = tokenize_pretrain_from_ids(ids=ids, seq_len=seq_len, pad_id=pad_id)
        elif task in ("sft", "r1"):
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
    bin_dataset: Optional[BinDataset] = None,
    bin2d_dataset: Optional[Bin2DDataset] = None,
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
    pack_sequences: bool = False,
    pack_eos: bool = True,
    pack_no_doc_split: bool = False,
    max_doc_tokens: int = 0,
    drop_long: bool = False,
) -> Iterator[MicroBatchGroup]:
    """
    Yield groups of `accum_steps` micro-batches with the same padded `seq_len`.

    If `bucket_sizes` is provided, each sample is assigned to the smallest bucket
    that can fit its (clipped) length, reducing padding and compute.

    If `return_label_positions` is True, the iterator also groups by the number of
    loss tokens (bucketed via `label_bucket_sizes`) and returns per-sample padded
    label positions + masks to enable sparse masked-loss computation.

    If `pack_sequences` is True (pretrain only), pack multiple documents into a
    fixed-length sequence to reduce padding. `pack_eos` appends EOS between docs.
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
    if task in ("sft", "r1"):
        bos_id = tokenizer.encode(f"{tokenizer.bos_token}assistant", add_special_tokens=False)
        eos_id = tokenizer.encode(f"{tokenizer.eos_token}", add_special_tokens=False)

    if bin_dataset is not None:
        stream = iter_bin_dataset(bin_dataset)
    else:
        stream = pretokenized if pretokenized is not None else iter_jsonl(paths)
    stream = shuffle_stream(stream, buffer_size=shuffle_buffer, seed=seed)

    max_doc = int(max_doc_tokens)
    use_pack = bool(pack_sequences) and str(task) == "pretrain"
    eos_token_id = tokenizer.eos_token_id

    def _clip_ids(ids: List[int]) -> Optional[List[int]]:
        if max_doc > 0 and len(ids) > max_doc:
            if bool(drop_long):
                return None
            ids = ids[:max_doc]
        return ids

    def iter_ids() -> Iterator[List[int]]:
        for obj in stream:
            ids = _encode_sample(obj, tokenizer=tokenizer, task=task)
            ids = _clip_ids(ids)
            if ids is None:
                continue
            yield ids

    if bin2d_dataset is not None:
        if int(max_seq_len) != int(bin2d_dataset.seq_len):
            raise ValueError(
                f"bin2d seq_len={bin2d_dataset.seq_len} != requested seq_len={max_seq_len}"
            )
        ids_iter = iter_bin2d_dataset(bin2d_dataset)
    elif bin_dataset is not None:
        ids_iter = iter_bin_dataset(bin_dataset)
    elif use_pack:
        ids_iter = pack_pretrain_ids(
            iter_ids(),
            seq_len=int(max_seq_len),
            pad_id=int(pad_id),
            eos_id=eos_token_id,
            add_eos=bool(pack_eos),
            allow_doc_split=not bool(pack_no_doc_split),
        )
    else:
        ids_iter = iter_ids()

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

    for obj in ids_iter:
        label_pos_obj: Optional[List[int]] = None
        if isinstance(obj, dict) and "ids" in obj:
            raw_ids = obj["ids"]
            if isinstance(raw_ids, (array, memoryview)):
                ids = raw_ids
            else:
                ids = [int(t) for t in cast(Sequence[Any], raw_ids)]
            if "label_pos" in obj:
                label_pos_obj = [int(t) for t in cast(Sequence[Any], obj["label_pos"])]
        else:
            ids = obj if isinstance(obj, (array, memoryview)) else list(cast(List[int], obj))

        if bin2d_dataset is None:
            ids = ids[: max_seq_len + 1]
        b = _pick_bucket(max(1, len(ids) - 1), buckets)

        if task == "pretrain":
            x, y, m = tokenize_pretrain_from_ids(ids=ids, seq_len=int(b), pad_id=pad_id)
        elif task in ("sft", "r1"):
            if bin2d_dataset is not None:
                x, y = tokenize_sft_from_ids_fixed(ids=ids)
                if return_label_positions and label_pos_obj is not None:
                    m = [0] * len(y)
                else:
                    loss_mask = _generate_sft_loss_mask(ids, bos_id or [], eos_id or [], len(ids))
                    m = loss_mask[1:]
            else:
                if return_label_positions and label_pos_obj is not None:
                    x, y = tokenize_sft_from_ids_fast(
                        ids=ids,
                        seq_len=int(b),
                        pad_id=pad_id,
                    )
                    m = [0] * len(y)
                else:
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
            if label_pos_obj is None:
                pos = [i for i, v in enumerate(m) if int(v) != 0]
            else:
                limit = max(1, len(y))
                pos = [i for i in label_pos_obj if int(i) < limit]
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


def pretokenize_dpo_jsonl(
    *,
    paths: Sequence[str],
    tokenizer,
) -> List[Dict[str, Any]]:
    """
    Pre-tokenize a DPO JSONL dataset once to avoid repeated text->ids work.

    Returns a list of {"chosen_ids": [...], "rejected_ids": [...]} items.

    Complexity: O(N * L) time, O(N * L) space for N samples with average length L.
    """
    cache: List[Dict[str, Any]] = []
    for obj in iter_jsonl(paths):
        chosen_ids, rejected_ids = _encode_dpo_pair(obj, tokenizer=tokenizer)
        cache.append({"chosen_ids": chosen_ids, "rejected_ids": rejected_ids})
    return cache


def make_dpo_microbatch_iterator(
    *,
    paths: Sequence[str],
    tokenizer,
    seq_len: int,
    batch_size: int,
    accum_steps: int,
    shuffle_buffer: int,
    seed: int,
    bucket_sizes: Optional[Sequence[int]] = None,
    pretokenized: Optional[Sequence[Dict[str, Any]]] = None,
) -> Iterator[MicroBatchGroup]:
    """
    Yield DPO micro-batches as concatenated chosen/rejected samples.

    Each micro-batch contains `batch_size` pairs, flattened into `2 * batch_size`
    sequences with the chosen samples first and rejected samples second.

    Complexity: O(N * L) time, O(B * L) space for N samples, sequence length L,
    and micro-batch size B (not counting cached dataset storage).
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

    pad_id = tokenizer.pad_token_id
    if pad_id is None:
        raise ValueError("Tokenizer must have pad_token_id set.")

    bos_id = tokenizer.encode(f"{tokenizer.bos_token}assistant", add_special_tokens=False)
    eos_id = tokenizer.encode(f"{tokenizer.eos_token}", add_special_tokens=False)

    stream: Iterable[Dict[str, Any]] = pretokenized if pretokenized is not None else iter_jsonl(paths)
    stream = shuffle_stream(stream, buffer_size=shuffle_buffer, seed=seed)

    buffers: Dict[int, List[Tuple[List[int], List[int], List[int], List[int], List[int], List[int]]]] = {
        b: [] for b in buckets
    }
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
            x_micro: List[List[int]] = []
            y_micro: List[List[int]] = []
            m_micro: List[List[int]] = []
            for x_c, y_c, m_c, x_r, y_r, m_r in chunk:
                x_micro.append(x_c)
                y_micro.append(y_c)
                m_micro.append(m_c)
                x_micro.append(x_r)
                y_micro.append(y_r)
                m_micro.append(m_r)
            xs.append(x_micro)
            ys.append(y_micro)
            ms.append(m_micro)
        return MicroBatchGroup(seq_len=int(b), micro_batches=int(accum_steps), x=xs, y=ys, loss_mask=ms)

    for obj in stream:
        chosen_ids, rejected_ids = _encode_dpo_pair(obj, tokenizer=tokenizer)
        max_len = max(len(chosen_ids), len(rejected_ids))
        b = _pick_bucket(max(1, max_len - 1), buckets)
        x_c, y_c, m_c = tokenize_sft_from_ids(
            ids=chosen_ids,
            seq_len=int(b),
            pad_id=pad_id,
            bos_id=bos_id,
            eos_id=eos_id,
        )
        x_r, y_r, m_r = tokenize_sft_from_ids(
            ids=rejected_ids,
            seq_len=int(b),
            pad_id=pad_id,
            bos_id=bos_id,
            eos_id=eos_id,
        )
        buffers[int(b)].append((x_c, y_c, m_c, x_r, y_r, m_r))
        out = maybe_yield_from_bucket(int(b))
        if out is not None:
            yield out

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
            x_micro: List[List[int]] = []
            y_micro: List[List[int]] = []
            m_micro: List[List[int]] = []
            for x_c, y_c, m_c, x_r, y_r, m_r in chunk:
                x_micro.append(x_c)
                y_micro.append(y_c)
                m_micro.append(m_c)
                x_micro.append(x_r)
                y_micro.append(y_r)
                m_micro.append(m_r)
            xs.append(x_micro)
            ys.append(y_micro)
            ms.append(m_micro)
        yield MicroBatchGroup(seq_len=int(b), micro_batches=int(micro_batches), x=xs, y=ys, loss_mask=ms)
