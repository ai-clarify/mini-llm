#!/usr/bin/env python
"""Construct Chinese data mixtures for MiniLLM."""
from __future__ import annotations

import argparse
import json
import random
from collections import defaultdict
from pathlib import Path
from typing import Dict, Iterable, Iterator, List, Sequence

from utils.conversation import flatten_conversation, normalize_conversation

DEFAULT_MIX = {
    "general": {"path": Path("data/chinese/general_conversations.jsonl"), "target": 460_000},
    "knowledge": {"path": Path("data/chinese/knowledge_qa.jsonl"), "target": 100_000},
    "math": {"path": Path("data/chinese/math_qa.jsonl"), "target": 8_000},
    "identity": {"path": Path("data/chinese/identity_conversations.jsonl"), "target": 1_000, "repeat": 2},
}
DEFAULT_PREFERENCE = {"preference": {"path": Path("data/chinese/preference_pairs.jsonl"), "target": 60_000}}

SAMPLE_IDENTITY_FALLBACK = Path("dataset/identity_cn_sample.jsonl")
SAMPLE_PREFERENCE_FALLBACK = Path("dataset/preference_cn_sample.jsonl")


def iter_jsonl(path: Path) -> Iterator[dict]:
    if not path.exists():
        return iter([])
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            yield json.loads(line)


def load_records(path: Path) -> List[dict]:
    return list(iter_jsonl(path))


def sample_records(records: Sequence[dict], target: int, *, rng: random.Random) -> List[dict]:
    if not records:
        return []
    if len(records) >= target:
        return rng.sample(records, target)
    result: List[dict] = []
    index = 0
    while len(result) < target:
        result.append(records[index % len(records)])
        index += 1
    rng.shuffle(result)
    return result


def build_pretrain_samples(mix: Dict[str, List[dict]], *, rng: random.Random) -> List[dict]:
    samples: List[dict] = []
    for source, records in mix.items():
        for record in records:
            try:
                messages = normalize_conversation(record)
                text = flatten_conversation(messages)
            except Exception:
                text = record.get("text") or record.get("content")
                if not text:
                    continue
            samples.append({"text": text, "source": source})
    rng.shuffle(samples)
    return samples


def build_sft_samples(mix: Dict[str, List[dict]], *, rng: random.Random) -> List[dict]:
    samples: List[dict] = []
    for source, records in mix.items():
        for record in records:
            try:
                messages = normalize_conversation(record)
            except Exception:
                continue
            samples.append({"conversations": [m.to_dict() for m in messages], "source": source})
    rng.shuffle(samples)
    return samples


def build_dpo_samples(records: List[dict], *, rng: random.Random) -> List[dict]:
    samples: List[dict] = []
    for record in records:
        chosen = record.get("chosen") or record.get("chosen_conversation")
        rejected = record.get("rejected") or record.get("rejected_conversation")
        prompt = record.get("prompt")
        if chosen is None or rejected is None:
            if prompt and "preferred" in record and "other" in record:
                chosen = [{"role": "user", "content": prompt}, {"role": "assistant", "content": record["preferred"]}]
                rejected = [{"role": "user", "content": prompt}, {"role": "assistant", "content": record["other"]}]
            else:
                continue
        samples.append({
            "chosen": chosen,
            "rejected": rejected,
            "source": record.get("source", "preference"),
        })
    rng.shuffle(samples)
    return samples


def ensure_identity_dataset(target_dir: Path) -> Path:
    target_path = DEFAULT_MIX["identity"]["path"]
    if target_path.exists():
        return target_path
    target_path.parent.mkdir(parents=True, exist_ok=True)
    if SAMPLE_IDENTITY_FALLBACK.exists():
        target_path.write_text(SAMPLE_IDENTITY_FALLBACK.read_text(encoding="utf-8"), encoding="utf-8")
    return target_path


def main() -> None:
    parser = argparse.ArgumentParser(description="Build Chinese data mixtures for MiniLLM")
    parser.add_argument("--output-dir", type=Path, default=Path("data/processed"), help="Directory to place processed datasets")
    parser.add_argument("--seed", type=int, default=42, help="Random seed for sampling")
    args = parser.parse_args()

    rng = random.Random(args.seed)
    args.output_dir.mkdir(parents=True, exist_ok=True)

    ensure_identity_dataset(args.output_dir)

    mix_records: Dict[str, List[dict]] = {}
    for name, cfg in DEFAULT_MIX.items():
        path = cfg["path"]
        records = load_records(path)
        if not records and name == "identity":
            records = load_records(SAMPLE_IDENTITY_FALLBACK)
        target = cfg.get("target", len(records))
        repeat = cfg.get("repeat", 1)
        sampled = sample_records(records, target, rng=rng)
        mix_records[name] = sampled * repeat
        print(f"Loaded {len(records)} records for {name}, sampled {len(mix_records[name])}")

    preference_cfg = DEFAULT_PREFERENCE["preference"]
    preference_records = load_records(preference_cfg["path"])
    if not preference_records and SAMPLE_PREFERENCE_FALLBACK.exists():
        preference_records = load_records(SAMPLE_PREFERENCE_FALLBACK)

    if preference_records:
        preference_samples = sample_records(preference_records, preference_cfg["target"], rng=rng)
    else:
        preference_samples = []
    preference_samples = build_dpo_samples(preference_samples, rng=rng)
    print(f"Loaded {len(preference_records)} preference records, sampled {len(preference_samples)}")

    pretrain_samples = build_pretrain_samples(mix_records, rng=rng)
    sft_samples = build_sft_samples(mix_records, rng=rng)

    pretrain_path = args.output_dir / "pretrain_chinese.jsonl"
    sft_path = args.output_dir / "sft_chinese.jsonl"
    dpo_path = args.output_dir / "dpo_chinese.jsonl"

    with pretrain_path.open("w", encoding="utf-8") as f:
        for sample in pretrain_samples:
            f.write(json.dumps(sample, ensure_ascii=False) + "\n")
    with sft_path.open("w", encoding="utf-8") as f:
        for sample in sft_samples:
            f.write(json.dumps(sample, ensure_ascii=False) + "\n")
    with dpo_path.open("w", encoding="utf-8") as f:
        for sample in preference_samples:
            f.write(json.dumps(sample, ensure_ascii=False) + "\n")

    print(f"Wrote {len(pretrain_samples)} pretrain samples -> {pretrain_path}")
    print(f"Wrote {len(sft_samples)} SFT samples -> {sft_path}")
    print(f"Wrote {len(preference_samples)} DPO pairs -> {dpo_path}")


if __name__ == "__main__":
    main()
