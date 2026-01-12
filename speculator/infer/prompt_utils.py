"""Prompt sampling utilities for speculator benchmarks."""
from __future__ import annotations

import json
import random
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Sequence

REPO_ROOT = Path(__file__).resolve().parents[2]

DEFAULT_PROMPTS = [
    "Explain the difference between overfitting and underfitting in one paragraph.",
    "Solve: (18 * 7) - (56 / 4) + 9. Show quick steps.",
    "Write a short email politely declining a meeting invite.",
    "List three pros and cons of renewable energy.",
    "What is a hash table? Give a concise definition and one use case.",
    "Summarize the plot of 'Cinderella' in two sentences.",
]

DEFAULT_PROMPT_DATASETS = [
    REPO_ROOT / "out/distill_ollama_qwen3_0.6b/synth.jsonl",
    REPO_ROOT / "dataset/minimind/sft_mini_512.jsonl",
    REPO_ROOT / "dataset/minimind/sft_512.jsonl",
    REPO_ROOT / "dataset/identity_cn_sample.jsonl",
]


def add_system(
    messages: List[Dict[str, Any]], system: Optional[str]
) -> List[Dict[str, Any]]:
    if system and (not messages or messages[0]["role"] != "system"):
        return [{"role": "system", "content": system}] + messages
    return messages


def iter_prompt_messages_from_record(
    record: Dict[str, Any], system: Optional[str]
) -> Iterable[List[Dict[str, Any]]]:
    if "conversations" in record:
        conversations = record["conversations"]
        if not isinstance(conversations, list):
            raise ValueError("Expected 'conversations' to be a list")
        yield add_system(list(conversations), system)
        return
    if "messages" in record:
        messages = record["messages"]
        if not isinstance(messages, list):
            raise ValueError("Expected 'messages' to be a list")
        yield add_system(list(messages), system)
        return
    if "turns" in record:
        turns = record["turns"]
        if not isinstance(turns, list):
            raise ValueError("Expected 'turns' to be a list")
        for turn in turns:
            yield add_system([{"role": "user", "content": str(turn)}], system)
        return
    if "prompt" in record:
        yield add_system([{"role": "user", "content": str(record["prompt"])}], system)
        return
    if "text" in record:
        yield add_system([{"role": "user", "content": str(record["text"])}], system)
        return
    raise ValueError(
        "Prompt record must contain 'conversations', 'messages', 'turns', 'prompt', or 'text'"
    )


def pick_default_prompts_jsonl() -> Optional[Path]:
    for path in DEFAULT_PROMPT_DATASETS:
        if path.is_file():
            return path
    return None


def _read_speculator_data_path(speculator_dir: Optional[Path]) -> Optional[Path]:
    if speculator_dir is None:
        return None
    cfg_path = speculator_dir / "speculator_config.json"
    if not cfg_path.is_file():
        return None
    data = json.loads(cfg_path.read_text(encoding="utf-8"))
    data_path = data.get("data_path")
    if not data_path:
        return None
    path = Path(data_path)
    if path.is_file():
        return path
    return None


def resolve_prompts_jsonl(
    prompts_jsonl: Optional[str], *, speculator_dir: Optional[Path]
) -> Optional[str]:
    """Resolve a prompt JSONL path. Time O(1) best/avg/worst, space O(1)."""
    if prompts_jsonl:
        return prompts_jsonl
    spec_path = _read_speculator_data_path(speculator_dir)
    if spec_path is not None:
        return str(spec_path)
    return None


def sample_prompt_messages(
    path: Path,
    system: Optional[str],
    *,
    max_samples: Optional[int],
    seed: Optional[int],
) -> List[List[Dict[str, Any]]]:
    """Reservoir-sample prompts. Time O(N) best/avg/worst, space O(k) or O(N) if k is None."""
    rng = random.Random(seed)
    messages: List[List[Dict[str, Any]]] = []
    seen = 0
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            record = json.loads(line)
            for msg in iter_prompt_messages_from_record(record, system):
                seen += 1
                if max_samples is None:
                    messages.append(msg)
                    continue
                if len(messages) < int(max_samples):
                    messages.append(msg)
                    continue
                idx = rng.randrange(seen)
                if idx < int(max_samples):
                    messages[idx] = msg
    if max_samples is not None and len(messages) > 1:
        rng.shuffle(messages)
    return messages


def load_prompt_messages(
    prompts_jsonl: Optional[str],
    system: Optional[str],
    *,
    max_samples: Optional[int],
    seed: Optional[int],
    default_prompts: Sequence[str] = DEFAULT_PROMPTS,
) -> List[List[Dict[str, Any]]]:
    """Load prompts from JSONL or defaults. Time O(N) best/avg/worst, space O(k) or O(N)."""
    path = Path(prompts_jsonl) if prompts_jsonl else pick_default_prompts_jsonl()
    if path is not None:
        if not path.is_file():
            raise FileNotFoundError(f"Prompts JSONL not found: {path}")
        return sample_prompt_messages(
            path, system, max_samples=max_samples, seed=seed
        )
    rng = random.Random(seed)
    messages = [add_system([{"role": "user", "content": p}], system) for p in default_prompts]
    if max_samples is not None and int(max_samples) < len(messages):
        rng.shuffle(messages)
        return messages[: int(max_samples)]
    return messages
