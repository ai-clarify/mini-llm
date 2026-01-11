#!/usr/bin/env python3
import argparse
import json
import sys
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional

import mlx.core as mx

ROOT = Path(__file__).resolve().parents[3]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from speculator.infer.mlx.common import (
    SpecStats,
    _apply_chat_template,
    _count_params_mlx,
    _load_target,
    _resolve_spec_config,
    baseline_decode,
    baseline_decode_minillm,
    load_speculator,
    speculative_decode_minillm_with_stats,
    speculative_decode_with_stats,
)

DEFAULT_PROMPTS = [
    "Explain the difference between overfitting and underfitting in one paragraph.",
    "Solve: (18 * 7) - (56 / 4) + 9. Show quick steps.",
    "Write a short email politely declining a meeting invite.",
    "List three pros and cons of renewable energy.",
    "What is a hash table? Give a concise definition and one use case.",
    "Summarize the plot of 'Cinderella' in two sentences.",
]


@dataclass(frozen=True)
class BenchResult:
    num_input_tokens: int
    num_output_tokens: int
    elapsed_s: float

    @property
    def tok_per_s(self) -> float:
        return self.num_output_tokens / max(self.elapsed_s, 1e-6)


@dataclass(frozen=True)
class SpecBenchResult:
    num_input_tokens: int
    num_output_tokens: int
    elapsed_s: float
    stats: SpecStats

    @property
    def tok_per_s(self) -> float:
        return self.num_output_tokens / max(self.elapsed_s, 1e-6)


def _render_progress(current: int, total: int, *, label: str = "bench") -> None:
    width = 28
    filled = int(width * current / total)
    bar = "#" * filled + "-" * (width - filled)
    percent = current / total * 100.0
    sys.stdout.write(f"\r[{label}] {current}/{total} {percent:5.1f}% |{bar}|")
    sys.stdout.flush()
    if current >= total:
        sys.stdout.write("\n")


def _add_system(
    messages: List[Dict[str, Any]], system: Optional[str]
) -> List[Dict[str, Any]]:
    if system and (not messages or messages[0]["role"] != "system"):
        return [{"role": "system", "content": system}] + messages
    return messages


def _iter_prompts_from_record(
    record: Dict[str, Any], system: Optional[str]
) -> Iterable[List[Dict[str, Any]]]:
    if "conversations" in record:
        conversations = record["conversations"]
        if not isinstance(conversations, list):
            raise ValueError("Expected 'conversations' to be a list")
        yield _add_system(list(conversations), system)
        return
    if "messages" in record:
        messages = record["messages"]
        if not isinstance(messages, list):
            raise ValueError("Expected 'messages' to be a list")
        yield _add_system(list(messages), system)
        return
    if "turns" in record:
        turns = record["turns"]
        if not isinstance(turns, list):
            raise ValueError("Expected 'turns' to be a list")
        for turn in turns:
            yield _add_system([{"role": "user", "content": str(turn)}], system)
        return
    raise ValueError(
        "Prompt record must contain 'conversations', 'messages', or 'turns'"
    )


def _load_prompt_messages(
    prompts_jsonl: Optional[str], system: Optional[str]
) -> List[List[Dict[str, Any]]]:
    if not prompts_jsonl:
        return [[{"role": "user", "content": p}] for p in DEFAULT_PROMPTS]
    path = Path(prompts_jsonl)
    if not path.is_file():
        raise FileNotFoundError(f"Prompts JSONL not found: {path}")
    messages: List[List[Dict[str, Any]]] = []
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            record = json.loads(line)
            messages.extend(list(_iter_prompts_from_record(record, system)))
    return messages


def _run_baseline_qwen3(
    *,
    target,
    input_ids: List[int],
    max_new_tokens: int,
    temperature: float,
    top_p: float,
    eos_token_id: Optional[int],
) -> BenchResult:
    start = time.perf_counter()
    output_ids = baseline_decode(
        target=target,
        input_ids=input_ids,
        max_new_tokens=max_new_tokens,
        temperature=temperature,
        top_p=top_p,
        eos_token_id=eos_token_id,
    )
    elapsed = time.perf_counter() - start
    num_input = len(input_ids)
    num_output = max(len(output_ids) - num_input, 0)
    return BenchResult(num_input, num_output, elapsed)


def _run_spec_qwen3(
    *,
    target,
    speculator,
    input_ids: List[int],
    max_new_tokens: int,
    spec_len: int,
    temperature: float,
    top_p: float,
    eos_token_id: Optional[int],
) -> SpecBenchResult:
    start = time.perf_counter()
    output_ids, stats = speculative_decode_with_stats(
        target=target,
        speculator=speculator,
        input_ids=input_ids,
        max_new_tokens=max_new_tokens,
        spec_len=spec_len,
        temperature=temperature,
        top_p=top_p,
        eos_token_id=eos_token_id,
        use_cache=True,
        optimized=True,
    )
    elapsed = time.perf_counter() - start
    num_input = len(input_ids)
    num_output = max(len(output_ids) - num_input, 0)
    return SpecBenchResult(num_input, num_output, elapsed, stats)


def _run_baseline_minillm(
    *,
    target,
    input_ids: List[int],
    max_new_tokens: int,
    temperature: float,
    top_p: float,
    eos_token_id: Optional[int],
) -> BenchResult:
    start = time.perf_counter()
    output_ids = baseline_decode_minillm(
        target=target,
        input_ids=input_ids,
        max_new_tokens=max_new_tokens,
        temperature=temperature,
        top_p=top_p,
        eos_token_id=eos_token_id,
    )
    elapsed = time.perf_counter() - start
    num_input = len(input_ids)
    num_output = max(len(output_ids) - num_input, 0)
    return BenchResult(num_input, num_output, elapsed)


def _run_spec_minillm(
    *,
    target,
    speculator,
    input_ids: List[int],
    max_new_tokens: int,
    spec_len: int,
    temperature: float,
    top_p: float,
    eos_token_id: Optional[int],
) -> SpecBenchResult:
    start = time.perf_counter()
    output_ids, stats = speculative_decode_minillm_with_stats(
        target=target,
        speculator=speculator,
        input_ids=input_ids,
        max_new_tokens=max_new_tokens,
        spec_len=spec_len,
        temperature=temperature,
        top_p=top_p,
        eos_token_id=eos_token_id,
        optimized=True,
    )
    elapsed = time.perf_counter() - start
    num_input = len(input_ids)
    num_output = max(len(output_ids) - num_input, 0)
    return SpecBenchResult(num_input, num_output, elapsed, stats)


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Benchmark EAGLE-3 speculator vs baseline (MLX backend)."
    )
    parser.add_argument(
        "--target_arch", type=str, choices=["qwen3", "minillm"], default="qwen3"
    )
    parser.add_argument("--hf_repo", type=str, default="Qwen/Qwen3-0.6B")
    parser.add_argument("--model_dir", type=str, default=None)
    parser.add_argument("--revision", type=str, default=None)
    parser.add_argument("--minillm_ckpt_dir", type=str, default=None)
    parser.add_argument("--minillm_tokenizer", type=str, default="./model")
    parser.add_argument(
        "--speculator_dir", type=str, default="out/eagle3_speculator_mlx/qwen3_0.6b"
    )
    parser.add_argument("--speculator_ckpt", type=str, default=None)
    parser.add_argument("--prompts_jsonl", type=str, default=None)
    parser.add_argument("--system", type=str, default=None)
    parser.add_argument("--max_samples", type=int, default=32)
    parser.add_argument("--max_new_tokens", type=int, default=1024)
    parser.add_argument("--temperature", type=float, default=1)
    parser.add_argument("--top_p", type=float, default=1.0)
    parser.add_argument(
        "--spec_len",
        type=int,
        default=None,
        help="Draft length for speculator (auto if unset).",
    )
    parser.add_argument(
        "--spec_layers",
        type=int,
        default=None,
        help="Transformer layers in speculator (auto if unset).",
    )
    parser.add_argument(
        "--head_rank",
        type=int,
        default=None,
        help="Low-rank speculator head size (overrides config if set; full head if unset).",
    )
    parser.add_argument("--no_speculator", action="store_true")
    parser.add_argument("--no_chat_template", action="store_true")
    parser.add_argument("--rounds", type=int, default=3)
    parser.add_argument("--seed", type=int, default=1337)
    args = parser.parse_args()

    target, tokenizer = _load_target(
        target_arch=args.target_arch,
        model_dir=args.model_dir,
        hf_repo=args.hf_repo,
        revision=args.revision,
        minillm_ckpt_dir=args.minillm_ckpt_dir,
        minillm_tokenizer=args.minillm_tokenizer,
    )

    speculator = None
    spec_len, spec_layers = _resolve_spec_config(
        args.spec_len, args.spec_layers, param_count=_count_params_mlx(target)
    )
    head_rank = (
        args.head_rank
        if args.head_rank is not None and int(args.head_rank) > 0
        else None
    )
    if not args.no_speculator:
        speculator_dir = Path(args.speculator_dir)
        speculator_ckpt = Path(args.speculator_ckpt) if args.speculator_ckpt else None
        speculator, spec_len = load_speculator(
            target_arch=args.target_arch,
            target=target,
            speculator_dir=speculator_dir,
            speculator_ckpt=speculator_ckpt,
            spec_len=spec_len,
            spec_layers=spec_layers,
            head_rank=head_rank,
        )

    prompt_messages = _load_prompt_messages(args.prompts_jsonl, args.system)
    if args.max_samples is not None:
        prompt_messages = prompt_messages[: int(args.max_samples)]

    prompt_inputs: List[List[int]] = []
    for messages in prompt_messages:
        if args.no_chat_template:
            prompt_text = messages[-1]["content"]
        else:
            prompt_text = _apply_chat_template(
                tokenizer, messages, add_generation_prompt=True
            )
        prompt_inputs.append(tokenizer.encode(prompt_text, add_special_tokens=False))

    baseline_results: List[BenchResult] = []
    spec_results: List[SpecBenchResult] = []

    total_samples = int(args.rounds) * len(prompt_inputs)
    if total_samples <= 0:
        raise ValueError(
            "No prompts to benchmark; check --max_samples or prompts file."
        )
    completed = 0

    for round_idx in range(int(args.rounds)):
        mx.random.seed(int(args.seed) + int(round_idx))
        for input_ids in prompt_inputs:
            if args.target_arch == "minillm":
                baseline = _run_baseline_minillm(
                    target=target,
                    input_ids=input_ids,
                    max_new_tokens=args.max_new_tokens,
                    temperature=args.temperature,
                    top_p=args.top_p,
                    eos_token_id=tokenizer.eos_token_id,
                )
            else:
                baseline = _run_baseline_qwen3(
                    target=target,
                    input_ids=input_ids,
                    max_new_tokens=args.max_new_tokens,
                    temperature=args.temperature,
                    top_p=args.top_p,
                    eos_token_id=tokenizer.eos_token_id,
                )
            baseline_results.append(baseline)

            if speculator is not None:
                if args.target_arch == "minillm":
                    spec = _run_spec_minillm(
                        target=target,
                        speculator=speculator,
                        input_ids=input_ids,
                        max_new_tokens=args.max_new_tokens,
                        spec_len=spec_len,
                        temperature=args.temperature,
                        top_p=args.top_p,
                        eos_token_id=tokenizer.eos_token_id,
                    )
                else:
                    spec = _run_spec_qwen3(
                        target=target,
                        speculator=speculator,
                        input_ids=input_ids,
                        max_new_tokens=args.max_new_tokens,
                        spec_len=spec_len,
                        temperature=args.temperature,
                        top_p=args.top_p,
                        eos_token_id=tokenizer.eos_token_id,
                    )
                spec_results.append(spec)
            completed += 1
            _render_progress(completed, total_samples)

    def summarize(rows: List[BenchResult]) -> Dict[str, float]:
        total_out = sum(r.num_output_tokens for r in rows)
        total_time = sum(r.elapsed_s for r in rows)
        total_in = sum(r.num_input_tokens for r in rows)
        return {
            "input_tokens": float(total_in),
            "output_tokens": float(total_out),
            "total_time_s": float(total_time),
            "tok_per_s": float(total_out / max(total_time, 1e-6)),
        }

    def summarize_acceptance(rows: List[SpecBenchResult]) -> Dict[str, float]:
        total_accept = sum(r.stats.total_accept for r in rows)
        total_draft = sum(r.stats.total_draft for r in rows)
        total_steps = sum(r.stats.steps for r in rows)
        zero_accept = sum(r.stats.zero_accept for r in rows)
        accept_rate = float(total_accept / total_draft) if total_draft else 0.0
        mean_accept = float(total_accept / total_steps) if total_steps else 0.0
        zero_rate = float(zero_accept / total_steps) if total_steps else 0.0
        return {
            "accept_rate": accept_rate,
            "mean_accept": mean_accept,
            "zero_rate": zero_rate,
        }

    base_stats = summarize(baseline_results)
    print(
        f"[bench] rounds={int(args.rounds)} samples={len(baseline_results)} prompts={len(prompt_inputs)} "
        f"temp={args.temperature} top_p={args.top_p}"
    )
    print(
        f"[bench] baseline output_tokens={base_stats['output_tokens']:.0f} "
        f"time_s={base_stats['total_time_s']:.2f} tok/s={base_stats['tok_per_s']:.2f}"
    )

    if spec_results:
        spec_stats = summarize(spec_results)
        speedup = base_stats["total_time_s"] / max(spec_stats["total_time_s"], 1e-6)
        accept_stats = summarize_acceptance(spec_results)
        print(
            f"[bench] spec_len={spec_len} output_tokens={spec_stats['output_tokens']:.0f} "
            f"time_s={spec_stats['total_time_s']:.2f} tok/s={spec_stats['tok_per_s']:.2f} "
            f"speedup={speedup:.2f}x"
        )
        print(
            f"[bench] acceptance mean={accept_stats['mean_accept']:.2f} "
            f"rate={accept_stats['accept_rate'] * 100:.2f}% "
            f"zero_accept={accept_stats['zero_rate'] * 100:.2f}%"
        )


if __name__ == "__main__":
    main()
