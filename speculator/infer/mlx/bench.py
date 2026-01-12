#!/usr/bin/env python3
import argparse
import sys
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional

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
from speculator.infer.prompt_utils import load_prompt_messages, resolve_prompts_jsonl


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
    parser.add_argument("--temperature", type=float, default=0.8)
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

    prompts_jsonl = resolve_prompts_jsonl(
        args.prompts_jsonl, speculator_dir=Path(args.speculator_dir)
    )
    prompt_messages = load_prompt_messages(
        prompts_jsonl,
        args.system,
        max_samples=args.max_samples,
        seed=args.seed,
    )

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

    prompt_count = len(prompt_inputs)
    for round_idx in range(int(args.rounds)):
        for prompt_idx, input_ids in enumerate(prompt_inputs):
            seed_base = int(args.seed) + (int(round_idx) * prompt_count) + int(prompt_idx)
            mx.random.seed(seed_base)
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
                mx.random.seed(seed_base)
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

    def summarize_tokens(rows: List[SpecBenchResult]) -> Dict[str, float]:
        """Aggregate token sources. Time O(n) best/avg/worst, space O(1)."""
        total_draft = sum(r.stats.total_draft for r in rows)
        total_accept = sum(r.stats.accepted_output for r in rows)
        total_target = sum(r.stats.target_generated for r in rows)
        return {
            "draft_tokens": float(total_draft),
            "accepted_tokens": float(total_accept),
            "target_generated": float(total_target),
        }

    def summarize_timing(rows: List[SpecBenchResult]) -> Dict[str, float]:
        """Aggregate speculator/target time. Time O(n) best/avg/worst, space O(1)."""
        total_spec = sum(r.stats.spec_time_s for r in rows)
        total_target = sum(r.stats.target_time_s for r in rows)
        return {"spec_time_s": float(total_spec), "target_time_s": float(total_target)}

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
        speedup_tok = spec_stats["tok_per_s"] / max(base_stats["tok_per_s"], 1e-6)
        accept_stats = summarize_acceptance(spec_results)
        token_stats = summarize_tokens(spec_results)
        timing_stats = summarize_timing(spec_results)
        target_per_out = token_stats["target_generated"] / max(spec_stats["output_tokens"], 1e-6)
        accept_per_out = token_stats["accepted_tokens"] / max(spec_stats["output_tokens"], 1e-6)
        draft_per_out = token_stats["draft_tokens"] / max(spec_stats["output_tokens"], 1e-6)
        print(
            f"[bench] spec_len={spec_len} output_tokens={spec_stats['output_tokens']:.0f} "
            f"time_s={spec_stats['total_time_s']:.2f} tok/s={spec_stats['tok_per_s']:.2f} "
            f"speedup_tok/s={speedup_tok:.2f}x"
        )
        print(
            f"[bench] acceptance mean={accept_stats['mean_accept']:.2f} "
            f"rate={accept_stats['accept_rate'] * 100:.2f}% "
            f"zero_accept={accept_stats['zero_rate'] * 100:.2f}%"
        )
        other_time = spec_stats["total_time_s"] - timing_stats["spec_time_s"] - timing_stats["target_time_s"]
        print(
            f"[bench] time_s baseline={base_stats['total_time_s']:.2f} "
            f"draft={timing_stats['spec_time_s']:.2f} "
            f"target={timing_stats['target_time_s']:.2f} other={other_time:.2f}"
        )
        print(
            f"[bench] tokens baseline_out={base_stats['output_tokens']:.0f} "
            f"accepted={token_stats['accepted_tokens']:.0f} "
            f"target_generated={token_stats['target_generated']:.0f} "
            f"draft_proposed={token_stats['draft_tokens']:.0f} "
            f"accepted/out={accept_per_out:.2f} "
            f"target/out={target_per_out:.2f} draft/out={draft_per_out:.2f}"
        )


if __name__ == "__main__":
    main()
