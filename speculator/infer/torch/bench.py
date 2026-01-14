#!/usr/bin/env python3
import argparse
import json
import string
import sys
import time
from collections import Counter
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional

import torch

ROOT = Path(__file__).resolve().parents[3]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from model.model_minillm import MiniLLMForCausalLM
from speculator.infer.prompt_utils import load_prompt_messages, resolve_prompts_jsonl
from speculator.infer.torch.common import (
    SpecStats,
    _apply_chat_template,
    _count_params_torch,
    _load_target_and_tokenizer,
    _resolve_dtype,
    _resolve_spec_config,
    baseline_decode,
    load_speculator,
    speculative_decode_with_stats,
    speculative_decode_with_trace,
)


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


class TokenLogWriter:
    def __init__(
        self,
        path: str,
        tokenizer,
        *,
        top_k: int = 50,
        summary_path: Optional[str] = None,
    ) -> None:
        self._path = Path(path)
        self._path.parent.mkdir(parents=True, exist_ok=True)
        self._fh = self._path.open("w", encoding="utf-8")
        if summary_path is None:
            self._summary_path = self._summary_path_for(self._path)
        else:
            self._summary_path = Path(summary_path)
        self._tokenizer = tokenizer
        self._top_k = int(top_k)
        self._accepted = Counter()
        self._target = Counter()
        self._rejected = Counter()
        self._cluster_accepted = Counter()
        self._cluster_target = Counter()
        self._cluster_rejected = Counter()
        self._token_cache: Dict[int, str] = {}

    def _decode_token(self, token_id: int) -> str:
        token_id = int(token_id)
        cached = self._token_cache.get(token_id)
        if cached is not None:
            return cached
        try:
            text = self._tokenizer.decode([token_id], skip_special_tokens=False)
        except TypeError:
            text = self._tokenizer.decode([token_id])
        self._token_cache[token_id] = text
        return text

    @staticmethod
    def _summary_path_for(path: Path) -> Path:
        if path.suffix == ".jsonl":
            base = path.with_suffix("")
            return Path(str(base) + ".summary.json")
        return Path(str(path) + ".summary.json")

    @property
    def summary_path(self) -> Path:
        return self._summary_path

    def _contains_cjk(self, text: str) -> bool:
        for ch in text:
            code = ord(ch)
            if (
                0x3400 <= code <= 0x4DBF
                or 0x4E00 <= code <= 0x9FFF
                or 0x3000 <= code <= 0x303F
                or 0xFF00 <= code <= 0xFFEF
            ):
                return True
        return False

    def _cluster_for(self, token: str) -> str:
        if not token:
            return "empty"
        if "\n" in token or "\r" in token:
            return "newline"
        stripped = token.strip()
        if not stripped:
            return "whitespace"
        if self._contains_cjk(stripped):
            return "cjk"
        if stripped.isdigit():
            return "digits"
        if stripped.isalpha():
            return "latin" if stripped.isascii() else "alpha"
        if stripped.isalnum():
            return "alnum"
        if all(ch in string.punctuation for ch in stripped):
            return "punct"
        return "mixed"

    def _top_tokens(self, counter: Counter) -> List[Dict[str, Any]]:
        top = []
        items = sorted(counter.items(), key=lambda kv: (-kv[1], kv[0]))
        for token_id, count in items[: self._top_k]:
            top.append(
                {
                    "token_id": int(token_id),
                    "token": self._decode_token(token_id),
                    "count": int(count),
                }
            )
        return top

    def _compare_tokens(self) -> Dict[str, List[Dict[str, Any]]]:
        rows = []
        for token_id in set(self._accepted) | set(self._target):
            acc = int(self._accepted.get(token_id, 0))
            tar = int(self._target.get(token_id, 0))
            total = acc + tar
            if total <= 0:
                continue
            acc_rate = acc / float(total)
            rows.append((token_id, acc, tar, total, acc_rate))

        high = sorted(rows, key=lambda r: (-r[4], -r[3], r[0]))
        low = sorted(rows, key=lambda r: (r[4], -r[3], r[0]))

        def build(entries: List[tuple]) -> List[Dict[str, Any]]:
            out = []
            for token_id, acc, tar, total, acc_rate in entries[: self._top_k]:
                out.append(
                    {
                        "token_id": int(token_id),
                        "token": self._decode_token(token_id),
                        "accepted": int(acc),
                        "target": int(tar),
                        "total": int(total),
                        "accept_rate": float(acc_rate),
                    }
                )
            return out

        return {
            "accept_rate_top": build(high),
            "accept_rate_bottom": build(low),
        }

    def _cluster_summary(self) -> List[Dict[str, Any]]:
        clusters = sorted(
            set(self._cluster_accepted)
            | set(self._cluster_target)
            | set(self._cluster_rejected)
        )
        rows = []
        for name in clusters:
            accepted = int(self._cluster_accepted.get(name, 0))
            target = int(self._cluster_target.get(name, 0))
            rejected = int(self._cluster_rejected.get(name, 0))
            emitted = accepted + target
            accept_rate = accepted / float(emitted) if emitted > 0 else 0.0
            rows.append(
                {
                    "cluster": name,
                    "accepted": accepted,
                    "target": target,
                    "rejected": rejected,
                    "accept_rate": float(accept_rate),
                    "emitted": int(emitted),
                }
            )
        rows.sort(key=lambda r: (-r["emitted"], r["cluster"]))
        return rows

    def log_event(self, event: Dict[str, Any]) -> None:
        token_id = int(event["token_id"])
        record = dict(event)
        record.setdefault("kind", "token")
        record["token_id"] = token_id
        token_text = self._decode_token(token_id)
        record["token"] = token_text
        record["cluster"] = self._cluster_for(token_text)
        self._fh.write(json.dumps(record, ensure_ascii=False) + "\n")
        source = record.get("source")
        if source == "accepted":
            self._accepted[token_id] += 1
            self._cluster_accepted[record["cluster"]] += 1
        elif source in ("target", "bonus", "fallback"):
            self._target[token_id] += 1
            self._cluster_target[record["cluster"]] += 1
        elif source == "rejected_draft":
            self._rejected[token_id] += 1
            self._cluster_rejected[record["cluster"]] += 1

    def trace_for(self, *, round_idx: int, prompt_idx: int) -> Callable[[Dict[str, Any]], None]:
        def _trace(event: Dict[str, Any]) -> None:
            payload = dict(event)
            payload["round"] = int(round_idx)
            payload["prompt"] = int(prompt_idx)
            self.log_event(payload)

        return _trace

    def close(self) -> None:
        accepted_total = int(sum(self._accepted.values()))
        target_total = int(sum(self._target.values()))
        denom = accepted_total + target_total
        summary = {
            "kind": "summary",
            "accepted_total": accepted_total,
            "target_total": target_total,
            "accept_rate": float(accepted_total / denom) if denom > 0 else 0.0,
            "rejected_total": int(sum(self._rejected.values())),
            "accepted_top": self._top_tokens(self._accepted),
            "target_top": self._top_tokens(self._target),
            "rejected_top": self._top_tokens(self._rejected),
            "target_compare": self._compare_tokens(),
            "clusters": self._cluster_summary(),
        }
        self._fh.write(json.dumps(summary, ensure_ascii=False) + "\n")
        self._fh.close()
        self._summary_path.parent.mkdir(parents=True, exist_ok=True)
        with self._summary_path.open("w", encoding="utf-8") as f:
            json.dump(summary, f, ensure_ascii=False, indent=2)


def _render_progress(current: int, total: int, *, label: str = "bench") -> None:
    width = 28
    filled = int(width * current / total)
    bar = "#" * filled + "-" * (width - filled)
    percent = current / total * 100.0
    sys.stdout.write(f"\r[{label}] {current}/{total} {percent:5.1f}% |{bar}|")
    sys.stdout.flush()
    if current >= total:
        sys.stdout.write("\n")


def _sync_device(device: torch.device) -> None:
    if device.type == "cuda":
        torch.cuda.synchronize()


def _run_baseline(
    *,
    target,
    input_ids: torch.Tensor,
    max_new_tokens: int,
    temperature: float,
    top_p: float,
    eos_token_id: Optional[int],
    use_cache: bool,
    device: torch.device,
) -> BenchResult:
    _sync_device(device)
    start = time.perf_counter()
    output_ids = baseline_decode(
        target=target,
        input_ids=input_ids,
        max_new_tokens=max_new_tokens,
        temperature=temperature,
        top_p=top_p,
        eos_token_id=eos_token_id,
        use_cache=use_cache,
    )
    _sync_device(device)
    elapsed = time.perf_counter() - start
    num_input = int(input_ids.shape[1])
    num_output = max(int(output_ids.shape[1]) - num_input, 0)
    return BenchResult(num_input, num_output, elapsed)


def _run_spec(
    *,
    target,
    speculator,
    input_ids: torch.Tensor,
    max_new_tokens: int,
    spec_len: int,
    temperature: float,
    top_p: float,
    eos_token_id: Optional[int],
    use_cache: bool,
    device: torch.device,
    trace: Optional[Callable[[Dict[str, Any]], None]] = None,
) -> SpecBenchResult:
    _sync_device(device)
    start = time.perf_counter()
    if trace is None:
        output_ids, stats = speculative_decode_with_stats(
            target=target,
            speculator=speculator,
            input_ids=input_ids,
            max_new_tokens=max_new_tokens,
            spec_len=spec_len,
            temperature=temperature,
            top_p=top_p,
            eos_token_id=eos_token_id,
            use_cache=use_cache,
            optimized=True,
        )
    else:
        output_ids, stats = speculative_decode_with_trace(
            target=target,
            speculator=speculator,
            input_ids=input_ids,
            max_new_tokens=max_new_tokens,
            spec_len=spec_len,
            temperature=temperature,
            top_p=top_p,
            eos_token_id=eos_token_id,
            use_cache=use_cache,
            optimized=True,
            trace=trace,
        )
    _sync_device(device)
    elapsed = time.perf_counter() - start
    num_input = int(input_ids.shape[1])
    num_output = max(int(output_ids.shape[1]) - num_input, 0)
    return SpecBenchResult(num_input, num_output, elapsed, stats)


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Benchmark MTP (MiniLLM) or EAGLE-3 speculator vs baseline (Torch backend)."
    )
    parser.add_argument(
        "--target_arch", type=str, choices=["qwen3", "minillm"], default="qwen3"
    )
    parser.add_argument("--target_model", type=str, default="Qwen/Qwen3-0.6B")
    parser.add_argument("--minillm_ckpt", type=str, default=None)
    parser.add_argument("--minillm_config", type=str, default=None)
    parser.add_argument("--minillm_tokenizer", type=str, default="./model")
    parser.add_argument(
        "--speculator_dir", type=str, default="out/eagle3_speculator/qwen3_0.6b"
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
    parser.add_argument(
        "--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu"
    )
    parser.add_argument("--dtype", type=str, default="bfloat16")
    parser.add_argument(
        "--token_log",
        type=str,
        default="out/bench_token_log_torch.jsonl",
        help="Write token-level acceptance log as JSONL (speculator only).",
    )
    args = parser.parse_args()

    torch.manual_seed(int(args.seed))
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(int(args.seed))

    device = torch.device(args.device)
    dtype = _resolve_dtype(args.dtype)

    target, tokenizer = _load_target_and_tokenizer(args, device, dtype)

    speculator = None
    spec_len, spec_layers = _resolve_spec_config(
        args.spec_len, args.spec_layers, param_count=_count_params_torch(target)
    )
    head_rank = (
        args.head_rank
        if args.head_rank is not None and int(args.head_rank) > 0
        else None
    )
    use_mtp = isinstance(target, MiniLLMForCausalLM)
    if not args.no_speculator and not use_mtp:
        speculator_dir = Path(args.speculator_dir)
        speculator_ckpt = Path(args.speculator_ckpt) if args.speculator_ckpt else None
        speculator, spec_len = load_speculator(
            target=target,
            speculator_dir=speculator_dir,
            speculator_ckpt=speculator_ckpt,
            spec_len=spec_len,
            spec_layers=spec_layers,
            spec_heads=0,
            head_rank=head_rank,
            dropout=0.0,
        )
        speculator = speculator.to(device)

    token_logger: Optional[TokenLogWriter] = None
    if args.token_log:
        if speculator is None:
            print("[warn] --token_log ignored because speculator is disabled", flush=True)
        else:
            token_log_path = Path(args.token_log)
            token_logger = TokenLogWriter(str(token_log_path), tokenizer)
            print(
                f"[bench] token_log={token_log_path} token_summary={token_logger.summary_path}",
                flush=True,
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

    prompt_inputs: List[torch.Tensor] = []
    for messages in prompt_messages:
        if args.no_chat_template:
            prompt_text = messages[-1]["content"]
        else:
            prompt_text = _apply_chat_template(
                tokenizer, messages, add_generation_prompt=True
            )
        input_ids = tokenizer(
            prompt_text, add_special_tokens=False, return_tensors="pt"
        ).input_ids
        prompt_inputs.append(input_ids.to(device))

    baseline_results: List[BenchResult] = []
    spec_results: List[SpecBenchResult] = []

    total_samples = int(args.rounds) * len(prompt_inputs)
    if total_samples <= 0:
        raise ValueError(
            "No prompts to benchmark; check --max_samples or prompts file."
        )
    completed = 0
    use_cache = args.target_arch != "minillm"
    prompt_count = len(prompt_inputs)
    try:
        for round_idx in range(int(args.rounds)):
            for prompt_idx, input_ids in enumerate(prompt_inputs):
                seed_base = (
                    int(args.seed) + (int(round_idx) * prompt_count) + int(prompt_idx)
                )
                torch.manual_seed(seed_base)
                if torch.cuda.is_available():
                    torch.cuda.manual_seed_all(seed_base)
                baseline = _run_baseline(
                    target=target,
                    input_ids=input_ids,
                    max_new_tokens=args.max_new_tokens,
                    temperature=args.temperature,
                    top_p=args.top_p,
                    eos_token_id=tokenizer.eos_token_id,
                    use_cache=use_cache,
                    device=device,
                )
                baseline_results.append(baseline)

                if speculator is not None:
                    torch.manual_seed(seed_base)
                    if torch.cuda.is_available():
                        torch.cuda.manual_seed_all(seed_base)
                    trace = (
                        token_logger.trace_for(round_idx=round_idx, prompt_idx=prompt_idx)
                        if token_logger is not None
                        else None
                    )
                    spec = _run_spec(
                        target=target,
                        speculator=speculator,
                        input_ids=input_ids,
                        max_new_tokens=args.max_new_tokens,
                        spec_len=spec_len,
                        temperature=args.temperature,
                        top_p=args.top_p,
                        eos_token_id=tokenizer.eos_token_id,
                        use_cache=use_cache,
                        device=device,
                        trace=trace,
                    )
                    spec_results.append(spec)
                completed += 1
                _render_progress(completed, total_samples)
    finally:
        if token_logger is not None:
            token_logger.close()

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
        total_accept = sum(r.stats.accepted_output for r in rows)
        total_target = sum(r.stats.target_generated for r in rows)
        total_steps = sum(r.stats.steps for r in rows)
        zero_accept = sum(r.stats.zero_accept for r in rows)
        denom = float(total_accept + total_target)
        accept_rate = float(total_accept / denom) if denom > 0 else 0.0
        mean_accept = float(total_accept / total_steps) if total_steps else 0.0
        zero_rate = float(zero_accept / total_steps) if total_steps else 0.0
        return {
            "accept_rate": accept_rate,
            "mean_accept": mean_accept,
            "zero_rate": zero_rate,
        }

    def summarize_tokens(rows: List[SpecBenchResult]) -> Dict[str, float]:
        total_draft = sum(r.stats.total_draft for r in rows)
        total_accept = sum(r.stats.accepted_output for r in rows)
        total_target = sum(r.stats.target_generated for r in rows)
        return {
            "draft_tokens": float(total_draft),
            "accepted_tokens": float(total_accept),
            "target_generated": float(total_target),
        }

    def summarize_timing(rows: List[SpecBenchResult]) -> Dict[str, float]:
        total_spec = sum(r.stats.spec_time_s for r in rows)
        total_target = sum(r.stats.target_time_s for r in rows)
        total_prefill = sum(r.stats.target_prefill_time_s for r in rows)
        total_verify = sum(r.stats.target_verify_time_s for r in rows)
        total_generate = sum(r.stats.target_generate_time_s for r in rows)
        return {
            "spec_time_s": float(total_spec),
            "target_time_s": float(total_target),
            "target_prefill_time_s": float(total_prefill),
            "target_verify_time_s": float(total_verify),
            "target_generate_time_s": float(total_generate),
        }

    def summarize_calls(rows: List[SpecBenchResult]) -> Dict[str, float]:
        total_prefill = sum(r.stats.target_prefill_calls for r in rows)
        total_verify = sum(r.stats.target_verify_calls for r in rows)
        total_generate = sum(r.stats.target_generate_calls for r in rows)
        return {
            "prefill": float(total_prefill),
            "verify": float(total_verify),
            "generate": float(total_generate),
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
        speedup_tok = spec_stats["tok_per_s"] / max(base_stats["tok_per_s"], 1e-6)
        accept_stats = summarize_acceptance(spec_results)
        token_stats = summarize_tokens(spec_results)
        timing_stats = summarize_timing(spec_results)
        call_stats = summarize_calls(spec_results)
        target_per_out = token_stats["target_generated"] / max(
            spec_stats["output_tokens"], 1e-6
        )
        accept_per_out = token_stats["accepted_tokens"] / max(
            spec_stats["output_tokens"], 1e-6
        )
        draft_per_out = token_stats["draft_tokens"] / max(
            spec_stats["output_tokens"], 1e-6
        )
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
        other_time = (
            spec_stats["total_time_s"]
            - timing_stats["spec_time_s"]
            - timing_stats["target_time_s"]
        )
        print(
            f"[bench] time_s baseline={base_stats['total_time_s']:.2f} "
            f"draft={timing_stats['spec_time_s']:.2f} "
            f"target={timing_stats['target_time_s']:.2f} other={other_time:.2f}"
        )
        print(
            f"[bench] target_time_s prefill={timing_stats['target_prefill_time_s']:.2f} "
            f"verify={timing_stats['target_verify_time_s']:.2f} "
            f"generate={timing_stats['target_generate_time_s']:.2f}"
        )
        baseline_calls = base_stats["output_tokens"]
        if args.target_arch == "qwen3":
            baseline_calls += len(baseline_results)
        print(
            f"[bench] forward_calls baseline={baseline_calls:.0f} "
            f"verify={call_stats['verify']:.0f} "
            f"generate={call_stats['generate']:.0f} "
            f"prefill={call_stats['prefill']:.0f}"
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
