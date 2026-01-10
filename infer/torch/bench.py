#!/usr/bin/env python3
import argparse
import sys
import json
import math
import time
from pathlib import Path
from types import SimpleNamespace
from typing import Any, Dict, Iterable, List, Optional, Tuple

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from model.model_minillm import MiniLLMConfig, MiniLLMForCausalLM


DEFAULT_PROMPTS = [
    "Explain the difference between overfitting and underfitting in one paragraph.",
    "Solve: (18 * 7) - (56 / 4) + 9. Show quick steps.",
    "Write a short email politely declining a meeting invite.",
    "List three pros and cons of renewable energy.",
    "What is a hash table? Give a concise definition and one use case.",
    "Summarize the plot of 'Cinderella' in two sentences.",
]


def _resolve_dtype(name: str) -> torch.dtype:
    name = str(name).lower()
    if name == "float16":
        return torch.float16
    if name == "bfloat16":
        return torch.bfloat16
    if name == "float32":
        return torch.float32
    raise ValueError(f"Unsupported dtype: {name}")


def _count_params_torch(model: torch.nn.Module) -> Optional[int]:
    try:
        return int(sum(p.numel() for p in model.parameters()))
    except Exception:
        return None


def _auto_spec_config(param_count: Optional[int]) -> Tuple[int, int]:
    if not param_count or param_count <= 0:
        return 1, 1
    params_b = float(param_count) / 1e9
    if params_b <= 1.0:
        return 1, 1
    if params_b <= 3.0:
        return 2, 1
    if params_b <= 7.0:
        return 3, 2
    if params_b <= 13.0:
        return 4, 2
    return 5, 2


def _resolve_spec_config(
    spec_len: Optional[int], spec_layers: Optional[int], *, param_count: Optional[int]
) -> Tuple[int, int]:
    auto_len, auto_layers = _auto_spec_config(param_count)
    if spec_len is None or int(spec_len) <= 0:
        spec_len = auto_len
    if spec_layers is None or int(spec_layers) <= 0:
        spec_layers = auto_layers
    return int(spec_len), int(spec_layers)


def _load_minillm_config(path: Optional[str]) -> MiniLLMConfig:
    if not path:
        return MiniLLMConfig()
    cfg_path = Path(path)
    with cfg_path.open("r", encoding="utf-8") as f:
        data = json.load(f)
    if not isinstance(data, dict):
        raise ValueError(f"MiniLLM config must be a JSON object: {cfg_path}")
    return MiniLLMConfig(**data)


def _extract_hidden_state(output: Any) -> torch.Tensor:
    return output.last_hidden_state


def _load_target_and_tokenizer(args, device: torch.device, dtype: torch.dtype):
    if args.target_arch == "minillm":
        tokenizer = AutoTokenizer.from_pretrained(args.minillm_tokenizer)
        if tokenizer.pad_token_id is None:
            tokenizer.pad_token_id = tokenizer.eos_token_id or 0
        tokenizer.padding_side = "right"

        cfg = _load_minillm_config(args.minillm_config)
        target = MiniLLMForCausalLM(cfg)
        if args.minillm_ckpt:
            state = torch.load(args.minillm_ckpt, map_location=device)
            target.load_state_dict(state, strict=False)
        else:
            print("[warn] MiniLLM checkpoint not provided; using random weights")
        target = target.to(device=device, dtype=dtype)
    else:
        tokenizer = AutoTokenizer.from_pretrained(args.target_model, trust_remote_code=True)
        if tokenizer.pad_token_id is None:
            tokenizer.pad_token_id = tokenizer.eos_token_id or 0
        tokenizer.padding_side = "right"

        target = AutoModelForCausalLM.from_pretrained(
            args.target_model, trust_remote_code=True, torch_dtype=dtype
        ).to(device)
    target.eval()
    return target, tokenizer


def _apply_chat_template(tokenizer, messages: List[Dict[str, Any]], *, add_generation_prompt: bool) -> str:
    return tokenizer.apply_chat_template(
        messages,
        tokenize=False,
        add_generation_prompt=add_generation_prompt,
        enable_thinking=False,
    )


def _add_system(messages: List[Dict[str, Any]], system: Optional[str]) -> List[Dict[str, Any]]:
    if system and (not messages or messages[0].get("role") != "system"):
        return [{"role": "system", "content": system}] + messages
    return messages


def _iter_prompts_from_record(record: Dict[str, Any], system: Optional[str]) -> Iterable[List[Dict[str, Any]]]:
    if isinstance(record.get("conversations"), list):
        yield _add_system(list(record["conversations"]), system)
        return
    if isinstance(record.get("messages"), list):
        yield _add_system(list(record["messages"]), system)
        return
    if isinstance(record.get("turns"), list):
        for turn in record["turns"]:
            yield _add_system([{"role": "user", "content": str(turn)}], system)
        return
    if isinstance(record.get("prompt"), str):
        yield _add_system([{"role": "user", "content": record["prompt"]}], system)
        return
    if isinstance(record.get("text"), str):
        yield _add_system([{"role": "user", "content": record["text"]}], system)
        return


def _load_prompt_messages(path: Optional[str], system: Optional[str]) -> List[List[Dict[str, Any]]]:
    prompts: List[List[Dict[str, Any]]] = []
    if path:
        with open(path, "r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                record = json.loads(line)
                for msg in _iter_prompts_from_record(record, system):
                    prompts.append(msg)
    else:
        for prompt in DEFAULT_PROMPTS:
            prompts.append(_add_system([{"role": "user", "content": prompt}], system))
    return prompts


def _stats(values: List[float]) -> Optional[Dict[str, float]]:
    if not values:
        return None
    vals = sorted(float(v) for v in values)
    n = len(vals)

    def pct(p: float) -> float:
        if n == 1:
            return vals[0]
        k = (n - 1) * p
        f = math.floor(k)
        c = math.ceil(k)
        if f == c:
            return vals[int(k)]
        d0 = vals[f] * (c - k)
        d1 = vals[c] * (k - f)
        return d0 + d1

    return {
        "mean": sum(vals) / n,
        "p50": pct(0.5),
        "p90": pct(0.9),
        "min": vals[0],
        "max": vals[-1],
    }


def _fmt(stats: Optional[Dict[str, float]], *, scale: float = 1.0, unit: str = "") -> str:
    if not stats:
        return "n/a"
    return (
        f"mean={stats['mean'] * scale:.2f}{unit} "
        f"p50={stats['p50'] * scale:.2f}{unit} "
        f"p90={stats['p90'] * scale:.2f}{unit} "
        f"min={stats['min'] * scale:.2f}{unit} "
        f"max={stats['max'] * scale:.2f}{unit}"
    )


def _clone_past_key_values(past):
    if past is None:
        return None
    return tuple((k.clone(), v.clone()) for (k, v) in past)


def sample_next_token(logits: torch.Tensor, *, temperature: float, top_p: float) -> int:
    if temperature <= 0:
        return int(torch.argmax(logits, dim=-1).item())
    logits = logits / float(temperature)
    if top_p >= 1.0:
        probs = torch.softmax(logits, dim=-1)
        return int(torch.multinomial(probs, num_samples=1).item())
    sorted_logits, sorted_idx = torch.sort(logits, descending=True)
    probs = torch.softmax(sorted_logits, dim=-1)
    cumprobs = torch.cumsum(probs, dim=-1)
    mask = cumprobs > float(top_p)
    if mask[..., 0].item():
        mask[..., 0] = False
    sorted_logits = torch.where(mask, torch.tensor(-1e9, device=logits.device), sorted_logits)
    probs = torch.softmax(sorted_logits, dim=-1)
    picked = torch.multinomial(probs, num_samples=1)
    return int(sorted_idx.gather(-1, picked).item())


class LowRankHead(torch.nn.Module):
    def __init__(self, *, hidden_size: int, vocab_size: int, rank: int) -> None:
        super().__init__()
        self.proj = torch.nn.Linear(hidden_size, int(rank), bias=False)
        self.out = torch.nn.Linear(int(rank), vocab_size, bias=False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.out(self.proj(x))


class Eagle3Speculator(torch.nn.Module):
    def __init__(
        self,
        *,
        hidden_size: int,
        vocab_size: int,
        spec_len: int,
        spec_layers: int,
        spec_heads: int,
        dropout: float,
        init_weight: Optional[torch.Tensor],
        head_rank: Optional[int] = None,
    ) -> None:
        super().__init__()
        self.layers = torch.nn.ModuleList(
            [
                torch.nn.TransformerEncoderLayer(
                    d_model=hidden_size,
                    nhead=spec_heads,
                    dim_feedforward=hidden_size * 4,
                    dropout=dropout,
                    activation="gelu",
                    batch_first=True,
                    norm_first=True,
                )
                for _ in range(int(spec_layers))
            ]
        )
        self.norm = torch.nn.LayerNorm(hidden_size)
        rank = int(head_rank) if head_rank is not None and int(head_rank) > 0 else 0
        if rank > 0:
            self.heads = torch.nn.ModuleList(
                [LowRankHead(hidden_size=hidden_size, vocab_size=vocab_size, rank=rank) for _ in range(int(spec_len))]
            )
        else:
            self.heads = torch.nn.ModuleList(
                [torch.nn.Linear(hidden_size, vocab_size, bias=False) for _ in range(int(spec_len))]
            )
            if init_weight is not None:
                for head in self.heads:
                    head.weight.data.copy_(init_weight)

    def forward(self, hidden: torch.Tensor) -> List[torch.Tensor]:
        x = hidden
        seq_len = x.shape[1]
        causal_mask = None
        if seq_len > 1:
            # Prevent peeking at future positions during multi-token drafting.
            causal_mask = torch.triu(torch.ones((seq_len, seq_len), device=x.device, dtype=torch.bool), diagonal=1)
        for layer in self.layers:
            x = layer(x, src_mask=causal_mask)
        x = self.norm(x)
        return [head(x) for head in self.heads]


def load_speculator(
    *,
    target: AutoModelForCausalLM,
    speculator_dir: str,
    speculator_ckpt: Optional[str],
    spec_len: int,
    spec_layers: int,
    spec_heads: int,
    head_rank: Optional[int],
    dropout: float,
) -> Tuple[Eagle3Speculator, int]:
    cfg_path = Path(speculator_dir) / "speculator_config.json"
    if cfg_path.is_file():
        with cfg_path.open("r", encoding="utf-8") as f:
            cfg = json.load(f)
        spec_len = int(cfg.get("spec_len", spec_len))
        spec_layers = int(cfg.get("spec_layers", spec_layers))
        spec_heads = int(cfg.get("spec_heads", spec_heads))
        if "head_rank" in cfg:
            head_rank = cfg.get("head_rank", head_rank)

    hidden_size = int(target.config.hidden_size)
    vocab_size = int(target.config.vocab_size)
    if spec_heads <= 0:
        cfg_heads = int(target.config.num_attention_heads)
        if cfg_heads > 0:
            spec_heads = cfg_heads
        else:
            spec_heads = max(1, hidden_size // 64)
        while spec_heads > 1 and hidden_size % spec_heads != 0:
            spec_heads -= 1

    init_weight = target.lm_head.weight.detach().clone()

    speculator = Eagle3Speculator(
        hidden_size=hidden_size,
        vocab_size=vocab_size,
        spec_len=spec_len,
        spec_layers=spec_layers,
        spec_heads=spec_heads,
        dropout=dropout,
        init_weight=init_weight,
        head_rank=head_rank,
    )

    ckpt = None
    if speculator_ckpt:
        ckpt = Path(speculator_ckpt)
    else:
        root = Path(speculator_dir) / "checkpoints"
        if root.is_dir():
            latest = None
            best = -1
            for child in root.iterdir():
                if not child.is_dir():
                    continue
                if child.name.startswith("step_"):
                    try:
                        step = int(child.name.split("_", 1)[1])
                    except (ValueError, IndexError):
                        continue
                    if step > best:
                        best = step
                        latest = child
            if latest is not None:
                ckpt = latest / "speculator.pt"
    if ckpt is None or not ckpt.is_file():
        raise FileNotFoundError(f"Speculator checkpoint not found under {speculator_dir}")

    state = torch.load(ckpt, map_location="cpu")
    speculator.load_state_dict(state, strict=True)
    speculator.eval()
    return speculator, spec_len


@torch.inference_mode()
def baseline_generate(
    *,
    target: AutoModelForCausalLM,
    input_ids: torch.Tensor,
    max_new_tokens: int,
    temperature: float,
    top_p: float,
    eos_token_id: Optional[int],
) -> SimpleNamespace:
    num_input_tokens = int(input_ids.shape[1])
    max_length = num_input_tokens + int(max_new_tokens)

    prefill_start = time.perf_counter()
    out = target(input_ids=input_ids, use_cache=True, return_dict=True)
    last_logits = out.logits[:, -1, :]
    past = out.past_key_values
    time_to_first_token = time.perf_counter() - prefill_start

    output_ids = input_ids[0].tolist()

    decode_start = time.perf_counter()
    while len(output_ids) < max_length:
        token = sample_next_token(last_logits, temperature=temperature, top_p=top_p)
        output_ids.append(int(token))
        if eos_token_id is not None and token == int(eos_token_id):
            break
        step_ids = torch.tensor([[int(token)]], device=input_ids.device, dtype=torch.long)
        out = target(input_ids=step_ids, past_key_values=past, use_cache=True, return_dict=True)
        past = out.past_key_values
        last_logits = out.logits[:, -1, :]

    total_decode_time = time.perf_counter() - decode_start
    num_output_tokens = max(len(output_ids) - num_input_tokens, 0)
    time_per_output_token = total_decode_time / max(num_output_tokens, 1)

    return SimpleNamespace(
        output_ids=output_ids,
        num_input_tokens=num_input_tokens,
        num_output_tokens=num_output_tokens,
        time_to_first_token=time_to_first_token,
        time_per_output_token=time_per_output_token,
        acceptance_lengths=[],
    )


@torch.inference_mode()
def speculative_generate(
    *,
    target: AutoModelForCausalLM,
    speculator: Eagle3Speculator,
    input_ids: torch.Tensor,
    max_new_tokens: int,
    spec_len: int,
    temperature: float,
    top_p: float,
    eos_token_id: Optional[int],
    use_cache: bool,
) -> SimpleNamespace:
    num_input_tokens = int(input_ids.shape[1])
    max_length = num_input_tokens + int(max_new_tokens)

    prefill_start = time.perf_counter()
    out = target(
        input_ids=input_ids,
        use_cache=bool(use_cache),
        output_hidden_states=True,
        return_dict=True,
    )
    past = out.past_key_values if use_cache else None
    last_hidden = _extract_hidden_state(out)[:, -1:, :]
    last_logits = out.logits[:, -1, :]
    time_to_first_token = time.perf_counter() - prefill_start

    output_ids = input_ids[0].tolist()
    acceptance_lengths: List[int] = []

    decode_start = time.perf_counter()
    while len(output_ids) < max_length:
        remaining = max_length - len(output_ids)
        block_len = min(int(spec_len), int(remaining))

        logits_list = speculator(last_hidden)
        draft_tokens = [
            sample_next_token(logits_list[i][:, -1, :], temperature=temperature, top_p=top_p)
            for i in range(block_len)
        ]

        if use_cache:
            past_snapshot = _clone_past_key_values(past)
            draft_tensor = torch.tensor([draft_tokens], dtype=torch.long, device=input_ids.device)
            block_out = target(
                input_ids=draft_tensor,
                past_key_values=past,
                use_cache=True,
                output_hidden_states=True,
                return_dict=True,
            )
            block_logits = block_out.logits
            block_hidden = _extract_hidden_state(block_out)
        else:
            draft_tensor = torch.tensor([draft_tokens], dtype=torch.long, device=input_ids.device)
            block_out = target(
                input_ids=torch.cat([input_ids.new_tensor([output_ids]), draft_tensor], dim=1),
                use_cache=False,
                output_hidden_states=True,
                return_dict=True,
            )
            block_logits = block_out.logits[:, -block_len:, :]
            block_hidden = _extract_hidden_state(block_out)[:, -block_len:, :]
            past_snapshot = None
        shifted_logits = torch.cat([last_logits.unsqueeze(1), block_logits[:, :-1, :]], dim=1)

        posterior_tokens = [
            sample_next_token(shifted_logits[:, i, :], temperature=temperature, top_p=top_p)
            for i in range(block_len)
        ]

        accept_len = 0
        for i in range(block_len):
            if draft_tokens[i] == posterior_tokens[i]:
                accept_len += 1
            else:
                break
        acceptance_lengths.append(int(accept_len))

        new_tokens = list(draft_tokens[:accept_len])
        if accept_len < block_len:
            new_tokens.append(int(posterior_tokens[accept_len]))
        if not new_tokens:
            break

        output_ids.extend(new_tokens)
        if eos_token_id is not None and eos_token_id in new_tokens:
            eos_idx = new_tokens.index(int(eos_token_id))
            tail = len(new_tokens) - eos_idx - 1
            if tail > 0:
                output_ids = output_ids[:-tail]
            break

        if use_cache:
            if accept_len == block_len:
                past = block_out.past_key_values
                last_hidden = block_hidden[:, -1:, :]
                last_logits = block_out.logits[:, -1, :]
            else:
                if past_snapshot is None:
                    out = target(
                        input_ids=input_ids.new_tensor([output_ids]),
                        use_cache=bool(use_cache),
                        output_hidden_states=True,
                        return_dict=True,
                    )
                    past = out.past_key_values if use_cache else None
                    last_hidden = _extract_hidden_state(out)[:, -1:, :]
                    last_logits = out.logits[:, -1, :]
                else:
                    past = past_snapshot
                    accept_tensor = torch.tensor([new_tokens], dtype=torch.long, device=input_ids.device)
                    accept_out = target(
                        input_ids=accept_tensor,
                        past_key_values=past,
                        use_cache=True,
                        output_hidden_states=True,
                        return_dict=True,
                    )
                    past = accept_out.past_key_values
                    last_hidden = _extract_hidden_state(accept_out)[:, -1:, :]
                    last_logits = accept_out.logits[:, -1, :]
        else:
            out = target(
                input_ids=input_ids.new_tensor([output_ids]),
                use_cache=False,
                output_hidden_states=True,
                return_dict=True,
            )
            last_hidden = _extract_hidden_state(out)[:, -1:, :]
            last_logits = out.logits[:, -1, :]

    total_decode_time = time.perf_counter() - decode_start
    num_output_tokens = max(len(output_ids) - num_input_tokens, 0)
    time_per_output_token = total_decode_time / max(num_output_tokens, 1)

    return SimpleNamespace(
        output_ids=output_ids,
        num_input_tokens=num_input_tokens,
        num_output_tokens=num_output_tokens,
        time_to_first_token=time_to_first_token,
        time_per_output_token=time_per_output_token,
        acceptance_lengths=acceptance_lengths,
    )


def main() -> None:
    parser = argparse.ArgumentParser(description="Benchmark EAGLE-3 speculator vs baseline.")
    parser.add_argument("--target_arch", type=str, choices=["qwen3", "minillm"], default="qwen3")
    parser.add_argument("--target_model", type=str, default="Qwen/Qwen3-0.6B")
    parser.add_argument("--minillm_ckpt", type=str, default=None)
    parser.add_argument("--minillm_config", type=str, default=None)
    parser.add_argument("--minillm_tokenizer", type=str, default="./model")
    parser.add_argument("--speculator_dir", type=str, default="out/eagle3_speculator/qwen3_0.6b")
    parser.add_argument("--speculator_ckpt", type=str, default=None)
    parser.add_argument("--prompts_jsonl", type=str, default=None)
    parser.add_argument("--system", type=str, default=None)
    parser.add_argument("--max_samples", type=int, default=32)
    parser.add_argument("--max_new_tokens", type=int, default=256)
    parser.add_argument("--temperature", type=float, default=0.8)
    parser.add_argument("--top_p", type=float, default=1.0)
    parser.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--dtype", type=str, default="bfloat16")
    parser.add_argument("--spec_len", type=int, default=None, help="Draft length for speculator (auto if unset).")
    parser.add_argument("--spec_layers", type=int, default=None, help="Transformer layers in speculator (auto if unset).")
    parser.add_argument("--spec_heads", type=int, default=0)
    parser.add_argument(
        "--head_rank",
        type=int,
        default=None,
        help="Low-rank speculator head size (overrides config if set; full head if unset).",
    )
    parser.add_argument("--spec_dropout", type=float, default=0.0)
    parser.add_argument("--no_speculator", action="store_true")
    parser.add_argument("--no_cache", action="store_true")
    parser.add_argument("--seed", type=int, default=1337)
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
    head_rank = args.head_rank if args.head_rank is not None and int(args.head_rank) > 0 else None
    if not args.no_speculator:
        speculator, spec_len = load_speculator(
            target=target,
            speculator_dir=args.speculator_dir,
            speculator_ckpt=args.speculator_ckpt,
            spec_len=spec_len,
            spec_layers=spec_layers,
            spec_heads=args.spec_heads,
            head_rank=head_rank,
            dropout=args.spec_dropout,
        )
        speculator = speculator.to(device)

    prompt_messages = _load_prompt_messages(args.prompts_jsonl, args.system)
    if args.max_samples is not None:
        prompt_messages = prompt_messages[: int(args.max_samples)]

    responses = []
    for messages in prompt_messages:
        prompt_text = _apply_chat_template(tokenizer, messages, add_generation_prompt=True)
        input_ids = tokenizer(prompt_text, add_special_tokens=False, return_tensors="pt").input_ids.to(device)

        baseline = baseline_generate(
            target=target,
            input_ids=input_ids,
            max_new_tokens=args.max_new_tokens,
            temperature=args.temperature,
            top_p=args.top_p,
            eos_token_id=tokenizer.eos_token_id,
        )

        if speculator is None:
            spec = baseline
        else:
            spec = speculative_generate(
                target=target,
                speculator=speculator,
                input_ids=input_ids,
                max_new_tokens=args.max_new_tokens,
                spec_len=spec_len,
                temperature=args.temperature,
                top_p=args.top_p,
                eos_token_id=tokenizer.eos_token_id,
                use_cache=not args.no_cache,
            )

        responses.append({1: baseline, spec_len: spec})

    prompt_lens = [r[1].num_input_tokens for r in responses]
    out_lens_base = [r[1].num_output_tokens for r in responses]
    out_lens_spec = [r[spec_len].num_output_tokens for r in responses]

    base_total_times = [
        r[1].time_to_first_token + (r[1].time_per_output_token * r[1].num_output_tokens)
        for r in responses
    ]
    spec_total_times = [
        r[spec_len].time_to_first_token + (r[spec_len].time_per_output_token * r[spec_len].num_output_tokens)
        for r in responses
    ]

    base_ttft = _stats([r[1].time_to_first_token for r in responses])
    spec_ttft = _stats([r[spec_len].time_to_first_token for r in responses])
    t1 = sum(r[1].time_per_output_token for r in responses) / max(len(responses), 1)
    tb = sum(r[spec_len].time_per_output_token for r in responses) / max(len(responses), 1)
    speedup = (t1 / tb) if tb > 0 else None

    acceptance_lengths = []
    for r in responses:
        acceptance_lengths.extend(r[spec_len].acceptance_lengths)
    accept_stats = _stats(acceptance_lengths)
    accept_rate = None
    zero_accept = None
    if acceptance_lengths:
        accept_rate = sum(acceptance_lengths) / (len(acceptance_lengths) * spec_len)
        zero_accept = sum(1 for x in acceptance_lengths if x == 0) / len(acceptance_lengths)

    print(f"[bench] spec_len={spec_len} temp={args.temperature} top_p={args.top_p} dtype={args.dtype}")
    print(f"Samples: {len(responses)} | prompts={len(prompt_messages)}")
    print(f"Prompt tokens: {_fmt(_stats(prompt_lens))}")
    print(f"Output tokens (baseline): {_fmt(_stats(out_lens_base))}")
    print(f"Output tokens (spec): {_fmt(_stats(out_lens_spec))}")
    print(f"Wall ms/sample (baseline): {_fmt(_stats(base_total_times), scale=1000.0, unit='ms')}")
    print(f"Wall ms/sample (spec): {_fmt(_stats(spec_total_times), scale=1000.0, unit='ms')}")
    print(f"TTFT ms (baseline): {_fmt(base_ttft, scale=1000.0, unit='ms')}")
    print(f"TTFT ms (spec): {_fmt(spec_ttft, scale=1000.0, unit='ms')}")
    if speedup is not None:
        print(f"Decoding speedup: {speedup:.2f}")
    else:
        print("Decoding speedup: n/a")

    if acceptance_lengths:
        print(f"Acceptance length: {_fmt(accept_stats)}")
        if accept_rate is not None:
            print(f"Acceptance rate (~mean/spec_len): {accept_rate * 100:.2f}% | zero-accept: {zero_accept * 100:.2f}%")
    else:
        print("Acceptance length: n/a")


if __name__ == "__main__":
    main()
