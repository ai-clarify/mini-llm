#!/usr/bin/env python3
import argparse
import json
import os
import time
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import torch
from torch import nn
from transformers import AutoModelForCausalLM, AutoTokenizer

from model.model_minillm import MiniLLMConfig, MiniLLMForCausalLM

os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")


def _resolve_dtype(name: str) -> torch.dtype:
    name = str(name).lower()
    if name == "float16":
        return torch.float16
    if name == "bfloat16":
        return torch.bfloat16
    if name == "float32":
        return torch.float32
    raise ValueError(f"Unsupported dtype: {name}")


def _count_params_torch(model: nn.Module) -> Optional[int]:
    try:
        return int(sum(p.numel() for p in model.parameters()))
    except Exception:
        return None


def _auto_spec_config(param_count: Optional[int]) -> Tuple[int, int]:
    if not param_count or param_count <= 0:
        return 2, 2
    params_b = float(param_count) / 1e9
    if params_b <= 1.0:
        return 2, 2
    if params_b <= 3.0:
        return 2, 2
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


def _pick_latest_checkpoint(ckpt_root: Path) -> Optional[Path]:
    if not ckpt_root.is_dir():
        return None
    best = None
    best_step = -1
    for child in ckpt_root.iterdir():
        if not child.is_dir():
            continue
        name = child.name
        if not name.startswith("step_"):
            continue
        try:
            step = int(name.split("_", 1)[1])
        except (ValueError, IndexError):
            continue
        if step > best_step:
            best_step = step
            best = child
    return best


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


def _token_prob_from_logits(
    logits: torch.Tensor,
    token: int,
    *,
    temperature: float,
    top_p: float,
) -> float:
    """Token probability under (temp, top_p). Time O(V) best/avg/worst, space O(V)."""
    token_id = int(token)
    logits = logits.reshape(-1)
    if temperature <= 0:
        return 1.0 if int(torch.argmax(logits, dim=-1).item()) == token_id else 0.0
    scaled = logits / float(temperature)
    if top_p >= 1.0:
        probs = torch.softmax(scaled, dim=-1)
        return float(probs[token_id].item())
    sorted_logits, sorted_idx = torch.sort(scaled, descending=True)
    probs = torch.softmax(sorted_logits, dim=-1)
    cumprobs = torch.cumsum(probs, dim=-1)
    mask = cumprobs > float(top_p)
    if mask[..., 0].item():
        mask[..., 0] = False
    filtered_logits = torch.where(
        mask, torch.tensor(-1e9, device=logits.device), sorted_logits
    )
    filtered_probs = torch.softmax(filtered_logits, dim=-1)
    token_mask = sorted_idx == token_id
    prob = (filtered_probs * token_mask).sum()
    return float(prob.item())


class LowRankHead(nn.Module):
    def __init__(self, *, hidden_size: int, vocab_size: int, rank: int) -> None:
        super().__init__()
        self.proj = nn.Linear(hidden_size, int(rank), bias=False)
        self.out = nn.Linear(int(rank), vocab_size, bias=False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.out(self.proj(x))


class Eagle3Speculator(nn.Module):
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
        self.layers = nn.ModuleList(
            [
                nn.TransformerEncoderLayer(
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
        self.norm = nn.LayerNorm(hidden_size)
        rank = int(head_rank) if head_rank is not None and int(head_rank) > 0 else 0
        if rank > 0:
            self.heads = nn.ModuleList(
                [LowRankHead(hidden_size=hidden_size, vocab_size=vocab_size, rank=rank) for _ in range(int(spec_len))]
            )
        else:
            self.heads = nn.ModuleList([nn.Linear(hidden_size, vocab_size, bias=False) for _ in range(int(spec_len))])
            if init_weight is not None:
                for head in self.heads:
                    head.weight.data.copy_(init_weight)

    def forward(self, hidden: torch.Tensor, attention_mask: Optional[torch.Tensor]) -> List[torch.Tensor]:
        x = hidden
        key_padding_mask = None
        if attention_mask is not None:
            key_padding_mask = attention_mask == 0
        seq_len = x.shape[1]
        causal_mask = None
        if seq_len > 1:
            # Prevent peeking at future positions during multi-token drafting.
            causal_mask = torch.triu(torch.ones((seq_len, seq_len), device=x.device, dtype=torch.bool), diagonal=1)
        for layer in self.layers:
            x = layer(x, src_mask=causal_mask, src_key_padding_mask=key_padding_mask)
        x = self.norm(x)
        return [head(x) for head in self.heads]


def load_speculator(
    *,
    target: AutoModelForCausalLM,
    speculator_dir: Path,
    speculator_ckpt: Optional[Path],
    spec_len: int,
    spec_layers: int,
    spec_heads: int,
    head_rank: Optional[int],
    dropout: float,
) -> Tuple[Eagle3Speculator, int]:
    cfg_path = speculator_dir / "speculator_config.json"
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

    if speculator_ckpt is None:
        latest = _pick_latest_checkpoint(speculator_dir / "checkpoints")
        if latest is None:
            raise FileNotFoundError(f"No checkpoints under {speculator_dir}/checkpoints")
        speculator_ckpt = latest / "speculator.pt"
    if not speculator_ckpt.is_file():
        raise FileNotFoundError(f"Speculator checkpoint not found: {speculator_ckpt}")

    state = torch.load(speculator_ckpt, map_location="cpu")
    speculator.load_state_dict(state, strict=True)
    speculator.eval()
    return speculator, spec_len


@torch.inference_mode()
def baseline_decode(
    *,
    target: AutoModelForCausalLM,
    input_ids: torch.Tensor,
    max_new_tokens: int,
    temperature: float,
    top_p: float,
    eos_token_id: Optional[int],
    use_cache: bool,
) -> torch.Tensor:
    device = input_ids.device
    output_ids = input_ids.clone()
    out = target(input_ids=output_ids, use_cache=bool(use_cache), return_dict=True)
    last_logits = out.logits[:, -1, :]
    past = out.past_key_values if use_cache else None
    for _ in range(int(max_new_tokens)):
        token = sample_next_token(last_logits, temperature=temperature, top_p=top_p)
        output_ids = torch.cat([output_ids, torch.tensor([[token]], device=device, dtype=torch.long)], dim=1)
        if eos_token_id is not None and token == eos_token_id:
            break
        if use_cache:
            out = target(input_ids=output_ids[:, -1:], past_key_values=past, use_cache=True, return_dict=True)
            past = out.past_key_values
            last_logits = out.logits[:, -1, :]
        else:
            out = target(input_ids=output_ids, use_cache=False, return_dict=True)
            last_logits = out.logits[:, -1, :]
    return output_ids


@torch.inference_mode()
def speculative_decode(
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
    optimized: bool,
    max_consecutive_misses: int = 2,
) -> torch.Tensor:
    device = input_ids.device
    output_ids = input_ids.clone()

    out = target(
        input_ids=output_ids,
        use_cache=bool(use_cache),
        output_hidden_states=True,
        return_dict=True,
    )
    past = out.past_key_values if use_cache else None
    hidden = _extract_hidden_state(out)
    last_hidden = hidden[:, -1:, :]
    last_logits = out.logits[:, -1, :]

    produced = 0
    consecutive_misses = 0
    while produced < int(max_new_tokens):
        remaining = int(max_new_tokens) - produced
        block_len = min(int(spec_len), int(remaining))
        logits_list = speculator(last_hidden, attention_mask=None)
        draft_tokens = [
            sample_next_token(logits_list[i][:, -1, :], temperature=temperature, top_p=top_p)
            for i in range(int(block_len))
        ]

        if use_cache:
            past_snapshot = _clone_past_key_values(past)
            draft_tensor = torch.tensor([draft_tokens], dtype=torch.long, device=device)
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
            draft_tensor = torch.tensor([draft_tokens], dtype=torch.long, device=device)
            block_out = target(
                input_ids=torch.cat([output_ids, draft_tensor], dim=1),
                use_cache=False,
                output_hidden_states=True,
                return_dict=True,
            )
            block_logits = block_out.logits[:, -len(draft_tokens) :, :]
            block_hidden = _extract_hidden_state(block_out)[:, -len(draft_tokens) :, :]
            past_snapshot = None
        shifted_logits = torch.cat([last_logits.unsqueeze(1), block_logits[:, :-1, :]], dim=1)

        accept_len = 0
        new_tokens: List[int] = []
        rejected = False
        for i in range(len(draft_tokens)):
            draft_token = int(draft_tokens[i])
            draft_logits = logits_list[i][:, -1, :]
            target_logits = shifted_logits[:, i, :]
            q_prob = _token_prob_from_logits(
                draft_logits, draft_token, temperature=temperature, top_p=top_p
            )
            p_prob = _token_prob_from_logits(
                target_logits, draft_token, temperature=temperature, top_p=top_p
            )
            accept_prob = 0.0 if q_prob <= 0.0 else min(1.0, p_prob / q_prob)
            u = float(torch.rand(1, device=device).item())
            if u < accept_prob:
                new_tokens.append(draft_token)
                accept_len += 1
                continue
            token = sample_next_token(target_logits, temperature=temperature, top_p=top_p)
            new_tokens.append(int(token))
            rejected = True
            break

        bonus_added = False
        if not rejected and accept_len == len(draft_tokens) and remaining > len(draft_tokens):
            bonus_logits = block_logits[:, -1, :]
            bonus_token = sample_next_token(bonus_logits, temperature=temperature, top_p=top_p)
            new_tokens.append(int(bonus_token))
            bonus_added = True

        if accept_len == 0:
            consecutive_misses += 1
        else:
            consecutive_misses = 0

        remaining = int(max_new_tokens) - produced
        if len(new_tokens) > remaining:
            new_tokens = new_tokens[:remaining]

        if not new_tokens:
            break

        output_ids = torch.cat(
            [output_ids, torch.tensor([new_tokens], dtype=torch.long, device=device)], dim=1
        )
        produced += len(new_tokens)

        if eos_token_id is not None and eos_token_id in new_tokens:
            eos_idx = new_tokens.index(eos_token_id)
            tail = len(new_tokens) - eos_idx - 1
            if tail > 0:
                output_ids = output_ids[:, :-tail]
            break

        if optimized and consecutive_misses >= max_consecutive_misses:
            break

        if use_cache:
            if accept_len == len(draft_tokens):
                past = block_out.past_key_values
                last_hidden = block_hidden[:, -1:, :]
                last_logits = block_out.logits[:, -1, :]
                if bonus_added:
                    bonus_tensor = torch.tensor(
                        [[new_tokens[-1]]], dtype=torch.long, device=device
                    )
                    bonus_out = target(
                        input_ids=bonus_tensor,
                        past_key_values=past,
                        use_cache=True,
                        output_hidden_states=True,
                        return_dict=True,
                    )
                    past = bonus_out.past_key_values
                    last_hidden = _extract_hidden_state(bonus_out)[:, -1:, :]
                    last_logits = bonus_out.logits[:, -1, :]
            else:
                if past_snapshot is None:
                    out = target(
                        input_ids=output_ids,
                        use_cache=bool(use_cache),
                        output_hidden_states=True,
                        return_dict=True,
                    )
                    past = out.past_key_values if use_cache else None
                    last_hidden = _extract_hidden_state(out)[:, -1:, :]
                    last_logits = out.logits[:, -1, :]
                else:
                    past = past_snapshot
                    accept_tensor = torch.tensor([new_tokens], dtype=torch.long, device=device)
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
                input_ids=output_ids,
                use_cache=False,
                output_hidden_states=True,
                return_dict=True,
            )
            last_hidden = _extract_hidden_state(out)[:, -1:, :]
            last_logits = out.logits[:, -1, :]

    if optimized and produced < int(max_new_tokens):
        fallback_ids = baseline_decode(
            target=target,
            input_ids=output_ids,
            max_new_tokens=int(max_new_tokens) - produced,
            temperature=temperature,
            top_p=top_p,
            eos_token_id=eos_token_id,
            use_cache=use_cache,
        )
        output_ids = fallback_ids

    return output_ids


def build_arg_parser(description: str) -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=description)
    parser.add_argument("--target_arch", type=str, choices=["qwen3", "minillm"], default="qwen3")
    parser.add_argument("--target_model", type=str, default="Qwen/Qwen3-0.6B")
    parser.add_argument("--minillm_ckpt", type=str, default=None)
    parser.add_argument("--minillm_config", type=str, default=None)
    parser.add_argument("--minillm_tokenizer", type=str, default="./model")
    parser.add_argument("--speculator_dir", type=str, default="out/eagle3_speculator/qwen3_0.6b")
    parser.add_argument("--speculator_ckpt", type=str, default=None)
    parser.add_argument("--prompt", type=str, default="Hello")
    parser.add_argument("--system", type=str, default=None)
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
    parser.add_argument("--seed", type=int, default=1337)
    parser.add_argument("--no_speculator", action="store_true")
    return parser


def run_cli() -> None:
    parser = build_arg_parser(
        description="Speculative decoding with EAGLE-3 speculator (Torch backend)."
    )
    args = parser.parse_args()
    optimized = True
    use_cache = True

    torch.manual_seed(int(args.seed))
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(int(args.seed))

    device = torch.device(args.device)
    dtype = _resolve_dtype(args.dtype)

    target, tokenizer = _load_target_and_tokenizer(args, device, dtype)

    messages: List[Dict[str, str]] = []
    if args.system:
        messages.append({"role": "system", "content": args.system})
    messages.append({"role": "user", "content": args.prompt})
    prompt_text = _apply_chat_template(tokenizer, messages, add_generation_prompt=True)
    input_ids = tokenizer(prompt_text, add_special_tokens=False, return_tensors="pt").input_ids.to(device)

    speculator = None
    spec_len, spec_layers = _resolve_spec_config(
        args.spec_len, args.spec_layers, param_count=_count_params_torch(target)
    )
    head_rank = args.head_rank if args.head_rank is not None and int(args.head_rank) > 0 else None
    if not args.no_speculator:
        speculator_dir = Path(args.speculator_dir)
        speculator_ckpt = Path(args.speculator_ckpt) if args.speculator_ckpt else None
        speculator, spec_len = load_speculator(
            target=target,
            speculator_dir=speculator_dir,
            speculator_ckpt=speculator_ckpt,
            spec_len=spec_len,
            spec_layers=spec_layers,
            spec_heads=args.spec_heads,
            head_rank=head_rank,
            dropout=args.spec_dropout,
        )
        speculator = speculator.to(device)

    start = time.time()
    if speculator is None:
        output_ids = baseline_decode(
            target=target,
            input_ids=input_ids,
            max_new_tokens=args.max_new_tokens,
            temperature=args.temperature,
            top_p=args.top_p,
            eos_token_id=tokenizer.eos_token_id,
            use_cache=use_cache,
        )
    else:
        output_ids = speculative_decode(
            target=target,
            speculator=speculator,
            input_ids=input_ids,
            max_new_tokens=args.max_new_tokens,
            spec_len=spec_len,
            temperature=args.temperature,
            top_p=args.top_p,
            eos_token_id=tokenizer.eos_token_id,
            use_cache=use_cache,
            optimized=optimized,
        )
    elapsed = time.time() - start
    out_text = tokenizer.decode(output_ids[0], skip_special_tokens=True)
    print(out_text)
    print(f"[time] {elapsed:.2f}s")
