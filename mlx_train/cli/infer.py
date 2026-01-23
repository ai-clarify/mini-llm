from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
from typing import List, Optional

import mlx.core as mx

from ..config import MiniLLMConfig
from ..models import MiniLLMForCausalLM, LayerKVCache
from ..nn.lora import merge_lora
from ..trace import ActivationTracer, TraceConfig, write_trace_outputs


def load_config(checkpoint_dir: Path) -> MiniLLMConfig:
    config_path = checkpoint_dir / "config.json"
    if not config_path.exists():
        raise FileNotFoundError(f"Missing config.json in checkpoint dir: {checkpoint_dir}")
    with open(config_path, "r", encoding="utf-8") as f:
        data = json.load(f)
    if not isinstance(data, dict):
        raise ValueError(f"config.json must be an object, got: {type(data).__name__}")
    return MiniLLMConfig.from_dict(data)


def load_state_seq_len(checkpoint_dir: Path) -> Optional[int]:
    state_path = checkpoint_dir / "state.json"
    if not state_path.exists():
        return None
    try:
        with open(state_path, "r", encoding="utf-8") as f:
            data = json.load(f)
    except Exception:
        return None
    if not isinstance(data, dict):
        return None
    args = data.get("args")
    if not isinstance(args, dict):
        return None
    seq_len = args.get("seq_len")
    if seq_len is None:
        return None
    try:
        seq_len = int(seq_len)
    except Exception:
        return None
    return seq_len if seq_len > 0 else None


def sample_next_token(
    logits: mx.array,
    *,
    temperature: float,
    top_p: float,
) -> int:
    if temperature <= 0:
        return int(mx.argmax(logits).item())

    logits = logits / temperature

    if top_p >= 1.0:
        token = mx.random.categorical(logits, axis=-1)
        return int(token.item())

    # Nucleus sampling (top_p) in sorted space, then map back.
    sorted_idx = mx.argsort(-logits, axis=-1)
    sorted_logits = mx.take_along_axis(logits, sorted_idx, axis=-1)
    sorted_probs = mx.softmax(sorted_logits, axis=-1)
    cumprobs = mx.cumsum(sorted_probs, axis=-1)

    # Remove tokens once cumprob passes top_p, but keep the first token that crosses the threshold.
    remove = cumprobs > top_p
    remove = mx.concatenate([mx.array([False]), remove[:-1]], axis=-1)
    neg_inf = mx.array(-1e9, dtype=sorted_logits.dtype)
    filtered_logits = mx.where(remove, neg_inf, sorted_logits)
    picked = mx.random.categorical(filtered_logits, axis=-1)
    picked_i = int(picked.item())
    return int(sorted_idx[picked_i].item())


def generate(
    model: MiniLLMForCausalLM,
    *,
    input_ids: List[int],
    eos_token_id: Optional[int],
    max_new_tokens: int,
    min_new_tokens: int,
    banned_token_ids: Optional[List[int]] = None,
    temperature: float,
    top_p: float,
    max_seq_len: Optional[int],
    trace: Optional[ActivationTracer] = None,
) -> List[int]:
    ids = list(input_ids)
    if max_new_tokens <= 0:
        return ids
    if max_seq_len is not None and len(ids) >= int(max_seq_len):
        return ids
    if not ids:
        raise ValueError("input_ids is empty; need at least 1 token for generation")

    cache_len = int(max_seq_len) if max_seq_len is not None else (len(ids) + int(max_new_tokens) + 1)
    cache: List[LayerKVCache] = model.allocate_kv_cache(batch_size=1, max_seq_len=cache_len)

    # Prefill: run the whole prompt once and populate KV cache.
    x0 = mx.array([ids], dtype=mx.int32)
    logits0, cache = model.forward_with_cache(x0, start_pos=0, cache=cache, trace=trace)
    logits = logits0[0, -1, :]

    produced = 0
    for _ in range(int(max_new_tokens)):
        if max_seq_len is not None and len(ids) >= int(max_seq_len):
            break

        if produced < int(min_new_tokens) and banned_token_ids:
            vocab = int(logits.shape[0])
            idx = mx.arange(vocab)
            mask = mx.zeros((vocab,), dtype=mx.bool_)
            for token_id in set(banned_token_ids):
                if 0 <= int(token_id) < vocab:
                    mask = mx.logical_or(mask, idx == int(token_id))
            logits = mx.where(mask, mx.array(-1e9, dtype=logits.dtype), logits)

        next_token = sample_next_token(logits, temperature=temperature, top_p=top_p)
        ids.append(int(next_token))
        produced += 1
        if eos_token_id is not None and produced >= int(min_new_tokens) and int(next_token) == int(eos_token_id):
            break
        if max_seq_len is not None and len(ids) >= int(max_seq_len):
            break

        # Decode: feed only the last generated token; KV cache makes it O(1) per step.
        pos = len(ids) - 1  # absolute position of the token we just appended
        x = mx.array([[int(next_token)]], dtype=mx.int32)
        logits1, cache = model.forward_with_cache(x, start_pos=int(pos), cache=cache, trace=trace)
        logits = logits1[0, -1, :]

    return ids


def main() -> None:
    parser = argparse.ArgumentParser(description="MiniLLM (MLX) inference")
    parser.add_argument("--tokenizer_path", type=str, default="./model")
    parser.add_argument(
        "--tokenizer_type",
        type=str,
        choices=["auto", "hf", "rustbpe"],
        default="auto",
        help="Tokenizer backend: auto (prefer RustBPE), hf, or rustbpe.",
    )
    parser.add_argument(
        "--checkpoint",
        type=str,
        required=True,
        help="Checkpoint directory produced by mlx_train/train.py (contains model.safetensors + config.json).",
    )
    parser.add_argument("--prompt", type=str, default="hi")
    parser.add_argument("--max_new_tokens", type=int, default=512)
    parser.add_argument("--min_new_tokens", type=int, default=0, help="Force at least N new tokens before EOS stop.")
    parser.add_argument("--temperature", type=float, default=0.0, help="0 for greedy; >0 for sampling.")
    parser.add_argument("--top_p", type=float, default=1.0, help="Nucleus sampling threshold (only if temperature>0).")
    parser.add_argument("--seed", type=int, default=1337)
    parser.add_argument(
        "--max_seq_len",
        type=int,
        default=None,
        help="Max total tokens (prompt + generation). Defaults to checkpoint seq_len when available.",
    )
    parser.add_argument(
        "--trace_out",
        type=str,
        default=None,
        help="Write per-token activation trace to a directory (trace.json/trace.html) or to a .json path.",
    )
    parser.add_argument("--trace_qkv", action="store_true", help="Record per-token Q/K/V RMS (mean over heads).")
    parser.add_argument(
        "--trace_qkv_per_head",
        action="store_true",
        help="Record per-head Q/K/V RMS (can be large; currently only saved in JSON).",
    )
    parser.add_argument(
        "--trace_mlp_topk",
        type=int,
        default=0,
        help="Record top-k MLP intermediate activations per token (indices + values).",
    )
    parser.add_argument("--trace_attn", action="store_true", help="Record attention pattern as top-k keys per head.")
    parser.add_argument(
        "--trace_attn_topk",
        type=int,
        default=16,
        help="Top-k keys to keep per head when --trace_attn is enabled.",
    )
    parser.add_argument(
        "--trace_attn_all_queries",
        action="store_true",
        help="Also trace attention for all prompt tokens during prefill (can be expensive).",
    )

    args = parser.parse_args()

    mx.random.seed(args.seed)

    from .tokenizer_utils import load_tokenizer

    ckpt = Path(args.checkpoint)
    if not ckpt.exists() or not ckpt.is_dir():
        raise FileNotFoundError(f"Checkpoint must be a directory: {ckpt}")

    cfg = load_config(ckpt)
    model = MiniLLMForCausalLM(cfg)
    model.load_weights(os.fspath(ckpt / "model.safetensors"))
    model.eval()
    if int(cfg.lora_r) > 0:
        merge_lora(model)

    tokenizer = load_tokenizer(
        args.tokenizer_path,
        tokenizer_type=str(args.tokenizer_type),
    )
    messages = [{"role": "user", "content": args.prompt}]
    prompt_text = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    prompt_ids: List[int] = tokenizer.encode(prompt_text, add_special_tokens=False)
    max_seq_len = args.max_seq_len
    if max_seq_len is None:
        max_seq_len = load_state_seq_len(ckpt)
    if max_seq_len is not None:
        max_seq_len = min(int(max_seq_len), int(cfg.max_position_embeddings))

    tracer: Optional[ActivationTracer] = None
    if args.trace_out:
        tracer = ActivationTracer(
            num_layers=int(cfg.num_hidden_layers),
            cfg=TraceConfig(
                record_qkv=bool(args.trace_qkv or args.trace_qkv_per_head),
                record_qkv_per_head=bool(args.trace_qkv_per_head),
                mlp_topk=int(args.trace_mlp_topk),
                record_attn=bool(args.trace_attn),
                record_attn_all_queries=bool(args.trace_attn_all_queries),
                attn_topk=int(args.trace_attn_topk),
            ),
        )

    output_ids = generate(
        model,
        input_ids=prompt_ids,
        eos_token_id=tokenizer.eos_token_id,
        max_new_tokens=args.max_new_tokens,
        min_new_tokens=args.min_new_tokens,
        banned_token_ids=list(tokenizer.all_special_ids),
        temperature=args.temperature,
        top_p=args.top_p,
        max_seq_len=max_seq_len,
        trace=tracer,
    )

    response_ids = output_ids[len(prompt_ids) :]
    response_text = tokenizer.decode(response_ids, skip_special_tokens=True)
    text = response_text.strip()
    if text:
        print(text)
    else:
        print("[infer] empty response (try --temperature 0.7 --top_p 1.0 or set --min_new_tokens)")

    if tracer is not None and args.trace_out:
        trace = tracer.to_dict(
            tokenizer=tokenizer,
            meta={
                "checkpoint": os.fspath(ckpt),
                "prompt": args.prompt,
                "prompt_tokens": int(len(prompt_ids)),
                "total_tokens": int(len(output_ids)),
            },
        )
        json_path, html_path = write_trace_outputs(out_path=args.trace_out, trace=trace)
        print(f"[trace] wrote {json_path}")
        print(f"[trace] wrote {html_path}")


if __name__ == "__main__":
    main()
