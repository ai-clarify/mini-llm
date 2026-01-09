import argparse
import json
import random
import sys
import time
from itertools import chain
from pathlib import Path
from types import SimpleNamespace
from typing import List, Optional, Tuple

import mlx.core as mx
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from mlx_train.config import MiniLLMConfig
from mlx_train.models.minillm import LayerKVCache, MiniLLMForCausalLM
from mlx_train.nn.lora import merge_lora

from dflash_mlx.utils import resolve_dataset


def _is_valid_ckpt_dir(path: Path) -> bool:
    return (path / "model.safetensors").is_file() and (path / "config.json").is_file()


def _step_num(path: Path) -> int:
    name = path.name
    if name.startswith("step_"):
        try:
            return int(name.split("_", 1)[1])
        except ValueError:
            return -1
    return -1


def _find_latest_checkpoint(path: Path) -> Optional[Path]:
    if not path.exists() or not path.is_dir():
        return None
    candidates = sorted(
        [p for p in path.glob("step_*") if p.is_dir()],
        key=_step_num,
        reverse=True,
    )
    for cand in candidates:
        if _is_valid_ckpt_dir(cand):
            return cand
    return None


def resolve_checkpoint_dir(path: Path) -> Path:
    if path.is_file():
        if path.name == "model.safetensors":
            path = path.parent
        else:
            raise FileNotFoundError(f"Checkpoint must be a directory: {path}")

    if _is_valid_ckpt_dir(path):
        return path

    # Try common layouts: stage_dir/checkpoints/step_*
    ckpt = _find_latest_checkpoint(path / "checkpoints")
    if ckpt is not None:
        return ckpt

    # Try stage dirs under a run root.
    for stage in ("sft", "r1", "pretrain"):
        ckpt = _find_latest_checkpoint(path / stage / "checkpoints")
        if ckpt is not None:
            return ckpt

    raise FileNotFoundError(
        "No valid checkpoint found. Expect a directory containing "
        "`model.safetensors` and `config.json`, or a stage dir with "
        "`checkpoints/step_*`."
    )


def load_config(checkpoint_dir: Path) -> MiniLLMConfig:
    config_path = checkpoint_dir / "config.json"
    if not config_path.exists():
        raise FileNotFoundError(f"Missing config.json in checkpoint dir: {checkpoint_dir}")
    with open(config_path, "r", encoding="utf-8") as f:
        data = json.load(f)
    if not isinstance(data, dict):
        raise ValueError(f"config.json must be an object, got: {type(data).__name__}")
    return MiniLLMConfig.from_dict(data)


def _dtype_map(name: str) -> mx.Dtype:
    name = str(name).lower()
    if name == "float16":
        return mx.float16
    if name == "bfloat16":
        return mx.bfloat16
    if name == "float32":
        return mx.float32
    raise ValueError(f"Unsupported dtype: {name}")


def _filter_layer_weights(weights, max_layers: int):
    filtered = {}
    for key, val in weights.items():
        if key.startswith("model.layers."):
            parts = key.split(".")
            if len(parts) > 2 and parts[2].isdigit():
                layer_id = int(parts[2])
                if layer_id >= int(max_layers):
                    continue
        filtered[key] = val
    return filtered


def load_mlx_model(
    checkpoint_dir: Path,
    *,
    dtype: str,
    override_layers: Optional[int] = None,
    filter_layers: bool = False,
) -> Tuple[MiniLLMForCausalLM, MiniLLMConfig]:
    resolved_dir = resolve_checkpoint_dir(checkpoint_dir)
    if resolved_dir != checkpoint_dir:
        print(f"[bench] resolved checkpoint: {resolved_dir}")
    cfg = load_config(resolved_dir)
    if override_layers is not None:
        cfg.num_hidden_layers = int(override_layers)
    model = MiniLLMForCausalLM(cfg)

    weights_path = resolved_dir / "model.safetensors"
    if not weights_path.exists():
        raise FileNotFoundError(f"Missing model.safetensors in {resolved_dir}")

    if filter_layers:
        weights = dict(mx.load(str(weights_path)))
        weights = _filter_layer_weights(weights, int(cfg.num_hidden_layers))
        model.load_weights(list(weights.items()), strict=False)
    else:
        model.load_weights(str(weights_path))

    model.apply(lambda p: p.astype(_dtype_map(dtype)))
    model.eval()
    if int(cfg.lora_r) > 0:
        merge_lora(model)
    return model, cfg


def clone_cache(cache: List[LayerKVCache]) -> List[LayerKVCache]:
    # MLX arrays are immutable; keep references to restore the cache snapshot cheaply.
    return list(cache)


def eval_cache(logits: mx.array, cache: List[LayerKVCache]) -> None:
    mx.eval(logits, *(c.k for c in cache), *(c.v for c in cache))


def sample_next_token(logits: mx.array, *, temperature: float, top_p: float) -> int:
    if temperature <= 0:
        return int(mx.argmax(logits).item())

    logits = logits / temperature

    if top_p >= 1.0:
        token = mx.random.categorical(logits, axis=-1)
        return int(token.item())

    sorted_idx = mx.argsort(-logits, axis=-1)
    sorted_logits = mx.take_along_axis(logits, sorted_idx, axis=-1)
    sorted_probs = mx.softmax(sorted_logits, axis=-1)
    cumprobs = mx.cumsum(sorted_probs, axis=-1)

    remove = cumprobs > top_p
    remove = mx.concatenate([mx.array([False]), remove[:-1]], axis=-1)
    neg_inf = mx.array(-1e9, dtype=sorted_logits.dtype)
    filtered_logits = mx.where(remove, neg_inf, sorted_logits)
    picked = mx.random.categorical(filtered_logits, axis=-1)
    picked_i = int(picked.item())
    return int(sorted_idx[picked_i].item())


def _find_stop(output_ids: List[int], stop_id: Optional[int], start: int) -> Optional[int]:
    if stop_id is None:
        return None
    try:
        return output_ids.index(int(stop_id), int(start))
    except ValueError:
        return None


def _sample_block(logits: mx.array, *, temperature: float, top_p: float) -> List[int]:
    tokens: List[int] = []
    for i in range(int(logits.shape[0])):
        tokens.append(sample_next_token(logits[i], temperature=temperature, top_p=top_p))
    return tokens


def _forward_with_cache(
    model: MiniLLMForCausalLM,
    x: mx.array,
    start_pos: int,
    cache: List[LayerKVCache],
) -> Tuple[mx.array, List[LayerKVCache]]:
    return model.forward_with_cache(x, start_pos=int(start_pos), cache=cache)


def baseline_generate(
    target: MiniLLMForCausalLM,
    input_ids: List[int],
    *,
    max_new_tokens: int,
    temperature: float,
    top_p: float,
    stop_token_id: Optional[int],
) -> SimpleNamespace:
    num_input_tokens = len(input_ids)
    max_length = num_input_tokens + max_new_tokens
    if num_input_tokens <= 0:
        raise ValueError("input_ids is empty")

    cache_len = max_length + 1
    cache = target.allocate_kv_cache(batch_size=1, max_seq_len=cache_len)

    prefill_start = time.perf_counter()
    prompt = mx.array([input_ids], dtype=mx.int32)
    logits, cache = _forward_with_cache(target, prompt, start_pos=0, cache=cache)
    eval_cache(logits, cache)
    last_logits = logits[0, -1, :]
    time_to_first_token = time.perf_counter() - prefill_start

    output_ids = list(input_ids)

    decode_start = time.perf_counter()
    while len(output_ids) < max_length:
        token = sample_next_token(last_logits, temperature=temperature, top_p=top_p)
        output_ids.append(int(token))
        stop_at = _find_stop(output_ids, stop_token_id, num_input_tokens)
        if stop_at is not None:
            output_ids = output_ids[: stop_at + 1]
            break
        if len(output_ids) >= max_length:
            break
        x = mx.array([[int(token)]], dtype=mx.int32)
        logits, cache = _forward_with_cache(target, x, start_pos=len(output_ids) - 1, cache=cache)
        eval_cache(logits, cache)
        last_logits = logits[0, -1, :]

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


def speculative_generate(
    target: MiniLLMForCausalLM,
    draft: MiniLLMForCausalLM,
    input_ids: List[int],
    *,
    max_new_tokens: int,
    block_size: int,
    temperature: float,
    top_p: float,
    stop_token_id: Optional[int],
) -> SimpleNamespace:
    num_input_tokens = len(input_ids)
    max_length = num_input_tokens + max_new_tokens
    if num_input_tokens <= 0:
        raise ValueError("input_ids is empty")

    cache_len = max_length + block_size + 1
    target_cache = target.allocate_kv_cache(batch_size=1, max_seq_len=cache_len)
    draft_cache = draft.allocate_kv_cache(batch_size=1, max_seq_len=cache_len)

    prefill_start = time.perf_counter()
    prompt = mx.array([input_ids], dtype=mx.int32)

    t_logits, target_cache = _forward_with_cache(target, prompt, start_pos=0, cache=target_cache)
    eval_cache(t_logits, target_cache)

    d_logits, draft_cache = _forward_with_cache(draft, prompt, start_pos=0, cache=draft_cache)
    eval_cache(d_logits, draft_cache)
    draft_logits = d_logits[0, -1, :]

    time_to_first_token = time.perf_counter() - prefill_start

    output_ids = list(input_ids)
    acceptance_lengths: List[int] = []

    decode_start = time.perf_counter()
    while len(output_ids) < max_length:
        remaining = max_length - len(output_ids)
        block_len = min(int(block_size), int(remaining))
        if block_len <= 0:
            break

        target_cache_snapshot = clone_cache(target_cache)
        draft_cache_snapshot = clone_cache(draft_cache)

        draft_tokens: List[int] = []
        draft_logits_local = draft_logits

        for i in range(block_len):
            token = sample_next_token(draft_logits_local, temperature=temperature, top_p=top_p)
            draft_tokens.append(int(token))
            x = mx.array([[int(token)]], dtype=mx.int32)
            d_logits, draft_cache = _forward_with_cache(
                draft,
                x,
                start_pos=len(output_ids) + i,
                cache=draft_cache,
            )
            eval_cache(d_logits, draft_cache)
            draft_logits_local = d_logits[0, -1, :]

        x_block = mx.array([draft_tokens], dtype=mx.int32)
        t_logits, target_cache = _forward_with_cache(
            target,
            x_block,
            start_pos=len(output_ids),
            cache=target_cache,
        )
        eval_cache(t_logits, target_cache)
        block_logits = t_logits[0]

        posterior_tokens = _sample_block(block_logits, temperature=temperature, top_p=top_p)

        accept_len = 0
        for i in range(block_len):
            if draft_tokens[i] == posterior_tokens[i]:
                accept_len += 1
            else:
                break

        acceptance_lengths.append(int(accept_len))

        new_tokens = list(draft_tokens[:accept_len])
        if accept_len < block_len and len(output_ids) + len(new_tokens) < max_length:
            new_tokens.append(int(posterior_tokens[accept_len]))

        if not new_tokens:
            break

        if accept_len == block_len:
            output_ids.extend(draft_tokens)
            draft_logits = draft_logits_local
            stop_at = _find_stop(output_ids, stop_token_id, num_input_tokens)
            if stop_at is not None:
                output_ids = output_ids[: stop_at + 1]
                break
            continue

        target_cache = target_cache_snapshot
        draft_cache = draft_cache_snapshot

        x_accept = mx.array([new_tokens], dtype=mx.int32)
        t_logits, target_cache = _forward_with_cache(
            target,
            x_accept,
            start_pos=len(output_ids),
            cache=target_cache,
        )
        eval_cache(t_logits, target_cache)

        d_logits, draft_cache = _forward_with_cache(
            draft,
            x_accept,
            start_pos=len(output_ids),
            cache=draft_cache,
        )
        eval_cache(d_logits, draft_cache)
        draft_logits = d_logits[0, -1, :]

        output_ids.extend(new_tokens)
        stop_at = _find_stop(output_ids, stop_token_id, num_input_tokens)
        if stop_at is not None:
            output_ids = output_ids[: stop_at + 1]
            break

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
    parser = argparse.ArgumentParser(description="DFlash-style speculative decoding (MLX)")
    parser.add_argument("--target_ckpt", type=str, required=True)
    parser.add_argument("--draft_ckpt", type=str, default=None)
    parser.add_argument("--draft_layers", type=int, default=None)
    parser.add_argument("--tokenizer_path", type=str, default="./model")
    parser.add_argument("--dataset", type=str, required=True)
    parser.add_argument("--max_samples", type=int, default=None)
    parser.add_argument("--max_new_tokens", type=int, default=256)
    parser.add_argument("--block_size", type=int, required=True)
    parser.add_argument("--temperature", type=float, default=0.0)
    parser.add_argument("--top_p", type=float, default=1.0)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--dtype", type=str, default="float16", choices=["float16", "bfloat16", "float32"])
    args = parser.parse_args()

    random.seed(args.seed)
    np.random.seed(args.seed)
    mx.random.seed(args.seed)

    try:
        from transformers import AutoTokenizer
    except ImportError as e:
        raise ImportError(
            "Failed to import `transformers`. Install MLX deps via "
            "`python3 -m pip install -r mlx_train/requirements.txt`."
        ) from e

    target_dir = Path(args.target_ckpt)
    target_model, target_cfg = load_mlx_model(target_dir, dtype=args.dtype)

    if args.draft_ckpt:
        draft_dir = Path(args.draft_ckpt)
        draft_model, _ = load_mlx_model(draft_dir, dtype=args.dtype)
    else:
        target_layers = int(target_cfg.num_hidden_layers)
        draft_layers = args.draft_layers
        if draft_layers is None:
            draft_layers = max(1, target_layers // 2)
        if draft_layers > target_layers:
            raise ValueError(f"draft_layers {draft_layers} > target_layers {target_layers}")
        draft_model, _ = load_mlx_model(
            target_dir,
            dtype=args.dtype,
            override_layers=int(draft_layers),
            filter_layers=True,
        )

    tokenizer = AutoTokenizer.from_pretrained(args.tokenizer_path)

    dataset = resolve_dataset(args.dataset)
    if args.max_samples is not None:
        if hasattr(dataset, "shuffle"):
            if len(dataset) > args.max_samples:
                dataset = dataset.shuffle(seed=args.seed).select(range(args.max_samples))
        else:
            dataset = dataset[: args.max_samples]

    responses = []
    for instance in dataset:
        messages = []
        for user_content in instance["turns"]:
            messages.append({"role": "user", "content": user_content})
            input_text = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
            input_ids = tokenizer.encode(input_text, add_special_tokens=False)

            response = {}
            response[1] = baseline_generate(
                target_model,
                input_ids,
                max_new_tokens=args.max_new_tokens,
                temperature=args.temperature,
                top_p=args.top_p,
                stop_token_id=tokenizer.eos_token_id,
            )

            response[args.block_size] = speculative_generate(
                target_model,
                draft_model,
                input_ids,
                max_new_tokens=args.max_new_tokens,
                block_size=args.block_size,
                temperature=args.temperature,
                top_p=args.top_p,
                stop_token_id=tokenizer.eos_token_id,
            )

            spec_response = response[args.block_size]
            generated_ids = spec_response.output_ids[spec_response.num_input_tokens :]
            output_text = tokenizer.decode(generated_ids, skip_special_tokens=True)
            messages.append({"role": "assistant", "content": output_text})
            responses.append(response)

    if not responses:
        print("[bench] no responses collected")
        return

    t1 = np.mean([r[1].time_per_output_token for r in responses])
    tb = np.mean([r[args.block_size].time_per_output_token for r in responses])
    if tb > 0:
        print(f"Decoding speedup: {t1 / tb:.2f}")
    else:
        print("Decoding speedup: n/a")

    acceptance_lengths = list(
        chain(*[r[args.block_size].acceptance_lengths for r in responses if r[args.block_size].acceptance_lengths])
    )
    if acceptance_lengths:
        tau = float(np.mean(acceptance_lengths))
        print(f"Average acceptance length: {tau:.2f}")
        histogram = [acceptance_lengths.count(b) / len(acceptance_lengths) for b in range(args.block_size + 1)]
        print(f"Acceptance length histogram: {[f'{x * 100:.1f}%' for x in histogram]}")
    else:
        print("Average acceptance length: n/a")


if __name__ == "__main__":
    main()
