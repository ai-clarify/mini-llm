#!/usr/bin/env python3
import json
import random
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import torch
from transformers import AutoTokenizer

from speculator.infer.torch.common import SpecStats, sample_next_token
from speculator.infer.torch.eagle3.draft import Llama3Eagle3Drafter
from speculator.infer.torch.eagle3.target import Qwen3ForCausalLM
from speculator.infer.torch.eagle3.utils.kv_cache import initialize_past_key_values
from speculator.infer.torch.eagle3.utils.util import (
    evaluate_posterior,
    prepare_logits_processor,
    reset_tree_mode,
    tree_decoding,
    update_inference_inputs,
)


@dataclass
class Eagle3DraftConfig:
    vocab_size: int
    draft_vocab_size: int
    hidden_size: int
    intermediate_size: int
    num_attention_heads: int
    num_key_value_heads: int
    max_position_embeddings: int
    rms_norm_eps: float
    rope_theta: float
    rope_scaling: Optional[Dict[str, Any]]
    hidden_act: str
    head_dim: int
    pretraining_tp: int = 1
    pad_token_id: int = 0
    target_hidden_size: Optional[int] = None

    @classmethod
    def from_dict(cls, data: Dict[str, Any], *, pad_token_id: int) -> "Eagle3DraftConfig":
        return cls(
            vocab_size=int(data.get("vocab_size", 0)),
            draft_vocab_size=int(data.get("draft_vocab_size", data.get("vocab_size", 0))),
            hidden_size=int(data.get("hidden_size", 0)),
            intermediate_size=int(data.get("intermediate_size", 0)),
            num_attention_heads=int(data.get("num_attention_heads", 0)),
            num_key_value_heads=int(data.get("num_key_value_heads", data.get("num_attention_heads", 0))),
            max_position_embeddings=int(data.get("max_position_embeddings", 2048)),
            rms_norm_eps=float(data.get("rms_norm_eps", 1e-6)),
            rope_theta=float(data.get("rope_theta", 10000.0)),
            rope_scaling=data.get("rope_scaling"),
            hidden_act=str(data.get("hidden_act", "silu")),
            head_dim=int(data.get("head_dim", 0)),
            pretraining_tp=int(data.get("pretraining_tp", 1)),
            pad_token_id=int(pad_token_id),
            target_hidden_size=data.get("target_hidden_size"),
        )


class Eagle3Runner(torch.nn.Module):
    def __init__(self, base_model: Qwen3ForCausalLM, eagle_layer: Llama3Eagle3Drafter) -> None:
        super().__init__()
        self.base_model = base_model
        self.eagle_layer = eagle_layer

    def forward(
        self,
        input_ids: torch.Tensor,
        *,
        attention_mask: Optional[torch.Tensor] = None,
        past_key_values: Optional[Any] = None,
        output_orig: bool = False,
        position_ids: Optional[torch.Tensor] = None,
    ) -> Tuple[Any, torch.Tensor, torch.Tensor]:
        outputs = self.base_model.model(
            input_ids=input_ids,
            attention_mask=attention_mask,
            past_key_values=past_key_values,
            position_ids=position_ids,
            output_hidden_states=False,
            return_dict=True,
        )
        hidden_states = outputs[0]
        if output_orig:
            orig = self.base_model.lm_head(hidden_states)
            return outputs, orig, hidden_states
        return outputs, hidden_states


def _load_eagle3_config(path: Path, *, pad_token_id: int) -> Eagle3DraftConfig:
    data = json.loads(path.read_text(encoding="utf-8"))
    cfg = Eagle3DraftConfig.from_dict(data, pad_token_id=pad_token_id)
    if cfg.vocab_size <= 0 or cfg.draft_vocab_size <= 0:
        raise ValueError(f"Invalid eagle3 config vocab sizes: {path}")
    return cfg


def load_eagle3_drafter(
    *,
    eagle3_dir: Path,
    target: Qwen3ForCausalLM,
    total_tokens: int,
    depth: int,
    top_k: int,
    threshold: float,
) -> Llama3Eagle3Drafter:
    cfg_path = eagle3_dir / "config.json"
    if not cfg_path.is_file():
        raise FileNotFoundError(f"Missing eagle3 config.json: {cfg_path}")
    pad_token_id = getattr(target.config, "pad_token_id", None)
    if pad_token_id is None:
        pad_token_id = 0
    cfg = _load_eagle3_config(cfg_path, pad_token_id=int(pad_token_id))
    if cfg.target_hidden_size is None:
        cfg.target_hidden_size = int(getattr(target.config, "hidden_size", cfg.hidden_size))

    drafter = Llama3Eagle3Drafter(
        cfg,
        total_tokens=int(total_tokens),
        depth=int(depth),
        top_k=int(top_k),
        threshold=float(threshold),
    )
    drafter.embed_tokens.weight = target.get_input_embeddings().weight

    state_path = eagle3_dir / "pytorch_model.bin"
    if not state_path.is_file():
        raise FileNotFoundError(f"Missing eagle3 weights: {state_path}")
    state = torch.load(state_path, map_location="cpu")
    drafter.load_state_dict(state, strict=False)
    drafter = drafter.to(device=next(target.parameters()).device, dtype=next(target.parameters()).dtype)
    drafter.eval()
    drafter.init_tree()
    return drafter


def load_eagle3_target(
    *,
    target_model: str,
    device: torch.device,
    dtype: torch.dtype,
) -> Tuple[Qwen3ForCausalLM, Any]:
    target = Qwen3ForCausalLM.from_pretrained(target_model, torch_dtype=dtype)
    target.config.output_hidden_states = False
    target = target.to(device)
    target.eval()

    tokenizer = AutoTokenizer.from_pretrained(target_model, trust_remote_code=True)
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token_id = tokenizer.eos_token_id or 0
    tokenizer.padding_side = "right"
    return target, tokenizer


def baseline_decode_eagle3(
    *,
    target: Qwen3ForCausalLM,
    input_ids: torch.Tensor,
    max_new_tokens: int,
    temperature: float,
    top_p: float,
    eos_token_id: Optional[int],
) -> torch.Tensor:
    """Baseline decode for Eagle3 target. Time O(T) best/avg/worst; space O(T)."""
    device = input_ids.device
    output_ids = input_ids.clone()
    max_length = int(output_ids.shape[1]) + int(max_new_tokens) + 16
    past_key_values, _, _ = initialize_past_key_values(target, max_length=max_length)

    outputs = target.model(input_ids=output_ids, past_key_values=past_key_values)
    last_logits = target.lm_head(outputs[0])[:, -1, :]

    for _ in range(int(max_new_tokens)):
        token = sample_next_token(last_logits, temperature=temperature, top_p=top_p)
        output_ids = torch.cat(
            [output_ids, torch.tensor([[token]], device=device, dtype=torch.long)], dim=1
        )
        if eos_token_id is not None and int(token) == int(eos_token_id):
            break
        step_ids = torch.tensor([[token]], device=device, dtype=torch.long)
        outputs = target.model(input_ids=step_ids, past_key_values=past_key_values)
        last_logits = target.lm_head(outputs[0])[:, -1, :]
    return output_ids


def eagle3_decode_with_stats(
    *,
    target: Qwen3ForCausalLM,
    drafter: Llama3Eagle3Drafter,
    input_ids: torch.Tensor,
    max_new_tokens: int,
    temperature: float,
    top_p: float,
    top_k: int,
    eos_token_id: Optional[int],
    seed: int,
) -> Tuple[torch.Tensor, SpecStats]:
    """Tree-based Eagle3 decode. Time O(S * L * V) avg; space O(L * V)."""
    random.seed(int(seed))
    device = input_ids.device
    output_ids: List[int] = input_ids[0].tolist()
    prompt_len = int(input_ids.shape[1])

    produced = 0
    steps = 0
    total_accept = 0
    total_draft = 0
    accepted_output = 0
    zero_accept = 0
    target_generated = 0
    spec_time_s = 0.0
    target_time_s = 0.0
    target_prefill_time_s = 0.0
    target_verify_time_s = 0.0
    target_generate_time_s = 0.0
    target_prefill_calls = 0
    target_verify_calls = 0
    target_generate_calls = 0

    logits_processor = (
        prepare_logits_processor(
            temperature=temperature,
            top_p=top_p,
            top_k=int(top_k),
        )
        if float(temperature) > 1e-5
        else None
    )

    max_length = int(prompt_len) + int(max_new_tokens) + int(drafter.total_tokens) + 16
    past_key_values, past_key_values_data, current_length_data = initialize_past_key_values(
        target, max_length=max_length
    )

    runner = Eagle3Runner(target, drafter)
    reset_tree_mode(runner)

    t0 = time.perf_counter()
    outputs, orig, _ = runner(
        input_ids,
        past_key_values=past_key_values,
        output_orig=True,
    )
    prefill_s = time.perf_counter() - t0
    target_time_s += prefill_s
    target_prefill_time_s += prefill_s
    target_prefill_calls += 1

    if logits_processor is None:
        token = torch.argmax(orig[:, -1], dim=-1)
        token = token[:, None]
    else:
        logits = orig[:, -1]
        logits = logits_processor(None, logits)
        probabilities = torch.nn.functional.softmax(logits, dim=1)
        token = torch.multinomial(probabilities, 1)
    target_generated += 1

    hidden_states = outputs.hidden_states
    if not hidden_states:
        raise RuntimeError("Eagle3 target did not return hidden states.")
    if len(hidden_states) < 3:
        raise RuntimeError("Eagle3 target hidden states missing required feature layers.")
    if len(hidden_states) > 3:
        hidden_states = hidden_states[:3]
    drafter_device = next(drafter.parameters()).device
    if hidden_states[0].device != drafter_device:
        hidden_states = [h.to(drafter_device) for h in hidden_states]
    hidden_concat = torch.cat(hidden_states, dim=-1)

    draft_input = torch.cat((input_ids, token.to(device)), dim=1)
    t0 = time.perf_counter()
    draft_tokens, retrieve_indices, tree_mask, tree_position_ids, _ = drafter.topK_genrate(
        hidden_concat, draft_input, logits_processor
    )
    spec_time_s += time.perf_counter() - t0
    retrieve_indices = retrieve_indices.to(device)
    tree_mask = tree_mask.to(device)
    tree_position_ids = tree_position_ids.to(device)

    padding_token = (torch.zeros(1, 1, dtype=torch.long, device=device) - 1)

    while produced < int(max_new_tokens):
        tree_len = int(draft_tokens.shape[1])
        target.model.tree_mask = tree_mask

        t0 = time.perf_counter()
        logits, hidden_state_new, _ = tree_decoding(
            runner,
            draft_tokens,
            past_key_values,
            tree_position_ids,
            input_ids,
            retrieve_indices,
        )
        verify_s = time.perf_counter() - t0
        target_time_s += verify_s
        target_verify_time_s += verify_s
        target_verify_calls += 1

        padded = torch.cat((draft_tokens, padding_token), dim=1)
        candidates = padded[0, retrieve_indices]

        best_candidate, accept_length, sample_token = evaluate_posterior(
            logits,
            candidates,
            logits_processor,
        )
        accept_length = int(accept_length)
        best_candidate = int(best_candidate)

        steps += 1
        total_draft += max(tree_len - 1, 0)
        total_accept += accept_length
        if accept_length == 0:
            zero_accept += 1

        new_tokens = candidates[best_candidate, : accept_length + 1].tolist()
        if new_tokens:
            output_ids.extend(int(t) for t in new_tokens)
            produced += len(new_tokens)
            accepted_output += len(new_tokens)

        eos_hit = False
        if eos_token_id is not None and int(eos_token_id) in new_tokens:
            eos_idx = new_tokens.index(int(eos_token_id))
            output_ids = output_ids[: len(output_ids) - len(new_tokens) + eos_idx + 1]
            produced -= len(new_tokens) - (eos_idx + 1)
            accepted_output -= len(new_tokens) - (eos_idx + 1)
            eos_hit = True

        if eos_hit or produced >= int(max_new_tokens):
            break

        t0 = time.perf_counter()
        (
            input_ids,
            draft_tokens,
            retrieve_indices,
            tree_mask,
            tree_position_ids,
            _,
            _,
        ) = update_inference_inputs(
            input_ids=input_ids,
            candidates=candidates,
            best_candidate=torch.tensor(best_candidate, device=device),
            accept_length=accept_length,
            retrieve_indices=retrieve_indices,
            logits_processor=logits_processor,
            new_token=0,
            past_key_values_data_list=past_key_values_data,
            current_length_data=current_length_data,
            model=runner,
            hidden_state_new=hidden_state_new,
            sample_token=sample_token,
        )
        retrieve_indices = retrieve_indices.to(device)
        tree_mask = tree_mask.to(device)
        tree_position_ids = tree_position_ids.to(device)
        spec_time_s += time.perf_counter() - t0
        target_generated += 1
        target_generate_calls += 1

    stats = SpecStats(
        total_accept=int(total_accept),
        total_draft=int(total_draft),
        accepted_output=int(accepted_output),
        zero_accept=int(zero_accept),
        steps=int(steps),
        spec_time_s=float(spec_time_s),
        target_time_s=float(target_time_s),
        target_prefill_time_s=float(target_prefill_time_s),
        target_verify_time_s=float(target_verify_time_s),
        target_generate_time_s=float(target_generate_time_s),
        target_prefill_calls=int(target_prefill_calls),
        target_verify_calls=int(target_verify_calls),
        target_generate_calls=int(target_generate_calls),
        target_generated=int(target_generated),
    )
    return torch.tensor([output_ids], device=device, dtype=torch.long), stats
