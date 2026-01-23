#!/usr/bin/env python3
"""Upgrade MLX MiniLLM checkpoints in-place.

Features:
- Expand MTP predictor layers (copy from layer 0 for new layers).
- Add missing value-mix projection weights (zero-init) when enabled.
- Write new config keys with defaults or provided overrides.
"""
from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path
from typing import Any, Dict, List, Optional, Set


def _load_json(path: Path) -> Dict[str, Any]:
    with path.open("r", encoding="utf-8") as f:
        data = json.load(f)
    if not isinstance(data, dict):
        raise ValueError(f"Expected JSON object in {path}")
    return data


def _save_json(path: Path, data: Dict[str, Any]) -> None:
    tmp = path.with_suffix(path.suffix + ".tmp")
    with tmp.open("w", encoding="utf-8") as f:
        json.dump(data, f, ensure_ascii=False, indent=2)
    tmp.replace(path)


def _find_mtp_layers(weights: Dict[str, Any]) -> Set[int]:
    layers: Set[int] = set()
    for name in weights:
        if not name.startswith("mtp_layers."):
            continue
        parts = name.split(".")
        if len(parts) < 2:
            continue
        try:
            idx = int(parts[1])
        except ValueError:
            continue
        layers.add(idx)
    return layers


def main() -> None:
    parser = argparse.ArgumentParser(description="Upgrade MLX MiniLLM checkpoints")
    parser.add_argument("--ckpt_dir", type=str, required=True, help="Checkpoint dir with model.safetensors")
    parser.add_argument("--mtp_layers", type=int, default=None, help="Target MTP layer count (copy from layer 0)")
    parser.add_argument("--value_mix", type=float, default=None, help="Set value_mix in config and init weights")
    parser.add_argument("--qk_norm", action=argparse.BooleanOptionalAction, default=None)
    parser.add_argument("--hidden_act", type=str, default=None, choices=["silu", "relu2"])
    parser.add_argument("--logit_softcap", type=float, default=None)
    parser.add_argument("--residual_scale", type=float, default=None)
    parser.add_argument("--zero_init_residual", action=argparse.BooleanOptionalAction, default=None)
    args = parser.parse_args()

    ckpt_dir = Path(args.ckpt_dir)
    if not ckpt_dir.exists() or not ckpt_dir.is_dir():
        raise FileNotFoundError(ckpt_dir)

    sys.path.insert(0, str(ckpt_dir.parent.parent))
    try:
        import mlx.core as mx
        from mlx_train.config import MiniLLMConfig
        from mlx_train.models import MiniLLMForCausalLM
    except Exception as exc:  # pragma: no cover - runtime env dependent
        raise RuntimeError("MLX is required to run this script") from exc

    config_path = ckpt_dir / "config.json"
    model_path = ckpt_dir / "model.safetensors"
    if not config_path.exists():
        raise FileNotFoundError(config_path)
    if not model_path.exists():
        raise FileNotFoundError(model_path)

    cfg_data = _load_json(config_path)

    def set_if(name: str, value: Optional[Any], default: Optional[Any] = None) -> None:
        if value is not None:
            cfg_data[name] = value
        elif name not in cfg_data and default is not None:
            cfg_data[name] = default

    set_if("hidden_act", args.hidden_act, "silu")
    set_if("qk_norm", args.qk_norm, False)
    set_if("qk_norm_eps", None, 1e-6)
    set_if("logit_softcap", args.logit_softcap, 0.0)
    set_if("value_mix", args.value_mix, 0.0)
    set_if("residual_scale", args.residual_scale, 1.0)
    set_if("zero_init_residual", args.zero_init_residual, False)
    if args.mtp_layers is not None:
        cfg_data["num_nextn_predict_layers"] = int(args.mtp_layers)

    cfg = MiniLLMConfig.from_dict(cfg_data)
    model = MiniLLMForCausalLM(cfg)

    weights = dict(mx.load(str(model_path)))
    existing_mtp = _find_mtp_layers(weights)
    model.load_weights(list(weights.items()), strict=False)

    target_mtp = int(args.mtp_layers) if args.mtp_layers is not None else int(cfg.num_nextn_predict_layers)
    if target_mtp > 0 and model.mtp_layers:
        src_layer = model.mtp_layers[0]
        for idx in range(target_mtp):
            if idx in existing_mtp:
                continue
            if idx >= len(model.mtp_layers):
                break
            dst = model.mtp_layers[idx]
            dst.norm.weight = mx.array(src_layer.norm.weight)
            dst.gate_proj.weight = mx.array(src_layer.gate_proj.weight)
            dst.up_proj.weight = mx.array(src_layer.up_proj.weight)
            dst.down_proj.weight = mx.array(src_layer.down_proj.weight)

    value_mix = float(cfg.value_mix)
    if value_mix > 0.0:
        for layer in model.model.layers:
            attn = layer.self_attn
            if getattr(attn, "v_mix_proj", None) is None:
                continue
            key = f"model.layers.{layer.layer_id}.self_attn.v_mix_proj.weight"
            if key not in weights:
                attn.v_mix_proj.weight = mx.zeros_like(attn.v_mix_proj.weight)

    model.save_weights(str(model_path))
    _save_json(config_path, cfg.to_dict())

    state_path = ckpt_dir / "state.json"
    if state_path.exists():
        state = _load_json(state_path)
        args_state = state.get("args")
        if isinstance(args_state, dict):
            if args.mtp_layers is not None:
                args_state["mtp_layers"] = int(args.mtp_layers)
            if args.value_mix is not None:
                args_state["value_mix"] = float(args.value_mix)
            if args.hidden_act is not None:
                args_state["hidden_act"] = str(args.hidden_act)
            if args.qk_norm is not None:
                args_state["qk_norm"] = bool(args.qk_norm)
            if args.logit_softcap is not None:
                args_state["logit_softcap"] = float(args.logit_softcap)
            if args.residual_scale is not None:
                args_state["residual_scale"] = float(args.residual_scale)
            if args.zero_init_residual is not None:
                args_state["zero_init_residual"] = bool(args.zero_init_residual)
            state["args"] = args_state
            _save_json(state_path, state)

    print(f"[done] upgraded checkpoint: {ckpt_dir}")


if __name__ == "__main__":
    main()
