#!/usr/bin/env python3
import argparse
import json
import os
import random
import subprocess
import sys
import time
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import torch
from torch import nn
from torch.utils.data import DataLoader, Dataset
from transformers import AutoModelForCausalLM, AutoTokenizer

ROOT = Path(__file__).resolve().parents[3]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from model.model_minillm import MiniLLMConfig, MiniLLMForCausalLM
from speculator.infer.torch.common import (
    Eagle3Speculator,
    _count_params_torch,
    _extract_hidden_layers,
    _minillm_forward_hidden_states,
    _resolve_dtype,
    _resolve_feature_layers,
    _resolve_spec_config,
    sample_next_token,
)

os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")


def _count_jsonl_lines(path: Path, *, max_lines: Optional[int] = None) -> int:
    if not path.is_file():
        return 0
    n = 0
    with path.open("r", encoding="utf-8") as f:
        for _ in f:
            n += 1
            if max_lines is not None and n >= max_lines:
                break
    return n


def _apply_chat_template(
    tokenizer, messages: List[Dict[str, Any]], *, add_generation_prompt: bool
) -> str:
    return tokenizer.apply_chat_template(
        messages,
        tokenize=False,
        add_generation_prompt=add_generation_prompt,
        enable_thinking=False,
    )


def _split_prompt(
    messages: List[Dict[str, Any]],
) -> Tuple[List[Dict[str, Any]], List[Dict[str, Any]]]:
    last_assistant = None
    for i in range(len(messages) - 1, -1, -1):
        if messages[i].get("role") == "assistant":
            last_assistant = i
            break
    if last_assistant is None:
        return messages, []
    return messages[:last_assistant], messages[last_assistant:]


class SyntheticChatDataset(Dataset):
    def __init__(self, jsonl_path: Path, tokenizer, *, max_seq_len: int) -> None:
        self.records = []
        with jsonl_path.open("r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                self.records.append(json.loads(line))
        self.tokenizer = tokenizer
        self.max_seq_len = int(max_seq_len)
        self.pad_id = (
            tokenizer.pad_token_id if tokenizer.pad_token_id is not None else 0
        )

    def __len__(self) -> int:
        return len(self.records)

    def __getitem__(self, idx: int):
        record = self.records[idx]
        conversations = record.get("conversations") or record.get("messages")
        if not conversations:
            raise ValueError("Missing conversations/messages in JSONL record")

        prompt_msgs, _ = _split_prompt(conversations)
        full_text = _apply_chat_template(
            self.tokenizer, conversations, add_generation_prompt=False
        )
        prompt_text = _apply_chat_template(
            self.tokenizer, prompt_msgs, add_generation_prompt=True
        )

        full_ids = self.tokenizer(full_text, add_special_tokens=False).input_ids
        prompt_ids = self.tokenizer(prompt_text, add_special_tokens=False).input_ids

        if len(full_ids) > self.max_seq_len:
            full_ids = full_ids[: self.max_seq_len]

        attention_mask = [1] * len(full_ids)
        loss_mask = [0.0] * len(full_ids)
        start = min(len(prompt_ids), len(full_ids))
        for i in range(start, len(full_ids)):
            loss_mask[i] = 1.0

        pad_len = self.max_seq_len - len(full_ids)
        if pad_len > 0:
            full_ids += [self.pad_id] * pad_len
            attention_mask += [0] * pad_len
            loss_mask += [0.0] * pad_len

        return (
            torch.tensor(full_ids, dtype=torch.long),
            torch.tensor(attention_mask, dtype=torch.long),
            torch.tensor(loss_mask, dtype=torch.float32),
        )


def _select_positions(
    loss_mask: torch.Tensor,
    attention_mask: torch.Tensor,
    *,
    spec_len: int,
    rng: random.Random,
) -> List[int]:
    batch, seq_len = loss_mask.shape
    positions: List[int] = []
    limit = int(seq_len) - int(spec_len) - 1
    for b in range(int(batch)):
        valid: List[int] = []
        for i in range(int(max(limit, 0))):
            if float(loss_mask[b, i].item()) <= 0.0:
                continue
            ok = True
            for j in range(1, int(spec_len) + 1):
                if float(loss_mask[b, i + j].item()) <= 0.0:
                    ok = False
                    break
                if float(attention_mask[b, i + j].item()) <= 0.0:
                    ok = False
                    break
            if ok:
                valid.append(i)
        if valid:
            positions.append(valid[rng.randrange(len(valid))])
        else:
            positions.append(-1)
    return positions


def _self_feed_prob(
    *,
    step: int,
    max_steps: int,
    offset: float,
    start: float,
    end: float,
) -> float:
    if max_steps <= 1:
        base = float(end)
    else:
        progress = float(step) / float(max_steps)
        base = float(start) + (float(end) - float(start)) * progress
    return max(0.0, min(1.0, base * float(offset)))


def _embed_tokens(
    target: AutoModelForCausalLM, token_ids: List[int]
) -> Optional[torch.Tensor]:
    if not token_ids:
        return None
    device = next(target.parameters()).device
    token_tensor = torch.tensor([token_ids], device=device, dtype=torch.long)
    return target.get_input_embeddings()(token_tensor)


def _qwen3_forward_hidden_states(
    target: AutoModelForCausalLM,
    input_ids: torch.Tensor,
    *,
    attention_mask: Optional[torch.Tensor],
    layer_ids: List[int],
) -> Tuple[torch.Tensor, List[torch.Tensor]]:
    out = target(
        input_ids=input_ids,
        attention_mask=attention_mask,
        use_cache=False,
        output_hidden_states=True,
        return_dict=True,
    )
    return out.last_hidden_state, _extract_hidden_layers(out, layer_ids)


def _spec_loss_autoregressive(
    *,
    speculator: Eagle3Speculator,
    target: AutoModelForCausalLM,
    input_ids: torch.Tensor,
    attention_mask: torch.Tensor,
    loss_mask: torch.Tensor,
    spec_len: int,
    step: int,
    max_steps: int,
    self_feed_start: float,
    self_feed_end: float,
    self_feed_temperature: float,
    loss_decay: float,
    rng: random.Random,
    target_arch: str,
    dtype: torch.dtype,
) -> torch.Tensor:
    device = input_ids.device
    positions = _select_positions(
        loss_mask,
        attention_mask,
        spec_len=spec_len,
        rng=rng,
    )
    max_pos = max((pos for pos in positions if pos >= 0), default=-1)
    if max_pos < 0:
        return torch.tensor(0.0, device=device)
    trim_len = int(max_pos) + 1
    trim_ids = input_ids[:, :trim_len]
    trim_attn = attention_mask[:, :trim_len]
    with torch.no_grad():
        if target_arch == "minillm":
            _, layer_hiddens = _minillm_forward_hidden_states(
                target,
                trim_ids,
                attention_mask=trim_attn,
                layer_ids=speculator.feature_layers,
            )
        else:
            _, layer_hiddens = _qwen3_forward_hidden_states(
                target,
                trim_ids,
                attention_mask=trim_attn,
                layer_ids=speculator.feature_layers,
            )
    if layer_hiddens and layer_hiddens[0].dtype != dtype:
        layer_hiddens = [h.to(dtype) for h in layer_hiddens]

    total_loss = torch.tensor(0.0, device=device)
    total_weight = torch.tensor(0.0, device=device)
    for b, pos in enumerate(positions):
        if pos < 0:
            continue
        context_hiddens = [h[b : b + 1, pos : pos + 1, :] for h in layer_hiddens]
        fused_context = speculator.fuse(context_hiddens, attention_mask=None)
        feed_tokens: List[int] = []
        draft_tokens: List[int] = []
        spec_logits_steps: List[torch.Tensor] = []
        for j in range(int(spec_len)):
            token_embeds = _embed_tokens(target, feed_tokens)
            logits = speculator.decode(
                fused_context=fused_context, token_embeds=token_embeds
            )
            spec_logits_steps.append(logits)
            pred_token = sample_next_token(
                logits[0],
                temperature=self_feed_temperature,
                top_p=1.0,
            )
            draft_tokens.append(int(pred_token))
            prob = _self_feed_prob(
                step=step,
                max_steps=max_steps,
                offset=float(j + 1) / float(spec_len),
                start=self_feed_start,
                end=self_feed_end,
            )
            target_id = int(input_ids[b, pos + j + 1].item())
            if rng.random() < prob:
                feed_tokens.append(int(pred_token))
            else:
                feed_tokens.append(target_id)

        for j, spec_logits in enumerate(spec_logits_steps):
            spec_log_probs = torch.log_softmax(spec_logits.float(), dim=-1)
            target_id = int(input_ids[b, pos + j + 1].item())
            loss = -spec_log_probs[0, target_id]
            weight = float(loss_decay) ** int(j)
            total_loss = total_loss + (loss * weight)
            total_weight = total_weight + weight

    denom = torch.clamp(total_weight, min=1.0)
    return total_loss / denom


def _count_tokens(*, batch_size: int, spec_len: int) -> float:
    return float(int(batch_size) * int(spec_len))


def _save_checkpoint(
    *, step: int, speculator: nn.Module, optimizer: torch.optim.Optimizer, out_dir: Path
) -> None:
    ckpt_dir = out_dir / "checkpoints" / f"step_{step:08d}"
    ckpt_dir.mkdir(parents=True, exist_ok=True)
    model_path = ckpt_dir / "speculator.pt"
    tmp_model = Path(os.fspath(model_path) + ".tmp")
    torch.save(speculator.state_dict(), tmp_model)
    os.replace(tmp_model, model_path)
    opt_path = ckpt_dir / "optimizer.pt"
    tmp_opt = Path(os.fspath(opt_path) + ".tmp")
    torch.save(optimizer.state_dict(), tmp_opt)
    os.replace(tmp_opt, opt_path)
    (ckpt_dir / "train_state.json").write_text(
        json.dumps({"step": step, "timestamp": time.time()}, indent=2),
        encoding="utf-8",
    )
    print(f"[ckpt] saved {ckpt_dir}")


def _find_latest_checkpoint(out_dir: Path) -> Optional[Tuple[int, Path]]:
    ckpt_root = out_dir / "checkpoints"
    if not ckpt_root.is_dir():
        return None
    best_step = -1
    best_dir: Optional[Path] = None
    for child in ckpt_root.iterdir():
        if not child.is_dir() or not child.name.startswith("step_"):
            continue
        try:
            step = int(child.name.split("_", 1)[1])
        except ValueError:
            continue
        if not (child / "speculator.pt").is_file():
            continue
        if step > best_step:
            best_step = step
            best_dir = child
    if best_dir is None:
        return None
    return best_step, best_dir


def _load_train_state_step(ckpt_dir: Path, *, fallback: int) -> int:
    state_path = ckpt_dir / "train_state.json"
    if not state_path.is_file():
        return fallback
    state = json.loads(state_path.read_text(encoding="utf-8"))
    return int(state.get("step", fallback))


def _load_speculator_config(out_dir: Path) -> Optional[Dict[str, Any]]:
    cfg_path = out_dir / "speculator_config.json"
    if not cfg_path.is_file():
        return None
    return json.loads(cfg_path.read_text(encoding="utf-8"))


def _load_minillm_config(path: Optional[str]) -> MiniLLMConfig:
    if not path:
        return MiniLLMConfig()
    cfg_path = Path(path)
    with cfg_path.open("r", encoding="utf-8") as f:
        data = json.load(f)
    if not isinstance(data, dict):
        raise ValueError(f"MiniLLM config must be a JSON object: {cfg_path}")
    return MiniLLMConfig(**data)


def _ensure_synth_data(args) -> None:
    data_path = Path(args.data_path)
    data_path.parent.mkdir(parents=True, exist_ok=True)
    existing = _count_jsonl_lines(data_path, max_lines=args.gen_total_samples)
    if existing >= args.min_samples:
        print(f"[data] found {existing} samples at {data_path}, skip generation")
        return
    target_total = max(int(args.gen_total_samples), int(args.min_samples))
    cmd = [
        sys.executable,
        "-m",
        "mlx_train.distill_data_ollama",
        "--out_jsonl",
        str(data_path),
        "--ollama_url",
        args.ollama_url,
        "--ollama_model",
        args.ollama_model,
        "--target_total_samples",
        str(target_total),
        "--num_workers",
        str(args.gen_workers),
        "--max_new_tokens",
        str(args.gen_max_new_tokens),
        "--temperature",
        str(args.gen_temperature),
        "--top_p",
        str(args.gen_top_p),
    ]
    print(f"[gen] running: {' '.join(cmd)}")
    subprocess.run(cmd, check=True)


def _infinite_loader(loader: DataLoader):
    while True:
        for batch in loader:
            yield batch


def _default_head_rank(hidden_size: int) -> int:
    return max(32, min(256, int(hidden_size) // 8))


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Train an EAGLE-3 style speculator for Qwen3-0.6B or MiniLLM using pure synthetic data."
    )
    parser.add_argument(
        "--target_arch", type=str, choices=["qwen3", "minillm"], default="qwen3"
    )
    parser.add_argument("--target_model", type=str, default="Qwen/Qwen3-0.6B")
    parser.add_argument("--minillm_ckpt", type=str, default=None)
    parser.add_argument("--minillm_config", type=str, default=None)
    parser.add_argument("--minillm_tokenizer", type=str, default="./model")
    parser.add_argument(
        "--data_path", type=str, default="out/distill_ollama_qwen3_0.6b/synth.jsonl"
    )
    parser.add_argument(
        "--out_dir", type=str, default="out/eagle3_speculator/qwen3_0.6b"
    )
    parser.add_argument(
        "--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu"
    )
    parser.add_argument("--dtype", type=str, default="bfloat16")
    parser.add_argument("--max_seq_len", type=int, default=512)
    parser.add_argument("--batch_size", type=int, default=2)
    parser.add_argument("--accum_steps", type=int, default=8)
    parser.add_argument("--learning_rate", type=float, default=2e-4)
    parser.add_argument("--weight_decay", type=float, default=0.1)
    parser.add_argument("--grad_clip", type=float, default=1.0)
    parser.add_argument("--max_steps", type=int, default=5500)
    parser.add_argument("--log_interval", type=int, default=50)
    parser.add_argument("--save_interval", type=int, default=500)
    parser.add_argument(
        "--head_rank",
        type=int,
        default=None,
        help="Low-rank speculator head size (auto if unset; set 0 to disable).",
    )
    parser.add_argument(
        "--early_stop_loss",
        type=float,
        default=None,
        help="Stop early when loss <= this value (disabled if unset).",
    )
    parser.add_argument(
        "--early_stop_patience",
        type=int,
        default=0,
        help="Consecutive steps meeting early_stop_loss before stopping (0 = immediate).",
    )
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
    parser.add_argument("--spec_heads", type=int, default=0)
    parser.add_argument("--spec_dropout", type=float, default=0.0)
    parser.add_argument(
        "--feature_layers",
        type=str,
        default=None,
        help="Comma-separated target layer indices for fusion (auto if unset).",
    )
    parser.add_argument("--self_feed_start", type=float, default=0.1)
    parser.add_argument("--self_feed_end", type=float, default=0.9)
    parser.add_argument("--self_feed_temperature", type=float, default=0.8)
    parser.add_argument("--loss_decay", type=float, default=0.9)
    parser.add_argument("--seed", type=int, default=1337)
    parser.add_argument("--min_samples", type=int, default=20000)

    parser.add_argument("--ollama_url", type=str, default="http://127.0.0.1:11434")
    parser.add_argument("--ollama_model", type=str, default="qwen3:0.6b")
    parser.add_argument("--gen_total_samples", type=int, default=20000)
    parser.add_argument("--gen_workers", type=int, default=8)
    parser.add_argument("--gen_max_new_tokens", type=int, default=512)
    parser.add_argument("--gen_temperature", type=float, default=0.2)
    parser.add_argument("--gen_top_p", type=float, default=0.95)
    parser.add_argument("--no_auto_generate", action="store_true")
    parser.add_argument("--no_resume", action="store_true")
    args = parser.parse_args()

    torch.manual_seed(int(args.seed))
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(int(args.seed))

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    resume_dir: Optional[Path] = None
    resume_step = 0
    resume_cfg: Optional[Dict[str, Any]] = None
    if not args.no_resume:
        resume_info = _find_latest_checkpoint(out_dir)
        if resume_info is not None:
            resume_step, resume_dir = resume_info
            resume_step = _load_train_state_step(resume_dir, fallback=resume_step)
            resume_cfg = _load_speculator_config(out_dir)

    if resume_cfg is not None:
        if resume_cfg.get("spec_len") is not None:
            args.spec_len = int(resume_cfg["spec_len"])
        if resume_cfg.get("spec_layers") is not None:
            args.spec_layers = int(resume_cfg["spec_layers"])
        if resume_cfg.get("spec_heads") is not None:
            args.spec_heads = int(resume_cfg["spec_heads"])
        if "head_rank" in resume_cfg:
            args.head_rank = resume_cfg["head_rank"]
        if resume_cfg.get("feature_layers"):
            args.feature_layers = ",".join(str(i) for i in resume_cfg["feature_layers"])
        if "self_feed_start" in resume_cfg:
            args.self_feed_start = float(resume_cfg["self_feed_start"])
        if "self_feed_end" in resume_cfg:
            args.self_feed_end = float(resume_cfg["self_feed_end"])
        if "self_feed_temperature" in resume_cfg:
            args.self_feed_temperature = float(resume_cfg["self_feed_temperature"])
        if "loss_decay" in resume_cfg:
            args.loss_decay = float(resume_cfg["loss_decay"])

    if not args.no_auto_generate:
        _ensure_synth_data(args)

    dtype = _resolve_dtype(args.dtype)
    device = torch.device(args.device)

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
        tokenizer = AutoTokenizer.from_pretrained(
            args.target_model, trust_remote_code=True
        )
        if tokenizer.pad_token_id is None:
            tokenizer.pad_token_id = tokenizer.eos_token_id or 0
        tokenizer.padding_side = "right"
        target = AutoModelForCausalLM.from_pretrained(
            args.target_model, trust_remote_code=True, torch_dtype=dtype
        ).to(device)
    target.eval()
    for p in target.parameters():
        p.requires_grad = False

    param_count = _count_params_torch(target)
    auto_spec = (
        args.spec_len is None
        or args.spec_layers is None
        or int(args.spec_len) <= 0
        or int(args.spec_layers) <= 0
    )
    spec_len, spec_layers = _resolve_spec_config(
        args.spec_len, args.spec_layers, param_count=param_count
    )
    args.spec_len = spec_len
    args.spec_layers = spec_layers
    if auto_spec:
        if param_count:
            print(
                f"[speculator] auto spec_len={spec_len} spec_layers={spec_layers} "
                f"(target_params={param_count / 1e9:.2f}B)"
            )
        else:
            print(f"[speculator] auto spec_len={spec_len} spec_layers={spec_layers}")

    hidden_size = int(target.config.hidden_size)
    vocab_size = int(target.config.vocab_size)
    spec_heads = int(args.spec_heads)
    if spec_heads <= 0:
        cfg_heads = int(target.config.num_attention_heads)
        if cfg_heads > 0:
            spec_heads = cfg_heads
        else:
            spec_heads = max(1, hidden_size // 64)
        while spec_heads > 1 and hidden_size % spec_heads != 0:
            spec_heads -= 1

    init_weight = target.lm_head.weight.detach().clone()

    if args.head_rank is None:
        head_rank = _default_head_rank(hidden_size)
    elif int(args.head_rank) <= 0:
        head_rank = None
    else:
        head_rank = int(args.head_rank)

    parsed_layers: Optional[List[int]] = None
    if args.feature_layers:
        parsed_layers = [
            int(x) for x in str(args.feature_layers).split(",") if x.strip()
        ]
    feature_layers = _resolve_feature_layers(
        parsed_layers, num_layers=int(target.config.num_hidden_layers)
    )

    speculator = Eagle3Speculator(
        hidden_size=hidden_size,
        vocab_size=vocab_size,
        spec_layers=args.spec_layers,
        spec_heads=spec_heads,
        dropout=args.spec_dropout,
        feature_layers=feature_layers,
        init_weight=init_weight,
        head_rank=head_rank,
    ).to(device)

    if resume_dir is not None:
        speculator.load_state_dict(
            torch.load(resume_dir / "speculator.pt", map_location=device)
        )
        print(f"[resume] step={resume_step} from {resume_dir}")

    trainable = sum(p.numel() for p in speculator.parameters() if p.requires_grad)
    head_note = f" head_rank={head_rank}" if head_rank else ""
    layer_note = ",".join(str(i) for i in feature_layers)
    print(
        f"[speculator] params={trainable / 1e6:.2f}M spec_len={args.spec_len} "
        f"spec_layers={args.spec_layers} feature_layers=[{layer_note}]{head_note}"
    )

    dataset = SyntheticChatDataset(
        Path(args.data_path), tokenizer, max_seq_len=args.max_seq_len
    )
    loader = DataLoader(
        dataset, batch_size=args.batch_size, shuffle=True, drop_last=False
    )
    data_iter = _infinite_loader(loader)

    optimizer = torch.optim.AdamW(
        speculator.parameters(), lr=args.learning_rate, weight_decay=args.weight_decay
    )
    if resume_dir is not None:
        opt_path = resume_dir / "optimizer.pt"
        if opt_path.is_file():
            optimizer.load_state_dict(torch.load(opt_path, map_location=device))

    use_amp = device.type == "cuda" and dtype in (torch.float16, torch.bfloat16)
    amp_dtype = dtype if dtype in (torch.float16, torch.bfloat16) else torch.float32
    scaler = torch.cuda.amp.GradScaler(enabled=use_amp and amp_dtype == torch.float16)

    config = {
        "target_arch": args.target_arch,
        "target_model": args.target_model,
        "minillm_ckpt": args.minillm_ckpt,
        "minillm_config": args.minillm_config,
        "minillm_tokenizer": args.minillm_tokenizer,
        "data_path": args.data_path,
        "max_seq_len": args.max_seq_len,
        "spec_len": args.spec_len,
        "spec_layers": args.spec_layers,
        "spec_heads": spec_heads,
        "feature_layers": feature_layers,
        "head_rank": head_rank,
        "learning_rate": args.learning_rate,
        "weight_decay": args.weight_decay,
        "batch_size": args.batch_size,
        "accum_steps": args.accum_steps,
        "dtype": args.dtype,
        "early_stop_loss": args.early_stop_loss,
        "early_stop_patience": args.early_stop_patience,
        "self_feed_start": args.self_feed_start,
        "self_feed_end": args.self_feed_end,
        "self_feed_temperature": args.self_feed_temperature,
        "loss_decay": args.loss_decay,
    }
    (out_dir / "speculator_config.json").write_text(
        json.dumps(config, indent=2), encoding="utf-8"
    )

    start_time = time.time()
    accum_steps = int(args.accum_steps)
    if accum_steps <= 0:
        raise ValueError("--accum_steps must be >= 1")

    optimizer.zero_grad(set_to_none=True)
    early_stop_loss = args.early_stop_loss
    early_stop_patience = int(args.early_stop_patience or 0)
    if early_stop_loss is not None and early_stop_patience <= 0:
        early_stop_patience = 1
    early_stop_hits = 0

    last_step = resume_step
    if resume_step >= int(args.max_steps):
        print(
            f"[resume] step={resume_step} >= max_steps={int(args.max_steps)}; nothing to do."
        )
        return
    try:
        for step in range(resume_step + 1, int(args.max_steps) + 1):
            loss_sum = 0.0
            tokens_seen = 0.0
            for accum_idx in range(accum_steps):
                input_ids, attention_mask, loss_mask = next(data_iter)
                input_ids = input_ids.to(device)
                attention_mask = attention_mask.to(device)
                loss_mask = loss_mask.to(device)
                tokens_seen += _count_tokens(
                    batch_size=input_ids.shape[0], spec_len=args.spec_len
                )
                step_seed = (
                    int(args.seed) + (int(step) * int(accum_steps)) + int(accum_idx)
                )
                step_rng = random.Random(step_seed)

                with torch.autocast(
                    device_type=device.type, dtype=amp_dtype, enabled=use_amp
                ):
                    loss = _spec_loss_autoregressive(
                        speculator=speculator,
                        target=target,
                        input_ids=input_ids,
                        attention_mask=attention_mask,
                        loss_mask=loss_mask,
                        spec_len=args.spec_len,
                        step=step,
                        max_steps=int(args.max_steps),
                        self_feed_start=args.self_feed_start,
                        self_feed_end=args.self_feed_end,
                        self_feed_temperature=args.self_feed_temperature,
                        loss_decay=args.loss_decay,
                        rng=step_rng,
                        target_arch=args.target_arch,
                        dtype=dtype,
                    )
                    loss = loss / float(accum_steps)

                if scaler.is_enabled():
                    scaler.scale(loss).backward()
                else:
                    loss.backward()
                loss_sum += float(loss.item() * float(accum_steps))

            if scaler.is_enabled():
                scaler.unscale_(optimizer)
            if args.grad_clip > 0:
                torch.nn.utils.clip_grad_norm_(speculator.parameters(), args.grad_clip)
            if scaler.is_enabled():
                scaler.step(optimizer)
                scaler.update()
            else:
                optimizer.step()
            optimizer.zero_grad(set_to_none=True)

            loss_val = loss_sum / float(accum_steps)
            stop_training = False
            if early_stop_loss is not None:
                if loss_val <= float(early_stop_loss):
                    early_stop_hits += 1
                    if early_stop_hits >= early_stop_patience:
                        print(
                            f"[train] early stop at step={step} loss={loss_val:.4f} "
                            f"target={float(early_stop_loss):.4f}"
                        )
                        stop_training = True
                else:
                    early_stop_hits = 0

            if step % args.log_interval == 0:
                elapsed = time.time() - start_time
                tok_s = tokens_seen / max(elapsed, 1e-6)
                print(f"[train] step={step} loss={loss_val:.4f} tok/s={tok_s:.2f}")
                start_time = time.time()

            saved = False
            if step % args.save_interval == 0 or step == int(args.max_steps):
                _save_checkpoint(
                    step=step,
                    speculator=speculator,
                    optimizer=optimizer,
                    out_dir=out_dir,
                )
                saved = True

            last_step = step
            if stop_training:
                if not saved:
                    _save_checkpoint(
                        step=step,
                        speculator=speculator,
                        optimizer=optimizer,
                        out_dir=out_dir,
                    )
                break
    except KeyboardInterrupt:
        if last_step > 0:
            print(f"[train] interrupted at step={last_step}, saving checkpoint")
            _save_checkpoint(
                step=last_step,
                speculator=speculator,
                optimizer=optimizer,
                out_dir=out_dir,
            )
        else:
            print("[train] interrupted before first step; no checkpoint saved")


if __name__ == "__main__":
    main()
