"""Utility helpers shared by the MiniLLM VLM training scripts."""
from __future__ import annotations

import math
import os
import random
from typing import Dict, Iterable, List, Optional

import numpy as np
import torch
import torch.distributed as dist
from torch.utils.data import Sampler
from transformers import AutoTokenizer

from model.model_vlm import MiniLLMVLM, VLMConfig


def is_main_process() -> bool:
    return not dist.is_initialized() or dist.get_rank() == 0


def Logger(content: str) -> None:
    if is_main_process():
        print(content)


def get_lr(current_step: int, total_steps: int, lr: float) -> float:
    return lr / 10 + 0.5 * lr * (1 + math.cos(math.pi * current_step / total_steps))


def init_distributed_mode() -> int:
    if int(os.environ.get("RANK", -1)) == -1:
        return 0
    dist.init_process_group(backend="nccl")
    local_rank = int(os.environ["LOCAL_RANK"])
    torch.cuda.set_device(local_rank)
    return local_rank


def setup_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def init_vlm_model(
    vlm_config: VLMConfig,
    from_weight: str = "pretrain_vlm",
    tokenizer_path: str = "../model",
    vision_model_path: str = "../model/vision_model/clip-vit-base-patch16",
    save_dir: str = "../out",
    device: str = "cuda",
    freeze_llm: bool = False,
):
    tokenizer = AutoTokenizer.from_pretrained(tokenizer_path)
    model = MiniLLMVLM(vlm_config, vision_model_path=vision_model_path)

    if from_weight != "none":
        moe_suffix = "_moe" if vlm_config.use_moe else ""
        weight_path = f"{save_dir}/{from_weight}_{vlm_config.hidden_size}{moe_suffix}.pth"
        if os.path.exists(weight_path):
            weights = torch.load(weight_path, map_location=device)
            model.load_state_dict(weights, strict=False)
        else:
            Logger(f"⚠️ 未找到预训练权重 {weight_path}，将从头训练。")

    if freeze_llm:
        for name, param in model.named_parameters():
            if "vision_proj" not in name:
                param.requires_grad = False

    Logger(
        f"所加载VLM Model可训练参数：{sum(p.numel() for p in model.parameters() if p.requires_grad) / 1e6:.3f} 百万"
    )
    preprocess = model.processor
    return model.to(device), tokenizer, preprocess


def vlm_checkpoint(
    vlm_config: VLMConfig,
    weight: str = "pretrain_vlm",
    model: Optional[torch.nn.Module] = None,
    optimizer: Optional[torch.optim.Optimizer] = None,
    epoch: int = 0,
    step: int = 0,
    wandb=None,
    save_dir: str = "../checkpoints",
    **kwargs,
) -> Optional[Dict[str, torch.Tensor]]:
    os.makedirs(save_dir, exist_ok=True)
    moe_path = "_moe" if vlm_config.use_moe else ""
    ckp_path = f"{save_dir}/{weight}_{vlm_config.hidden_size}{moe_path}.pth"
    resume_path = f"{save_dir}/{weight}_{vlm_config.hidden_size}{moe_path}_resume.pth"

    if model is not None:
        from torch.nn.parallel import DistributedDataParallel

        state_dict = model.module.state_dict() if isinstance(model, DistributedDataParallel) else model.state_dict()
        clean_state_dict = {k: v for k, v in state_dict.items() if not k.startswith("vision_encoder.")}
        ckp_tmp = ckp_path + ".tmp"
        torch.save({k: v.half() for k, v in clean_state_dict.items()}, ckp_tmp)
        os.replace(ckp_tmp, ckp_path)

        wandb_id = None
        if wandb:
            wandb_id = wandb.run.id

        resume_data = {
            "model": state_dict,
            "optimizer": optimizer.state_dict() if optimizer else None,
            "epoch": epoch,
            "step": step,
            "world_size": dist.get_world_size() if dist.is_initialized() else 1,
            "wandb_id": wandb_id,
        }
        for key, value in kwargs.items():
            if value is None:
                continue
            resume_data[key] = value.state_dict()

        resume_tmp = resume_path + ".tmp"
        torch.save(resume_data, resume_tmp)
        os.replace(resume_tmp, resume_path)
        return None

    if os.path.exists(resume_path):
        ckp_data = torch.load(resume_path, map_location="cpu")
        saved_ws = ckp_data.get("world_size", 1)
        current_ws = dist.get_world_size() if dist.is_initialized() else 1
        if saved_ws != current_ws and ckp_data.get("step") is not None:
            ckp_data["step"] = ckp_data["step"] * saved_ws // max(current_ws, 1)
            Logger(f"GPU数量变化({saved_ws}→{current_ws})，step已自动转换为{ckp_data['step']}")
        return ckp_data
    return None


class SkipBatchSampler(Sampler[List[int]]):
    """Sampler that skips the first N batches when resuming training."""

    def __init__(self, sampler: Iterable[int], batch_size: int, skip_batches: int = 0) -> None:
        self.sampler = list(sampler)
        self.batch_size = batch_size
        self.skip_batches = skip_batches

    def __iter__(self):
        batch: List[int] = []
        skipped = 0
        for idx in self.sampler:
            batch.append(idx)
            if len(batch) == self.batch_size:
                if skipped < self.skip_batches:
                    skipped += 1
                    batch = []
                    continue
                yield batch
                batch = []
        if batch and skipped >= self.skip_batches:
            yield batch

    def __len__(self) -> int:  # pragma: no cover - deterministic calculation
        total_batches = (len(self.sampler) + self.batch_size - 1) // self.batch_size
        return max(0, total_batches - self.skip_batches)
