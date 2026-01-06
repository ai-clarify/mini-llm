"""Smoke tests ensuring the MiniMind VLM training & inference loops run locally."""
from __future__ import annotations

import json
import sys
from pathlib import Path

import pytest
import torch
from PIL import Image
from torch.utils.data import DataLoader

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

pytest.importorskip(
    "mlx.core",
    reason="libmlx.so not available in the current environment; skipping VLM smoke tests.",
    exc_type=ImportError,
)

from dataset.vlm_dataset import VLMDataset
from model.model_vlm import VLMConfig
from trainer.trainer_utils import init_vlm_model


def _create_dummy_dataset(root: Path) -> tuple[Path, Path]:
    """Create a tiny JSONL dataset plus a single RGB image for smoke tests."""

    images_dir = root / "images"
    images_dir.mkdir(parents=True, exist_ok=True)
    image_path = images_dir / "dummy.png"
    Image.new("RGB", (224, 224), color=(255, 0, 0)).save(image_path)

    sample = {
        "image": image_path.name,
        "conversations": [
            {"content": "<image> 请描述图片里的颜色。"},
            {"content": "图片主要是红色。"},
        ],
    }

    data_path = root / "samples.jsonl"
    with data_path.open("w", encoding="utf-8") as f:
        f.write(json.dumps(sample, ensure_ascii=False) + "\n")

    return data_path, images_dir


def _init_tiny_model(tmp_path: Path):
    """Initialise a CPU-only MiniMindVLM with extremely small settings."""

    vlm_config = VLMConfig(
        hidden_size=64,
        num_attention_heads=4,
        num_key_value_heads=4,
        num_hidden_layers=1,
        intermediate_size=128,
        max_seq_len=128,
        image_ids=[34] * 16,
        image_special_token="@" * 16,
    )

    model, tokenizer, preprocess = init_vlm_model(
        vlm_config,
        from_weight="none",
        tokenizer_path=str(Path("model")),
        vision_model_path=str(tmp_path / "missing_vision"),
        save_dir=str(tmp_path / "out"),
        device="cpu",
    )
    return model, tokenizer, preprocess


def test_vlm_training_smoke(tmp_path):
    """Run a single optimisation step to validate the training loop wiring."""

    data_path, images_dir = _create_dummy_dataset(tmp_path)
    model, tokenizer, preprocess = _init_tiny_model(tmp_path)

    dataset = VLMDataset(
        str(data_path),
        str(images_dir),
        tokenizer,
        preprocess,
        max_length=96,
        image_special_token=model.params.image_special_token,
    )
    loader = DataLoader(dataset, batch_size=1, shuffle=False)
    input_ids, labels, loss_mask, pixel_values = next(iter(loader))

    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3)
    criterion = torch.nn.CrossEntropyLoss(reduction="none")

    model.train()
    outputs = model(input_ids, pixel_values=pixel_values)
    logits = outputs.logits

    # Align logits and labels for next-token prediction.
    shift_logits = logits[:, :-1, :]
    shift_labels = labels[:, : shift_logits.size(1)]
    shift_mask = loss_mask[:, : shift_logits.size(1)]

    loss = criterion(shift_logits.reshape(-1, shift_logits.size(-1)), shift_labels.reshape(-1))
    loss = loss.view_as(shift_labels)
    loss = (loss * shift_mask).sum() / torch.clamp(shift_mask.sum(), min=1.0)

    loss.backward()
    optimizer.step()

    assert torch.isfinite(loss), "Loss should be a finite value after one optimisation step."


def test_vlm_inference_smoke(tmp_path):
    """Ensure greedy decoding works and returns tokens that include the prompt."""

    data_path, images_dir = _create_dummy_dataset(tmp_path)
    model, tokenizer, preprocess = _init_tiny_model(tmp_path)

    dataset = VLMDataset(
        str(data_path),
        str(images_dir),
        tokenizer,
        preprocess,
        max_length=96,
        image_special_token=model.params.image_special_token,
    )
    pixel_values = dataset[0][3].unsqueeze(0)

    messages = [{"role": "user", "content": "<image> 请描述图片里的颜色。"}]
    prompt = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    encoded = tokenizer(prompt, return_tensors="pt")

    model.eval()
    with torch.no_grad():
        generated = model.generate(
            encoded["input_ids"],
            attention_mask=encoded["attention_mask"],
            max_new_tokens=4,
            do_sample=False,
            pixel_values=pixel_values,
            pad_token_id=tokenizer.pad_token_id,
            eos_token_id=tokenizer.eos_token_id,
        )

    assert generated.shape[1] >= encoded["input_ids"].shape[1]
