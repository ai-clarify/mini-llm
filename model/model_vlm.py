"""MiniLLM vision-language model integration.

This module adapts the upstream vision-language implementation to fit the
MiniLLM codebase. The CLIP encoder is kept frozen by default to keep the
dependency surface and training costs low.
"""
from __future__ import annotations

import os
from types import SimpleNamespace
from typing import List, Optional, Tuple, Union

import numpy as np
import torch
from torch import nn
from transformers import CLIPModel, CLIPProcessor

from .model_minillm import MiniLLMConfig, MiniLLMForCausalLM, MOEFeedForward


class VLMConfig(MiniLLMConfig):
    """Configuration for the MiniLLM vision-language model."""

    model_type = "minillm-v"

    def __init__(
        self,
        image_special_token: str = "@" * 196,
        image_ids: Optional[List[int]] = None,
        **kwargs,
    ) -> None:
        self.image_special_token = image_special_token
        self.image_ids = image_ids or [34] * 196
        super().__init__(**kwargs)


class _FallbackProcessor:
    """Minimal image processor used when CLIP assets are unavailable."""

    def __init__(self, image_size: int = 224) -> None:
        self.image_size = image_size
        self.mean = torch.tensor([0.48145466, 0.4578275, 0.40821073], dtype=torch.float32).view(3, 1, 1)
        self.std = torch.tensor([0.26862954, 0.26130258, 0.27577711], dtype=torch.float32).view(3, 1, 1)

    def _prepare(self, image) -> torch.Tensor:
        if image.mode != "RGB":
            image = image.convert("RGB")
        image = image.resize((self.image_size, self.image_size))
        array = np.array(image, dtype=np.float32) / 255.0
        tensor = torch.from_numpy(array).permute(2, 0, 1)
        tensor = (tensor - self.mean) / self.std
        return tensor

    def __call__(self, images, return_tensors: str = "pt"):
        if not isinstance(images, (list, tuple)):
            images = [images]
        tensors = [self._prepare(image) for image in images]
        pixel_values = torch.stack(tensors, dim=0)
        if return_tensors == "pt":
            return {"pixel_values": pixel_values}
        raise ValueError("_FallbackProcessor only supports return_tensors='pt'")


class _FallbackVisionBackbone(nn.Module):
    """Very small vision backbone producing CLIP-like patch embeddings."""

    def __init__(self, embed_dim: int = 768) -> None:
        super().__init__()
        self.patch_embed = nn.Conv2d(3, embed_dim, kernel_size=16, stride=16, bias=False)
        self.norm = nn.LayerNorm(embed_dim)
        self.cls_token = nn.Parameter(torch.zeros(1, 1, embed_dim))

    def forward(self, pixel_values: torch.Tensor) -> SimpleNamespace:  # type: ignore[override]
        patches = self.patch_embed(pixel_values)
        patches = patches.flatten(2).transpose(1, 2)
        patches = self.norm(patches)
        cls_token = self.cls_token.expand(pixel_values.size(0), -1, -1)
        embeddings = torch.cat([cls_token, patches], dim=1)
        return SimpleNamespace(last_hidden_state=embeddings)


class _FallbackVisionEncoder(nn.Module):
    """Wrapper mimicking the CLIP vision encoder interface."""

    def __init__(self, embed_dim: int = 768) -> None:
        super().__init__()
        self.vision_model = _FallbackVisionBackbone(embed_dim)
        for param in self.parameters():
            param.requires_grad = False


class VisionProj(nn.Module):
    """Projects CLIP vision embeddings into the language model space."""

    def __init__(self, ve_hidden_size: int = 768, hidden_size: int = 512) -> None:
        super().__init__()
        self.vision_proj = nn.Linear(ve_hidden_size, hidden_size)

    def forward(self, image_encoders: torch.Tensor) -> torch.Tensor:  # type: ignore[override]
        return self.vision_proj(image_encoders)


class MiniLLMVLM(MiniLLMForCausalLM):
    """MiniLLM vision-language model with a frozen CLIP encoder."""

    config_class = VLMConfig

    def __init__(
        self,
        params: Optional[VLMConfig] = None,
        vision_model_path: str = "./model/vision_model/clip-vit-base-patch16",
    ) -> None:
        params = params or VLMConfig()
        super().__init__(params)
        self.params: VLMConfig = params
        self.vision_encoder, self.processor = self.get_vision_model(vision_model_path)
        self.vision_proj = VisionProj(hidden_size=params.hidden_size)

    # ------------------------------------------------------------------
    # Vision helpers
    # ------------------------------------------------------------------
    @staticmethod
    def get_vision_model(model_path: str) -> Tuple[Optional[CLIPModel], Optional[CLIPProcessor]]:
        """Load a CLIP model/processor if available on disk."""

        if not os.path.exists(model_path):
            return _FallbackVisionEncoder(), _FallbackProcessor()

        model = CLIPModel.from_pretrained(model_path)
        processor = CLIPProcessor.from_pretrained(model_path)
        for param in model.parameters():
            param.requires_grad = False
        return model.eval(), processor

    @staticmethod
    def image2tensor(image, processor: Optional[CLIPProcessor]) -> torch.Tensor:
        if processor is None:
            raise RuntimeError("CLIP processor has not been initialised. Did you download the vision encoder?")
        if image.mode in {"RGBA", "LA"}:
            image = image.convert("RGB")
        inputs = processor(images=image, return_tensors="pt")["pixel_values"]
        return inputs

    @staticmethod
    def get_image_embeddings(image_tensors: torch.Tensor, vision_model: Optional[CLIPModel]) -> torch.Tensor:
        if vision_model is None:
            raise RuntimeError("CLIP vision model has not been initialised. Did you download the vision encoder?")
        with torch.no_grad():
            outputs = vision_model.vision_model(pixel_values=image_tensors)
        img_embedding = outputs.last_hidden_state[:, 1:, :].squeeze()
        return img_embedding

    # ------------------------------------------------------------------
    # Language model integration
    # ------------------------------------------------------------------
    def _inject_vision_embeddings(
        self,
        tokens: torch.Tensor,
        hidden_states: torch.Tensor,
        vision_tensors: Optional[torch.Tensor] = None,
        seqlen: int = 512,
    ) -> torch.Tensor:
        def find_indices(tok: torch.Tensor, image_ids: List[int]):
            # Look for contiguous runs of placeholder image token ids.
            image_ids_tensor = torch.tensor(image_ids, device=tok.device)
            token_windows = tok.unfold(1, len(image_ids), 1)
            matches = (token_windows == image_ids_tensor).all(dim=2)
            results = {}
            for batch_idx in range(tok.size(0)):
                indices = matches[batch_idx].nonzero(as_tuple=True)[0]
                if indices.numel():
                    results[batch_idx] = [
                        (idx.item(), idx.item() + len(image_ids) - 1) for idx in indices
                    ]
            return results or None

        image_indices = find_indices(tokens, self.params.image_ids)
        if vision_tensors is not None and image_indices:
            projected = self.vision_proj(vision_tensors)
            if projected.dim() == 3:
                projected = projected.unsqueeze(0)
            updated_states = []
            for batch_idx in range(hidden_states.size(0)):
                if batch_idx not in image_indices:
                    updated_states.append(hidden_states[batch_idx])
                    continue
                current = hidden_states[batch_idx]
                img_idx = 0
                for start_idx, end_idx in image_indices[batch_idx]:
                    if img_idx < projected.size(1):
                        replacement = projected[batch_idx][img_idx]
                        current = torch.cat((current[:start_idx], replacement, current[end_idx + 1 :]), dim=0)[:seqlen]
                        img_idx += 1
                updated_states.append(current)
            hidden_states = torch.stack(updated_states, dim=0)
        return hidden_states

    def forward(
        self,
        input_ids: Optional[torch.Tensor] = None,
        attention_mask: Optional[torch.Tensor] = None,
        past_key_values: Optional[List[Tuple[torch.Tensor, torch.Tensor]]] = None,
        use_cache: bool = False,
        logits_to_keep: Union[int, torch.Tensor] = 0,
        pixel_values: Optional[torch.FloatTensor] = None,
        **kwargs,
    ) -> torch.Tensor:
        if input_ids is None:
            raise ValueError("input_ids must not be None")
        _, seq_length = input_ids.shape
        past_key_values_list: List[Optional[Tuple[torch.Tensor, torch.Tensor]]]
        if past_key_values is None:
            past_key_values_list = [None] * len(self.model.layers)
        else:
            past_key_values_list = list(past_key_values)

        start_pos = past_key_values_list[0][0].shape[1] if past_key_values_list[0] is not None else 0

        hidden_states = self.model.dropout(self.model.embed_tokens(input_ids))

        if pixel_values is not None and start_pos == 0:
            pixel_values_tensor: torch.Tensor = pixel_values
            if pixel_values_tensor.dim() == 6:
                pixel_values_tensor = pixel_values_tensor.squeeze(2)
            bs, num_images, _, _, _ = pixel_values_tensor.shape
            stack_dim = 1 if bs > 1 else 0
            vision_tensors = torch.stack(
                [
                    self.get_image_embeddings(pixel_values_tensor[:, idx, :, :, :], self.vision_encoder)
                    for idx in range(num_images)
                ],
                dim=stack_dim,
            )
            hidden_states = self._inject_vision_embeddings(
                tokens=input_ids, hidden_states=hidden_states, vision_tensors=vision_tensors, seqlen=seq_length
            )

        position_embeddings = (
            self.model.freqs_cos[start_pos : start_pos + seq_length],
            self.model.freqs_sin[start_pos : start_pos + seq_length],
        )

        presents: List[Tuple[torch.Tensor, torch.Tensor]] = []
        for layer, past_key_value in zip(self.model.layers, past_key_values_list):
            hidden_states, present = layer(
                hidden_states,
                position_embeddings,
                past_key_value=past_key_value,
                use_cache=use_cache,
                attention_mask=attention_mask,
            )
            presents.append(present)

        hidden_states = self.model.norm(hidden_states)
        aux_loss = sum(
            layer.mlp.aux_loss
            for layer in self.model.layers
            if isinstance(layer.mlp, MOEFeedForward)
        )

        slice_indices = slice(-logits_to_keep, None) if isinstance(logits_to_keep, int) else logits_to_keep
        logits = self.lm_head(hidden_states[:, slice_indices, :])
        self.OUT.__setitem__("last_hidden_state", hidden_states)
        self.OUT.__setitem__("logits", logits)
        self.OUT.__setitem__("aux_loss", aux_loss)
        self.OUT.__setitem__("past_key_values", presents)
        return self.OUT
