#!/usr/bin/env python
"""
LoRA utilities for adapting Swin-UNETR attention blocks (Q/V projections only).
Default behavior matches the original model when rank == 0.
"""
from __future__ import annotations

import math
from pathlib import Path
from typing import Iterable, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F
from monai.networks.nets.swin_unetr import WindowAttention


class LoRALinear(nn.Module):
    """
    Low-Rank Adaptation (LoRA) wrapper for a linear projection.
    When r == 0, behaves exactly like the frozen base linear.
    """

    def __init__(
        self,
        in_features: int,
        out_features: int,
        r: int = 0,
        lora_alpha: float = 1.0,
        bias: bool = True,
        base_layer: nn.Linear | None = None,
    ) -> None:
        super().__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.r = int(r)
        self.lora_alpha = float(lora_alpha)
        self.scale = self.lora_alpha / self.r if self.r > 0 else 0.0

        freeze_base = self.r > 0
        if base_layer is None:
            base_linear = nn.Linear(in_features, out_features, bias=bias)
            weight = base_linear.weight.data.clone()
            bias_data = base_linear.bias.data.clone() if bias else None
        else:
            weight = base_layer.weight.data.clone()
            bias_data = base_layer.bias.data.clone() if base_layer.bias is not None else None

        self.weight = nn.Parameter(weight, requires_grad=not freeze_base)
        if bias_data is None:
            self.register_parameter("bias", None)
        else:
            self.bias = nn.Parameter(bias_data, requires_grad=not freeze_base)

        if self.r > 0:
            self.lora_A = nn.Parameter(torch.zeros((out_features, self.r)))
            self.lora_B = nn.Parameter(torch.zeros((self.r, in_features)))
            nn.init.kaiming_uniform_(self.lora_A, a=math.sqrt(5))
            nn.init.zeros_(self.lora_B)
            # Mark for easy filtering
            self.lora_A._is_lora_param = True  # type: ignore[attr-defined]
            self.lora_B._is_lora_param = True  # type: ignore[attr-defined]
        else:
            self.register_parameter("lora_A", None)
            self.register_parameter("lora_B", None)

    @classmethod
    def from_linear(cls, linear: nn.Linear, r: int, lora_alpha: float = 1.0) -> "LoRALinear":
        """Convenience constructor that wraps an existing Linear layer."""
        return cls(
            linear.in_features,
            linear.out_features,
            r=r,
            lora_alpha=lora_alpha,
            bias=linear.bias is not None,
            base_layer=linear,
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:  # type: ignore[override]
        base_out = F.linear(x, self.weight, self.bias)
        if self.r > 0 and self.lora_A is not None and self.lora_B is not None:
            lora_out = (x @ self.lora_B.t()) @ self.lora_A.t()
            lora_out = lora_out * self.scale
            if lora_out.dtype != base_out.dtype:
                lora_out = lora_out.to(dtype=base_out.dtype)
            return base_out + lora_out
        return base_out


def _wrap_window_attention_with_lora(attn: WindowAttention, lora_rank: int, lora_alpha: float) -> None:
    """
    Attach LoRA adapters to a WindowAttention module (Q and V projections only).
    Keeps behavior identical when lora_rank == 0.
    """
    if getattr(attn, "_lora_wrapped", False):
        return

    attn.lora_rank = int(lora_rank)
    attn.lora_alpha = float(lora_alpha)
    attn.use_lora = attn.lora_rank > 0
    attn.q_lora = (
        LoRALinear(attn.dim, attn.dim, r=attn.lora_rank, lora_alpha=attn.lora_alpha, bias=False)
        if attn.use_lora
        else None
    )
    attn.v_lora = (
        LoRALinear(attn.dim, attn.dim, r=attn.lora_rank, lora_alpha=attn.lora_alpha, bias=False)
        if attn.use_lora
        else None
    )
    if attn.use_lora:
        target_device = attn.qkv.weight.device
        target_dtype = attn.qkv.weight.dtype
        if attn.q_lora is not None:
            attn.q_lora.to(device=target_device, dtype=target_dtype)
        if attn.v_lora is not None:
            attn.v_lora.to(device=target_device, dtype=target_dtype)

    def forward_lora(x: torch.Tensor, mask: torch.Tensor | None) -> torch.Tensor:  # type: ignore[override]
        b, n, c = x.shape
        qkv = attn.qkv(x)
        if attn.use_lora and attn.q_lora is not None and attn.v_lora is not None:
            delta_q = attn.q_lora(x)
            delta_v = attn.v_lora(x)
            delta_q = delta_q.to(dtype=qkv.dtype, device=qkv.device)
            delta_v = delta_v.to(dtype=qkv.dtype, device=qkv.device)
            q_base = qkv[..., :c]
            k_base = qkv[..., c : 2 * c]
            v_base = qkv[..., 2 * c :]
            qkv = torch.cat((q_base + delta_q, k_base, v_base + delta_v), dim=2)

        qkv = qkv.reshape(b, n, 3, attn.num_heads, c // attn.num_heads).permute(2, 0, 3, 1, 4)
        q, k, v = qkv[0], qkv[1], qkv[2]
        q = q * attn.scale
        attn_scores = q @ k.transpose(-2, -1)
        relative_position_bias = attn.relative_position_bias_table[
            attn.relative_position_index.clone()[:n, :n].reshape(-1)  # type: ignore[index]
        ].reshape(n, n, -1)
        relative_position_bias = relative_position_bias.permute(2, 0, 1).contiguous()
        attn_scores = attn_scores + relative_position_bias.unsqueeze(0)
        if mask is not None:
            nw = mask.shape[0]
            attn_scores = attn_scores.view(b // nw, nw, attn.num_heads, n, n) + mask.unsqueeze(1).unsqueeze(0)
            attn_scores = attn_scores.view(-1, attn.num_heads, n, n)
        attn_scores = attn.softmax(attn_scores)
        attn_scores = attn.attn_drop(attn_scores).to(v.dtype)
        x_out = (attn_scores @ v).transpose(1, 2).reshape(b, n, c)
        x_out = attn.proj(x_out)
        x_out = attn.proj_drop(x_out)
        return x_out

    attn.forward = forward_lora.__get__(attn, attn.__class__)
    attn._lora_wrapped = True


def add_lora_to_swin_unetr(
    model: nn.Module,
    lora_rank: int = 4,
    lora_alpha: float = 1.0,
    lora_targets: Iterable[str] | None = ("WindowAttention",),
    freeze_backbone: bool = True,
) -> nn.Module:
    """
    Inject LoRA adapters into Swin-UNETR attention blocks (Q/V only).
    Optionally freezes the backbone so only LoRA (and any intentionally unfrozen heads) train.
    """
    if lora_rank <= 0:
        return model

    target_names: Tuple[str, ...] = tuple(lora_targets) if lora_targets is not None else tuple()
    for module in model.modules():
        name = module.__class__.__name__
        if isinstance(module, WindowAttention) or name in target_names:
            _wrap_window_attention_with_lora(module, lora_rank, lora_alpha)

    if freeze_backbone:
        for param in model.parameters():
            param.requires_grad = False
        for _, param in model.named_parameters():
            if getattr(param, "_is_lora_param", False):
                param.requires_grad = True

    return model


def get_lora_params(model: nn.Module) -> list[nn.Parameter]:
    """Return parameters marked as LoRA trainables."""
    return [p for _, p in model.named_parameters() if getattr(p, "_is_lora_param", False) and p.requires_grad]


def count_parameters(model: nn.Module) -> tuple[int, int]:
    """Return (total_params, trainable_params)."""
    total = sum(p.numel() for p in model.parameters())
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    return total, trainable


def log_model_params(model: nn.Module) -> None:
    """Print total, trainable, and LoRA parameter counts."""
    total, trainable = count_parameters(model)
    lora_params = get_lora_params(model)
    lora_count = sum(p.numel() for p in lora_params)
    print(
        f"Params - total: {total:,} | trainable: {trainable:,} | LoRA trainable: {lora_count:,}"
    )


def load_checkpoint_then_add_lora(
    model: nn.Module,
    checkpoint_path: str | Path,
    lora_rank: int,
    lora_alpha: float = 1.0,
    freeze_backbone: bool = True,
    map_location: str | torch.device = "cpu",
) -> nn.Module:
    """
    Convenience helper for inference: load a baseline checkpoint, then attach LoRA.
    """
    state = torch.load(checkpoint_path, map_location=map_location)
    state_dict = state["state_dict"] if isinstance(state, dict) and "state_dict" in state else state
    model.load_state_dict(state_dict, strict=False)
    return add_lora_to_swin_unetr(
        model,
        lora_rank=lora_rank,
        lora_alpha=lora_alpha,
        freeze_backbone=freeze_backbone,
    )
