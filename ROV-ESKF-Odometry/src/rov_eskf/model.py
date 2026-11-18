"""V20/V24 CNN-Transformer, including checkpoint-declared modality ablations.

Model input contract (fixed by the training data contract):
- ``rpm [B,8,256]``: eight motor speeds (/1000)
- ``accel_xy [B,2,256]``: horizontal body-frame acceleration (training z-score)
- ``gyro_z [B,1,256]``: yaw angular velocity (training z-score)
- ``depth [B,1,256]``: physical mean depth (training z-score)
V24 ablations select only config.model.modalities, in its declared order;
omitted branches are not constructed or passed zero-valued substitutes.

Output: ``(mean[B,3], log_std[B,3])`` for body-frame ``[vx_b, vy_b, r]`` at the window end.
Standard deviation is ``exp(log_std)``. The three output axes are FLU body-frame longitudinal velocity,
lateral velocity, and yaw-rate, respectively.

This module is the sole entry point for model definition and checkpoint loading. ESKF/replay use
``windows.network_observations`` for inference without accessing model internals.
"""
from __future__ import annotations

from pathlib import Path
from typing import Any

import torch
from torch import nn


class LocalCNN(nn.Module):
    """V20 two-layer local temporal encoder: downsample 256 samples to 64 tokens."""

    def __init__(self, input_channels: int) -> None:
        super().__init__()
        self.layers = nn.Sequential(
            nn.Conv1d(input_channels, 32, kernel_size=7, stride=2, padding=3),
            nn.BatchNorm1d(32), nn.ReLU(),
            nn.Conv1d(32, 64, kernel_size=5, stride=2, padding=2),
            nn.BatchNorm1d(64), nn.ReLU(),
        )

    def forward(self, values: torch.Tensor) -> torch.Tensor:
        # [B, C, 256] -> [B, 64, 64] -> transpose to [B, 64 tokens, 64]
        return self.layers(values).transpose(1, 2)


class VelocityHead(nn.Module):
    """Shared MLP with mean/log_std heads; clip log_std for numerical stability."""

    def __init__(self, width: int, dropout: float) -> None:
        super().__init__()
        self.shared = nn.Sequential(nn.Linear(width, 64), nn.ReLU(), nn.Dropout(dropout))
        self.mean = nn.Linear(64, 3)
        self.log_std = nn.Linear(64, 3)

    def forward(self, latent: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        value = self.shared(latent)
        # Bounds keep exp(log_std) numerically safe while retaining a wide range.
        return self.mean(value), torch.clamp(self.log_std(value), min=-6.0, max=3.0)


class V20FourBranchTransformer(nn.Module):
    """Four-branch CNN-Transformer (RPM / accel_xy / gyro_z / depth).

    All four branches retain the same 64-token temporal resolution and are summed elementwise after projection.
    The causal Transformer uses an upper-triangular mask to prevent token i from reading future tokens j>i.
    The last token feeds the output head. Training and deployment share this implementation for consistent causality.
    """
    input_keys = ("rpm", "accel_xy", "gyro_z", "depth")

    def __init__(self, d_model: int, layers: int, heads: int, ffn_dim: int,
                 dropout: float, max_tokens: int) -> None:
        super().__init__()
        self.rpm = LocalCNN(8)
        self.accel_xy = LocalCNN(2)
        self.gyro_z = LocalCNN(1)
        self.depth = LocalCNN(1)
        self.rpm_project = nn.Linear(64, d_model)
        self.accel_project = nn.Linear(64, d_model)
        self.gyro_project = nn.Linear(64, d_model)
        self.depth_project = nn.Linear(64, d_model)
        self.position = nn.Parameter(torch.zeros(1, max_tokens, d_model))
        layer = nn.TransformerEncoderLayer(d_model=d_model, nhead=heads, dim_feedforward=ffn_dim,
                                           dropout=dropout, batch_first=True, norm_first=True)
        self.transformer = nn.TransformerEncoder(layer, num_layers=layers)
        self.head = VelocityHead(d_model, dropout)

    def forward(self, rpm: torch.Tensor, accel_xy: torch.Tensor, gyro_z: torch.Tensor,
                depth: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        tokens = (self.rpm_project(self.rpm(rpm)) + self.accel_project(self.accel_xy(accel_xy))
                  + self.gyro_project(self.gyro_z(gyro_z)) + self.depth_project(self.depth(depth)))
        count = tokens.shape[1]
        if count > self.position.shape[1]:
            raise ValueError(f"{count} tokens exceed the configured max_tokens")
        tokens = tokens + self.position[:, :count]
        mask = torch.triu(torch.ones(count, count, dtype=torch.bool, device=tokens.device), diagonal=1)
        latent = self.transformer(tokens, mask=mask)[:, -1]
        return self.head(latent)


class ModalCNNTransformer(nn.Module):
    """V24 modality ablation, with training-identical module names and sums."""

    def __init__(self, cfg: dict[str, Any]) -> None:
        super().__init__()
        channels = {"rpm": 8, "throttle": 8, "accel_xy": 2, "accel_xyz": 3, "gyro_z": 1, "gyro_xyz": 3, "depth": 1}
        self.project_names = {"rpm": "rpm_project", "accel_xy": "accel_project",
                              "gyro_z": "gyro_project", "depth": "depth_project"}
        self.project_names.update(throttle='throttle_project',accel_xyz='accel_project',gyro_xyz='gyro_project')
        self.input_keys = tuple(cfg["modalities"])
        if (not self.input_keys or len(set(self.input_keys)) != len(self.input_keys)
                or any(key not in channels for key in self.input_keys)):
            raise ValueError("checkpoint modalities must be unique supported sensor names")
        width = int(cfg["d_model"])
        for key in self.input_keys:
            setattr(self, key, LocalCNN(channels[key]))
        for key in self.input_keys:
            setattr(self, self.project_names[key], nn.Linear(64, width))
        self.position = nn.Parameter(torch.zeros(1, int(cfg["max_tokens"]), width))
        layer = nn.TransformerEncoderLayer(
            width, int(cfg["transformer_heads"]), int(cfg["transformer_ffn_dim"]),
            float(cfg["dropout"]), batch_first=True, norm_first=True)
        self.transformer = nn.TransformerEncoder(layer, int(cfg["transformer_layers"]))
        self.head = VelocityHead(width, float(cfg["dropout"]))

    def forward(self, *inputs: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        if len(inputs) != len(self.input_keys):
            raise ValueError("input modality count differs from checkpoint")
        encoded = [getattr(self, self.project_names[key])(getattr(self, key)(value))
                   for key, value in zip(self.input_keys, inputs)]
        tokens = encoded[0]
        for value in encoded[1:]:
            tokens = tokens + value
        count = tokens.shape[1]
        if count > self.position.shape[1]:
            raise ValueError("token count exceeds positional encoding length")
        tokens = tokens + self.position[:, :count]
        mask = torch.triu(torch.ones(count, count, dtype=torch.bool,
                                    device=tokens.device), diagonal=1)
        return self.head(self.transformer(tokens, mask=mask)[:, -1])


def build_model(model_cfg: dict[str, Any]) -> nn.Module:
    """Rebuild the exact legacy or modality-specific CNN--Transformer checkpoint."""
    model = model_cfg["model"]
    if "modalities" in model:
        if model.get("architecture") != "cnn_transformer":
            raise ValueError("ESKF currently supports only cnn_transformer checkpoints")
        return ModalCNNTransformer(model)
    return V20FourBranchTransformer(d_model=int(model["d_model"]),
                                    layers=int(model["transformer_layers"]),
                                    heads=int(model["transformer_heads"]),
                                    ffn_dim=int(model["transformer_ffn_dim"]),
                                    dropout=float(model["dropout"]),
                                    max_tokens=int(model["max_tokens"]))


def load_checkpoint(path: str | Path, device: str | torch.device = "cpu") -> tuple[nn.Module, dict[str, Any]]:
    """Load best.pt from the training repository and return ``(model in eval mode, checkpoint dict)``.

    Key checkpoint fields: ``format_version``, ``model`` (state_dict),
    ``normalizer``({rpm_*, accel_xy_*, gyro_z_*, depth_*}), ``config``.
    """
    device = torch.device(device)
    payload = torch.load(Path(path), map_location=device, weights_only=False)
    if payload.get("format_version") != 1:
        raise ValueError("Unsupported checkpoint format (format_version != 1)")
    model = build_model(payload["config"]).to(device)
    model.load_state_dict(payload["model"], strict=True)
    model.eval()
    return model, payload


def parameter_count(model: nn.Module) -> int:
    """Return the number of trainable parameters for model-size reporting."""
    return sum(parameter.numel() for parameter in model.parameters() if parameter.requires_grad)
