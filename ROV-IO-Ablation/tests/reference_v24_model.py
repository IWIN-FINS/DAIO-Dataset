# Frozen from the existing ROV-IO-Training/model.py solely for full V24 forward regression checks.
"""V20-compatible four-branch CNN--Transformer velocity observation network."""
from __future__ import annotations

import torch
from torch import nn


class LocalCNN(nn.Module):
    """V11/V20 two-layer local temporal encoder: downsample 256 samples to 64 tokens."""
    def __init__(self, input_channels: int) -> None:
        super().__init__()
        self.layers = nn.Sequential(
            nn.Conv1d(input_channels, 32, kernel_size=7, stride=2, padding=3), nn.BatchNorm1d(32), nn.ReLU(),
            nn.Conv1d(32, 64, kernel_size=5, stride=2, padding=2), nn.BatchNorm1d(64), nn.ReLU(),
        )

    def forward(self, values: torch.Tensor) -> torch.Tensor:
        return self.layers(values).transpose(1, 2)


class VelocityHead(nn.Module):
    def __init__(self, width: int, dropout: float) -> None:
        super().__init__()
        self.shared = nn.Sequential(nn.Linear(width, 64), nn.ReLU(), nn.Dropout(dropout))
        self.mean = nn.Linear(64, 3)
        self.log_std = nn.Linear(64, 3)

    def forward(self, latent: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        value = self.shared(latent)
        # Bounds make exp(log_std) numerically safe while retaining a wide range.
        return self.mean(value), torch.clamp(self.log_std(value), min=-6.0, max=3.0)


class V20FourBranchTransformer(nn.Module):
    """Inputs are RPM, horizontal acceleration, yaw angular velocity, and physical mean depth.

    The fixed input order is ``rpm [B,8,256]``, ``accel_xy [B,2,256]``,
    ``gyro_z [B,1,256]``, ``depth [B,1,256]``, matching the data contract.
    Four CNNs extract local temporal features that are summed elementwise after projection. The final
    causal Transformer token predicts body-frame ``[vx, vy, r]`` and diagonal log-std at the window end.
    """
    input_keys = ("rpm", "accel_xy", "gyro_z", "depth")

    def __init__(self, d_model: int, layers: int, heads: int, ffn_dim: int, dropout: float, max_tokens: int) -> None:
        super().__init__()
        self.rpm = LocalCNN(8); self.accel_xy = LocalCNN(2); self.gyro_z = LocalCNN(1); self.depth = LocalCNN(1)
        self.rpm_project = nn.Linear(64, d_model); self.accel_project = nn.Linear(64, d_model)
        self.gyro_project = nn.Linear(64, d_model); self.depth_project = nn.Linear(64, d_model)
        self.position = nn.Parameter(torch.zeros(1, max_tokens, d_model))
        layer = nn.TransformerEncoderLayer(d_model=d_model, nhead=heads, dim_feedforward=ffn_dim, dropout=dropout,
                                           batch_first=True, norm_first=True)
        self.transformer = nn.TransformerEncoder(layer, num_layers=layers)
        self.head = VelocityHead(d_model, dropout)

    def forward(self, rpm: torch.Tensor, accel_xy: torch.Tensor, gyro_z: torch.Tensor,
                depth: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        # All four branches must retain the same 64-token temporal resolution for elementwise fusion.
        tokens = (self.rpm_project(self.rpm(rpm)) + self.accel_project(self.accel_xy(accel_xy)) +
                  self.gyro_project(self.gyro_z(gyro_z)) + self.depth_project(self.depth(depth)))
        count = tokens.shape[1]
        if count > self.position.shape[1]: raise ValueError(f"{count} tokens exceed configured max_tokens")
        tokens = tokens + self.position[:, :count]
        # The upper-triangular mask prevents token i from reading future tokens j>i, preserving deployment causality.
        mask = torch.triu(torch.ones(count, count, dtype=torch.bool, device=tokens.device), diagonal=1)
        latent = self.transformer(tokens, mask=mask)[:, -1]
        return self.head(latent)


def build_model(cfg: dict) -> V20FourBranchTransformer:
    model = cfg["model"]
    return V20FourBranchTransformer(d_model=int(model["d_model"]), layers=int(model["transformer_layers"]),
                                    heads=int(model["transformer_heads"]), ffn_dim=int(model["transformer_ffn_dim"]),
                                    dropout=float(model["dropout"]), max_tokens=int(model["max_tokens"]))


def parameter_count(model: nn.Module) -> int:
    return sum(parameter.numel() for parameter in model.parameters() if parameter.requires_grad)
