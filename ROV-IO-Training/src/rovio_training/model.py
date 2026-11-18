"""V20 兼容的四分支 CNN--Transformer 速度观测网络。"""
from __future__ import annotations

import torch
from torch import nn


class LocalCNN(nn.Module):
    """V11/V20 的两层局部时间编码器：256 个采样点下采样为 64 个 token。"""
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
    """输入为 RPM、水平加速度、偏航角速度和物理平均深度。

    输入顺序被固定为 ``rpm [B,8,256]``、``accel_xy [B,2,256]``、
    ``gyro_z [B,1,256]``、``depth [B,1,256]``，并与数据契约一一对应。
    四个 CNN 分别抽取局部时间特征，投影后逐元素相加；因果 Transformer 的最后
    一个 token 预测窗口末端的机体系 ``[vx, vy, r]`` 以及对角 log-std。
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
        # 四支特征必须保持同一 64-token 时间分辨率，才能逐元素相加融合。
        tokens = (self.rpm_project(self.rpm(rpm)) + self.accel_project(self.accel_xy(accel_xy)) +
                  self.gyro_project(self.gyro_z(gyro_z)) + self.depth_project(self.depth(depth)))
        count = tokens.shape[1]
        if count > self.position.shape[1]: raise ValueError(f"{count} tokens exceed configured max_tokens")
        tokens = tokens + self.position[:, :count]
        # 上三角 mask 禁止第 i 个 token 读取未来 j>i 的数据，保证部署时严格因果。
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
