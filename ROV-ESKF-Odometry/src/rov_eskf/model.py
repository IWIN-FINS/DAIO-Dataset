"""V20/V24 CNN-Transformer, including checkpoint-declared modality ablations.

模型输入契约（训练数据契约固定）：
- ``rpm [B,8,256]``：八路电机转速（/1000）
- ``accel_xy [B,2,256]``：机体系水平加速度（train z-score）
- ``gyro_z [B,1,256]``：偏航角速度（train z-score）
- ``depth [B,1,256]``：物理平均深度（train z-score）
V24 ablations select only config.model.modalities, in its declared order;
omitted branches are not constructed or passed zero-valued substitutes.

输出：窗口末端的机体系 ``[vx_b, vy_b, r]`` 的 ``(mean[B,3], log_std[B,3])``，
标准差为 ``exp(log_std)``。三个输出轴依次为 FLU 机体系纵向速度、横向速度、
偏航角速度。

本模块是网络定义与 checkpoint 加载的唯一入口；ESKF / replay 只调用
``windows.network_observations`` 得到推理结果，不直接触碰模型细节。
"""
from __future__ import annotations

from pathlib import Path
from typing import Any

import torch
from torch import nn


class LocalCNN(nn.Module):
    """V20 的两层局部时间编码器：256 个采样点下采样为 64 个 token。"""

    def __init__(self, input_channels: int) -> None:
        super().__init__()
        self.layers = nn.Sequential(
            nn.Conv1d(input_channels, 32, kernel_size=7, stride=2, padding=3),
            nn.BatchNorm1d(32), nn.ReLU(),
            nn.Conv1d(32, 64, kernel_size=5, stride=2, padding=2),
            nn.BatchNorm1d(64), nn.ReLU(),
        )

    def forward(self, values: torch.Tensor) -> torch.Tensor:
        # [B, C, 256] -> [B, 64, 64] -> 转置为 [B, 64 token, 64]
        return self.layers(values).transpose(1, 2)


class VelocityHead(nn.Module):
    """共享 MLP -> mean / log_std 双头；log_std 被裁剪以保持数值安全。"""

    def __init__(self, width: int, dropout: float) -> None:
        super().__init__()
        self.shared = nn.Sequential(nn.Linear(width, 64), nn.ReLU(), nn.Dropout(dropout))
        self.mean = nn.Linear(64, 3)
        self.log_std = nn.Linear(64, 3)

    def forward(self, latent: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        value = self.shared(latent)
        # 上下界让 exp(log_std) 数值安全且保留较宽范围。
        return self.mean(value), torch.clamp(self.log_std(value), min=-6.0, max=3.0)


class V20FourBranchTransformer(nn.Module):
    """四支 CNN-Transformer 网络（RPM / accel_xy / gyro_z / depth）。

    四支特征保持同一 64-token 时间分辨率，投影后逐元素相加；因果 Transformer
    的上三角 mask 禁止第 i 个 token 读取未来 j>i 的数据，最后一个 token 送进
    输出头。训练与部署使用同一实现，保证因果一致性。
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
            raise ValueError(f"{count} 个 token 超过配置的 max_tokens")
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
    """加载训练仓库的 best.pt 并返回 ``(eval 模式的模型, checkpoint dict)``。

    checkpoint dict 关键字段：``format_version``、``model``（state_dict）、
    ``normalizer``（{rpm_*, accel_xy_*, gyro_z_*, depth_*}）、``config``。
    """
    device = torch.device(device)
    payload = torch.load(Path(path), map_location=device, weights_only=False)
    if payload.get("format_version") != 1:
        raise ValueError("不支持的 checkpoint 格式（format_version != 1）")
    model = build_model(payload["config"]).to(device)
    model.load_state_dict(payload["model"], strict=True)
    model.eval()
    return model, payload


def parameter_count(model: nn.Module) -> int:
    """返回可训练参数量，便于打印模型规模。"""
    return sum(parameter.numel() for parameter in model.parameters() if parameter.requires_grad)
