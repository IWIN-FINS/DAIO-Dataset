"""五组实验共享模型组件；删模态实际删除分支，不以零输入冒充消融。"""
from __future__ import annotations
import torch
from torch import nn

CHANNELS = {"rpm": 8, "throttle": 8, "accel_xy": 2, "accel_xyz": 3, "gyro_z": 1, "gyro_xyz": 3, "depth": 1}
PROJECT_NAMES = {"rpm": "rpm_project", "throttle": "throttle_project", "accel_xy": "accel_project", "accel_xyz": "accel_project", "gyro_z": "gyro_project", "gyro_xyz": "gyro_project", "depth": "depth_project"}

class LocalCNN(nn.Module):
    """保持 V24 卷积定义，256 点得到 64 token。整体窗口截止于预测时刻。"""
    def __init__(self, input_channels):
        super().__init__()
        self.layers = nn.Sequential(
            nn.Conv1d(input_channels, 32, 7, stride=2, padding=3), nn.BatchNorm1d(32), nn.ReLU(),
            nn.Conv1d(32, 64, 5, stride=2, padding=2), nn.BatchNorm1d(64), nn.ReLU())
    def forward(self, values):
        return self.layers(values).transpose(1, 2)

class VelocityHead(nn.Module):
    """各模型共用相同的速度均值和对角 log-std 输出头。"""
    def __init__(self, width, dropout):
        super().__init__()
        self.shared = nn.Sequential(nn.Linear(width, 64), nn.ReLU(), nn.Dropout(dropout))
        self.mean = nn.Linear(64, 3)
        self.log_std = nn.Linear(64, 3)
    def forward(self, latent):
        value = self.shared(latent)
        return self.mean(value), self.log_std(value).clamp(-6., 3.)

class CNNEncoder(nn.Module):
    def __init__(self, cfg):
        super().__init__()
        self.input_keys = tuple(cfg["modalities"])
        # 保持完整 V24 的模块命名和初始化顺序，便于严格权重/前向回归。
        for key in self.input_keys:
            setattr(self, key, LocalCNN(CHANNELS[key]))
        for key in self.input_keys:
            setattr(self, PROJECT_NAMES[key], nn.Linear(64, int(cfg["d_model"])))
    def encode(self, inputs):
        if len(inputs) != len(self.input_keys):
            raise ValueError("输入模态数量与 checkpoint 契约不符")
        tokens = [getattr(self, PROJECT_NAMES[k])(getattr(self, k)(x)) for k, x in zip(self.input_keys, inputs)]
        value = tokens[0]
        for token in tokens[1:]:
            value = value + token
        return value

class CNNTransformer(CNNEncoder):
    def __init__(self, cfg):
        super().__init__(cfg)
        width = int(cfg["d_model"])
        self.position = nn.Parameter(torch.zeros(1, int(cfg["max_tokens"]), width))
        layer = nn.TransformerEncoderLayer(width, int(cfg["transformer_heads"]),
            int(cfg["transformer_ffn_dim"]), float(cfg["dropout"]), batch_first=True, norm_first=True)
        self.transformer = nn.TransformerEncoder(layer, int(cfg["transformer_layers"]))
        self.head = VelocityHead(width, float(cfg["dropout"]))
    def forward(self, *inputs):
        tokens = self.encode(inputs)
        count = tokens.shape[1]
        if count > self.position.shape[1]:
            raise ValueError("token 数超过位置编码长度")
        tokens = tokens + self.position[:, :count]
        mask = torch.triu(torch.ones(count, count, dtype=torch.bool, device=tokens.device), 1)
        return self.head(self.transformer(tokens, mask=mask)[:, -1])

class FlatMLP(nn.Module):
    def __init__(self, cfg):
        super().__init__()
        self.input_keys = tuple(cfg["modalities"])
        self.encoder = nn.Sequential(nn.Linear(12 * 256, 64), nn.ReLU(),
            nn.Dropout(float(cfg["dropout"])), nn.Linear(64, 64), nn.ReLU())
        self.head = VelocityHead(64, float(cfg["dropout"]))
    def forward(self, *inputs):
        # 固定通道顺序；展平保留时刻位置，但不显式共享时间局部权重。
        return self.head(self.encoder(torch.cat(inputs, dim=1).flatten(1)))

class CNNGRU(CNNEncoder):
    def __init__(self, cfg):
        super().__init__(cfg)
        width = int(cfg["d_model"])
        self.gru = nn.GRU(width, width, num_layers=3, dropout=float(cfg["dropout"]), batch_first=True)
        self.head = VelocityHead(width, float(cfg["dropout"]))
    def forward(self, *inputs):
        # 不传入上一窗口隐藏状态：每次由 PyTorch 以零状态开始。
        sequence, _ = self.gru(self.encode(inputs))
        return self.head(sequence[:, -1])

REGISTRY = {"cnn_transformer": CNNTransformer, "flat_mlp": FlatMLP, "cnn_gru": CNNGRU}

def build_model(cfg):
    model = cfg["model"]
    return REGISTRY[model["architecture"]](model)

def parameter_count(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)
