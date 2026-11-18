"""Shared components for five experiments; removing a modality removes its branch, rather than substituting zeros."""
from __future__ import annotations
import torch
from torch import nn

CHANNELS = {"rpm": 8, "throttle": 8, "accel_xy": 2, "accel_xyz": 3, "gyro_z": 1, "gyro_xyz": 3, "depth": 1}
PROJECT_NAMES = {"rpm": "rpm_project", "throttle": "throttle_project", "accel_xy": "accel_project", "accel_xyz": "accel_project", "gyro_z": "gyro_project", "gyro_xyz": "gyro_project", "depth": "depth_project"}

class LocalCNN(nn.Module):
    """Retain V24 convolutions: 256 samples produce 64 tokens; the full window ends at prediction time."""
    def __init__(self, input_channels):
        super().__init__()
        self.layers = nn.Sequential(
            nn.Conv1d(input_channels, 32, 7, stride=2, padding=3), nn.BatchNorm1d(32), nn.ReLU(),
            nn.Conv1d(32, 64, 5, stride=2, padding=2), nn.BatchNorm1d(64), nn.ReLU())
    def forward(self, values):
        return self.layers(values).transpose(1, 2)

class VelocityHead(nn.Module):
    """Use the same velocity-mean and diagonal log-std output heads for all models."""
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
        # Retain full V24 module names and initialization order for exact weight/forward regression checks.
        for key in self.input_keys:
            setattr(self, key, LocalCNN(CHANNELS[key]))
        for key in self.input_keys:
            setattr(self, PROJECT_NAMES[key], nn.Linear(64, int(cfg["d_model"])))
    def encode(self, inputs):
        if len(inputs) != len(self.input_keys):
            raise ValueError("The number of input modalities does not match the checkpoint contract")
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
            raise ValueError("Token count exceeds the positional encoding length")
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
        # Fix channel order; flattening preserves temporal position without explicitly sharing local temporal weights.
        return self.head(self.encoder(torch.cat(inputs, dim=1).flatten(1)))

class CNNGRU(CNNEncoder):
    def __init__(self, cfg):
        super().__init__(cfg)
        width = int(cfg["d_model"])
        self.gru = nn.GRU(width, width, num_layers=3, dropout=float(cfg["dropout"]), batch_first=True)
        self.head = VelocityHead(width, float(cfg["dropout"]))
    def forward(self, *inputs):
        # Do not pass hidden state from the previous window; PyTorch starts each window with zero state.
        sequence, _ = self.gru(self.encode(inputs))
        return self.head(sequence[:, -1])

REGISTRY = {"cnn_transformer": CNNTransformer, "flat_mlp": FlatMLP, "cnn_gru": CNNGRU}

def build_model(cfg):
    model = cfg["model"]
    return REGISTRY[model["architecture"]](model)

def parameter_count(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)
