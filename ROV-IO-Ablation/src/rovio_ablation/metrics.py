"""Losses and velocity-observation metrics shared by validation and final evaluation."""
from __future__ import annotations

from typing import Any

import numpy as np
import torch
import torch.nn.functional as F


AXES = ("vx_b_mps", "vy_b_mps", "r_radps")


def hybrid_loss(mean: torch.Tensor, log_std: torch.Tensor, target: torch.Tensor, phase: int,
                huber_delta: float, nll_weight: float) -> tuple[torch.Tensor, dict[str, float]]:
    huber = F.huber_loss(mean, target, delta=huber_delta)
    if phase == 1: return huber, {"huber": float(huber.detach()), "nll": 0.0}
    nll = 0.5 * ((mean - target).square() * torch.exp(-2.0 * log_std) + 2.0 * log_std)
    nll = nll.mean()
    return huber + nll_weight * nll, {"huber": float(huber.detach()), "nll": float(nll.detach())}


def network_metrics(prediction: np.ndarray, std: np.ndarray, target: np.ndarray) -> dict[str, Any]:
    prediction, std, target = (np.asarray(value, dtype=np.float64) for value in (prediction, std, target))
    if prediction.shape != std.shape or prediction.shape != target.shape or prediction.ndim != 2 or prediction.shape[1] != 3:
        raise ValueError("prediction, std, target must be matching [N,3]")
    error = prediction - target; axes: dict[str, Any] = {}
    for index, name in enumerate(AXES):
        axis_error, axis_std = error[:, index], np.maximum(std[:, index], 1e-8)
        axes[name] = {"rmse": float(np.sqrt(np.mean(axis_error**2))), "mae": float(np.mean(np.abs(axis_error))),
                      "mean_error": float(np.mean(axis_error)),
                      "gaussian_nll": float(np.mean(0.5 * ((axis_error / axis_std) ** 2 + 2.0 * np.log(axis_std)))),
                      "calibration_ratio": float(np.sqrt(np.mean(axis_error**2)) / np.mean(axis_std))}
    return {"samples": int(len(target)), "axes": axes,
            "body_xy_rmse": float(np.sqrt(np.mean(error[:, :2] ** 2))),
            "all_axis_rmse": float(np.sqrt(np.mean(error ** 2)))}
