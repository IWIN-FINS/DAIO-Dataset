"""Ablation contract tests for fixed inputs, branch removal, and reload behavior."""
from pathlib import Path
import pytest
import torch
from rovio_ablation.config import load_config
from rovio_ablation.model import build_model

ROOT = Path(__file__).resolve().parents[1]
EXPERIMENTS = ["full_v24", "no_depth", "no_rpm", "flat_mlp", "cnn_gru"]

@pytest.mark.parametrize("name", EXPERIMENTS)
def test_models(name):
    cfg = load_config(ROOT / "experiments" / name / "config.yaml")
    model = build_model(cfg).eval()
    channels = {"rpm": 8, "accel_xy": 2, "gyro_z": 1, "depth": 1}
    inputs = [torch.randn(2, channels[k], 256) for k in model.input_keys]
    with torch.no_grad():
        mean, log_std = model(*inputs)
        again = model(*inputs)[0]
    assert mean.shape == log_std.shape == (2, 3)
    assert torch.isfinite(mean).all() and (log_std.exp() > 0).all()
    torch.testing.assert_close(mean, again)
    clone = build_model(cfg).eval()
    clone.load_state_dict(model.state_dict(), strict=True)
    with torch.no_grad():
        torch.testing.assert_close(mean, clone(*inputs)[0])
    if name == "no_depth":
        assert "depth" not in model.input_keys
        assert not any("depth" in key for key in model.state_dict())
    if name == "no_rpm":
        assert "rpm" not in model.input_keys
        assert not any("rpm" in key for key in model.state_dict())
    if name == "cnn_gru":
        assert not hasattr(model, "position") and not hasattr(model, "transformer")


def test_full_matches_frozen_v24():
    from reference_v24_model import build_model as old_build
    cfg = load_config(ROOT / "experiments/full_v24/config.yaml")
    original = old_build(cfg).eval()
    current = build_model(cfg).eval()
    current.load_state_dict(original.state_dict(), strict=True)
    x = [torch.randn(2, c, 256) for c in (8, 2, 1, 1)]
    with torch.no_grad():
        for a, b in zip(original(*x), current(*x)):
            torch.testing.assert_close(a, b, rtol=0, atol=0)


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA unavailable")
@pytest.mark.parametrize("name", EXPERIMENTS)
def test_cuda_small_backward(name):
    cfg = load_config(ROOT / "experiments" / name / "config.yaml")
    model = build_model(cfg).cuda().train()
    channels = {"rpm": 8, "accel_xy": 2, "gyro_z": 1, "depth": 1}
    x = [torch.randn(2, channels[k], 256, device="cuda") for k in model.input_keys]
    mean, std = model(*x)
    (mean.square().mean() + std.square().mean()).backward()
    assert all(torch.isfinite(p.grad).all() for p in model.parameters() if p.grad is not None)
