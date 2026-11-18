"""Audit ESKF loading/preprocessing against the frozen paper training artifacts."""
from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import torch

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT.parent / "ROV-IO-Ablation" / "src"))
sys.path.insert(0, str(ROOT.parent / "ROV-IO-Training" / "src"))

from rov_eskf.data import load_sequence
from rov_eskf.model import load_checkpoint, parameter_count
from rov_eskf.windows import V20WindowBuilder
from rovio_ablation.data import V20WindowDataset
from rovio_ablation.model import build_model as build_ablation_model
from rovio_training.model import build_model as build_full_model
from plot_paper_odometry_test1_4 import PAPER_CHECKPOINT, TEST_SOURCES, sha256

NO_RPM_CHECKPOINT = (ROOT.parent / "ROV-IO-Ablation" / "experiments" / "no_rpm"
                     / "runs" / "20260915-222643_seed42" / "best.pt")


def main() -> None:
    torch.set_num_threads(4)
    torch.manual_seed(42)
    report = {"models": {}, "records": [], "device": "cpu"}
    inventory = json.loads((NO_RPM_CHECKPOINT.parent / "source_inventory.json").read_text(encoding="utf-8"))
    report["frozen_ablation_source_sha256"] = {}
    for relative in ("src/rovio_ablation/model.py", "src/rovio_ablation/data.py"):
        current = ROOT.parent / "ROV-IO-Ablation" / relative
        snapshot = NO_RPM_CHECKPOINT.parent / "source_snapshot" / relative
        expected_hash = inventory["files"][relative]
        assert sha256(current) == sha256(snapshot) == expected_hash
        report["frozen_ablation_source_sha256"][relative] = expected_hash
    for label, path, build in (("full", PAPER_CHECKPOINT, build_full_model),
                                ("no_rpm", NO_RPM_CHECKPOINT, build_ablation_model)):
        runtime, payload = load_checkpoint(path, "cpu")
        reference = build(payload["config"]).eval()
        reference.load_state_dict(payload["model"], strict=True)
        channels = {"rpm": 8, "accel_xy": 2, "gyro_z": 1, "depth": 1}
        inputs = [torch.randn(7, channels[key], 256) for key in runtime.input_keys]
        with torch.no_grad():
            actual, expected = runtime(*inputs), reference(*inputs)
        for left, right in zip(actual, expected):
            torch.testing.assert_close(left, right, rtol=1e-6, atol=1e-7)
        report["models"][label] = {
            "checkpoint": str(path), "sha256": sha256(path),
            "input_keys": list(runtime.input_keys), "parameters": parameter_count(runtime),
            "forward_max_abs_difference": max(float((a - b).abs().max())
                                               for a, b in zip(actual, expected)),
            "checkpoint_epoch": int(payload["epoch"]),
        }
        if label == "no_rpm":
            no_rpm_model, no_rpm_payload = runtime, payload
            assert "rpm" not in runtime.input_keys and not hasattr(runtime, "rpm")
            assert not any(key.startswith("rpm_") for key in payload["normalizer"])

    full_meta = json.loads((PAPER_CHECKPOINT.parent / "run_metadata.json").read_text(encoding="utf-8"))
    ablation_meta = json.loads((NO_RPM_CHECKPOINT.parent / "run_metadata.json").read_text(encoding="utf-8"))
    assert full_meta["manifest_sha256"] == ablation_meta["manifest_sha256"]
    report["shared_training_manifest_sha256"] = full_meta["manifest_sha256"]
    normalizer = {key: np.asarray(value) for key, value in no_rpm_payload["normalizer"].items()}
    for number, original_name in TEST_SOURCES.items():
        test_path = ROOT.parent / "dataset" / "paper_split_v24" / "test" / f"test{number}.csv"
        original = ROOT.parent / "dataset" / "processed" / "v2" / original_name
        assert sha256(test_path) == sha256(original)
        seq = load_sequence(test_path, require_reference=True)
        builder = V20WindowBuilder(seq, no_rpm_payload)
        assert "rpm" not in builder.windows
        arrays = builder.feature_arrays()
        training = V20WindowDataset([original], no_rpm_payload["config"], normalizer)
        exported_path = (NO_RPM_CHECKPOINT.parent / "evaluation" / "demo"
                         / f"{original.stem}_predictions.npz")
        with np.load(exported_path) as stored:
            exported = {key: stored[key] for key in ("time", "prediction", "std")}
        training_times = np.array([entry[2] for entry in training.entries])
        np.testing.assert_array_equal(training_times, exported["time"])
        selection = np.linspace(0, len(training) - 1, 31, dtype=int)
        runtime_indices = np.searchsorted(builder.ts, training_times[selection])
        np.testing.assert_array_equal(builder.ts[runtime_indices], training_times[selection])
        samples = [training[int(index)] for index in selection]
        feature_error = {}
        for key, array in zip(builder.input_keys, arrays):
            expected = np.stack([sample[key].numpy() for sample in samples])
            np.testing.assert_allclose(array[runtime_indices], expected, rtol=0, atol=1e-6)
            feature_error[key] = float(np.max(np.abs(array[runtime_indices] - expected)))
        with torch.no_grad():
            prediction, log_std = no_rpm_model(*[torch.from_numpy(array[runtime_indices])
                                                for array in arrays])
        prediction = prediction.numpy()
        std = log_std.exp().numpy()
        np.testing.assert_allclose(prediction, exported["prediction"][selection], rtol=1e-5, atol=1e-6)
        np.testing.assert_allclose(std, exported["std"][selection], rtol=1e-5, atol=1e-6)
        # Perturb only in-memory RPM; the three selected feature windows must stay identical.
        seq.rpm[:] = np.random.default_rng(42).normal(1e5, 1e5, seq.rpm.shape)
        changed = V20WindowBuilder(seq, no_rpm_payload).feature_arrays()
        for left, right in zip(arrays, changed):
            np.testing.assert_array_equal(left, right)
        row = {"test": f"Test {number}", "source_sha256": sha256(test_path),
               "training_export_samples": len(training), "parity_samples": len(selection),
               "feature_max_abs_difference": feature_error,
               "exported_mean_max_abs_difference": float(np.max(np.abs(
                   prediction - exported["prediction"][selection]))),
               "exported_std_max_abs_difference": float(np.max(np.abs(
                   std - exported["std"][selection]))), "rpm_perturbation_independent": True}
        report["records"].append(row)
        print(json.dumps(row))
    destination = ROOT / "reports" / "paper_no_rpm_inference_audit.json"
    destination.parent.mkdir(parents=True, exist_ok=True)
    destination.write_text(json.dumps(report, indent=2, ensure_ascii=False), encoding="utf-8")
    print(f"Audit passed: {destination}")


if __name__ == "__main__":
    main()
