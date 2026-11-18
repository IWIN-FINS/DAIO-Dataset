from __future__ import annotations

import json
import csv
from pathlib import Path

import h5py
import numpy as np
import yaml

from rovio_training.config import load_config
from rovio_training.data import V20WindowDataset, manifest_has_shared_validation_demo, physical_mean_depth, resolve_manifest
from rovio_training.engine import evaluate_existing_run, load_checkpoint, train_new
from rovio_training.model import build_model, parameter_count


def _write_record(path: Path, phase: float) -> None:
    count, hz = 360, 200.0
    ts = np.arange(count) / hz; depth_ts = np.arange(0.0, ts[-1], 1 / 25.0); tag_ts = np.arange(0.0, ts[-1], 1 / 30.0)
    scale = 1.0 / (1000.0 * 9.80665); zero = np.full(4, 101325.0)
    with h5py.File(path, "w") as handle:
        imu = handle.create_group("imu"); imu["ts"] = ts
        imu["accel"] = np.column_stack((.2 * np.sin(ts + phase), .1 * np.cos(ts + phase), np.full(count, 9.80665)))
        imu["gyro"] = np.column_stack((np.zeros(count), np.zeros(count), .1 * np.cos(ts + phase)))
        rpm = handle.create_group("rpm"); rpm["ts"] = ts; rpm["rpm"] = np.tile(np.linspace(-500, 500, count)[:, None], (1, 8))
        depth = handle.create_group("depth"); depth["ts"] = depth_ts; depth["depth4_raw"] = np.full((len(depth_ts), 4), 1.25)
        tag = handle.create_group("apriltag"); tag["ts"] = tag_ts
        tag["xy"] = np.column_stack((.1 * tag_ts + .01 * np.sin(tag_ts + phase), .03 * np.cos(tag_ts + phase)))
        tag["yaw"] = .1 * tag_ts; tag["yaw_rate"] = np.full(len(tag_ts), .1)
        handle.attrs["depth_calibration"] = json.dumps({"sensor_zero_pressure_pa": zero.tolist(), "pressure_to_depth_m_per_pa": scale})


def _fixture(tmp_path: Path) -> tuple[Path, Path, Path]:
    root = tmp_path / "data"; processed = root / "processed"
    for directory, phase in ((processed / "train", 0.0), (processed / "evaluation" / "paper", .3), (processed / "demo", .6)):
        directory.mkdir(parents=True); _write_record(directory / "record.h5", phase)
    manifest = tmp_path / "manifest.yaml"
    manifest.write_text(yaml.safe_dump({"offline_apriltag_only": True, "processed_root": "processed", "splits": {
        "train": {"glob": "train/**/*.h5", "expected_count": 1}, "val": {"glob": "evaluation/paper/**/*.h5", "expected_count": 1},
        "demo": {"glob": "demo/**/*.h5", "expected_count": 1}}}), encoding="utf-8")
    config = tmp_path / "config.yaml"
    config.write_text(yaml.safe_dump({"experiment": {"name": "smoke", "seed": 7, "device": "cpu", "deterministic": False, "tf32": False},
        "data": {"imu_hz": 200.0, "network_hz": 50.0, "window_samples": 256, "label_savgol_window": 15, "label_savgol_polyorder": 3,
                 "pressure": {"atmosphere_pa": 101325.0, "water_density_kg_m3": 1000.0, "gravity_mps2": 9.80665}, "preprocessing": {}},
        "model": {"name": "v20_four_branch_cnn_transformer", "d_model": 32, "transformer_layers": 1, "transformer_heads": 4,
                  "transformer_ffn_dim": 64, "dropout": 0.0, "max_tokens": 160},
        "training": {"epochs": 2, "phase1_epochs": 1, "batch_size": 8, "num_workers": 0, "pin_memory": False, "lr": 1e-3,
                     "weight_decay": 1e-4, "grad_clip_norm": 5.0, "huber_delta": .1, "nll_weight": .05, "log_every_batches": 0},
        "evaluation": {"selection_metric": "body_xy_rmse", "evaluate_demo_after_training": True, "plot_dpi": 50}}), encoding="utf-8")
    return root, manifest, config


def test_v20_features_are_shape_safe_and_causal(tmp_path: Path) -> None:
    root, manifest, config_path = _fixture(tmp_path); config = load_config(config_path)
    paths = resolve_manifest(manifest, root); dataset = V20WindowDataset(paths["train"], config); normalizer = dataset.fit_normalizer(); item = dataset[0]
    assert item["rpm"].shape == (8, 256) and item["accel_xy"].shape == (2, 256)
    assert item["gyro_z"].shape == (1, 256) and item["depth"].shape == (1, 256) and item["target"].shape == (3,)
    assert np.all(normalizer["rpm_mean"] == 0.0) and np.all(normalizer["rpm_std"] == 1000.0)
    np.testing.assert_allclose(item["rpm"].numpy(), dataset.windows[0]["rpm"][0] / 1000.0)
    raw_depth = dataset.windows[0]["depth"][0][0, 0]
    assert abs(raw_depth - 1.25) < 1e-6
    model = build_model(config); mean, log_std = model(*(item[key].unsqueeze(0) for key in model.input_keys))
    assert mean.shape == log_std.shape == (1, 3) and parameter_count(model) > 0


def test_explicit_files_support_a_shared_validation_demo_without_train_leakage(tmp_path: Path) -> None:
    root, manifest, _ = _fixture(tmp_path)
    shared = {
        "offline_apriltag_only": True,
        "processed_root": "processed",
        "splits": {
            "train": {"files": ["train/record.h5"], "expected_count": 1},
            "val": {"files": ["evaluation/paper/record.h5"], "expected_count": 1},
            "demo": {"alias_of": "val", "expected_count": 1},
        },
    }
    manifest.write_text(yaml.safe_dump(shared), encoding="utf-8")
    paths = resolve_manifest(manifest, root)
    assert paths["val"] == paths["demo"]
    assert set(paths["train"]).isdisjoint(paths["val"])
    assert manifest_has_shared_validation_demo(manifest)


def test_two_epoch_cpu_smoke_saves_best_and_demo_once(tmp_path: Path) -> None:
    root, manifest, config_path = _fixture(tmp_path); config = load_config(config_path)
    run = train_new(config, config_path, manifest, root, tmp_path / "runs", tmp_path)
    assert (run / "last.pt").is_file() and (run / "best.pt").is_file() and (run / "evaluation" / "demo" / "network_summary.json").is_file()
    assert (run / "manifest_snapshot.yaml").is_file() and (run / "dataset_inventory.json").is_file()
    _, checkpoint = load_checkpoint(run / "best.pt", device=__import__("torch").device("cpu"))
    assert checkpoint["epoch"] == 2 and np.all(np.asarray(checkpoint["normalizer"]["rpm_std"]) == 1000.0)
    summary = evaluate_existing_run(run, root, "demo", "best", None, tmp_path)
    assert summary["split"] == "demo" and (run / "evaluation" / "manual_demo_best" / "record_predictions.npz").is_file()


def _write_processed_csv(path: Path, phase: float, count: int = 360) -> None:
    """Minimal new-CSV fixture: pressure is already physical depth and must not be converted again."""
    hz = 200.0
    ts = np.arange(count) / hz
    names = ["sample_time_s", "pressure", "acc_x", "acc_y", "acc_z", "gyro_z"]
    names += [f"int_rpm_{index}" for index in range(1, 9)]
    names += ["tag_x_m", "tag_y_m", "tag_yaw_rad"]
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=names); writer.writeheader()
        for index, time_s in enumerate(ts):
            row = {"sample_time_s": time_s, "pressure": 2.5 + 0.01 * np.sin(time_s),
                   "acc_x": 0.2 * np.sin(time_s + phase), "acc_y": 0.1 * np.cos(time_s + phase),
                   "acc_z": 9.80665, "gyro_z": 0.1 * np.cos(time_s + phase),
                   "tag_x_m": 0.1 * time_s + 0.01 * np.sin(time_s + phase),
                   "tag_y_m": 0.03 * np.cos(time_s + phase), "tag_yaw_rad": 0.1 * time_s}
            row.update({f"int_rpm_{channel}": -600.0 + index + channel for channel in range(1, 9)})
            writer.writerow(row)


def _csv_fixture(tmp_path: Path, count: int = 360) -> tuple[Path, Path, Path]:
    root, processed = tmp_path / "dataset", tmp_path / "dataset" / "processed"
    (processed / "v1").mkdir(parents=True); (processed / "v2").mkdir(parents=True)
    _write_processed_csv(processed / "v1" / "001_train__segment_001.csv", 0.0, count)
    _write_processed_csv(processed / "v2" / "025_demo__segment_001.csv", 0.3, count)
    with (processed / "manifest.csv").open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=("processed_relative_path", "network_window_exclusion_end_intervals_s"))
        writer.writeheader()
        writer.writerow({"processed_relative_path": "v1/001_train__segment_001.csv", "network_window_exclusion_end_intervals_s": "[[1.40, 1.60]]"})
        writer.writerow({"processed_relative_path": "v2/025_demo__segment_001.csv", "network_window_exclusion_end_intervals_s": "[]"})
    manifest = tmp_path / "csv_manifest.yaml"
    manifest.write_text(yaml.safe_dump({"data_format": "processed_csv_200hz", "offline_apriltag_only": True,
        "processed_root": "processed", "splits": {
            "train": {"glob": "v*/*.csv", "exclude_files": ["v2/025_demo__segment_001.csv"], "expected_count": 1},
            "val": {"files": ["v2/025_demo__segment_001.csv"], "expected_count": 1},
            "demo": {"alias_of": "val", "expected_count": 1},
        }}), encoding="utf-8")
    config = tmp_path / "csv_config.yaml"
    config.write_text(yaml.safe_dump({"experiment": {"name": "csv-smoke", "seed": 7, "device": "cpu", "deterministic": False, "tf32": False},
        "data": {"source_format": "processed_csv_200hz", "imu_hz": 200.0, "network_hz": 50.0, "window_samples": 256,
                 "label_savgol_window_s": 0.5, "label_savgol_polyorder": 3, "preprocessing": {}},
        "model": {"name": "v20_four_branch_cnn_transformer", "d_model": 32, "transformer_layers": 1,
                  "transformer_heads": 4, "transformer_ffn_dim": 64, "dropout": 0.0, "max_tokens": 160},
        "training": {"epochs": 2, "phase1_epochs": 1, "batch_size": 8, "num_workers": 0, "pin_memory": False,
                     "lr": 1e-3, "weight_decay": 1e-4, "grad_clip_norm": 5.0, "huber_delta": .1,
                     "nll_weight": .05, "log_every_batches": 0},
        "evaluation": {"selection_metric": "body_xy_rmse", "evaluate_demo_after_training": True, "plot_dpi": 50}}), encoding="utf-8")
    return root, manifest, config


def test_processed_csv_uses_direct_depth_and_excludes_rpm_windows(tmp_path: Path) -> None:
    root, manifest, config_path = _csv_fixture(tmp_path); config = load_config(config_path)
    paths = resolve_manifest(manifest, root)
    dataset = V20WindowDataset(paths["train"], config); normalizer = dataset.fit_normalizer()
    # RPM is divided only by 1000; pressure=2.5 m must not be interpreted as Pa and reconverted.
    item = dataset[0]
    np.testing.assert_allclose(item["rpm"].numpy(), dataset.windows[0]["rpm"][0] / 1000.0)
    assert 2.45 < float(dataset.windows[0]["depth"][0, 0, 0]) < 2.55
    assert np.all(normalizer["rpm_std"] == 1000.0)
    # Without anomaly intervals there are 27 windows at 50 Hz; 1.40--1.60 s excludes 10 endpoints.
    assert len(dataset) == 17
    assert paths["val"] == paths["demo"] and set(paths["train"]).isdisjoint(paths["val"])


def test_v21_uses_512_past_raw_samples_and_expands_rpm_exclusion(tmp_path: Path) -> None:
    root, manifest, config_path = _csv_fixture(tmp_path, count=720)
    # Provide exact source-row metadata: processed sample 598 is the only saturated RPM sample.
    processed_manifest = root / "processed" / "manifest.csv"
    with processed_manifest.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=("processed_relative_path", "source_row_start",
                                                     "rpm_saturation_source_rows", "network_window_exclusion_end_intervals_s"))
        writer.writeheader()
        writer.writerow({"processed_relative_path": "v1/001_train__segment_001.csv", "source_row_start": 2,
                         "rpm_saturation_source_rows": "[600]", "network_window_exclusion_end_intervals_s": "[]"})
        writer.writerow({"processed_relative_path": "v2/025_demo__segment_001.csv", "source_row_start": 2,
                         "rpm_saturation_source_rows": "[]", "network_window_exclusion_end_intervals_s": "[]"})
    raw_config = yaml.safe_load(config_path.read_text(encoding="utf-8"))
    raw_config["experiment"]["name"] = "v21-smoke"
    raw_config["data"].update({"source_sample_hz": 200.0, "network_input_hz": 100.0})
    raw_config["model"]["name"] = "v21_four_branch_cnn_transformer"
    config_path.write_text(yaml.safe_dump(raw_config), encoding="utf-8")

    config = load_config(config_path)
    dataset = V20WindowDataset(resolve_manifest(manifest, root)["train"], config)
    normalizer = dataset.fit_normalizer()
    assert dataset.raw_window_samples == 512 and dataset.downsample_factor == 2 and dataset.stride == 4
    # The first output is raw sample 511; its final pair is exactly [510,511], with no future samples.
    first = dataset[0]
    expected = (dataset.windows[0]["rpm"][0, :, 510] + dataset.windows[0]["rpm"][0, :, 511]) / 2.0 / 1000.0
    np.testing.assert_allclose(first["rpm"].numpy()[:, -1], expected)
    assert float(first["time"]) == 511 / 200.0
    assert first["rpm"].shape == (8, 256) and first["accel_xy"].shape == (2, 256)
    model = build_model(config)
    assert model.rpm(first["rpm"].unsqueeze(0)).shape == (1, 64, 64)
    # Saturation occurs at 2.99 s; all window endpoints over approximately the next 2.555 s must be masked.
    output_times = np.asarray([entry[2] for entry in dataset.entries])
    assert not np.any((output_times >= 598 / 200.0) & (output_times <= (598 + 511) / 200.0))
    assert np.all(normalizer["rpm_std"] == 1000.0)


def test_v22_uses_1024_past_raw_samples_and_four_sample_means(tmp_path: Path) -> None:
    root, manifest, config_path = _csv_fixture(tmp_path, count=1600)
    processed_manifest = root / "processed" / "manifest.csv"
    with processed_manifest.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=("processed_relative_path", "source_row_start",
                                                     "rpm_saturation_source_rows", "network_window_exclusion_end_intervals_s"))
        writer.writeheader()
        writer.writerow({"processed_relative_path": "v1/001_train__segment_001.csv", "source_row_start": 2,
                         "rpm_saturation_source_rows": "[1302]", "network_window_exclusion_end_intervals_s": "[]"})
        writer.writerow({"processed_relative_path": "v2/025_demo__segment_001.csv", "source_row_start": 2,
                         "rpm_saturation_source_rows": "[]", "network_window_exclusion_end_intervals_s": "[]"})
    raw_config = yaml.safe_load(config_path.read_text(encoding="utf-8"))
    raw_config["experiment"]["name"] = "v22-smoke"
    raw_config["data"].update({"source_sample_hz": 200.0, "network_input_hz": 50.0})
    raw_config["model"]["name"] = "v22_four_branch_cnn_transformer"
    config_path.write_text(yaml.safe_dump(raw_config), encoding="utf-8")

    config = load_config(config_path)
    dataset = V20WindowDataset(resolve_manifest(manifest, root)["train"], config)
    dataset.fit_normalizer()
    assert dataset.raw_window_samples == 1024 and dataset.downsample_factor == 4 and dataset.stride == 4
    first = dataset[0]
    expected = dataset.windows[0]["rpm"][0, :, 1020:1024].mean(axis=1) / 1000.0
    np.testing.assert_allclose(first["rpm"].numpy()[:, -1], expected)
    assert float(first["time"]) == 1023 / 200.0
    assert first["rpm"].shape == (8, 256) and first["depth"].shape == (1, 256)
    output_times = np.asarray([entry[2] for entry in dataset.entries])
    assert not np.any((output_times >= 1300 / 200.0) & (output_times <= (1300 + 1023) / 200.0))


def test_v23_uses_1280_past_raw_samples_and_five_sample_means(tmp_path: Path) -> None:
    root, manifest, config_path = _csv_fixture(tmp_path, count=2100)
    processed_manifest = root / "processed" / "manifest.csv"
    with processed_manifest.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=("processed_relative_path", "source_row_start",
                                                     "rpm_saturation_source_rows", "network_window_exclusion_end_intervals_s"))
        writer.writeheader()
        writer.writerow({"processed_relative_path": "v1/001_train__segment_001.csv", "source_row_start": 2,
                         "rpm_saturation_source_rows": "[1702]", "network_window_exclusion_end_intervals_s": "[]"})
        writer.writerow({"processed_relative_path": "v2/025_demo__segment_001.csv", "source_row_start": 2,
                         "rpm_saturation_source_rows": "[]", "network_window_exclusion_end_intervals_s": "[]"})
    raw_config = yaml.safe_load(config_path.read_text(encoding="utf-8"))
    raw_config["experiment"]["name"] = "v23-smoke"
    raw_config["data"].update({"source_sample_hz": 200.0, "network_input_hz": 40.0})
    raw_config["model"]["name"] = "v23_four_branch_cnn_transformer"
    config_path.write_text(yaml.safe_dump(raw_config), encoding="utf-8")

    config = load_config(config_path)
    dataset = V20WindowDataset(resolve_manifest(manifest, root)["train"], config)
    dataset.fit_normalizer()
    assert dataset.raw_window_samples == 1280 and dataset.downsample_factor == 5 and dataset.stride == 4
    first = dataset[0]
    expected = dataset.windows[0]["rpm"][0, :, 1275:1280].mean(axis=1) / 1000.0
    np.testing.assert_allclose(first["rpm"].numpy()[:, -1], expected)
    assert float(first["time"]) == 1279 / 200.0
    assert first["rpm"].shape == (8, 256) and first["gyro_z"].shape == (1, 256)
    output_times = np.asarray([entry[2] for entry in dataset.entries])
    assert not np.any((output_times >= 1700 / 200.0) & (output_times <= (1700 + 1279) / 200.0))
