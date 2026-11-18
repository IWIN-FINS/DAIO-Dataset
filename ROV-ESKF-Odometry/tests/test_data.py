"""Contract tests ensuring runtime data does not depend on AprilTag."""
from __future__ import annotations

import json
import csv

import h5py
import numpy as np
import pytest

from rov_eskf.data import height_measurement, load_sequence, physical_mean_depth
from rov_eskf.evaluate import RecordWindow, _effective_window, _slice_output, _window
from rov_eskf.replay import Replay11Output, replay_attitude_aided_11
from rov_eskf.windows import V20WindowBuilder
from rov_eskf.yaw_source import causal_yaw_rate_source


def _write_runtime_only(path) -> None:
    ts = np.arange(60, dtype=float) / 200.
    with h5py.File(path, "w") as h:
        h.create_dataset("imu/ts", data=ts); h.create_dataset("imu/accel", data=np.zeros((60, 3))); h.create_dataset("imu/gyro", data=np.zeros((60, 3)))
        h.create_dataset("rpm/ts", data=ts); h.create_dataset("rpm/rpm", data=np.zeros((60, 8)))
        h.create_dataset("depth/ts", data=np.arange(0., .30, .05)); h.create_dataset("depth/depth4", data=np.zeros((6, 4))); h.create_dataset("depth/depth4_raw", data=np.full((6, 4), 100.))
        h.create_dataset("aux/native_ts", data=np.arange(0., .30, .03)); h.create_dataset("aux/rpy_logged_rad", data=np.zeros((10, 3)))
        h.attrs["depth_calibration"] = json.dumps({"sensor_zero_pressure_pa": [101325.] * 4, "pressure_to_depth_m_per_pa": 0.1})


def test_runtime_h5_without_apriltag_can_be_loaded(tmp_path) -> None:
    path = tmp_path / "runtime.h5"; _write_runtime_only(path)
    seq = load_sequence(path)
    assert seq.tag_ts is None
    depth = physical_mean_depth(seq, np.array([.02]), 101325., 1000., 9.80665)
    assert np.isfinite(depth[0])
    with pytest.raises(ValueError, match="AprilTag"):
        load_sequence(path, require_reference=True)


def _replay_config() -> dict:
    return {
        "initialization": {"startup_s": .02, "position_std_m": .5, "velocity_std_mps": .3,
                           "accel_bias_std_mps2": .2, "yaw_std_rad": .1,
                           "yaw_rate_correction_std_radps": .05},
        "filter": {"gravity_flu": [0., 0., -9.80665], "accel_sample_covariance": np.eye(3) * 1e-4,
                   "accel_bias_rw_psd": np.eye(3) * 1e-6, "yaw_rate_source_psd_rad2_per_s": 1e-6,
                   "yaw_rate_correction_rw_psd_rad2_per_s3": 1e-7,
                   "yaw_rate_source": {"backward_window_s": .01, "median_filter_samples": 1,
                                       "max_abs_rate_radps": 1.},
                   "height_variance_m2": 1e-3, "depth_nis_gate_1d": 10., "velocity_nis_gate_2d": 10.,
                   "yaw_rate_nis_gate_1d": 10., "network": {"xy": {"min_std": .001, "max_std": 1.,
                   "covariance_scale": 1., "max_step_mps": 1.}, "yaw_rate": {"min_std": .001,
                   "max_std": 1., "variance_scale": 1.}}},
        "benchmark": {"yaw_reanchor_std_rad": .1},
        "debug": {"max_imu_gap_s": .0125},
    }


def test_window_is_aligned_to_imu_and_enforces_initialization(tmp_path) -> None:
    path = tmp_path / "runtime.h5"; _write_runtime_only(path); seq = load_sequence(path)
    cfg = {"initialization": {"startup_s": .02}}
    window = _window(RecordWindow(path, .03, .07), seq, cfg)
    start, end, actual_start, actual_end = _effective_window(seq, window)
    assert (start, end, actual_start, actual_end) == (6, 14, .03, .07)
    with pytest.raises(ValueError, match="initialization window"):
        _window(RecordWindow(path, .01, .07), seq, cfg)
    with pytest.raises(ValueError, match="exceeds the record duration"):
        _window(RecordWindow(path, .03, .30), seq, cfg)


def test_replay_stops_at_end_and_reanchors_at_requested_time(tmp_path) -> None:
    path = tmp_path / "runtime.h5"; _write_runtime_only(path); seq = load_sequence(path)
    config = _replay_config(); config["initialization"]["startup_s"] = .10
    output = replay_attitude_aided_11(seq, config, reanchor_yaw=.4, reanchor_time=.14, end_time=.20)
    assert np.isclose(output.ts[-1], .20)
    assert np.isclose(output.reanchor_time, .14)
    assert np.isclose(output.state[28, 13], .4)


def test_scoring_slice_also_slices_event_log() -> None:
    output = Replay11Output(np.array([0., .005, .01]), np.zeros((3, 15)), np.zeros((3, 11)),
                            np.array([0., .01]), np.array(["height", "network_xy"]), np.array([True, True]),
                            np.zeros(2), np.zeros((2, 2)), [{"time": 0.}, {"time": .01}], "", np.zeros(3))
    sliced = _slice_output(output, 2)
    assert len(sliced.update_ts) == len(sliced.event_meta) == 1
    assert sliced.event_meta[0]["time"] == .01


def _write_processed_csv(path, count: int = 1600) -> None:
    fields = ("sample_time_s", "pressure", "acc_x", "acc_y", "acc_z", "gyro_z", "yaw",
              "int_rpm_1", "int_rpm_2", "int_rpm_3", "int_rpm_4", "int_rpm_5", "int_rpm_6",
              "int_rpm_7", "int_rpm_8", "tag_x_m", "tag_y_m", "tag_yaw_rad")
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields); writer.writeheader()
        for index in range(count):
            time = index / 200.0
            writer.writerow({"sample_time_s": time, "pressure": 1.25 + .01 * time,
                             "acc_x": .1 * index, "acc_y": -.2 * index, "acc_z": 9.80665,
                             "gyro_z": .1, "yaw": .1 * time,
                             **{f"int_rpm_{axis}": 100 * axis + index for axis in range(1, 9)},
                             "tag_x_m": .5 * time, "tag_y_m": 0., "tag_yaw_rad": .1 * time})


def test_processed_csv_v24_contract_and_six_sample_windows(tmp_path) -> None:
    path = tmp_path / "025_demo__segment_001.csv"; _write_processed_csv(path)
    sequence = load_sequence(path, require_reference=True)
    assert sequence.source_format == "processed_csv_200hz"
    assert len(sequence.depth_ts) == 400 and np.isclose(sequence.attitude_rpy[-1, 2], .1 * 1599 / 200.)
    # CSV depth updates only at the end of each 20 ms interval; physical depth is not reconverted from pressure.
    assert np.isclose(height_measurement(sequence, 0), -(1.25 + .01 * 3 / 200.))
    source = causal_yaw_rate_source(sequence, {"backward_window_s": .05, "median_filter_samples": 1,
                                                "max_abs_rate_radps": 1.0})
    assert np.isclose(source.rate_at(.20), .1, atol=1e-12)

    checkpoint = {"normalizer": {"rpm_mean": np.zeros(8), "rpm_std": np.full(8, 1000.),
                                  "accel_xy_mean": np.zeros(2), "accel_xy_std": np.ones(2),
                                  "gyro_z_mean": np.zeros(1), "gyro_z_std": np.ones(1),
                                  "depth_mean": np.zeros(1), "depth_std": np.ones(1)},
                  "config": {"data": {"source_sample_hz": 200., "network_input_hz": 200. / 6.,
                                         "network_hz": 50., "window_samples": 256}}}
    builder = V20WindowBuilder(sequence, checkpoint)
    rpm, _, _, depth = builder.feature_arrays()
    assert builder.raw_window_samples == 1536 and builder.downsample_factor == 6 and builder.stride == 4
    # The final V24 token uses the current output sample and the five preceding raw samples, never future samples.
    expected = np.mean(sequence.rpm[1530:1536, 0]) / 1000.
    assert np.isclose(rpm[0, 0, -1], expected)
    assert np.isclose(builder.window_times()[0], 1535 / 200.)
    assert rpm.shape == (17, 8, 256) and depth.shape == (17, 1, 256)
