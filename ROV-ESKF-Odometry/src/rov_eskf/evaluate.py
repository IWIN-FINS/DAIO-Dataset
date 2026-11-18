"""Offline benchmarks, metrics, and plots; the runtime replay module does not read AprilTag."""
from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass, replace
from pathlib import Path
from typing import Any

import numpy as np
import torch
from scipy.signal import savgol_filter
from scipy.spatial.transform import Rotation

from .data import Sequence, load_sequence, require_apriltag, resolve_manifest
from .replay import Replay11Output, replay_attitude_aided_11, replay_direct_imu_yaw_9, save_replay11
from .windows import network_observations


@dataclass(frozen=True)
class RecordWindow:
    """Offline benchmark window for one record; all times are relative to the first IMU sample."""

    path: Path
    requested_start_s: float
    requested_end_s: float | None


def angle_diff(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    return (np.asarray(a) - np.asarray(b) + np.pi) % (2 * np.pi) - np.pi


def _interpolate(ts: np.ndarray, values: np.ndarray, query: np.ndarray) -> np.ndarray:
    values = np.asarray(values, dtype=float)
    if values.ndim == 1:
        return np.interp(query, ts, values)
    return np.column_stack([np.interp(query, ts, values[:, col]) for col in range(values.shape[1])])


def offline_reference(sequence: Sequence, query_ts: np.ndarray, window: int, polyorder: int) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Offline AprilTag reference: smoothed XY/yaw and world-frame XY velocity."""
    tag_ts, tag_xy, tag_yaw, _ = require_apriltag(sequence)
    width = min(int(window), len(tag_ts) if len(tag_ts) % 2 else len(tag_ts) - 1)
    if width < 5 or width <= polyorder:
        raise ValueError("Insufficient AprilTag samples for an offline reference")
    xy = savgol_filter(tag_xy, width, polyorder, axis=0, mode="interp")
    yaw = savgol_filter(np.unwrap(tag_yaw), width, polyorder, mode="interp")
    velocity = np.gradient(xy, tag_ts, axis=0, edge_order=1)
    return _interpolate(tag_ts, xy, query_ts), np.interp(query_ts, tag_ts, yaw), _interpolate(tag_ts, velocity, query_ts)


def state_projection(raw: np.ndarray) -> np.ndarray:
    yaw = Rotation.from_quat(np.asarray(raw)[:, 6:10]).as_euler("xyz", degrees=False)[:, 2]
    return np.c_[raw[:, :6], yaw]


def _slice_output(output: Replay11Output, start_index: int) -> Replay11Output:
    mask = output.update_ts >= output.ts[start_index] - 1e-9
    return replace(output, ts=output.ts[start_index:], state=output.state[start_index:],
                   covariance_diag=output.covariance_diag[start_index:], yaw_source_age_s=output.yaw_source_age_s[start_index:],
                   update_ts=output.update_ts[mask], update_kind=output.update_kind[mask],
                   update_accepted=output.update_accepted[mask], update_nis=output.update_nis[mask],
                   update_residual=output.update_residual[mask],
                   event_meta=[row for row, keep in zip(output.event_meta, mask) if keep])


def acceptance_metrics(kinds: np.ndarray, accepted: np.ndarray, nis: np.ndarray) -> dict[str, Any]:
    result: dict[str, Any] = {}
    for kind in sorted(set(np.asarray(kinds, dtype=str))):
        mask = np.asarray(kinds, dtype=str) == kind
        values = np.asarray(nis, dtype=float)[mask]
        result[kind] = {"events": int(mask.sum()), "acceptance_rate": float(np.mean(np.asarray(accepted)[mask])),
                        "nis_median": float(np.median(values[np.isfinite(values)])) if np.any(np.isfinite(values)) else float("nan")}
    return result


def odometry_metrics(state: np.ndarray, ref_xy: np.ndarray, ref_yaw: np.ndarray, ref_velocity: np.ndarray,
                     *, yaw_absolute: bool) -> dict[str, float]:
    estimate_xy = state[:, :2]
    reference_xy = ref_xy - ref_xy[0]
    position_error = estimate_xy - reference_xy
    yaw_reference = ref_yaw if yaw_absolute else ref_yaw - ref_yaw[0]
    yaw_error = angle_diff(state[:, 6], yaw_reference)
    velocity_error = state[:, 3:5] - ref_velocity
    return {
        "xy_position_rmse_m": float(np.sqrt(np.mean(position_error ** 2))),
        "final_position_error_m": float(np.linalg.norm(position_error[-1])),
        "yaw_rmse_rad": float(np.sqrt(np.mean(yaw_error ** 2))),
        "final_yaw_error_rad": float(abs(yaw_error[-1])),
        "world_vx_rmse_mps": float(np.sqrt(np.mean(velocity_error[:, 0] ** 2))),
        "world_vy_rmse_mps": float(np.sqrt(np.mean(velocity_error[:, 1] ** 2))),
        "world_xy_velocity_rmse_mps": float(np.sqrt(np.mean(velocity_error ** 2))),
    }


def _save(fig: Any, path: Path, dpi: int) -> None:
    plt = _pyplot()
    fig.tight_layout(); fig.savefig(path, dpi=dpi); plt.close(fig)


def _pyplot() -> Any:
    """Return a headless Matplotlib interface for saving figures from CLI and Jetson replay."""
    import matplotlib
    matplotlib.use("Agg", force=True)
    import matplotlib.pyplot as plt
    return plt


def network_world_velocity(state_raw: np.ndarray, network_ts: np.ndarray, network_body_xy: np.ndarray,
                           query_ts: np.ndarray) -> np.ndarray:
    """Rotate network body-XY velocity using ESKF attitude and align it with the world-frame timeline."""
    rotations = Rotation.from_quat(np.asarray(state_raw)[:, 6:10]).as_matrix()
    body = np.column_stack((np.interp(query_ts, network_ts, network_body_xy[:, 0]),
                            np.interp(query_ts, network_ts, network_body_xy[:, 1]),
                            np.zeros(len(query_ts))))
    return np.einsum("nij,nj->ni", rotations, body)[:, :2]


def network_body_velocity_reference(sequence: Sequence, network_ts: np.ndarray, config: dict) -> tuple[np.ndarray, np.ndarray]:
    """Offline plots only: construct AprilTag-derived body-XY ground truth at network timestamps."""
    _, yaw, velocity_world = offline_reference(sequence, network_ts,
                                                int(config["evaluation"]["reference_savgol_window"]),
                                                int(config["evaluation"]["reference_savgol_polyorder"]))
    cosine, sine = np.cos(yaw), np.sin(yaw)
    velocity_body = np.column_stack((cosine * velocity_world[:, 0] + sine * velocity_world[:, 1],
                                     -sine * velocity_world[:, 0] + cosine * velocity_world[:, 1]))
    return velocity_body, yaw


def save_network_body_velocity(sequence: Sequence, network, config: dict, start_time: float, end_time: float,
                               directory: Path, dpi: int) -> dict[str, float]:
    """Save network body-XY predictions, offline AprilTag ground truth, standard deviations, and comparison plots."""
    mask = (network.ts >= start_time - 1e-9) & (network.ts <= end_time + 1e-9)
    ts, prediction, std = network.ts[mask], network.mean[mask, :2], network.std[mask, :2]
    reference, _ = network_body_velocity_reference(sequence, ts, config)
    error = prediction - reference
    np.savetxt(directory / "network_body_velocity.csv", np.c_[ts, prediction, std, reference, error], delimiter=",",
               header="time,vx_body_network,vy_body_network,vx_std,vy_std,vx_body_apriltag,vy_body_apriltag,vx_error,vy_error", comments="")
    plt = _pyplot()
    fig, axes = plt.subplots(2, 1, sharex=True, figsize=(10, 5.2))
    rel = ts - ts[0]
    for col, axis in enumerate(axes):
        axis.plot(rel, reference[:, col], "k", lw=1.0, label="AprilTag offline ground truth")
        axis.plot(rel, prediction[:, col], color="#2ca02c", lw=1.0, label="network prediction")
        axis.fill_between(rel, prediction[:, col] - std[:, col], prediction[:, col] + std[:, col],
                          color="#2ca02c", alpha=.18, linewidth=0, label="network ±1σ")
        axis.set_ylabel(f"body v_{'xy'[col]} [m/s]"); axis.grid(alpha=.25)
    axes[0].legend(fontsize=8); axes[-1].set_xlabel("time after scoring start [s]")
    _save(fig, directory / "network_body_velocity.png", dpi)
    return {"network_body_vx_rmse_mps": float(np.sqrt(np.mean(error[:, 0] ** 2))),
            "network_body_vy_rmse_mps": float(np.sqrt(np.mean(error[:, 1] ** 2))),
            "network_body_xy_rmse_mps": float(np.sqrt(np.mean(error ** 2)))}


def _imu_yaw_aligned(sequence: Sequence, query_ts: np.ndarray, reference_yaw: np.ndarray, *, yaw_absolute: bool) -> np.ndarray:
    """Causally hold onboard yaw on the IMU timeline and align it with the plotted world heading.

    In benchmark mode, the ESKF re-anchors to AprilTag yaw at the scoring start. Raw IMU yaw
    is compared after a constant offset; deployment mode instead sets the scoring-start heading to zero.
    This is for offline visualization only and never provides an additional filter observation.
    """
    query = np.asarray(query_ts, dtype=float)
    indices = np.searchsorted(sequence.attitude_ts, query, side="right") - 1
    if np.any(indices < 0):
        raise ValueError("Scoring starts before the first IMU attitude packet; cannot plot IMU yaw")
    yaw = np.unwrap(np.asarray(sequence.attitude_rpy[indices, 2], dtype=float))
    offset = float(reference_yaw[0] - yaw[0]) if yaw_absolute else float(-yaw[0])
    return yaw + offset


def _wrapped_yaw_for_plot(yaw: np.ndarray) -> np.ndarray:
    """Wrap yaw to [0, 2π) and break curves at boundary crossings to avoid artificial vertical lines."""
    wrapped = np.mod(np.asarray(yaw, dtype=float), 2.0 * np.pi)
    if len(wrapped) > 1:
        wrapped[1:][np.abs(np.diff(wrapped)) > np.pi] = np.nan
    return wrapped


def _plots(ts: np.ndarray, state: np.ndarray, ref_xy: np.ndarray, ref_yaw: np.ndarray, ref_velocity: np.ndarray,
           directory: Path, dpi: int, *, yaw_absolute: bool, imu_yaw: np.ndarray | None,
           network_velocity_world: np.ndarray | None = None) -> None:
    plt = _pyplot()
    rel, xy_ref = ts - ts[0], ref_xy - ref_xy[0]
    fig, ax = plt.subplots(figsize=(6.5, 5.5)); ax.plot(xy_ref[:, 0], xy_ref[:, 1], "k", label="AprilTag offline reference")
    ax.plot(state[:, 0], state[:, 1], label="11-state ESKF"); ax.set(xlabel="world x [m]", ylabel="world y [m]"); ax.axis("equal"); ax.grid(alpha=.25); ax.legend(); _save(fig, directory / "xy.png", dpi)
    yaw_ref = ref_yaw if yaw_absolute else ref_yaw - ref_yaw[0]
    fig, axes = plt.subplots(3, 1, sharex=True, figsize=(10, 7))
    for axis, label, estimate, reference in zip(axes[:2], ("world x [m]", "world y [m]"),
                                                  (state[:, 0], state[:, 1]), (xy_ref[:, 0], xy_ref[:, 1])):
        axis.plot(rel, reference, "k", lw=1., label="AprilTag offline reference")
        axis.plot(rel, estimate, lw=1., label="11-state ESKF")
        axis.set_ylabel(label); axis.grid(alpha=.25)
    axes[2].plot(rel, _wrapped_yaw_for_plot(yaw_ref), "k", lw=1., label="AprilTag offline reference")
    axes[2].plot(rel, _wrapped_yaw_for_plot(state[:, 6]), lw=1., label="11-state ESKF")
    if imu_yaw is not None:
        axes[2].plot(rel, _wrapped_yaw_for_plot(imu_yaw), color="#d95f02", lw=.9, alpha=.9,
                     label="IMU solved yaw (aligned; diagnostic only)")
    axes[2].set_ylabel("yaw [rad]")
    axes[2].set_ylim(0.0, 2.0 * np.pi)
    axes[2].set_yticks((0.0, .5 * np.pi, np.pi, 1.5 * np.pi, 2.0 * np.pi),
                       ("0", "π/2", "π", "3π/2", "2π"))
    axes[2].grid(alpha=.25)
    axes[0].legend(fontsize=8)
    axes[2].legend(fontsize=8, loc="best")
    axes[-1].set_xlabel("time after scoring start [s]"); _save(fig, directory / "x_y_yaw.png", dpi)
    fig, axes = plt.subplots(2, 1, sharex=True, figsize=(10, 5.2))
    for col, axis in enumerate(axes):
        axis.plot(rel, ref_velocity[:, col], "k", lw=1., label="AprilTag offline reference")
        axis.plot(rel, state[:, 3 + col], lw=1., label="11-state ESKF")
        if network_velocity_world is not None:
            axis.plot(rel, network_velocity_world[:, col], color="#2ca02c", lw=.9, alpha=.9,
                      label="network prediction (world)")
        axis.set_ylabel(f"world v_{'xy'[col]} [m/s]"); axis.grid(alpha=.25)
    axes[0].legend(fontsize=8); axes[-1].set_xlabel("time after scoring start [s]"); _save(fig, directory / "world_velocity.png", dpi)


def _json(value: Any) -> Any:
    if isinstance(value, np.ndarray): return value.tolist()
    if isinstance(value, np.generic): return value.item()
    if isinstance(value, dict): return {str(k): _json(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)): return [_json(v) for v in value]
    return value


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _window(record: str | Path | RecordWindow, sequence: Sequence, config: dict) -> RecordWindow:
    """Map user-specified seconds to a validated window relative to the record start."""
    if isinstance(record, RecordWindow):
        requested = record
    else:
        requested = RecordWindow(Path(record), float(config["initialization"]["startup_s"]), None)
    duration = float(sequence.imu_ts[-1] - sequence.imu_ts[0])
    start = float(requested.requested_start_s)
    end = duration if requested.requested_end_s is None else float(requested.requested_end_s)
    startup = float(config["initialization"]["startup_s"])
    if start < startup:
        raise ValueError(f"{requested.path}: start_s={start:g} precedes the end of the {startup:g} s initialization window")
    if not np.isfinite(end) or end <= start:
        raise ValueError(f"{requested.path}: end_s must be greater than start_s")
    if end > duration + 1e-9:
        raise ValueError(f"{requested.path}: end_s={end:g} exceeds the record duration of {duration:g} s")
    return RecordWindow(requested.path, start, None if requested.requested_end_s is None else end)


def _effective_window(sequence: Sequence, window: RecordWindow) -> tuple[int, int, float, float]:
    """Align requested bounds inward to actual 200 Hz IMU timestamps to avoid partial integration intervals."""
    origin = float(sequence.imu_ts[0])
    requested_end = float(sequence.imu_ts[-1] - origin) if window.requested_end_s is None else window.requested_end_s
    start_index = int(np.searchsorted(sequence.imu_ts, origin + window.requested_start_s, side="left"))
    end_index = int(np.searchsorted(sequence.imu_ts, origin + requested_end, side="right") - 1)
    if start_index >= len(sequence.imu_ts) or end_index <= start_index:
        raise ValueError(f"{window.path}: the valid IMU window is empty")
    return start_index, end_index, float(sequence.imu_ts[start_index]), float(sequence.imu_ts[end_index])


def _window_metadata(sequence: Sequence, window: RecordWindow, full: Replay11Output, start_index: int,
                     end_index: int, start_time: float, end_time: float, config: dict) -> dict[str, Any]:
    """Save reviewable cropping, initialization, and start-time posterior details together."""
    origin = float(sequence.imu_ts[0])
    return {
        "source_hdf5": str(sequence.path),
        "source_hdf5_sha256": _sha256(sequence.path),
        "requested_start_s": float(window.requested_start_s),
        "requested_end_s": None if window.requested_end_s is None else float(window.requested_end_s),
        "effective_start_s": float(start_time - origin),
        "effective_end_s": float(end_time - origin),
        "scoring_duration_s": float(end_time - start_time),
        "initialization_window_s": [0.0, float(config["initialization"]["startup_s"])],
        "reanchor_time_s": None if full.reanchor_time is None else float(full.reanchor_time - origin),
        "state_full_coordinate_note": "Before start_s: provisional world; after yaw re-anchoring at start_s: benchmark local world.",
        "accel_bias_initial_mps2": full.state[0, 10:13].tolist(),
        "accel_bias_at_start_mps2": full.state[start_index, 10:13].tolist(),
        "yaw_rate_correction_at_start_radps": float(full.state[start_index, 14]),
        "imu_start_index": int(start_index),
        "imu_end_index": int(end_index),
    }


def evaluate_record(record: str | Path | RecordWindow, config: dict, model: torch.nn.Module | None,
                    checkpoint: dict[str, Any] | None, device: str | torch.device, out_dir: str | Path) -> dict[str, Any]:
    mode = config["evaluation"]["mode"]
    record_path = record.path if isinstance(record, RecordWindow) else Path(record)
    sequence = load_sequence(record_path, require_reference=(mode == "benchmark_apriltag_init"))
    window = _window(record, sequence, config)
    start_index, end_index, t_start, t_end = _effective_window(sequence, window)
    reanchor_yaw = None
    if mode == "benchmark_apriltag_init":
        reanchor_yaw = float(offline_reference(sequence, np.array([t_start]), config["evaluation"]["reference_savgol_window"], config["evaluation"]["reference_savgol_polyorder"])[1][0])
    network = None if model is None else network_observations(sequence, model, checkpoint, device,
                                                                batch_size=int(config["model"]["batch_size"]), pressure_cfg=config["data"]["pressure"])
    replay = replay_direct_imu_yaw_9 if config["filter"].get("yaw_mode", "estimated_network_r") == "direct_imu" else replay_attitude_aided_11
    full = replay(sequence, config, model, checkpoint, device, reanchor_yaw=reanchor_yaw,
                  network=network, reanchor_time=t_start, end_time=t_end)
    directory = Path(out_dir) / record_path.stem
    save_replay11(full, directory, save_covariance=bool(config["debug"]["save_covariance"]),
                  save_event_trace=bool(config["debug"]["save_event_trace"]), suffix="_full")
    scored = _slice_output(full, start_index); state = state_projection(scored.state)
    metadata = _window_metadata(sequence, window, full, start_index, end_index, t_start, t_end, config)
    (directory / "window_metadata.json").write_text(json.dumps(_json(metadata), indent=2), encoding="utf-8")
    save_replay11(scored, directory, save_covariance=bool(config["debug"]["save_covariance"]),
                  save_event_trace=bool(config["debug"]["save_event_trace"]))
    metrics: dict[str, Any] = {"scoring_start_s": metadata["effective_start_s"],
                               "scoring_end_s": metadata["effective_end_s"],
                               "scoring_duration_s": metadata["scoring_duration_s"], "mode": mode,
                               "startup_initialization": full.startup_source,
                               "updates": acceptance_metrics(scored.update_kind, scored.update_accepted, scored.update_nis)}
    if sequence.tag_ts is not None:
        ref_xy, ref_yaw, ref_velocity = offline_reference(sequence, scored.ts, config["evaluation"]["reference_savgol_window"], config["evaluation"]["reference_savgol_polyorder"])
        metrics.update(odometry_metrics(state, ref_xy, ref_yaw, ref_velocity, yaw_absolute=(mode == "benchmark_apriltag_init")))
        net_world = None if network is None else network_world_velocity(scored.state, network.ts, network.mean[:, :2], scored.ts)
        if network is not None:
            metrics.update(save_network_body_velocity(sequence, network, config, scored.ts[0], scored.ts[-1], directory,
                                                      int(config["evaluation"]["plot_dpi"])))
        if bool(config["debug"]["save_plots"]):
            # Do not read or plot IMU-solved yaw in the raw_gyro_z experiment, to avoid implying it is a filter input.
            source_type = str(config["filter"].get("yaw_rate_source", {}).get("type", "solved_yaw_backward_difference"))
            imu_yaw = None if source_type == "raw_gyro_z" else _imu_yaw_aligned(
                sequence, scored.ts, ref_yaw, yaw_absolute=(mode == "benchmark_apriltag_init"))
            _plots(scored.ts, state, ref_xy, ref_yaw, ref_velocity, directory, int(config["evaluation"]["plot_dpi"]),
                   yaw_absolute=(mode == "benchmark_apriltag_init"), imu_yaw=imu_yaw,
                   network_velocity_world=net_world)
    else:
        metrics["offline_reference_available"] = False
    (directory / "metrics.json").write_text(json.dumps(_json(metrics), indent=2), encoding="utf-8")
    return metrics


def _records(config: dict, split: str) -> list[RecordWindow]:
    explicit = config["data"].get("records", [])
    if explicit:
        return [RecordWindow(Path(item["path"]), float(config["initialization"]["startup_s"])
                             if item.get("start_s") is None else float(item["start_s"]),
                             None if item.get("end_s") is None else float(item["end_s"])) for item in explicit]
    return [RecordWindow(path, float(config["initialization"]["startup_s"]), None)
            for path in resolve_manifest(config["data"]["manifest"], config["data"]["root"])[split]]


def evaluate_split(split: str, config: dict, out_dir: str | Path, device: str | torch.device) -> dict[str, Any]:
    model = checkpoint = None
    if config["model"].get("checkpoint"):
        from .model import load_checkpoint
        model, checkpoint = load_checkpoint(config["model"]["checkpoint"], device)
    rows = [{"record": item.path.stem, "metrics": evaluate_record(item, config, model, checkpoint, device, out_dir)}
            for item in _records(config, split)]
    direct_yaw = config["filter"].get("yaw_mode", "estimated_network_r") == "direct_imu"
    summary = {"experiment": "direct_imu_yaw_ekf9" if direct_yaw else "attitude_aided_eskf11",
               "state_definition": "[p_w(3),v_w(3),b_a(3)] + external_imu_yaw" if direct_yaw else "[p_w(3),v_w(3),b_a(3),yaw,yaw_rate_correction]",
               "mode": config["evaluation"]["mode"], "split": split, "records": rows}
    directory = Path(out_dir); directory.mkdir(parents=True, exist_ok=True)
    (directory / "replay_summary.json").write_text(json.dumps(_json(summary), indent=2), encoding="utf-8")
    _write_markdown_summary(summary, directory)
    return summary


def _write_markdown_summary(summary: dict[str, Any], directory: Path) -> None:
    """Write a reviewable per-trajectory summary without labeling shared demonstrations as blind tests."""
    rows = summary["records"]
    direct_yaw = summary["experiment"] == "direct_imu_yaw_ekf9"
    observation_text = "50 Hz mean depth and network XY velocity; yaw comes directly from the IMU solution, with no yaw estimation or network yaw-rate fusion." if direct_yaw else "50 Hz mean depth, network XY velocity, and network yaw-rate."
    lines = ["# V24 ESKF Replay Summary", "",
             "These results use shared validation/demonstration records, not independent blind tests. Warm up over 0--8 s and re-anchor AprilTag yaw once at 8 s; "
             f"afterward, AprilTag is used only for offline scoring. Runtime fusion uses {observation_text}", "",
             "| Record | XY position RMSE [m] | World XY velocity RMSE [m/s] | Yaw RMSE [rad] | Final position [m] | XY accept |", 
             "|---|---:|---:|---:|---:|---:|"]
    for row in rows:
        metrics = row["metrics"]; updates = metrics["updates"]
        lines.append(
            f"| {row['record']} | {metrics.get('xy_position_rmse_m', float('nan')):.4f} | "
            f"{metrics.get('world_xy_velocity_rmse_mps', float('nan')):.4f} | "
            f"{metrics.get('yaw_rmse_rad', float('nan')):.4f} | "
            f"{metrics.get('final_position_error_m', float('nan')):.4f} | "
            f"{updates.get('network_xy', {}).get('acceptance_rate', float('nan')):.3f} |"
        )
    (directory / "REPORT.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
