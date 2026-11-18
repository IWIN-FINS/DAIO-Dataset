"""Small, dependency-light utilities shared by training and evaluation."""
from __future__ import annotations

import hashlib
import json
import os
import random
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np
import torch
import yaml


def read_yaml(path: str | Path) -> dict[str, Any]:
    with Path(path).open("r", encoding="utf-8") as handle:
        value = yaml.safe_load(handle)
    if not isinstance(value, dict):
        raise ValueError(f"{path}: expected a YAML mapping")
    return value


def canonical_json(value: Any) -> str:
    def default(item: Any) -> Any:
        if isinstance(item, Path):
            return item.as_posix()
        if isinstance(item, np.generic):
            return item.item()
        if isinstance(item, np.ndarray):
            return item.tolist()
        raise TypeError(f"cannot serialize {type(item)!r}")
    return json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":"), default=default)


def sha256_file(path: str | Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for block in iter(lambda: handle.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def atomic_json(path: str | Path, value: Any) -> None:
    path = Path(path); path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(value, ensure_ascii=False, indent=2, default=_json_default), encoding="utf-8")
    os.replace(temporary, path)


def atomic_yaml(path: str | Path, value: Any) -> None:
    path = Path(path); path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(yaml.safe_dump(value, sort_keys=False, allow_unicode=True), encoding="utf-8")
    os.replace(temporary, path)


def _json_default(value: Any) -> Any:
    if isinstance(value, np.ndarray): return value.tolist()
    if isinstance(value, np.generic): return value.item()
    if isinstance(value, Path): return value.as_posix()
    raise TypeError(f"cannot JSON serialize {type(value)!r}")


def set_seed(seed: int, deterministic: bool) -> None:
    random.seed(seed); np.random.seed(seed); torch.manual_seed(seed)
    if torch.cuda.is_available(): torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.benchmark = not deterministic
    torch.use_deterministic_algorithms(deterministic, warn_only=False)


def choose_device(requested: str) -> torch.device:
    requested = str(requested).lower()
    if requested == "auto": return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if requested == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("config requires CUDA but torch.cuda.is_available() is false")
    if requested not in {"cuda", "cpu"}: raise ValueError("experiment.device must be auto, cuda, or cpu")
    return torch.device(requested)


def git_revision(root: str | Path) -> str | None:
    try:
        return subprocess.check_output(["git", "-C", str(root), "rev-parse", "HEAD"], text=True,
                                       stderr=subprocess.DEVNULL).strip()
    except (OSError, subprocess.CalledProcessError):
        return None


def environment_snapshot(device: torch.device, project_root: str | Path) -> dict[str, Any]:
    gpu: dict[str, Any] | None = None
    if device.type == "cuda":
        properties = torch.cuda.get_device_properties(device)
        gpu = {"name": properties.name, "total_memory_bytes": int(properties.total_memory),
               "capability": [int(properties.major), int(properties.minor)]}
    return {"created_utc": datetime.now(timezone.utc).isoformat(), "python": sys.version,
            "torch": torch.__version__, "cuda_runtime": torch.version.cuda, "device": str(device), "gpu": gpu,
            "git_revision": git_revision(project_root)}


def rng_state() -> dict[str, Any]:
    return {"python": random.getstate(), "numpy": np.random.get_state(), "torch": torch.get_rng_state(),
            "cuda": torch.cuda.get_rng_state_all() if torch.cuda.is_available() else None}


def restore_rng_state(state: dict[str, Any]) -> None:
    random.setstate(state["python"]); np.random.set_state(state["numpy"]); torch.set_rng_state(state["torch"])
    if torch.cuda.is_available() and state.get("cuda") is not None: torch.cuda.set_rng_state_all(state["cuda"])
