"""Source snapshots support provenance and resumption checks; never include data or existing training artifacts."""
import hashlib
import shutil
from pathlib import Path
from .utils import atomic_json, canonical_json, sha256_file

def source_inventory(root):
    root = Path(root)
    paths = list((root / "src").rglob("*.py"))
    paths += list((root / "experiments").glob("*/train.py"))
    paths += list((root / "experiments" / "new_v1v2").glob("*/train.py"))
    paths += list(root.glob("*.py"))
    paths += list((root / "tests").rglob("*.py"))
    paths += [p for p in (root / "pyproject.toml", root / "requirements-lock.txt") if p.is_file()]
    records = {p.relative_to(root).as_posix(): sha256_file(p) for p in sorted(paths)}
    return {"files": records, "sha256": hashlib.sha256(canonical_json(records).encode("utf-8")).hexdigest()}

def snapshot_source(root, run):
    root, run = Path(root), Path(run)
    inventory = source_inventory(root)
    for relative in inventory["files"]:
        destination = run / "source_snapshot" / relative
        destination.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(root / relative, destination)
    atomic_json(run / "source_inventory.json", inventory)
    return inventory

def assert_source(root, run):
    import json
    saved = json.loads((Path(run) / "source_inventory.json").read_text(encoding="utf-8"))
    if source_inventory(root)["sha256"] != saved["sha256"]:
        raise RuntimeError("Source code has changed; refusing to resume. Restore source_snapshot or start a new experiment.")
