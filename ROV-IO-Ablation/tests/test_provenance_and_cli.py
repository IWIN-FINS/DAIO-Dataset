"""Standalone entry-point, manifest, and summary-tool regression tests."""
import importlib.util
import json
from pathlib import Path
import subprocess
import sys
import pytest
from rovio_ablation.provenance import snapshot_source, assert_source
from rovio_ablation.engine import _dataset_inventory, _assert_dataset_snapshot
from rovio_ablation.data import resolve_manifest

ROOT = Path(__file__).resolve().parents[1]

def test_source_change_rejected(tmp_path):
    root = tmp_path / "project"
    (root / "src").mkdir(parents=True)
    source = root / "src" / "model.py"
    source.write_text("x = 1\n", encoding="utf-8")
    run = tmp_path / "run"
    snapshot_source(root, run)
    assert_source(root, run)
    source.write_text("x = 2\n", encoding="utf-8")
    with pytest.raises(RuntimeError, match="Source code"):
        assert_source(root, run)

def test_data_manifest_change_rejected(csv_data, tmp_path):
    root, manifest = csv_data
    inventory = _dataset_inventory(resolve_manifest(manifest, root), root)
    path = root / "processed" / "manifest.csv"
    path.write_text(path.read_text(encoding="utf-8")+"\n", encoding="utf-8")
    with pytest.raises(RuntimeError, match="SHA-256"):
        _assert_dataset_snapshot(tmp_path, manifest, root, inventory["snapshot_sha256"])

@pytest.mark.parametrize("name",["full_v24","no_depth","no_rpm","flat_mlp","cnn_gru"])
def test_independent_cli(name):
    result = subprocess.run([sys.executable, "train.py", "--help"],
        cwd=ROOT/"experiments"/name, capture_output=True, timeout=30)
    assert result.returncode == 0, result.stderr
    assert b"resume" in result.stdout and b"inspect" in result.stdout

def test_summary_marks_missing_and_complete(tmp_path):
    spec=importlib.util.spec_from_file_location("summary",ROOT/"summarize.py")
    module=importlib.util.module_from_spec(spec); spec.loader.exec_module(module)
    root=tmp_path/"repo"
    run=root/"experiments"/"full_v24"/"runs"/"20260915-150000_seed42"
    (run/"evaluation"/"val").mkdir(parents=True)
    (run/"final_summary.json").write_text("{}",encoding="utf-8")
    (run/"model_summary.json").write_text('{"parameters":226182}',encoding="utf-8")
    (run/"evaluation"/"val"/"network_summary.json").write_text(json.dumps({
        "combined":{"body_xy_rmse":.02,"all_axis_rmse":.03,"axes":{}}}),encoding="utf-8")
    rows=module.summarize(root,tmp_path/"out")
    assert len(rows)==30
    assert sum(r["status"]=="complete" for r in rows)==1
    assert (tmp_path/"out"/"comparison.md").is_file()
