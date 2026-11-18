"""数据契约、两阶段训练、断点恢复与回溯测试。"""
from pathlib import Path
import copy
import json
import numpy as np
import pytest
import torch
from rovio_ablation.config import load_config
from rovio_ablation.data import V20WindowDataset, resolve_manifest
from rovio_ablation.engine import train_new, resume_run, load_checkpoint
from rovio_ablation import engine

ROOT = Path(__file__).resolve().parents[1]
NAMES = ["full_v24","no_depth","no_rpm","flat_mlp","cnn_gru"]

def config(name):
    return load_config(ROOT / "experiments" / name / "config.yaml")

def test_same_windows_and_no_inactive_statistics(csv_data):
    root, manifest = csv_data
    splits = resolve_manifest(manifest,root)
    reference = None
    for name in NAMES:
        cfg = config(name)
        dataset = V20WindowDataset(splits["train"],cfg)
        normalizer = dataset.fit_normalizer()
        assert set(normalizer) == {f"{k}_{suffix}" for k in dataset.input_keys for suffix in ("mean","std")}
        targets = np.array([x[3] for x in dataset.entries])
        ends = np.array([x[2] for x in dataset.entries])
        if reference is None: reference = ends,targets
        else:
            np.testing.assert_array_equal(reference[0],ends)
            np.testing.assert_array_equal(reference[1],targets)
        assert ends[0] > 1535/200 # 第0帧异常覆盖首个窗口，所有消融均排除它。
        for key in dataset.input_keys:
            idx,start,_,_ = dataset.entries[0]
            raw = dataset.windows[idx][key][start]
            expected = raw.reshape(raw.shape[0],256,6).mean(axis=2)
            np.testing.assert_array_equal(dataset._feature_window(idx,start,key),expected)
        item = dataset[0]
        if "rpm" in dataset.input_keys:
            np.testing.assert_allclose(item["rpm"],dataset._feature_window(0,dataset.entries[0][1],"rpm")/1000,rtol=1e-6)
        frozen = {k:v.copy() for k,v in normalizer.items()}
        val = V20WindowDataset(splits["val"],cfg,normalizer)
        val[0]
        for k in frozen: np.testing.assert_array_equal(normalizer[k],frozen[k])

@pytest.mark.parametrize("name",NAMES)
def test_two_phase_smoke(csv_data,tmp_path,name):
    root, manifest = csv_data
    cfg = config(name)
    cfg["experiment"].update(device="cpu", deterministic=True)
    cfg["training"].update(epochs=2,phase1_epochs=1,batch_size=8)
    cfg["evaluation"]["plot_dpi"]=30
    run = train_new(cfg,ROOT/"experiments"/name/"config.yaml",manifest,root,tmp_path/name,ROOT)
    model,payload = load_checkpoint(run/"best.pt",torch.device("cpu"))
    assert payload["epoch"] == 2 and "shuffle_rng_state" in payload
    assert (run/"source_inventory.json").is_file()
    assert (run/"source_snapshot"/"src"/"rovio_ablation"/"model.py").is_file()
    assert (run/"evaluation"/"demo"/"network_summary.json").is_file()
    assert (run/"evaluation"/"val"/"val_network.png").is_file()
    rows = [json.loads(line) for line in (run/"metrics.jsonl").read_text(encoding="utf-8").splitlines()]
    assert not rows[0]["is_best"] and rows[1]["is_best"]

def test_resume_matches_continuous(csv_data,tmp_path,monkeypatch):
    root,manifest = csv_data
    cfg=config("flat_mlp")
    cfg["experiment"].update(device="cpu",deterministic=True)
    cfg["training"].update(epochs=3,phase1_epochs=1,batch_size=8)
    cfg["evaluation"].update(plot_dpi=25,evaluate_demo_after_training=False)
    cfgpath=ROOT/"experiments"/"flat_mlp"/"config.yaml"
    complete=train_new(cfg,cfgpath,manifest,root,tmp_path/"complete",ROOT)
    original=engine._train_epoch
    def interrupted(*args,**kwargs):
        epoch=kwargs.get("epoch",args[-1])
        if epoch==2: raise RuntimeError("controlled interruption")
        return original(*args,**kwargs)
    monkeypatch.setattr(engine,"_train_epoch",interrupted)
    with pytest.raises(RuntimeError,match="controlled interruption"):
        train_new(cfg,cfgpath,manifest,root,tmp_path/"interrupted",ROOT)
    run=next((tmp_path/"interrupted").iterdir())
    monkeypatch.setattr(engine,"_train_epoch",original)
    resume_run(run,root,ROOT)
    a=torch.load(complete/"last.pt",weights_only=False)
    b=torch.load(run/"last.pt",weights_only=False)
    for k in a["model"]: torch.testing.assert_close(a["model"][k],b["model"][k],rtol=0,atol=0)
    assert a["best_metric"]==b["best_metric"]
