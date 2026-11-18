"""仅构造短小合成 CSV fixture，不读取或更改实机数据。"""
from pathlib import Path
import csv
import numpy as np
import pytest
import torch
import yaml

@pytest.fixture(autouse=True)
def cpu_threads():
    previous = torch.get_num_threads()
    torch.set_num_threads(2)
    yield
    torch.set_num_threads(previous)

@pytest.fixture
def csv_data(tmp_path):
    root = tmp_path / "dataset"
    processed = root / "processed"
    processed.mkdir(parents=True)
    fields = ["sample_time_s", "pressure", "acc_x", "acc_y", "acc_z", "gyro_z"]
    fields += [f"int_rpm_{i}" for i in range(1,9)] + ["tag_x_m", "tag_y_m", "tag_yaw_rad"]
    rows = []
    for split, phase in (("train",0.),("val",.4),("demo",.8)):
        path = processed / f"{split}.csv"
        with path.open("w",newline="",encoding="utf-8") as handle:
            writer = csv.DictWriter(handle, fieldnames=fields)
            writer.writeheader()
            for i in range(1620):
                t = i / 200.
                row = dict(sample_time_s=t, pressure=1.5+.05*np.sin(t+phase),
                           acc_x=.2*np.sin(t+phase), acc_y=.1*np.cos(t), acc_z=9.80665,
                           gyro_z=.1, tag_x_m=.1*t, tag_y_m=.02*np.sin(t), tag_yaw_rad=.1*t)
                row.update({f"int_rpm_{j}": -300 + 2*i + j for j in range(1,9)})
                writer.writerow(row)
        rows.append(dict(processed_relative_path=path.name, source_row_start=0,
                         rpm_saturation_source_rows="[0]", network_window_exclusion_end_intervals_s="[]"))
    with (processed / "manifest.csv").open("w",newline="",encoding="utf-8") as h:
        writer = csv.DictWriter(h,fieldnames=list(rows[0])); writer.writeheader(); writer.writerows(rows)
    manifest = tmp_path / "manifest.yaml"
    manifest.write_text(yaml.safe_dump(dict(data_format="processed_csv_200hz",offline_apriltag_only=True,processed_root="processed",
        splits={k:dict(files=[f"{k}.csv"],expected_count=1) for k in ("train","val","demo")})),encoding="utf-8")
    return root, manifest
