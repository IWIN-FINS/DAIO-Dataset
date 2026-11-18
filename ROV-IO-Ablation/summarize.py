"""汇总各实验/种子最新一次完成运行；绝不按 demo 误差选择运行。"""
from pathlib import Path
import argparse
import csv
import json
import statistics

EXPERIMENTS = ("full_v24","no_depth","no_rpm","flat_mlp","cnn_gru")
SEEDS = (42,2025,2026)

def summarize(root, output):
    root, output = Path(root),Path(output)
    output.mkdir(parents=True,exist_ok=True)
    rows = []
    for experiment in EXPERIMENTS:
        for seed in SEEDS:
            runs = sorted((root/"experiments"/experiment/"runs").glob(f"*_seed{seed}"),reverse=True)
            completed = [p for p in runs if (p/"final_summary.json").is_file()]
            run = completed[0] if completed else None
            for split in ("val","demo"):
                row = dict(experiment=experiment,seed=seed,split=split,status="未完成",
                           run=str(run.relative_to(root)) if run else "", completed_runs=len(completed))
                path = run/"evaluation"/split/"network_summary.json" if run else None
                if path and path.is_file():
                    metrics=json.loads(path.read_text(encoding="utf-8"))["combined"]
                    row.update(status="完成",body_xy_rmse=metrics["body_xy_rmse"],all_axis_rmse=metrics["all_axis_rmse"])
                    for axis,values in metrics["axes"].items():
                        for key in ("rmse","mae","mean_error","gaussian_nll","calibration_ratio"):
                            row[f"{axis}_{key}"]=values[key]
                    model=json.loads((run/"model_summary.json").read_text(encoding="utf-8"))
                    row["parameters"]=model["parameters"]
                rows.append(row)
    keys=list(dict.fromkeys(k for row in rows for k in row))
    with (output/"comparison.csv").open("w",newline="",encoding="utf-8-sig") as h:
        writer=csv.DictWriter(h,fieldnames=keys);writer.writeheader();writer.writerows(rows)
    lines=["# 消融实验汇总","",
           "每个种子取时间戳最新的已完成运行，不按 demo 指标挑选；多次完成运行的数量见 CSV。四条 demo 曾用于历史选模，本轮仅为留出展示集。","",
           "| 实验 | split | 已完成种子/3 | 联合 XY RMSE 均值 ± 样本标准差 [m/s] |","|---|---|---:|---:|"]
    for name in EXPERIMENTS:
        for split in ("val","demo"):
            values=[r["body_xy_rmse"] for r in rows if r["experiment"]==name and r["split"]==split and r["status"]=="完成"]
            score=f"{statistics.mean(values):.6f} ± {statistics.stdev(values):.6f}" if len(values)>1 else (f"{values[0]:.6f} (单种子)" if values else "未完成")
            lines.append(f"| {name} | {split} | {len(values)}/3 | {score} |")
    (output/"comparison.md").write_text("\n".join(lines)+"\n",encoding="utf-8")
    return rows

if __name__=="__main__":
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root",type=Path,default=Path(__file__).resolve().parent)
    parser.add_argument("--output",type=Path,default=None)
    args=parser.parse_args()
    summarize(args.root,args.output or args.root/"reports"/"comparison")
    print("汇总完成；未训练实验保留为“未完成”。")
