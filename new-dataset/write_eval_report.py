"""Evidence-bound results summary; no manual editing of scientific metrics."""
from pathlib import Path
import csv
import numpy as np
ROOT=Path(__file__).resolve().parents[1];OUT=ROOT/'ROV-ESKF-Odometry/reports/new_v1v2';DATA=ROOT/'new-dataset'
NAMES={'horizontal_rpm':'Horizontal IMU + RPM','full_imu_rpm':'Full six-axis IMU + RPM','horizontal_command':'Horizontal IMU + commands','airio_planar':'AirIO (Planar ROV)','deepvl_planar':'DeepVL (Planar ROV)'}
NAMES['airio_3d']='AirIO (3D mixed-reference supervision)'

def main():
    network=list(csv.DictReader((OUT/'network_ablation_metrics.csv').open(encoding='utf-8-sig')))
    odom=list(csv.DictReader((OUT/'metrics.csv').open(encoding='utf-8-sig')))
    methods=list(dict.fromkeys(r['method'] for r in odom));diagnostic=[]
    for method in methods:
        for test in range(1,5):
            path=OUT/method/f'test{test}_replay/updates.csv'
            if not path.exists():path=OUT/method/f'test{test}_updates.csv'
            if not path.exists():continue
            events=[r for r in csv.DictReader(path.open(encoding='utf-8-sig')) if r['kind']=='network_xy' and float(r['time'])>=8]
            t=np.array([float(r['time']) for r in events]);i=int(np.argmax(np.diff(t)))
            diagnostic.append({'method':method,'test':f'Test {test}','events':len(events),
                'accepted':sum(r['accepted']=='True' for r in events),'largest_update_gap_s':float(t[i+1]-t[i]),
                'gap_start_record_s':float(t[i]),'gap_end_record_s':float(t[i+1])})
    with (OUT/'network_update_availability.csv').open('w',encoding='utf-8-sig',newline='') as f:
        w=csv.DictWriter(f,fieldnames=list(diagnostic[0]));w.writeheader();w.writerows(diagnostic)
    ntable='\n'.join('| '+NAMES[r['model']]+' | '+f"{float(r['checkpoint_epoch']):.0f}"+' | '+' | '.join(f'{float(r[k]):.5f}' for k in ('xy_vector_rmse_mps','xy_component_mean_rmse_mps','yaw_rate_rmse_radps'))+' |' for r in network if r['test']=='Mean')
    otable='\n'.join('| '+NAMES[r['method']]+' | '+' | '.join(f'{float(r[k]):.5f}' for k in ('velocity_xy_vector_rmse_mps','position_xy_vector_rmse_m','final_position_error_m','rpe_5s_xy_vector_rmse_m'))+' |' for r in odom if r['test']=='Mean' and r['scoring']=='all_samples')
    detail='\n'.join('| '+NAMES[r['method']]+' | '+r['test']+' | '+' | '.join(f'{float(r[k]):.5f}' for k in ('velocity_xy_vector_rmse_mps','position_xy_vector_rmse_m','rpe_5s_xy_vector_rmse_m'))+' |' for r in odom if r['test']!='Mean' and r['scoring']=='all_samples')
    text=f'''# V1/V2 Evaluation Results

## Reproduce the Evaluation

Run commands from the repository root:

```sh
python verify_release.py
python evaluate_network.py --model new/horizontal_rpm --test 1
python evaluate_network.py --model new/full_imu_rpm --test 1
python evaluate_network.py --model new/horizontal_command --test 1
python replay.py --method daio --test 1
python replay.py --method no_rpm --test 1
python replay.py --method deepvl --test 1
python replay.py --method airio --test 1
```

Repeat for Test 2--4. Follow `new-dataset/processed/split.yaml`: validation uses V2 026/027 and Test 1--4 use V2 025/028/029/030. Select checkpoints using validation data. Use the checkpoint mapping in the repository README and `models/index.csv`; pass `--checkpoint` to DAIO/no-RPM replay when selecting a different model.

For the paper DAIO checkpoint, run:

```sh
python replay.py --method daio --test 1 --checkpoint models/combined/daio.pt
```

## Network Results

Read `network_ablation_metrics.csv` for per-trajectory results. The table below reports arithmetic means across the four test trajectories. Use all valid 50 Hz windows for metrics.

| Model | Best epoch | XY vector RMSE (m/s) | Mean XY component RMSE (m/s) | Yaw-rate RMSE (rad/s) |
|---|---:|---:|---:|---:|
{ntable}

## Odometry Results

Replay continuously from each record's beginning. Use an 8 s warm-up and one-time yaw/XY-position alignment at the scoring start. Compute metrics over the remaining 200 Hz samples using the supplied reference processing. Report the arithmetic mean across trajectories.

| Method | Velocity RMSE (m/s) | Position RMSE (m) | Final position error (m) | 5 s relative displacement RMSE (m) |
|---|---:|---:|---:|---:|
{otable}

| Method | Trajectory | Velocity RMSE (m/s) | Position RMSE (m) | 5 s relative displacement RMSE (m) |
|---|---|---:|---:|---:|
{detail}

Read `metrics.csv` for component metrics and per-trajectory scores. Use `provenance.json` to identify the checkpoints, data, and implementation used for this evaluation.

## Plot the Released Paper Results

```sh
python results/odometry/plot_odometry.py
```

Set tests, display ranges, and figure sizes in the script's `USER CONFIGURATION` section. Find the generated figures in `outputs/odometry_figures/`.
'''
    (OUT/'README.md').write_text(text,encoding='utf-8')
    print('Wrote evaluation report:',OUT/'README.md')

if __name__=='__main__':main()
