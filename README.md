# DAIO Dataset

Processed underwater ROV data, pretrained models, and code for network training and odometry evaluation.

## 1. Download and Install

```sh
git clone https://github.com/IWIN-FINS/DAIO-Dataset.git
cd DAIO-Dataset
python -m pip install -r requirements.txt
python verify_release.py
```

Use Python 3.10 and a CUDA-enabled PyTorch environment for training. Use `--device cpu` for CPU evaluation or `--device cuda` for GPU evaluation. Run the commands below from the repository root.

## 2. Use the Dataset

Load the processed CSV files from `new-dataset/processed/v1/` and `new-dataset/processed/v2/`. Follow the split in `new-dataset/processed/split.yaml`; `datasets.csv` provides a per-file index.

| Split | Segments | Records |
|---|---:|---|
| Training | 49 | Listed under `train` |
| Validation | 2 | V2 026, 027 |
| Test (`demo`) | 4 | Test 1--4: V2 025, 028, 029, 030 |

Use `sample_time_s` as the 200 Hz timeline. Read acceleration in m/s², angular velocity in rad/s, and attitude in radians. The `pressure` field is already depth in meters. RPM and throttle commands are signed values. Use AprilTag XY/yaw for offline labels and evaluation. Fit normalization statistics on the training split and use the validation split for checkpoint selection.

## 3. Evaluate the Released Models

Run network evaluation for a selected model and test trajectory:

```sh
python evaluate_network.py --model new/horizontal_rpm --test 1
python evaluate_network.py --model new/full_imu_rpm --test 1
python evaluate_network.py --model new/horizontal_command --test 1
```

Run odometry replay:

```sh
python replay.py --method daio --test 1
python replay.py --method no_rpm --test 1
python replay.py --method deepvl --test 1
python replay.py --method airio --test 1
```

Repeat with `--test 2`, `--test 3`, and `--test 4`. Network outputs are written to `outputs/network/`; replay CSV files are written to `outputs/`. Use `--output PATH` to choose another destination.

Select checkpoints as follows:

| Task | Checkpoint |
|---|---|
| DAIO trained on the released V1/V2 data; default DAIO replay | `models/new/horizontal_rpm.pt` |
| Six-axis IMU + RPM network | `models/new/full_imu_rpm.pt` |
| Horizontal IMU + command network | `models/new/horizontal_command.pt` |
| Paper DAIO replay; joint legacy + V1/V2 training | `models/combined/daio.pt` |
| Paper no-RPM replay; joint legacy + V1/V2 training; default no-RPM replay | `models/combined/no_rpm.pt` |
| DeepVL | `models/deepvl/ensemble.pt` |
| AirIO / AirIMU | `models/airio/airio.pt`, `models/airio/airimu.pt` |

To reproduce the paper's DAIO replay, select its checkpoint explicitly:

```sh
python replay.py --method daio --test 1 --checkpoint models/combined/daio.pt
```

Use `models/index.csv` for checkpoint identifiers and training-data provenance. For a no-RPM comparison trained only on the released data, train the no-RPM configuration below and pass its `best.pt` using `--checkpoint`.

## 4. Train on the Released Data

Validate the dataset, then train the desired configuration:

```sh
python ROV-IO-Ablation/experiments/new_v1v2/horizontal_rpm/train.py validate

# DAIO: horizontal IMU + RPM + depth
python ROV-IO-Ablation/experiments/new_v1v2/horizontal_rpm/train.py train --seed 42

# No RPM: horizontal IMU + depth
python ROV-IO-Ablation/experiments/new_v1v2/no_rpm/train.py train --seed 42

# Input ablations
python ROV-IO-Ablation/experiments/new_v1v2/full_imu_rpm/train.py train --seed 42
python ROV-IO-Ablation/experiments/new_v1v2/horizontal_command/train.py train --seed 42
```

Keep the supplied configuration for the default 80-epoch schedule: 40 Huber-only epochs followed by Huber plus uncertainty loss. The model uses 256 causal input points spanning approximately 7.68 s and predicts at 50 Hz. Training outputs are saved under each experiment's `runs/` directory.

Resume or evaluate a run using the same experiment entry point:

```sh
python ROV-IO-Ablation/experiments/new_v1v2/horizontal_rpm/train.py resume --run-dir PATH_TO_RUN
python ROV-IO-Ablation/experiments/new_v1v2/horizontal_rpm/train.py evaluate --run-dir PATH_TO_RUN --split demo
python replay.py --method daio --test 1 --checkpoint PATH_TO_RUN/best.pt
python replay.py --method no_rpm --test 1 --checkpoint PATH_TO_NO_RPM_RUN/best.pt
```

Train the comparison methods with:

```sh
python new-dataset/deepvl/prepare.py
python new-dataset/deepvl/train.py
python new-dataset/airio_3d/prepare.py
python new-dataset/airio_3d/train.py airimu
python train_airio.py
```

`train_airio.py` uses the released AirIMU checkpoint by default. Pass `--airimu PATH_TO_AIRIMU_CHECKPOINT` to use your trained AirIMU model. Use `--help` on each command-line entry point to inspect its options.

## 5. Compare Results and Draw Figures

Use the supplied replay protocol: process each record from its start, warm up for 8 s, align coordinates once at the scoring start, and score the remaining trajectory. Replay CSV files include reference and predicted XY position, XY velocity, and yaw.

Use `results/odometry/test1_odometry.csv` through `test4_odometry.csv` and `odometry_metrics_updated.csv` for the archived paper comparison. Regenerate its figures with:

```sh
python results/odometry/plot_odometry.py
```

Edit the script's `USER CONFIGURATION` section to select tests, display ranges, and figure sizes. Figures are saved to `outputs/odometry_figures/`.

## Repository Layout

| Path | Use |
|---|---|
| `new-dataset/processed/` | Dataset, split, and preprocessing metadata |
| `ROV-IO-Ablation/experiments/new_v1v2/` | Current training entry points |
| `ROV-IO-Ablation/src/rovio_ablation/` | Models, features, losses, and training engine |
| `ROV-ESKF-Odometry/src/rov_eskf/` | ESKF and replay implementation |
| `models/` | Released checkpoints and index |
| `new-dataset/airio_3d*/`, `new-dataset/deepvl/`, `new-dataset/baselines/` | Comparison methods |
| `results/odometry/` | Paper results and plotting script |
| `SHA256SUMS`, `verify_release.py` | Release integrity verification |

## Licensing

Read `LICENSE`, `LICENSE_STATUS.md`, and `THIRD_PARTY.md` before reuse or redistribution. Retain the license notices supplied with third-party components.
