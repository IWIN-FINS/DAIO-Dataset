# DAIO：最新数据集、网络训练与里程计代码

本资料包只包含最新的、有 AprilTag 参考的新 V1/V2 **处理后数据**。不包含旧数据集、原始采集 CSV、无人机视频、环境安装目录、逐轮训练断点或缓存。开放水域 V5 演示没有同等级的独立定位参考，不混入本论文训练/测试划分。

## 数据与文件

| 内容 | 位置 |
|---|---|
| 55 段处理后 CSV、清洗审计、修复与源行映射 | `new-dataset/processed/` |
| 49 训练 / 2 验证 / 4 测试划分 | `new-dataset/processed/split.yaml`、`datasets.csv` |
| DAIO、无 RPM、六轴 IMU、推进器指令训练入口 | `ROV-IO-Ablation/experiments/new_v1v2/` |
| 网络、损失、数据窗口与训练器 | `ROV-IO-Ablation/src/rovio_ablation/` |
| 11 状态 ESKF、网络观测、离线回放 | `ROV-ESKF-Odometry/src/rov_eskf/` |
| 模型及训练来源 | `models/`、`models/index.csv` |
| AirIO / AirIMU 与 DeepVL 适配代码 | `new-dataset/airio_3d*`、`new-dataset/deepvl/`、`new-dataset/baselines/` |
| 论文四方法逐样本结果及绘图代码 | `results/odometry/` |
| 第三方来源和许可证 | `THIRD_PARTY.md` |

验证轨迹为 V2 026、027；Test 1–4 对应 025、028、029、030。划分文件中的 `demo` 即测试集。54 条源记录产生 55 个连续段，其中 V2 019 长中断切成两段；短段保留但不构造不足长度的网络窗口。三集合无源记录交叉。

采样时间 `sample_time_s` 步长 0.005 s（200 Hz）；`pressure` 已是深度米值。IMU 加速度单位 m/s²，角速度和姿态为 rad/s、rad，RPM 为有符号转速，指令为有符号 throttle。完整字段包含六轴 IMU、roll/pitch/yaw、8 路 RPM、8 路指令、电压、AprilTag XY/yaw 和原时间字段。AprilTag 仅用于离线监督/评估。DAIO 不输入电压、指令或 AprilTag；DeepVL 的输入不同，详见其配置。

## 安装与完整性检查

推荐 Python 3.10。在包根目录运行；训练需要 PyTorch CUDA 环境，CPU 可用于回放检查。

```sh
python -m pip install -r requirements.txt
python verify_release.py
```

`SHA256SUMS` 覆盖发布文件；数据与模型另有逐文件索引。模型为本项目可信来源的 PyTorch checkpoint，包含归一化和配置；不要用这些加载入口读取来源不明的 pickle 权重。

## 仅使用最新数据训练

下列命令从头训练，默认 seed 42、80 轮；前 40 轮 Huber，随后加入不确定度损失。原始 200 Hz 每 6 点因果均值，256 个网络输入点、约 7.68 s 历史，50 Hz 预测。归一化由训练集计算，验证指标用于选模。此次整理没有启动正式训练。

```sh
# DAIO：水平 IMU + RPM + 深度
python ROV-IO-Ablation/experiments/new_v1v2/horizontal_rpm/train.py validate
python ROV-IO-Ablation/experiments/new_v1v2/horizontal_rpm/train.py train --seed 42

# DAIO without RPM：水平 IMU + 深度
python ROV-IO-Ablation/experiments/new_v1v2/no_rpm/train.py train --seed 42

# 输入消融
python ROV-IO-Ablation/experiments/new_v1v2/full_imu_rpm/train.py train --seed 42
python ROV-IO-Ablation/experiments/new_v1v2/horizontal_command/train.py train --seed 42
```

各实验 `runs/` 保存配置、日志、最佳权重、断点和自动测试结果。恢复命令为对应入口的 `resume --run-dir PATH`；评估为 `evaluate --run-dir PATH --split demo`。先执行对应入口 `--help` 查看参数。

```sh
python new-dataset/deepvl/prepare.py
python new-dataset/deepvl/train.py
python new-dataset/airio_3d/prepare.py
python new-dataset/airio_3d/train.py airimu
python train_airio.py
```

`train_airio.py` 默认使用随包冻结 AirIMU；要改用重训权重，传入 `--airimu PATH`。三维参考采用 AprilTag XY/yaw、压力计 Z 和 IMU 解算 roll/pitch，不能表述为完全独立的三维动捕真值。

## 网络与里程计评估

```sh
python evaluate_network.py --model new/horizontal_rpm --test 1
python evaluate_network.py --model new/full_imu_rpm --test 1
python replay.py --method daio --test 1
python replay.py --method deepvl --test 1
python replay.py --method airio --test 1
python replay.py --method no_rpm --test 1
```

`--test` 为 1–4，`--device` 为 cpu 或 cuda。结果写入 `outputs/`。从记录起点连续回放，前 8 秒预热，评分起点一次坐标对齐，后续不使用持续定位真值更新。回放 CSV 包括参考及预测的世界系 XY 位置、速度和 yaw。

**权重来源必须区分：**

- 默认 `--method daio` 使用 `models/new/horizontal_rpm.pt`，仅在最新数据训练。
- `models/new/` 三组、AirIO/AirIMU、DeepVL 均来自新数据训练。
- `models/combined/daio.pt` 与 `models/combined/no_rpm.pt` 是论文历史联合训练权重，使用过旧＋新训练集。旧数据不随本包提供，无法仅靠本包复现这两个权重的联合训练过程。
- 目前随包没有经过确认的“新数据单独训练”的 no-RPM 权重。默认 `--method no_rpm` 明确使用上述历史联合权重。要进行严格新数据训练对比，先运行新增 no-RPM 训练入口，再执行 `python replay.py --method no_rpm --test 1 --checkpoint PATH_TO_BEST_PT`。
- 若复现论文联合 DAIO 回放，显式执行 `python replay.py --method daio --test 1 --checkpoint models/combined/daio.pt`。

`results/odometry/` 是论文历史对比的归档结果，其中 DAIO/no-RPM 为联合训练模型；它们不是本包默认新数据 DAIO 的重算结果。`new-dataset/combined_training/` 仅保存这两个历史模型的训练源码供查阅，缺少旧数据时不要启动该入口。`ROV-IO-Training` 保留共享代码/历史格式兼容层；当前训练入口以上述 `new_v1v2` 为准。

## 发布范围与许可

本次为本地开源资料整理，尚未上传公开仓库。第三方代码保留原许可证。项目自有代码、数据和模型的公开许可证尚需权利人确定，见 `LICENSE_STATUS.md`；没有替你擅自授予 MIT/Apache/CC 等许可。
