# Orin NX：V20 在线部署升级到 V24——仅变更项

本文件只列出相对旧版 [V20 Orin 交接说明](ORIN_NX_DEPLOYMENT_HANDOFF.md) 和
[V20 在线部署提示词](../ORIN_NX_在线ESKF部署提示词.md) 的变化。未在本文明确
修改的 11 状态 ESKF、FLU 坐标系、队列顺序消费、NIS 日志、降级策略及测试要求，
均按旧文档执行。

## 1. 模型与时间窗

| 项目 | V20 | V24（新） |
|---|---:|---:|
| checkpoint | V20 `best.pt` | `ROV-IO-Training/runs/v24_33p33hz_7p68s/20260830-172129_seed42/best.pt` |
| 原始输入频率 | 200 Hz | 200 Hz |
| 网络输入频率 | 200 Hz | 33.333 Hz：每连续 6 个原始样本取因果均值 |
| token 数 | 256 | 256（模型结构不变） |
| 原始历史长度 | 256 样本，1.28 s | 1536 样本，7.68 s |
| 网络输出频率 | 50 Hz | 50 Hz；每 4 个原始样本推理一次 |

输入/输出张量仍为：

```text
输入: [1,8,256] RPM, [1,2,256] accel_xy, [1,1,256] gyro_z, [1,1,256] depth
输出: mean=[v_x^B,v_y^B,r], std=[σ_vx,σ_vy,σ_r]
```

## 2. 在线特征构造替换

修改 `src/rov_eskf/online_features.py`：

- 把 V20 的 `deque(maxlen=256)` 改为至少 `deque(maxlen=1536)`；
- 当前网络时刻为原始样本索引 `e` 时，取 `[e-1535,e]` 的 1536 点；
- 由旧到新分成 256 个不重叠 6 点块，每块计算算术均值；
- 最后一个 token 必须是当前样本及前 5 个样本的均值；不得使用未来样本；
- 首次推理必须在收满 1536 个健康包后，约启动 7.68 s；
- 后续只在每第 4 个原始包触发一次推理。

对于每个六点均值 `mean_6(x)`，特征为：

```text
RPM      = mean_6(signed_rpm) / 1000
acc_x/y  = (mean_6(acc_x/y) - accel_xy_mean) / accel_xy_std
gyro_z   = (mean_6(gyro_z) - gyro_z_mean) / gyro_z_std
depth_m  = (mean_6(depth_m) - depth_mean) / depth_std
```

不要在六点均值后再做去偏、坐标翻转、重采样或在线统计。

## 3. V24 冻结 z-score

修改在线 checkpoint 加载/特征初始化逻辑：从 `checkpoint["normalizer"]` 读取统计量，
不要手写或重新计算。当前 V24 checkpoint 的值为：

| 特征 | 均值 μ | 标准差 σ |
|---|---:|---:|
| `acc_x` | -0.03506604 | 0.08320995 |
| `acc_y` | 0.00973243 | 0.05856332 |
| `gyro_z` | -0.00524616 | 0.11925314 |
| `depth_m` | 0.28159106 | 0.02481415 |
| 8 路 RPM | 0 | 1000（即仅 `/1000`） |

启动时必须检查 checkpoint 的输入契约：

```text
source_sample_hz = 200
network_input_hz = 200 / 6
network_hz = 50
window_samples = 256
```

任一不一致即拒绝启动。

## 4. 深度字段语义变化

修改 `src/rov_eskf/online_types.py` 与 `src/rov_eskf/queue_adapter.py`：

- 统一在线字段为 `depth_m`：四压力计均值已经换算得到的物理深度，**向下为正**；
- 新数据集 CSV 的 `pressure` 就是这个 `depth_m`，不能再次减 `101325` 或除以 `1000×9.80665`；
- 若队列实际给出四个绝对压力值，只能在 `queue_adapter.py` 按下式换算一次：
  \[
  depth_m=(\mathrm{mean}(P_{1:4})-101325)/(1000\times9.80665).
  \]
- 若真实压力仅 50 Hz 刷新，在其余 200 Hz 包使用最近过去 `depth_m` 零阶保持；
- 50 Hz 高度观测使用 `filter.update_height(-depth_m, ...)`。

## 5. yaw 传播源（V24 当前默认）

修改 `src/rov_eskf/online_yaw.py` 与 `src/rov_eskf/online.py`：

- 保留 IMU 解算 yaw 的严格因果后向差分作为传播源；
- `yaw_imu_rad` 已是多圈弧度。新姿态包到达后先 `unwrap`，再仅与过去
  `backward_window_s=0.05 s` 前最近的包求差分，并使用过去 5 个候选的因果中值滤波；
- 得到的 yaw-rate 从当前姿态包时刻起零阶保持：
  \[
  \dot\psi=\dot\psi_{\mathrm{imu}}+d_{\dot\psi}.
  \]
- 每个 IMU 区间调用：

  ```python
  filter.predict_interval(acc_prev, acc_now, causal_yaw_rate, 0.0, 0.0, dt)
  ```

- `gyro_z_radps` 仍仅用于 V24 网络特征，不得替换该传播源；
- 网络 `r` 仍通过 `update_yaw_rate()` 与同一 `causal_yaw_rate` 独立更新
  `yaw_rate_correction`。

## 6. 50 Hz 同步更新与网络协方差

修改 `src/rov_eskf/online.py` 和 `src/rov_eskf/observations.py`：

```text
height(-depth_m) → network XY velocity → network yaw-rate
```

- 三个更新每个 50 Hz tick 各执行一次；
- 网络 XY 与网络 yaw-rate 保持独立 NIS 门控；
- V24 使用：

  ```yaml
  filter:
    yaw_rate_source:
      type: solved_yaw_backward_difference
      backward_window_s: 0.05
      median_filter_samples: 5
      max_abs_rate_radps: 1.0
    network:
      xy: {covariance_scale: 1.0, min_std: 0.005, max_std: 2.0, max_step_mps: 0.10}
      yaw_rate: {variance_scale: 1.0, min_std: 0.005, max_std: 2.0}
  ```

- 不要沿用旧 V20 部署配置的网络协方差倍率 `16.0`。

## 7. 配置文件与测试更新

新增 `configs/online_v24_relative_solved_yaw.yaml`，至少替换为：

```yaml
model:
  checkpoint: "/home/<user>/rovio/models/v24_best.pt"
  device: "cuda"
  batch_size: 1
online:
  source_hz: 200.0
  input_hz: 33.333333333333336
  network_hz: 50.0
  window_tokens: 256
  raw_window_samples: 1536
  downsample_factor: 6
  network_stride_samples: 4
  startup_s: 8.0
  depth_representation: "mean_depth_m_down_positive"
  yaw_rate_source: "solved_yaw_backward_difference"
```

替换/新增测试：

1. `tests/test_online_features_parity.py`：在线 1536 点缓冲与离线
   `src/rov_eskf/windows.py::V20WindowBuilder` 在 V24 checkpoint 下的四个输入逐元素一致；
2. `tests/test_online_yaw_causality.py`：未来 `yaw_imu_rad` 改动不得改变当前状态；
   新 yaw 包到达后才允许改变后续传播率；`gyro_z` 改动不得改变 ESKF yaw；
3. `tests/test_online_runner.py`：第 1536 包首次推理，之后每 4 包一次；
4. `tests/test_online_runner.py`：每个 tick 的更新顺序为高度、XY、yaw-rate。

## 8. 启动期唯一变化

部署模式由旧 V20 的 10 s 改为 **8 s**：前 8 s 收集静止偏置、初始深度与 V24 历史，
不发布正式状态；8 s 后以 `yaw=0` 的相对世界系发布。在线部署不允许 AprilTag 重锚。
