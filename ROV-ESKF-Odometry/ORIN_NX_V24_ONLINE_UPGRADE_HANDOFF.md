# Orin NX: V24 Online Deployment Steps

Use these settings when implementing a V24 online runner on Orin NX. Load `models/new/horizontal_rpm.pt`, read its configuration and normalizer, and use the FLU body frame.

## 1. Model and Time Window

| Item | V20 | V24 (new) |
|---|---:|---:|
| Checkpoint | V20 `best.pt` | `models/new/horizontal_rpm.pt` |
| Raw input frequency | 200 Hz | 200 Hz |
| Network input frequency | 200 Hz | 33.333 Hz: causal mean of each six consecutive raw samples |
| Tokens | 256 | 256 (unchanged architecture) |
| Raw history | 256 samples, 1.28 s | 1536 samples, 7.68 s |
| Network output frequency | 50 Hz | 50 Hz; infer every four raw samples |

Input/output tensors remain:

```text
Inputs: [1,8,256] RPM, [1,2,256] accel_xy, [1,1,256] gyro_z, [1,1,256] depth
Outputs: mean=[v_x^B,v_y^B,r], std=[σ_vx,σ_vy,σ_r]
```

## 2. Replace Online Feature Construction

In the online feature builder:

- Replace V20's `deque(maxlen=256)` with at least `deque(maxlen=1536)`.
- At raw sample index `e`, take the 1536 samples in `[e-1535,e]`.
- From oldest to newest, form 256 nonoverlapping blocks of six samples and compute each arithmetic mean.
- The final token must average the current sample and the five preceding samples; never use future samples.
- Run the first inference only after receiving 1536 healthy packets, approximately 7.68 s after startup.
- Trigger subsequent inference every fourth raw packet.

For each six-sample mean `mean_6(x)`, construct:

```text
RPM      = mean_6(signed_rpm) / 1000
acc_x/y  = (mean_6(acc_x/y) - accel_xy_mean) / accel_xy_std
gyro_z   = (mean_6(gyro_z) - gyro_z_mean) / gyro_z_std
depth_m  = (mean_6(depth_m) - depth_mean) / depth_std
```

Do not add debiasing, axis flips, resampling, or online statistics after the six-sample means.

## 3. Frozen V24 Z-Score Statistics

Read all normalization statistics from `checkpoint["normalizer"]` when loading the model. Apply the saved means and standard deviations after causal block averaging.

Check the checkpoint input contract at startup:

```text
source_sample_hz = 200
network_input_hz = 200 / 6
network_hz = 50
window_samples = 256
```

Reject startup if any value differs.

## 4. Depth Field Semantics

In the online sensor interface and queue adapter:

- Use `depth_m` consistently: physical depth converted from the four-sensor mean pressure, **positive downward**.
- The new dataset CSV field `pressure` already represents `depth_m`; do not subtract `101325` or divide by `1000×9.80665` again.
- If the queue supplies four absolute pressures, convert exactly once in the queue adapter:
  \[
  depth_m=(\mathrm{mean}(P_{1:4})-101325)/(1000\times9.80665).
  \]
- If pressure refreshes at only 50 Hz, use zero-order hold of the latest past `depth_m` in other 200 Hz packets.
- Apply 50 Hz height observations using `filter.update_height(-depth_m, ...)`.

## 5. Yaw Propagation Source (V24 Default)

In the online yaw source and runner:

- Retain strictly causal backward differences of IMU-solved yaw as the propagation source.
- `yaw_imu_rad` is already in multi-turn radians. On each new attitude packet, unwrap yaw, then difference it against the latest packet at or before `backward_window_s=0.05 s` earlier. Apply a causal median filter over the past five candidates.
- Hold the resulting yaw-rate from the current attitude timestamp:
  \[
  \dot\psi=\dot\psi_{\mathrm{imu}}+d_{\dot\psi}.
  \]
- For each IMU interval, call:

  ```python
  filter.predict_interval(acc_prev, acc_now, causal_yaw_rate, 0.0, 0.0, dt)
  ```

- `gyro_z_radps` remains a V24 network feature only; do not substitute it for this propagation source.
- Network `r` still updates `yaw_rate_correction` independently through `update_yaw_rate()` using the same `causal_yaw_rate`.

## 6. Synchronized 50 Hz Updates and Network Covariance

Apply observations in this order:

```text
height(-depth_m) → network XY velocity → network yaw-rate
```

- Execute each of the three updates once per 50 Hz tick.
- Keep separate NIS gates for network XY and network yaw-rate.
- Use the V24 settings:

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

- Do not retain the old V20 deployment network covariance multiplier of `16.0`.

## 7. Configuration and Test Updates

Add `configs/online_v24_relative_solved_yaw.yaml` with at least these replacements:

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

Replace or add the following tests:

1. `tests/test_online_features_parity.py`: all four inputs from the online 1536-sample buffer must match `src/rov_eskf/windows.py::V20WindowBuilder` elementwise with the V24 checkpoint.
2. `tests/test_online_yaw_causality.py`: future `yaw_imu_rad` changes must not affect the current state. Only newly arrived yaw packets may change subsequent propagation rates; changes to `gyro_z` must not change ESKF yaw.
3. `tests/test_online_runner.py`: first inference at packet 1536, then every four packets.
4. `tests/test_online_runner.py`: update order at every tick is height, XY velocity, then yaw-rate.

## 8. Startup Change

Reduce deployment startup from V20's 10 s to **8 s**. Collect stationary bias, initial depth, and V24 history during the first 8 s without publishing operational states. Then publish in a relative world frame with `yaw=0`. AprilTag re-anchoring is not allowed in online deployment.
