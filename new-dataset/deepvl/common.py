"""Pinned DeepVL model and the single causal feature/label contract."""
from pathlib import Path
import csv, hashlib, importlib.util, json, sys, types
from dataclasses import dataclass
import numpy as np
import torch
from scipy.signal import savgol_filter

HERE = Path(__file__).resolve().parent
DATA = HERE.parent
ROOT = DATA.parent
sys.path.insert(0, str(DATA))
from reliable_io import atomic_json, torch_save

RECIPE = dict(seed=42, member_seeds=list(range(42, 50)), ensemble_size=8,
              input_size=16, output_size=2, layers=[40, 40, 40], dropout=.1,
              network_hz=20, imu_hz=200, sequence_points=300,
              batch_size=128, updates=2500, mse_updates=1250, lr=.001,
              lr_milestones=[1000, 2000], lr_gamma=.1, validate_every=20,
              sample_density_per_second=7.5, precision='FP32; TF32 disabled',
              online_inputs=['acc_xyz', 'gyro_xyz', 'throttle_1..8', 'battery_valid', 'battery_voltage'],
              target='yaw-heading XY velocity; SG101/poly3 AprilTag labels',
              augmentation='none; vehicle-specific PWM/vertical-thruster augmentation disabled')


def sha(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as f:
        for chunk in iter(lambda: f.read(1048576), b''): h.update(chunk)
    return h.hexdigest()


def splits():
    import yaml
    d = yaml.safe_load((DATA/'processed/split.yaml').read_text(encoding='utf-8'))
    return {k: [DATA/'processed'/p for p in v['files']] for k, v in d['splits'].items()}


def fingerprint():
    return dict(recipe=RECIPE, torch=torch.__version__,
                sources={str(p.relative_to(HERE)): sha(p) for p in sorted(HERE.glob('*.py'))},
                vendor={str(p.relative_to(HERE)): sha(p) for p in sorted((HERE/'vendor').rglob('*'))
                        if p.is_file() and p.suffix in ('.py', '.hpp', '.info', '.yaml', '.json')},
                manifest=sha(DATA/'processed/manifest.csv'), split=sha(DATA/'processed/split.yaml'),
                data={str(p.relative_to(DATA)): sha(p) for paths in splits().values() for p in paths})


@dataclass
class Record:
    time: np.ndarray
    accel: np.ndarray
    gyro: np.ndarray
    command: np.ndarray
    voltage: np.ndarray
    depth: np.ndarray
    tag_xy: np.ndarray | None = None
    tag_yaw: np.ndarray | None = None


def load_record(path, reference=False):
    # Online parsing does not load RPM, AprilTag or solved attitude at all.
    names = ['sample_time_s'] + [f'acc_{a}' for a in 'xyz'] + [f'gyro_{a}' for a in 'xyz']
    names += [f'throttle_{i}' for i in range(1, 9)] + ['battery_voltage', 'pressure']
    if reference: names += ['tag_x_m', 'tag_y_m', 'tag_yaw_rad']
    with Path(path).open(encoding='utf-8-sig') as f: header = next(csv.reader(f))
    a = np.loadtxt(path, delimiter=',', skiprows=1, usecols=[header.index(n) for n in names], encoding='utf-8-sig', ndmin=2)
    if not np.isfinite(a).all(): raise ValueError(f'Nonfinite telemetry: {path}')
    np.testing.assert_allclose(np.diff(a[:, 0]), .005, rtol=0, atol=1e-10)
    return Record(a[:, 0], a[:, 1:4], a[:, 4:7], a[:, 7:15], a[:, 15], a[:, 16],
                  a[:, 17:19] if reference else None, np.unwrap(a[:, 19]) if reference else None)


def feature_points(record):
    """Each 20Hz packet is stamped at the LAST of its ten past/current IMU rows."""
    end = np.arange(9, len(record.time), 10)
    n = len(end)*10
    imu = np.column_stack((record.accel[:n], record.gyro[:n])).reshape(-1, 10, 6).mean(axis=1)
    return end, np.column_stack((imu, record.command[end], np.ones(len(end)), record.voltage[end]))


def reference(record):
    width = min(101, len(record.time) if len(record.time) % 2 else len(record.time)-1)
    if width < 5: raise ValueError('Reference too short')
    xy = savgol_filter(record.tag_xy, width, 3, axis=0, mode='interp')
    yaw = savgol_filter(record.tag_yaw, width, 3, mode='interp')
    velocity = np.gradient(xy, record.time, axis=0, edge_order=1)
    c, s = np.cos(yaw), np.sin(yaw)
    heading = np.column_stack((c*velocity[:, 0]+s*velocity[:, 1], -s*velocity[:, 0]+c*velocity[:, 1]))
    return xy, yaw, velocity, heading


_vendor = None
_model_type = None
def model(device='cpu'):
    global _vendor, _model_type
    if _vendor is None:
        p = HERE/'vendor/DeepVL/scripts/networks/networks.py'
        _vendor = types.ModuleType('deepvl_official_networks'); _vendor.__file__ = str(p)
        # The upstream constructor hardcodes CUDA. This device-only compatibility
        # substitution leaves its GRU, PReLU, unused decoders and heads intact.
        exec(compile(p.read_text(encoding='utf-8').replace('.cuda()', ''), str(p), 'exec'), _vendor.__dict__)
        class PlanarRNN(_vendor.RNN):
            def forward(self, x, h0=None):
                # Official real_time=False discards h, while real_time=True
                # discards all but the final output. Expose both without altering
                # the active GRU/PReLU/head calculation (skip_connection=False).
                out, h = self.rnn(x, h0)
                out = self.activation1(out)
                return self.linear_vel(out), self.linear_std(out), h
        _model_type = PlanarRNN
    return _model_type(16, 2, [40, 40, 40], 3, .1, skip_connection=False).to(device)


def normalize(x, stats):
    return ((x-np.asarray(stats['input_mean']))/np.asarray(stats['input_std'])).astype(np.float32)


def ensemble_moments(means, logstd, stats):
    """Moment match in PHYSICAL units, including both sources of uncertainty."""
    scale = torch.as_tensor(stats['output_std'], device=means.device, dtype=means.dtype)
    center = torch.as_tensor(stats['output_mean'], device=means.device, dtype=means.dtype)
    physical = means*scale+center
    within = torch.exp(2*logstd.clamp(min=float(np.log(1e-6))))*scale.square()
    mean = physical.mean(dim=0)
    # Centered form avoids catastrophic cancellation of E[mu^2]-E[mu]^2.
    variance = (within+(physical-mean).square()).mean(dim=0)
    return mean, variance.clamp(min=1e-12)


@torch.no_grad()
def predict(models, x, stats, device, chunk=512):
    """Continuous hidden states, also used for full-trajectory validation."""
    states = [None]*len(models); means=[]; variances=[]
    for start in range(0, len(x), chunk):
        a = torch.as_tensor(x[start:start+chunk], dtype=torch.float32, device=device)[None]
        mus=[]; logs=[]
        for i, m in enumerate(models):
            m.eval(); mu, log, h = m(a, states[i]); states[i]=h
            mus.append(mu[0]); logs.append(log[0])
        mean, var = ensemble_moments(torch.stack(mus), torch.stack(logs), stats)
        means.append(mean.cpu().numpy()); variances.append(var.cpu().numpy())
    return np.concatenate(means), np.concatenate(variances)


def load_ensemble(checkpoint=None, device='cpu'):
    p = Path(checkpoint or HERE/'runs/best.pt')
    payload = torch.load(p, map_location='cpu', weights_only=False)
    members=[]
    for state in payload['models']:
        m=model(device); m.load_state_dict(state); m.eval(); members.append(m)
    return members, payload


def observations(record, models, stats, device):
    ends, raw=feature_points(record)
    mean,var=predict(models, normalize(raw, stats), stats, device)
    return ends,mean,var
