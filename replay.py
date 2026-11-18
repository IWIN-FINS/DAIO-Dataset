"""Replay one new Test trajectory with a released checkpoint. Run from any directory."""
import argparse
import json
import sys
from pathlib import Path

import numpy as np
import torch
import yaml

ROOT = Path(__file__).resolve().parent
NEW = ROOT / 'new-dataset'


def daio(path, method, device, checkpoint=None):
    sys.path.insert(0, str(ROOT/'ROV-ESKF-Odometry/src'))
    from rov_eskf.data import load_sequence
    from rov_eskf.evaluate import offline_reference, state_projection
    from rov_eskf.model import load_checkpoint
    from rov_eskf.replay import replay_attitude_aided_11
    from rov_eskf.windows import network_observations
    record = load_sequence(path, require_reference=True)
    xy, yaw, vel = offline_reference(record, record.imu_ts, 101, 3)
    cfg = yaml.safe_load((ROOT/'ROV-ESKF-Odometry/configs/benchmark_paper_test1_4_current_model.yaml').read_text(encoding='utf8'))
    selected = checkpoint or (ROOT/'models/new/horizontal_rpm.pt' if method == 'daio' else ROOT/'models/combined/no_rpm.pt')
    print(f'Checkpoint: {selected}; see models/index.csv for training-data provenance')
    net, ckpt = load_checkpoint(selected, device)
    obs = network_observations(record, net, ckpt, device, batch_size=128)
    state = replay_attitude_aided_11(record, cfg, network=obs, reanchor_yaw=float(yaw[1600]), reanchor_time=8.)
    scored = state_projection(state.state)[1600:]
    pos = scored[:, :2] + (xy[1600] - scored[0, :2])
    return np.column_stack([record.imu_ts[1600:],xy[1600:],vel[1600:],yaw[1600:],pos,scored[:, 3:5],scored[:, 6]])


def deepvl(path, device):
    sys.path[:0] = [str(NEW/'deepvl'),str(NEW)]
    import common as deep
    from deepvl_filter_runtime import replay
    record = deep.load_record(path)
    xy, yaw, vel, _ = deep.reference(deep.load_record(path, reference=True))
    members, ckpt = deep.load_ensemble(ROOT/'models/deepvl/ensemble.pt', device)
    end, mean, var = deep.observations(record, members, ckpt['normalization'], device)
    cfg = json.loads((NEW/'deepvl_filter_recovery_v3/selected_filter_config.json').read_text(encoding='utf8'))
    state, _, _, _ = replay(record, end, mean, var, yaw[1600], xy[1600], cfg)
    return np.column_stack([record.time,xy,vel,yaw,state[:, :2],state[:, 3:5],state[:, 6]])[1600:]


def airio(path, device, end_s):
    sys.path.insert(0, str(NEW))
    from airio_six_probes import common, data, make_model
    import runtime_filter
    from scipy.spatial.transform import Rotation
    raw = common.c.read_source(path)
    ref = common.c.make_reference(raw)
    stop = len(raw['time']) if end_s is None else np.searchsorted(raw['time'],end_s,side='right')
    imu = {k:raw[k][:stop].astype(np.float32) for k in ('acc','gyro')}
    imu['time'] = raw['time'][:stop]
    imu_net = common.vendor.airimu(device)
    imu_net.load_state_dict(torch.load(ROOT/'models/airio/airimu.pt',map_location='cpu',weights_only=False)['model'])
    bias = json.loads((ROOT/'models/airio/warmup_calibration.json').read_text(encoding='utf8'))['gyro_bias']
    ckpt = torch.load(ROOT/'models/airio/airio.pt',map_location='cpu',weights_only=False)
    net = make_model(ckpt['spec'],42,device)
    net.load_state_dict(ckpt['model'])
    ori = common.causal.propagate(imu, imu_net.eval(), bias, device)
    ix, mean, var = data.causal_motion(imu,ori['rotvec'],net.eval())
    yaw = ref['reference_rpy'][:,2]
    replay = runtime_filter.replay(imu,ori,ix,mean,var,yaw[1600])
    state = replay['state'][1600:]
    pos = state[:,6:8] + (ref['position_world'][1600,:2] - state[0,6:8])
    angle = Rotation.from_quat(replay['quaternion'][1600:]).as_euler('xyz')[:,2]
    score = slice(1600,stop)
    return np.column_stack([imu['time'][1600:],ref['position_world'][score,:2],ref['velocity_world'][score,:2],yaw[score],pos,state[:,3:5],angle])


if __name__ == '__main__':
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--method',choices=['daio','no_rpm','deepvl','airio'],required=True)
    p.add_argument('--test',type=int,choices=range(1,5),required=True)
    p.add_argument('--device',choices=['cpu','cuda'],default='cpu')
    p.add_argument('--airio-end-s',type=float,help='Optional short AirIO prefix for an installation check')
    p.add_argument('--output',type=Path)
    p.add_argument('--checkpoint',type=Path,help='DAIO/no_rpm checkpoint, e.g. a newly trained best.pt')
    a = p.parse_args()
    if a.airio_end_s is not None and (a.method != 'airio' or a.airio_end_s <= 8):
        p.error('--airio-end-s requires AirIO and a time greater than 8 seconds')
    if a.checkpoint and a.method not in ('daio','no_rpm'):p.error('--checkpoint is only for DAIO/no_rpm')
    torch.set_num_threads(4)
    files = yaml.safe_load((NEW/'processed/split.yaml').read_text(encoding='utf8'))['splits']['demo']['files']
    path = NEW/'processed'/files[a.test-1]
    if a.method == 'airio':
        output = airio(path,a.device,a.airio_end_s)
    elif a.method == 'deepvl':
        output = deepvl(path,a.device)
    else:
        output = daio(path,a.method,a.device,a.checkpoint)
    suffix = '' if a.airio_end_s is None else '_prefix'
    dest = a.output or ROOT/'outputs'/f'{a.method}_test{a.test}{suffix}.csv'
    dest.parent.mkdir(exist_ok=True,parents=True)
    np.savetxt(dest,output,fmt='%.17g',delimiter=',',header='time_s,gt_x_m,gt_y_m,gt_vx_mps,gt_vy_mps,gt_yaw_rad,x_m,y_m,vx_mps,vy_mps,yaw_rad',comments='')
    print(f'{len(output)} scored samples -> {dest}')
