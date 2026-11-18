"""Six single-factor AirIO diagnostic arms. Original dataset, code, and checkpoints are read-only."""
from pathlib import Path
import argparse
import sys
import numpy as np
import torch
from torch.nn import functional as F

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT / 'airio_3d_recovery'))
import common
import data
import trainer
import metrics

OUT = ROOT / 'airio_six_probes'
BASE = common.CANDIDATES[1]
ARMS = ['imu_scaling', 'training_pose', 'causal_average', 'tilt_only', 'encoder_dropout', 'endpoint_supervision']


def fit_scaling():
    total = np.zeros(6)
    square = np.zeros(6)
    count = 0
    sources = []
    for path in common.c.split_paths()['train']:
        r = common.c.load_runtime(path)
        vals = np.column_stack([r['acc'], r['gyro']]).astype(float)
        total += vals.sum(0)
        square += (vals ** 2).sum(0)
        count += len(vals)
        sources.append(str(path))
    mean = total / count
    scale = np.sqrt(square / count - mean**2)
    assert np.isfinite(scale).all() and min(scale) > 1e-5
    return dict(samples=count, sources=sources, mean=mean.tolist(), scale=scale.tolist())


def tilt_vector(r):
    # Shortest rotation that takes the body gravity direction back to world up.
    theta = r.norm(dim=-1, keepdim=True)
    up = torch.zeros_like(r)
    up[..., 2] = 1
    cross = torch.linalg.cross(r, up)
    body_up = up - cross * torch.sinc(theta / torch.pi) + torch.linalg.cross(r, cross) * (.5 * torch.sinc(theta / (2 * torch.pi)).square())
    axis = torch.linalg.cross(body_up, up)
    sine = axis.norm(dim=-1, keepdim=True)
    return axis * (torch.atan2(sine, body_up[..., 2:3]) / sine.clamp_min(1e-12))


def moving_average(a):
    x = a.transpose(1, 2)
    return F.avg_pool1d(F.pad(x, (19, 0), mode='replicate'), 20, stride=1).transpose(1, 2)


def make_model(spec, seed, device):
    net = common.model(spec, seed, device)
    mode = spec['arm']
    if mode == 'imu_scaling':
        net.register_buffer('imu_mean', torch.tensor(spec['mean'], device=device))
        net.register_buffer('imu_scale', torch.tensor(spec['scale'], device=device))
        def hook(module, args):
            d, r = args
            x = (torch.cat([d['acc'], d['gyro']], -1) - module.imu_mean) / module.imu_scale
            return dict(acc=x[..., :3], gyro=x[..., 3:]), r
        net.register_forward_pre_hook(hook)
    elif mode == 'causal_average':
        net.register_forward_pre_hook(lambda m, args: ({k: moving_average(v) for k,v in args[0].items()}, args[1]))
    elif mode == 'tilt_only':
        net.register_forward_pre_hook(lambda m, args: (args[0], tilt_vector(args[1])))
    elif mode == 'encoder_dropout':
        for m in net.modules():
            if isinstance(m, torch.nn.Dropout):
                m.p = 0.
    return net


class CausalTrainingBank(data.Bank):
    def __init__(self, split, device, dtype):
        super().__init__(split, device, dtype)
        if split != 'train':
            return
        for row in self.records:
            path = Path(row['source'])
            ori = data.orientation(path)
            start = row['offset']
            assert len(ori['rotvec']) == row['samples']
            self.arrays['rotvec'][start:start+row['samples']] = torch.as_tensor(ori['rotvec'], device=device, dtype=dtype)


def use_arm(arm):
    trainer.HERE = OUT
    trainer.Bank = CausalTrainingBank if arm == 'training_pose' else data.Bank
    trainer.model = make_model


def verify(result, baseline, spec, phase, seed, ctx):
    if result['signature'] != common.digest(dict(context=ctx['signature'], spec=spec, phase=phase, seed=seed)):
        raise RuntimeError('Unexpected candidate signature')
    directory = trainer.run_path(phase, spec, seed)
    old_folder = common.HERE/'runs'/phase/BASE['name']/f'seed_{seed}'
    old = common.read(old_folder/'config.json')
    assert old['inventory_signature'] == common.read(directory/'config.json')['inventory_signature']
    gate = result['best_qualified'] or result['best_diagnostic']
    key = 'best_qualified.pt' if result['best_qualified'] else 'best_diagnostic.pt'
    assert common.c.sha(directory/key) == result['checkpoint_hashes'][key]
    v = gate['validation']
    return dict(seed=seed,phase=phase,checkpoint=str(directory/key),sha256=result['checkpoint_hashes'][key],
                score=v['xyz_velocity_rmse'],baseline=baseline['best_diagnostic']['validation']['xyz_velocity_rmse'],
                zero=v['zero_xyz_rmse'],ratio=v['zero_ratio'],records=v['records'],checks=v['qualification'],
                fixed_initial=result['initial']['all_rmse'],fixed_best=result['best_fit']['train']['all_rmse'],
                updates=result['updates'],inventory=old['inventory_signature'])


def main():
    OUT.mkdir(exist_ok=True)
    legacy = common.context()
    common.require_preflight(legacy)
    ctx = dict(signature=common.digest([legacy['signature'],common.c.sha(__file__)]))
    scaling = fit_scaling()
    common.write(OUT/'scaling.json',scaling)
    summary = []
    for arm in ARMS:
        use_arm(arm)
        spec = dict(BASE, arm=arm, name=arm, endpoint_weight=float(arm=='endpoint_supervision'))
        if arm=='imu_scaling':
            spec.update(mean=scaling['mean'],scale=scaling['scale'])
        runs = {}
        def run(phase,seed):
            baseline=common.read(common.HERE/'runs'/phase/BASE['name']/f'seed_{seed}'/'complete.json')
            result=trainer.train_run(ctx,spec,phase,seed)
            return verify(result,baseline,spec,phase,seed,ctx)
        for seed in (42,43):
            runs[str(seed)]=run('representative',seed)
        passed=all(r['checks']['passed'] and r['score'] < r['baseline']-.0001 for r in runs.values())
        if passed:
            runs['fit']=run('fixed',42)
            passed=runs['fit']['fixed_best'] <= .5*runs['fit']['fixed_initial']
        if passed:
            runs['pilot']=run('pilot',42)
            passed=runs['pilot']['checks']['passed'] and runs['pilot']['score'] < runs['pilot']['baseline']-.0001
        if passed:
            # Rebuild the winning model from disk, verify both records and pure past-only inference.
            r=runs['pilot'];net=make_model(spec,42,'cuda')
            net.load_state_dict(torch.load(r['checkpoint'],map_location='cpu',weights_only=False)['model'])
            repeated=metrics.validation(net,data.Bank('val','cuda',torch.float32))
            np.testing.assert_allclose(repeated['xyz_velocity_rmse'],r['score'],atol=1e-7,rtol=0)
            assert repeated['qualification']['passed']
            source=common.c.split_paths()['val'][0]
            rt={k:v[:2400].copy() for k,v in common.c.load_runtime(source).items()}
            rot=data.orientation(source)['rotvec'][:2400].copy()
            ix,a,_=data.causal_motion(rt,rot,net)
            rt['acc'][1800:]+=3;rt['gyro'][1800:]+=1;rot[1800:]+=.5
            j,b,_=data.causal_motion(rt,rot,net)
            np.testing.assert_array_equal(ix,j);np.testing.assert_array_equal(a[ix<=1800],b[ix<=1800])
            common.write(OUT/'verified_candidate.json',dict(spec=spec,checkpoint=r['checkpoint'],sha256=r['sha256'],repeated=repeated,future_invariance=True))
        summary.append(dict(arm=arm,passed=passed,runs=runs))
        common.write(OUT/'results.json',summary)
        if passed:break
    common.write(OUT/'decision.json',dict(attempts=len(summary),solution=summary[-1]['arm'] if summary[-1]['passed'] else None,
            signature=ctx['signature'],original_context=legacy['signature'],formal_training_started=False))
if __name__=='__main__':main()
