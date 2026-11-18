"""Qualified IMU-standardized AirIO: isolated formal training and gated tests."""
from pathlib import Path
import argparse
import contextlib
import csv
import gc
import os
import sys
import traceback

HERE = Path(__file__).resolve().parent
DATASET = HERE.parent
sys.path.insert(0, str(DATASET))
import airio_six_probes as probes
from airio_six_probes import common, data, metrics, trainer
import numpy as np
import torch

REPORT = common.c.ROOT / 'ROV-ESKF-Odometry/reports/airio_3d_formal_scaled'
EVIDENCE = DATASET / 'airio_six_probes'


@contextlib.contextmanager
def process_lock():
    import msvcrt
    with (HERE / 'pipeline.lock').open('a+b') as file:
        file.seek(0)
        file.write(b'0')
        file.flush()
        file.seek(0)
        try:
            msvcrt.locking(file.fileno(), msvcrt.LK_NBLCK, 1)
        except OSError as exc:
            raise RuntimeError('This formal pipeline is already running') from exc
        try:
            yield
        finally:
            file.seek(0)
            msvcrt.locking(file.fileno(), msvcrt.LK_UNLCK, 1)


def evidence_context():
    legacy = common.context()
    common.require_preflight(legacy)
    decision = common.read(EVIDENCE / 'decision.json')
    verified = common.read(EVIDENCE / 'verified_candidate.json')
    scaling = common.read(EVIDENCE / 'scaling.json')
    results = common.read(EVIDENCE / 'results.json')
    expected = common.digest([legacy['signature'], common.c.sha(probes.__file__)])
    if decision['signature'] != expected or decision['original_context'] != legacy['signature']:
        raise RuntimeError('Diagnostic source/data context changed')
    if decision['solution'] != 'imu_scaling' or not verified['future_invariance']:
        raise RuntimeError('No verified causal IMU-standardization solution')
    if not verified['repeated']['qualification']['passed']:
        raise RuntimeError('Diagnostic qualification failed')
    arm = next(r for r in results if r['arm'] == 'imu_scaling')
    if not arm['passed']:
        raise RuntimeError('Diagnostic confirmation failed')
    spec = verified['spec']
    if spec['mean'] != scaling['mean'] or spec['scale'] != scaling['scale']:
        raise RuntimeError('Diagnostic preprocessing mismatch')
    if set(scaling['sources']) != {str(p) for p in common.c.split_paths()['train']}:
        raise RuntimeError('Normalization must use training records only')
    if spec['weight_decay'] != .0001 or spec['endpoint_weight'] != 0 or spec['dtype'] != 'float32':
        raise RuntimeError('Unexpected candidate settings')
    for phase in ('42', '43', 'pilot'):
        r = arm['runs'][phase]
        if not r['checks']['passed'] or r['score'] >= r['baseline'] - .0001:
            raise RuntimeError('Single-factor improvement not confirmed: ' + phase)
    fit = arm['runs']['fit']
    if fit['fixed_best'] > .5 * fit['fixed_initial']:
        raise RuntimeError('Fixed-window fit gate failed')
    evidence = {str(EVIDENCE / name): common.c.sha(EVIDENCE / name) for name in
                ('decision.json', 'verified_candidate.json', 'scaling.json', 'results.json')}
    for r in arm['runs'].values():
        if common.c.sha(r['checkpoint']) != r['sha256']:
            raise RuntimeError('Diagnostic checkpoint changed')
        evidence[r['checkpoint']] = r['sha256']
    if common.c.sha(verified['checkpoint']) != verified['sha256']:
        raise RuntimeError('Pilot candidate checkpoint changed')
    settings = dict(common.CONFIG, auto_start_formal=True)
    ctx = dict(legacy_signature=legacy['signature'], diagnostic_signature=expected,
               code={str(Path(__file__)): common.c.sha(__file__),
                     str(Path(probes.__file__)): common.c.sha(probes.__file__)},
               evidence=evidence, spec=spec, settings=settings,
               authorization='User requested formal training and testing on 2026-11-04',
               initialization='fresh seed 42; diagnostic weights only used for preflight')
    ctx['signature'] = common.digest(ctx)
    target = HERE / 'formal_config.lock.json'
    if target.exists():
        if common.read(target)['signature'] != ctx['signature']:
            raise RuntimeError('Formal source/config/evidence changed; refusing automatic resume')
    else:
        common.write(target, ctx)
    return ctx


def configure(ctx):
    trainer.HERE = HERE
    trainer.CONFIG = ctx['settings']
    trainer.Bank = data.Bank
    trainer.model = probes.make_model


def formal_lock(ctx):
    lock = common.read(HERE / 'formal_config.lock.json')
    if lock['signature'] != ctx['signature']:
        raise RuntimeError('Stale formal lock')
    return lock


def preflight(ctx, device):
    cached = HERE / 'preflight.json'
    if cached.exists():
        saved = common.read(cached)
        if saved['signature'] == ctx['signature'] and saved['passed']:
            return saved
    # Re-fit from train only and reproduce the diagnostic checkpoint's deployed score.
    fitted = probes.fit_scaling()
    np.testing.assert_allclose(fitted['mean'], ctx['spec']['mean'], atol=1e-12, rtol=0)
    np.testing.assert_allclose(fitted['scale'], ctx['spec']['scale'], atol=1e-12, rtol=0)
    net = probes.make_model(ctx['spec'], 42, device)
    candidate = common.read(EVIDENCE / 'verified_candidate.json')
    payload = torch.load(candidate['checkpoint'], map_location='cpu', weights_only=False)
    net.load_state_dict(payload['model'])
    val = data.Bank('val', device, torch.float32)
    gate = metrics.validation(net, val)
    if not gate['qualification']['passed']:
        raise RuntimeError('Candidate no longer passes deployment validation')
    np.testing.assert_allclose(gate['xyz_velocity_rmse'], candidate['repeated']['xyz_velocity_rmse'], atol=1e-7, rtol=0)
    # Shared hook must be equivalent to explicitly standardizing the exact same raw batch.
    raw, rotation, _ = val.batch(np.arange(2))
    plain = common.model(ctx['spec'], 42, device)
    plain.load_state_dict({k: v for k, v in payload['model'].items() if k not in ('imu_mean', 'imu_scale')})
    x = (torch.cat([raw['acc'], raw['gyro']], -1) - net.imu_mean) / net.imu_scale
    plain.eval()
    net.eval()
    with torch.no_grad():
        direct = plain(dict(acc=x[..., :3], gyro=x[..., 3:]), rotation)
        shared = net(raw, rotation)
    for key in ('net_vel', 'cov'):
        torch.testing.assert_close(direct[key], shared[key], atol=0, rtol=0)
    output = dict(passed=True, signature=ctx['signature'], candidate_validation=gate,
                  normalization_refit=True, preprocessing_parity=True, train_from_scratch=True,
                  time=common.now())
    common.write(cached, output)
    del net, plain, val, payload
    gc.collect()
    torch.cuda.empty_cache()
    print('PREFLIGHT PASSED', flush=True)
    return output


def training_report():
    import reporting
    reporting.HERE = HERE
    reporting.REPORT = REPORT
    common.write(HERE / 'diagnosis_result.json', dict(ready=True, reason='imu_standardization_confirmed'))
    reporting.diagnostic_report()


def evaluate(ctx, device):
    import evaluation
    evaluation.HERE = HERE
    evaluation.REPORT = REPORT
    evaluation.model = probes.make_model
    evaluation.formal_lock = formal_lock
    result = evaluation.run(ctx, device)
    # Describe the shared input transform explicitly; the filter still sees physical IMU units.
    result['normalization'] = dict(mean=ctx['spec']['mean'], scale=ctx['spec']['scale'],
        scope='AirIO velocity network only; raw physical IMU and AirIMU correction unchanged')
    common.write(REPORT / 'evaluation/provenance.json', result)
    return result


def network_tests(ctx, device):
    """Save direct causal network performance separately from fused odometry."""
    import evaluation
    from scipy.spatial.transform import Rotation
    evaluation.HERE = HERE
    evaluation.REPORT = REPORT
    evaluation.model = probes.make_model
    evaluation.formal_lock = formal_lock
    net, checkpoint, _ = evaluation.qualified_model(ctx, device)
    out = REPORT / 'network'
    out.mkdir(parents=True, exist_ok=True)
    rows = []
    for number, path in enumerate(common.c.split_paths()['demo'], 1):
        ref = common.c.load_record(path)
        runtime = common.c.load_runtime(path)
        ori = data.orientation(path)
        ix, body, var = data.causal_motion(runtime, ori['rotvec'], net)
        keep = ix >= 1600
        ix, body, var = ix[keep], body[keep], var[keep]
        q = Rotation.from_quat(ori['quaternion'])
        delta = ref['reference_rpy'][1600, 2] - q[1600].as_euler('xyz')[2]
        world = Rotation.from_euler('z', delta).apply(q[ix].apply(body))
        truth = ref['velocity_world'][ix]
        err = world - truth
        rows.append(dict(test=f'Test {number}', samples=len(ix),
            velocity_x_rmse=float(np.sqrt(np.mean(err[:, 0] ** 2))),
            velocity_y_rmse=float(np.sqrt(np.mean(err[:, 1] ** 2))),
            velocity_z_rmse=float(np.sqrt(np.mean(err[:, 2] ** 2))),
            velocity_xy_rmse=float(np.sqrt(np.mean(np.sum(err[:, :2] ** 2, axis=1)))),
            velocity_xyz_rmse=float(np.sqrt(np.mean(np.sum(err ** 2, axis=1))))))
        common.c.save_npz(out / f'test{number}_network.npz', time=runtime['time'][ix],
            reference_velocity=truth, velocity=world, body_velocity=body, body_variance=var,
            checkpoint_sha256=common.c.sha(checkpoint), context=ctx['signature'])
        np.savetxt(out / f'test{number}_network.csv', np.c_[runtime['time'][ix], truth, world],
            delimiter=',', fmt='%.17g', comments='',
            header='time_s,gt_vx,gt_vy,gt_vz,airio_vx,airio_vy,airio_vz')
    rows.append(dict(test='Mean', **{k: float(np.mean([r[k] for r in rows])) for k in rows[0] if k != 'test'}))
    with (out / 'metrics.csv').open('w', encoding='utf-8-sig', newline='') as file:
        writer = csv.DictWriter(file, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    return rows


def status(stage, **extra):
    common.write(HERE / 'status.json', dict(stage=stage, pid=os.getpid(), time=common.now(), **extra))
    print(stage.upper(), flush=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('command', choices=['all', 'preflight', 'train', 'evaluate', 'report', 'status'])
    parser.add_argument('--device', default='cuda')
    args = parser.parse_args()
    if args.command == 'status':
        for name in ('status.json', 'progress.json'):
            if (HERE / name).exists():
                print(name, common.read(HERE / name), flush=True)
        return
    with process_lock():
        try:
            status('verifying', command=args.command)
            ctx = evidence_context()
            configure(ctx)
            if args.command in ('all', 'preflight', 'train'):
                preflight(ctx, args.device)
            if args.command in ('all', 'train'):
                status('training', signature=ctx['signature'], max_epochs=200)
                result = trainer.train_run(ctx, ctx['spec'], 'formal', 42, args.device)
                training_report()
                if result['best_qualified'] is None:
                    raise RuntimeError('Formal training has no qualified checkpoint; tests blocked')
                if result['reason'] != 'budget_complete':
                    raise RuntimeError('Formal training stopped on collapse; tests blocked pending diagnosis')
            if args.command in ('all', 'evaluate'):
                complete = common.read(trainer.run_path('formal', ctx['spec'], 42) / 'complete.json')
                if complete['reason'] != 'budget_complete':
                    raise RuntimeError('Incomplete or collapsed formal training; tests blocked')
                status('testing_network', signature=ctx['signature'])
                network_tests(ctx, args.device)
                status('testing_odometry', signature=ctx['signature'])
                evaluate(ctx, args.device)
                status('exporting_figures', signature=ctx['signature'])
                import reporting
                reporting.HERE = HERE
                reporting.REPORT = REPORT
                reporting.comparison_figures()
            if args.command == 'report':
                training_report()
            status('complete', command=args.command, signature=ctx['signature'], report=str(REPORT))
        except BaseException as exc:
            status('failed', command=args.command, error=repr(exc), traceback=traceback.format_exc())
            raise


if __name__ == '__main__':
    for stream in (sys.stdout, sys.stderr):
        if hasattr(stream, 'reconfigure'):
            stream.reconfigure(encoding='utf-8')
    main()
