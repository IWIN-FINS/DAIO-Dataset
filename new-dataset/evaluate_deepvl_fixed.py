"""Fixed DeepVL replay with train-calibrated priors, selected on VAL only."""
from pathlib import Path
import csv,json,sys,time,hashlib
import numpy as np
import torch
DATA=Path(__file__).resolve().parent
sys.path.insert(0,str(DATA/'deepvl'))
import common as c
import filter as core
from deepvl_filter_runtime import replay

OUT=DATA.parent/'ROV-ESKF-Odometry/reports/new_v1v2/deepvl_planar'
CONFIG_DIR=DATA/'deepvl_filter_recovery_v3'


def runtime_signature(checkpoint):
    return hashlib.sha256(json.dumps(dict(checkpoint=c.sha(checkpoint),
       network_contract=c.sha(c.HERE/'common.py'),original_core=c.sha(c.HERE/'filter.py'),
       calibrated_runtime=c.sha(DATA/'deepvl_filter_runtime.py'),
       calibration=c.sha(CONFIG_DIR/'train_sensor_calibration.json'),
       config=c.sha(CONFIG_DIR/'selected_filter_config.json'),
       selection=c.sha(CONFIG_DIR/'selection.json'),evaluator=c.sha(Path(__file__))),sort_keys=True).encode()).hexdigest()


def evaluate():
    selection=json.loads((CONFIG_DIR/'selection.json').read_text(encoding='utf-8'))
    assert selection['selected_before_test'] and not selection['test_data_used']
    for key,file in [('config_sha256','selected_filter_config.json'),('calibration_sha256','train_sensor_calibration.json'),
                     ('validation_results_sha256','validation_candidates.json')]:
        if selection[key]!=c.sha(CONFIG_DIR/file):raise RuntimeError('Frozen validation selection changed')
    if selection['runtime_sha256']!=c.sha(DATA/'deepvl_filter_runtime.py'):raise RuntimeError('Runtime changed after validation selection')
    calibration=json.loads((CONFIG_DIR/'train_sensor_calibration.json').read_text(encoding='utf-8'))
    if not calibration.get('independent_training_yaw') or not calibration.get('hierarchical_bias_prior'):
        raise RuntimeError('Require independent TRAIN yaw and record-level bias prior; internal AHRS yaw is not a calibration reference')
    allowed={str(p) for p in c.splits()['train']}
    for source in calibration['sources']:
        path = (DATA.parent / source['path']).resolve()
        if str(path) not in allowed or c.sha(path)!=source['sha256']:raise RuntimeError('Calibration source mismatch/leakage')
    config=json.loads((CONFIG_DIR/'selected_filter_config.json').read_text(encoding='utf-8'))
    checkpoint=c.HERE/'runs/best.pt';device='cuda' if torch.cuda.is_available() else 'cpu';torch.set_num_threads(4)
    models,payload=c.load_ensemble(checkpoint,device)
    if payload['context']!=c.fingerprint():raise RuntimeError('Frozen DeepVL model/data source changed')
    signature=runtime_signature(checkpoint)
    OUT.mkdir(exist_ok=True,parents=True);timings=[];findings=[]
    for test,path in enumerate(c.splits()['demo'],1):
        dest=OUT/f'test{test}_samples.npz'
        if dest.exists():
            with np.load(dest) as z:
                if str(z['method_signature'])!=signature or str(z['source_sha256'])!=c.sha(path):raise RuntimeError('Stale DeepVL output; preserve prior run before replacement')
        else:
            online=c.load_record(path,reference=False);offline=c.load_record(path,reference=True)
            xy,yaw,_,_=c.reference(offline)
            begin=time.perf_counter();ends,mu,var=c.observations(online,models,payload['normalization'],device)
            inference=time.perf_counter()-begin;begin=time.perf_counter()
            projection,state,cov,events=replay(online,ends,mu,var,yaw[1600],xy[1600],config)
            np.savez_compressed(OUT/f'test{test}_full_state.npz',time=online.time,state=state,covariance_diag=cov,
                observation_times=online.time[ends],observation_mean=mu,observation_variance=var,
                checkpoint_sha256=c.sha(checkpoint),source_sha256=c.sha(path),method_signature=signature)
            np.savez_compressed(dest,time=online.time[1600:],position=projection[1600:,:2],velocity=projection[1600:,3:5],yaw=projection[1600:,6],
                 checkpoint_sha256=c.sha(checkpoint),source_sha256=c.sha(path),method_signature=signature)
            with (OUT/f'test{test}_updates.csv').open('w',encoding='utf-8-sig',newline='') as f:
                w=csv.DictWriter(f,fieldnames=['time','kind','accepted','nis','residual_x','residual_y','variance_x','variance_y']);w.writeheader();w.writerows(events)
            timings.append(dict(test=test,network_seconds=inference,filter_seconds=time.perf_counter()-begin))
        saved=np.load(dest);full=np.load(OUT/f'test{test}_full_state.npz')
        events=list(csv.DictReader((OUT/f'test{test}_updates.csv').open(encoding='utf-8-sig')))
        network=[e for e in events if e['kind']=='network_xy' and float(e['time'])>=8]
        finding=dict(test=f'Test {test}',status='stable_after_train_sensor_calibration',
            network_updates=len(network),accepted_updates=sum(e['accepted']=='True' for e in network),
            acceptance_fraction=sum(e['accepted']=='True' for e in network)/len(network),
            max_scored_world_xy_speed_mps=float(np.max(np.linalg.norm(saved['velocity'],axis=1))),
            maximum_gyro_bias_abs=np.max(abs(full['state'][:,9:12]),axis=0).tolist(),
            covariance_diagonal_min=float(full['covariance_diag'].min()))
        if finding['max_scored_world_xy_speed_mps']>=2 or finding['acceptance_fraction']<.95:
            raise RuntimeError('Test replay remains unstable; do not mark recovery complete: '+str(finding))
        findings.append(finding);print('DeepVL repaired',finding,flush=True)
    original=json.loads((DATA/'deepvl_filter_recovery/original_failure_provenance.json').read_text(encoding='utf-8'))
    provenance=dict(original,adaptation='Official DeepVL ensemble + original ReAqROVIO core with TRAIN-calibrated IMU bias prior/noise; VAL selection; heading XY observation',
         filter_config=config,filter_config_sha256=c.sha(CONFIG_DIR/'selected_filter_config.json'),
         filter_calibration=str(CONFIG_DIR/'train_sensor_calibration.json'),filter_selection=selection,
         runtime_sources={str(p.relative_to(DATA)):c.sha(p) for p in (DATA/'deepvl_filter_runtime.py',Path(__file__),c.HERE/'filter.py')},
         method_signature=signature,timings=timings,network_weights_unchanged=True)
    c.atomic_json(OUT/'provenance.json',provenance)
    c.atomic_json(OUT/'filter_diagnostics.json',dict(checkpoint_sha256=c.sha(checkpoint),findings=findings,
         conclusion='Recovered using train-calibrated IMU bias priors/noise selected on validation only; no test tuning.',
         original_failure_archive='diagnostic_deepvl_original_priors_20261102'))
    with (OUT/'filter_diagnostics.csv').open('w',encoding='utf-8-sig',newline='') as f:
        w=csv.DictWriter(f,fieldnames=list(findings[0]));w.writeheader();w.writerows(findings)
    return provenance


if __name__=='__main__':evaluate()
