"""Common full-sample scoring for two ESKF variants, planar AirIO and DeepVL."""
from pathlib import Path
import sys,json,csv,copy,time,argparse,hashlib
import numpy as np
import torch,yaml
ROOT=Path(__file__).resolve().parents[1];DATA=ROOT/'new-dataset';BASE=DATA/'baselines'
sys.path[:0]=[str(BASE),str(ROOT/'ROV-ESKF-Odometry/src')]
import planar_common as pc
from rov_eskf.data import load_sequence as load_runtime
from rov_eskf.model import load_checkpoint
from rov_eskf.windows import network_observations
from rov_eskf.replay import replay_attitude_aided_11,save_replay11
from rov_eskf.evaluate import offline_reference,state_projection
from infer_planar import motion_observations
from airio_filter import replay as replay_airio

OUT=ROOT/'ROV-ESKF-Odometry/reports/new_v1v2'

def metrics(position,velocity,yaw,reference_xy,reference_velocity,reference_yaw,mask):
    e=position-reference_xy;v=velocity-reference_velocity
    y=np.arctan2(np.sin(yaw-reference_yaw),np.cos(yaw-reference_yaw))
    lag=1000;valid=mask[lag:]&mask[:-lag]
    r=(position[lag:]-position[:-lag])-(reference_xy[lag:]-reference_xy[:-lag])
    return {'sample_count':int(mask.sum()),'velocity_x_rmse_mps':float(np.sqrt(np.mean(v[mask,0]**2))),
        'velocity_y_rmse_mps':float(np.sqrt(np.mean(v[mask,1]**2))),
        'velocity_xy_vector_rmse_mps':float(np.sqrt(np.mean(np.sum(v[mask]**2,axis=1)))),
        'position_x_rmse_m':float(np.sqrt(np.mean(e[mask,0]**2))),
        'position_y_rmse_m':float(np.sqrt(np.mean(e[mask,1]**2))),
        'position_xy_vector_rmse_m':float(np.sqrt(np.mean(np.sum(e[mask]**2,axis=1)))),
        'final_position_error_m':float(np.linalg.norm(e[-1])),
        'rpe_5s_xy_vector_rmse_m':float(np.sqrt(np.mean(np.sum(r[valid]**2,axis=1)))),
        'yaw_rmse_rad':float(np.sqrt(np.mean(y[mask]**2)))}

def config_for(checkpoint):
    c=yaml.safe_load((ROOT/'ROV-ESKF-Odometry/configs/benchmark_paper_test1_4_current_model.yaml').read_text(encoding='utf-8'))
    c['data'].update(root=str(DATA),manifest=str(DATA/'processed/split.yaml'),records=[])
    c['model'].update(checkpoint=str(checkpoint),device='cuda' if torch.cuda.is_available() else 'cpu')
    return c

def main(methods=('horizontal_rpm','horizontal_command')):
    if 'airio_3d' in methods:
        # Dedicated source-frozen 3D pipeline preserves existing three replays.
        if set(methods)!=set(('horizontal_rpm','horizontal_command','airio_3d','deepvl_planar')):
            raise ValueError('airio_3d promotion requires the complete four-method comparison')
        import subprocess
        subprocess.run([sys.executable,str(DATA/'airio_3d/run_pipeline.py'),'--through','evaluate'],check=True)
        return
    torch.set_num_threads(4);OUT.mkdir(exist_ok=True,parents=True)
    status=json.loads((DATA/'training_status.json').read_text(encoding='utf-8'))
    rows=[];timing=[];sources={}
    manifest={r['processed_relative_path']:r for r in csv.DictReader((DATA/'processed/manifest.csv').open(encoding='utf-8'))}
    paths=pc.resolve_manifest(DATA/'processed/split.yaml',DATA)['demo']
    runtime_sources={str(p.relative_to(ROOT)):pc.file_sha(p) for p in (ROOT/'ROV-ESKF-Odometry/src/rov_eskf').glob('*.py')}
    device='cuda' if torch.cuda.is_available() else 'cpu'
    airio_ckpt=BASE/'runs/airio/best.pt'
    deepvl_source=None
    if 'deepvl_planar' in methods:
        sys.path.insert(0,str(DATA/'deepvl'))
        import importlib.util
        entry=DATA/'evaluate_deepvl_fixed.py' if (DATA/'deepvl_filter_recovery/selection.json').exists() else DATA/'deepvl/evaluate.py'
        spec=importlib.util.spec_from_file_location('deepvl_evaluator',entry)
        deepvl_module=importlib.util.module_from_spec(spec);spec.loader.exec_module(deepvl_module)
        deepvl_source=deepvl_module.evaluate()
    if 'airio_planar' in methods:
        if not airio_ckpt.exists():raise FileNotFoundError('AirIO velocity weights do not exist; choose existing ESKF methods explicitly')
        airio=pc.official_airio()
        airio.load_state_dict(torch.load(airio_ckpt,map_location='cpu',weights_only=False)['model']);airio.eval();airio=airio.to(device)
    for method in methods:
        directory=OUT/method;directory.mkdir(exist_ok=True)
        if method=='deepvl_planar':sources[method]=deepvl_source
        elif method!='airio_planar':
            checkpoint=Path(status[method]['run_dir'])/'best.pt'
            model,payload=load_checkpoint(checkpoint,device);c=config_for(checkpoint)
            (directory/'config.yaml').write_text(yaml.safe_dump(c,sort_keys=False),encoding='utf-8')
            sources[method]={'checkpoint':str(checkpoint),'sha256':pc.file_sha(checkpoint),'epoch':payload['epoch']}
        else:sources[method]={'checkpoint':str(airio_ckpt),'sha256':pc.file_sha(airio_ckpt),'adaptation':'XY/yaw supervision and 2D velocity observation'}
        signature=hashlib.sha256(json.dumps({'method':method,'checkpoint_sha256':sources[method]['sha256'],'runtime':runtime_sources,
            'evaluation_sha256':pc.file_sha(Path(__file__))},sort_keys=True).encode()).hexdigest()
        accepted_signatures={signature}
        # Read-only migration for the three already-scored methods. Their NPZ
        # files must remain byte-identical. Accept the original evaluator only
        # when its archived runtime hashes are still EXACTLY the current ones.
        archive=DATA/'deepvl/previous_evaluation'
        prior_path=archive/'ROV-ESKF-Odometry/reports/new_v1v2/provenance.json'
        if method!='deepvl_planar' and prior_path.exists():
            prior=json.loads(prior_path.read_text(encoding='utf-8'))
            if prior['runtime_sources']==runtime_sources and prior['models'][method]['sha256']==sources[method]['sha256']:
                accepted_signatures.add(hashlib.sha256(json.dumps({'method':method,'checkpoint_sha256':sources[method]['sha256'],
                    'runtime':runtime_sources,'evaluation_sha256':prior['evaluation_source']},sort_keys=True).encode()).hexdigest())
        for index,path in enumerate(paths,1):
            seq=load_runtime(path,require_reference=True);ref_xy,ref_yaw,ref_velocity=offline_reference(seq,seq.imu_ts,101,3)
            output=directory/f'test{index}_samples.npz'
            if output.exists():
                saved=np.load(output)
                if str(saved['checkpoint_sha256'])!=sources[method]['sha256'] or str(saved['source_sha256'])!=pc.file_sha(path):raise RuntimeError('Existing evaluation mismatch')
                if method!='deepvl_planar' and ('method_signature' not in saved or str(saved['method_signature']) not in accepted_signatures):raise RuntimeError('Runtime implementation changed; archive prior diagnostic outputs before recomputing')
                time_s=saved['time'];position=saved['position'];velocity=saved['velocity'];yaw=saved['yaw']
            else:
                if method=='deepvl_planar':raise RuntimeError('DeepVL adapter did not produce the common sample interface')
                begin=time.perf_counter()
                if method!='airio_planar':
                    net=network_observations(seq,model,payload,device,batch_size=128)
                    full=replay_attitude_aided_11(seq,c,network=net,reanchor_yaw=float(ref_yaw[1600]),reanchor_time=8.0)
                    states=state_projection(full.state)
                    save_replay11(full,directory/f'test{index}_replay',save_covariance=True,save_event_trace=True)
                else:
                    cache_path=BASE/'orientation_cache'/path.parent.name/(path.stem+'.npz')
                    cache=np.load(cache_path)
                    if str(cache['checkpoint_sha256'])!=pc.file_sha(BASE/'runs/airimu/best.pt'):raise RuntimeError('Stale AirIMU cache')
                    t,mean,var=motion_observations(seq,cache,airio,device)
                    states,cov=replay_airio(seq,cache,t,mean,var,float(ref_yaw[1600]))
                    np.savez_compressed(directory/f'test{index}_full_state.npz',time=seq.imu_ts,state=states,covariance_diag=cov,observation_times=t,observation_mean=mean,observation_variance=var)
                time_s=seq.imu_ts[1600:];position=states[1600:,:2].copy();velocity=states[1600:,3:5];yaw=states[1600:,6]
                position+=ref_xy[1600]-position[0]
                np.savez_compressed(output,time=time_s,position=position,velocity=velocity,yaw=yaw,
                                    checkpoint_sha256=sources[method]['sha256'],source_sha256=pc.file_sha(path),method_signature=signature)
                timing.append({'method':method,'test':index,'replay_seconds':time.perf_counter()-begin})
            mask=np.ones(len(time_s),bool)
            rec=manifest[path.relative_to(DATA/'processed').as_posix()]
            repaired=np.asarray(json.loads(rec['repaired_sample_indices']),dtype=int)-1600
            repaired=repaired[(repaired>=0)&(repaired<len(mask))]
            without=mask.copy();without[repaired]=False
            for label,m in [('all_samples',mask),('without_repaired_samples',without)]:
                rows.append({'method':method,'test':f'Test {index}','scoring':label,'start_s':8.,'end_s':float(time_s[-1]),
                    **metrics(position,velocity,yaw,ref_xy[1600:],ref_velocity[1600:],ref_yaw[1600:],m)})
            print(method,'Test',index,rows[-2],flush=True)
    for method in methods:
        for score in ('all_samples','without_repaired_samples'):
            selected=[r for r in rows if r['method']==method and r['scoring']==score]
            rows.append({'method':method,'test':'Mean','scoring':score,**{k:float(np.mean([r[k] for r in selected])) for k in selected[0] if k.endswith(('_m','_mps','_rad'))}})
    with (OUT/'metrics.csv').open('w',encoding='utf-8-sig',newline='') as f:
        writer=csv.DictWriter(f,fieldnames=list(rows[0]));writer.writeheader();writer.writerows(rows)
    pc.atomic_json(OUT/'provenance.json',{'models':sources,'evaluated_methods':list(methods),'runtime_sources':runtime_sources,'data_manifest':pc.file_sha(DATA/'processed/manifest.csv'),
                    'evaluation_source':pc.file_sha(Path(__file__)),'timings':timing,'alignment':'once at 8s, yaw and XY only; no scale or repeated alignment'})
    (OUT/'metric_definitions.md').write_text('''# Metric definitions
All scoring uses 200 Hz samples from 8 s to record end. Position is translated once at 8 s, preserving the AprilTag world origin. Yaw is anchored once by the benchmark layer. No truth velocity is used to initialize inference.

Let e_p = p_hat - p_GT and e_v = v_hat - v_GT in the world XY plane.
Component RMSE = sqrt(mean(e_component^2)); vector RMSE = sqrt(mean(e_x^2+e_y^2)). These differ from sqrt(mean over both components) by sqrt(2).
Final position error = norm(e_p[-1]). Five-second relative displacement RMSE = sqrt(mean(norm((p_hat[t+5]-p_hat[t])-(p_GT[t+5]-p_GT[t]))^2)), using all overlapping pairs at a 1000-sample offset.
Yaw RMSE uses wrapped yaw residuals in radians. Mean is the arithmetic mean of four per-trajectory metrics, not a pooled statistic.
The second score excludes inserted/repaired endpoints, including both endpoints for relative displacement. Final position error still describes the actual recorded endpoint. Labels use SG101/poly3. No extra display smoothing changes these metrics.
AirIO is a planar ROV adaptation of the pinned official method, not an unmodified full-3D benchmark. Its only online sensors are six-axis IMU. The two DAIO ESKF variants also use depth and solved IMU yaw; this is a system comparison, while the RPM-vs-command pair isolates the feedback input.
DeepVL (Planar ROV) uses the pinned official 8-GRU ensemble and a port of ReAqROVIO's no-vision inertial core, with heading XY velocity supervision/update. Online sensors are six-axis IMU, 8 commands, battery voltage and relative depth. Network updates are 20 Hz; IMU propagation/scoring is 200 Hz. It does not use RPM or ongoing solved/AprilTag yaw. This comparison includes different sensors, backends and observation rates; it is not a controlled network-only ablation.
For repaired DeepVL, IMU gyro bias priors and white-noise PSDs are calibrated on TRAIN raw IMU, IMU-derived tilt and independent AprilTag yaw only, then frozen using VAL 026/027. IMU-internal yaw is not used as an absolute calibration reference. Bias uncertainty uses between-record calibration variability, not noisy per-sample yaw-rate residuals. No test data estimate/tune parameters. Network weights and predictions are unchanged. The original failure is archived separately.
''',encoding='utf-8')

if __name__=='__main__':
    parser=argparse.ArgumentParser()
    parser.add_argument('--methods',nargs='+',choices=['horizontal_rpm','horizontal_command','airio_planar','airio_3d','deepvl_planar'],default=['horizontal_rpm','horizontal_command'])
    args=parser.parse_args();main(tuple(args.methods))
