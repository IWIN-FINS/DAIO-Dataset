"""Sensor-prior diagnosis on TRAIN and VAL only. No test data tuning."""
from pathlib import Path
import csv,json,sys,copy,time,argparse
import numpy as np
import torch
from scipy.spatial.transform import Rotation
from scipy.signal import savgol_filter
DATA=Path(__file__).resolve().parent
sys.path.insert(0,str(DATA/'deepvl'))
import common as c
import deepvl_filter_runtime as runtime

OUT=DATA/'deepvl_filter_recovery'


def calibrate(independent_yaw=False,hierarchical_prior=False):
    OUT.mkdir(exist_ok=True)
    if (OUT/'train_sensor_calibration.json').exists():
        result=json.loads((OUT/'train_sensor_calibration.json').read_text(encoding='utf-8'))
        for source in result['sources']:
            if c.sha(Path(source['path']))!=source['sha256']:raise RuntimeError('Calibration source changed')
        return result
    residuals=[];second=[];sources=[];per_record=[]
    for path in c.splits()['train']:
        record=c.load_record(path,reference=independent_yaw)
        if len(record.time)<200:continue
        with path.open(encoding='utf-8-sig') as f:header=next(csv.reader(f))
        # Sensor-provided attitude is used OFFLINE ON TRAIN ONLY to characterize
        # raw gyro bias. It is not independent 3D ground truth or a model label,
        # and is never fed into validation/test propagation or initialization.
        rpy=np.loadtxt(path,delimiter=',',skiprows=1,usecols=[header.index(k) for k in ('roll','pitch','yaw')],encoding='utf-8-sig')
        if independent_yaw:
            # Internal AHRS heading drifts and is NOT an absolute heading
            # calibration reference. The existing independent TRAIN yaw label
            # corrects this axis; test-time attitude is still propagated IMU-only.
            width=min(101,len(rpy) if len(rpy)%2 else len(rpy)-1)
            rpy[:,2]=savgol_filter(record.tag_yaw,width,3)
        rotations=Rotation.from_euler('xyz',rpy)
        expected=(rotations[:-1].inv()*rotations[1:]).as_rotvec()/.005
        residual=record.gyro[1:]-expected
        residuals.append(residual);per_record.append(np.median(residual,axis=0))
        second.append(np.diff(np.column_stack((record.accel,record.gyro)),n=2,axis=0))
        sources.append(dict(path=str(path),sha256=c.sha(path)))
    residual=np.concatenate(residuals);bias=np.median(residual,axis=0)
    # Conservative initial uncertainty from raw gyro minus sensor-attitude rate;
    # includes record-to-record changes and quantization, not a presumed zero bias.
    uncertainty=np.quantile(abs(residual-bias),.95,axis=0)
    pooled_bias=bias.copy();sample_spread=uncertainty.copy()
    if hierarchical_prior:
        # Initial bias is a RECORD-level latent parameter. Per-sample residuals
        # also contain differentiated AprilTag noise and real motion; using that
        # spread as bias uncertainty lets velocity errors spuriously steer yaw.
        # Use dispersion of independent per-record calibration estimates instead.
        record_bias=np.asarray(per_record)
        bias=np.median(record_bias,axis=0)
        uncertainty=np.quantile(abs(record_bias-bias),.95,axis=0)
    delta=np.concatenate(second)
    sigma=np.median(abs(delta-np.median(delta,axis=0)),axis=0)/(.6744897501960817*np.sqrt(6))
    result=dict(fit_split='train only',sources=sources,initial_gyro_bias=bias.tolist(),
        gyro_bias_prior_std=np.maximum(uncertainty,1e-4).tolist(),sample_white_noise_std=sigma.tolist(),
        continuous_noise_psd=(sigma**2*.005).tolist(),per_record_bias_median=np.asarray(per_record).tolist(),
        independent_training_yaw=independent_yaw,
        hierarchical_bias_prior=hierarchical_prior,pooled_residual_median=pooled_bias.tolist(),
        per_sample_residual_spread95=sample_spread.tolist(),
        interpretation=('Offline TRAIN raw gyro vs IMU-derived tilt and independent AprilTag yaw; no vertical/zero-tilt truth, no test statistics or online truth' if independent_yaw else 'Offline sensor calibration using training raw IMU and internal attitude; no independent yaw correction'),
        estimator=('Median of per-record gyro-rate bias estimates; 95th between-record deviation for initial bias prior; MAD second difference / sqrt(6) white noise' if hierarchical_prior else 'global median gyro-rate residual; 95th absolute residual prior bound; MAD second difference / sqrt(6) white-noise estimate'))
    c.atomic_json(OUT/'train_sensor_calibration.json',result);return result


def validation_cache():
    device='cuda' if torch.cuda.is_available() else 'cpu';torch.set_num_threads(4)
    models,payload=c.load_ensemble(device=device)
    if payload['context']!=c.fingerprint():raise RuntimeError('Trained model source mismatch')
    arrays=[]
    for index,path in enumerate(c.splits()['val'],1):
        record=c.load_record(path,reference=False);offline=c.load_record(path,reference=True)
        xy,yaw,velocity,_=c.reference(offline)
        cache=OUT/f'val{index}_observations.npz'
        if cache.exists():
            z=np.load(cache);assert str(z['source_sha256'])==c.sha(path)
            assert str(z['checkpoint_sha256'])==c.sha(c.HERE/'runs/best.pt')
            ends=z['ends'];mean=z['mean'];variance=z['variance']
        else:
            ends,mean,variance=c.observations(record,models,payload['normalization'],device)
            np.savez_compressed(cache,ends=ends,mean=mean,variance=variance,
                 source_sha256=c.sha(path),checkpoint_sha256=c.sha(c.HERE/'runs/best.pt'))
        arrays.append((record,ends,mean,variance,xy,yaw,velocity))
    return arrays


def main(independent_yaw=False,hierarchical_prior=False):
    if (OUT/'selection.json').exists():
        selected=json.loads((OUT/'selection.json').read_text(encoding='utf-8'))
        calibration=calibrate(independent_yaw,hierarchical_prior)
        if calibration.get('independent_training_yaw',False)!=independent_yaw or calibration.get('hierarchical_bias_prior',False)!=hierarchical_prior:
            raise RuntimeError('Frozen calibration mode changed; use a NEW --output-dir')
        for key,name in [('calibration_sha256','train_sensor_calibration.json'),('config_sha256','selected_filter_config.json'),('validation_results_sha256','validation_candidates.json')]:
            if selected[key]!=c.sha(OUT/name):raise RuntimeError('Frozen artifact changed: '+name)
        if selected['runtime_sha256']!=c.sha(DATA/'deepvl_filter_runtime.py'):raise RuntimeError('Validated runtime changed')
        for i,path in enumerate(c.splits()['val'],1):
            with np.load(OUT/f'val{i}_observations.npz') as z:
                assert str(z['source_sha256'])==c.sha(path)
                assert str(z['checkpoint_sha256'])==c.sha(c.HERE/'runs/best.pt')
        print('Verified existing frozen selection; no files overwritten:',selected['candidate'],flush=True)
        return
    result=calibrate(independent_yaw,hierarchical_prior);print('Train gyro bias',result['initial_gyro_bias'],'prior std',result['gyro_bias_prior_std'],flush=True)
    base=runtime.original_config();candidates={'original':base}
    prior=copy.deepcopy(base);prior['initial_covariance_diag'][9:12]=(np.asarray(result['gyro_bias_prior_std'])**2).tolist()
    candidates['gyro_prior_uncertainty_only']=prior
    mean=copy.deepcopy(prior);mean['initial_gyro_bias']=result['initial_gyro_bias']
    candidates['gyro_bias_and_prior']=mean
    noise=copy.deepcopy(mean);psd=np.asarray(result['continuous_noise_psd'])
    noise['process_noise_diag'][3:6]=np.maximum(psd[:3],1e-12).tolist()
    noise['process_noise_diag'][12:15]=np.maximum(psd[3:],1e-12).tolist()
    candidates['train_calibrated_imu']=noise
    c.atomic_json(OUT/'candidate_configs.json',candidates)
    validation=validation_cache();rows=[]
    for name,config in candidates.items():
        for i,(record,ends,mu,var,xy,yaw,velocity) in enumerate(validation,1):
            projection,states,cov,events=runtime.replay(record,ends,mu,var,yaw[1600],xy[1600],config)
            p=projection[1600:,:2]-xy[1600:];v=projection[1600:,3:5]-velocity[1600:]
            ne=[e for e in events if e['kind']=='network_xy' and e['time']>=8]
            yaw_error=np.arctan2(np.sin(projection[1600:,6]-yaw[1600:]),np.cos(projection[1600:,6]-yaw[1600:]))
            row=dict(candidate=name,validation=i,velocity_vector_rmse=float(np.sqrt(np.mean(np.sum(v*v,axis=1)))),
                position_vector_rmse=float(np.sqrt(np.mean(np.sum(p*p,axis=1)))),
                yaw_rmse_rad=float(np.sqrt(np.mean(yaw_error**2))),
                accepted_network_fraction=sum(e['accepted'] for e in ne)/len(ne),
                maximum_speed=float(np.linalg.norm(projection[:,3:6],axis=1).max()),
                gyro_bias_final=states[-1,9:12].tolist())
            print(json.dumps(row),flush=True);rows.append(row)
            np.savez_compressed(OUT/f'{name}_val{i}.npz',time=record.time,projection=projection,state=states,covariance_diag=cov)
    c.atomic_json(OUT/'validation_candidates.json',dict(fit_split='train',selection_split='val',test_data_used=False,results=rows))
    eligible=[]
    for name in candidates:
        selected=[r for r in rows if r['candidate']==name]
        if all(r['accepted_network_fraction']>=.95 and r['maximum_speed']<2 for r in selected):
            eligible.append((float(np.mean([r['velocity_vector_rmse'] for r in selected])),name))
    if not eligible:raise RuntimeError('No stable validation configuration; do not proceed to test')
    score,name=min(eligible)
    c.atomic_json(OUT/'selected_filter_config.json',candidates[name])
    c.atomic_json(OUT/'selection.json',dict(candidate=name,validation_velocity_vector_rmse_mean=score,
        selected_before_test=True,test_data_used=False,time_unix=time.time(),
        selection_rule='Lowest mean validation XY velocity vector RMSE among configurations with >=95% network acceptance and <2m/s max speed on both VAL records',
        calibration_sha256=c.sha(OUT/'train_sensor_calibration.json'),validation_results_sha256=c.sha(OUT/'validation_candidates.json'),
        config_sha256=c.sha(OUT/'selected_filter_config.json'),runtime_sha256=c.sha(DATA/'deepvl_filter_runtime.py')))
    print('Selected and frozen using VAL only:',name,score,flush=True)


if __name__=='__main__':
    parser=argparse.ArgumentParser();parser.add_argument('--independent-yaw',action='store_true');parser.add_argument('--hierarchical-prior',action='store_true')
    parser.add_argument('--output-dir',type=Path,help='New directory for an independent recalibration; existing frozen results are read-only')
    args=parser.parse_args()
    if args.independent_yaw:OUT=DATA/'deepvl_filter_recovery_v2'
    if args.hierarchical_prior:
        args.independent_yaw=True;OUT=DATA/'deepvl_filter_recovery_v3'
    if args.output_dir:OUT=args.output_dir.resolve()
    main(args.independent_yaw,args.hierarchical_prior)
