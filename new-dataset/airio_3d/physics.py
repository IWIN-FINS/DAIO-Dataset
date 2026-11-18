"""Validation-only hard gates. Reference states are used for diagnostics, never online."""
import numpy as np
import torch
from scipy.spatial.transform import Rotation
import contracts as c
import vendor,causal
import pypose as pp


def controlled(record,orientation,device):
    starts=np.arange(1600,len(record['time'])-1000,200,dtype=int)
    if len(starts)==0:raise ValueError('No post-warm-up five-second checks')
    errors={};covariance_ok=True
    for name,corrected in [('raw',False),('corrected',True)]:
        velocity=[];position=[]
        for offset in range(0,len(starts),64):
            start=starts[offset:offset+64];ix=start[:,None]+np.arange(1000)[None]
            acc=record['acc'][ix].copy();gyro=record['gyro'][ix].copy()
            if corrected:acc=acc+orientation['correction_acc'][ix];gyro=gyro+orientation['correction_gyro'][ix]
            tensor=lambda a:torch.tensor(a,device=device,dtype=torch.float32)
            initial=dict(pos=tensor(record['position'][start,None]),vel=tensor(record['velocity'][start,None]),rot=pp.SO3(tensor(record['quaternion'][start,None])))
            with torch.no_grad():
                out=vendor.integrate(tensor(acc),tensor(gyro),initial,
                    acc_cov=tensor(orientation['acc_cov'][ix]) if corrected else None,
                    gyro_cov=tensor(orientation['gyro_cov'][ix]) if corrected else None)
            velocity.append(out['vel'][:,-1].cpu().numpy()-record['velocity'][start+1000])
            position.append(out['pos'][:,-1].cpu().numpy()-record['position'][start+1000])
            cov=out['cov'].cpu().numpy();sym=(cov+cov.swapaxes(-1,-2))*.5
            covariance_ok &= bool(np.isfinite(cov).all() and np.max(np.abs(cov-cov.swapaxes(-1,-2)))<1e-3 and np.linalg.eigvalsh(sym).min()>-1e-5)
        v=np.concatenate(velocity);p=np.concatenate(position)
        errors[name]=dict(xyz_velocity_rmse=float(np.sqrt(np.mean(np.sum(v*v,axis=1)))),
            z_velocity_rmse=float(np.sqrt(np.mean(v[:,2]**2))),z_displacement_rmse=float(np.sqrt(np.mean(p[:,2]**2))),windows=len(v))
    return errors,covariance_ok


def validate(model,bias,device):
    reports=[];orientations={};g=c.CONFIG['gates']
    for path in c.split_paths()['val']:
        record=c.load_record(path);orientation=causal.propagate(c.load_runtime(path),model,bias,device)
        orientations[str(path)]=orientation;static=record['static'];nstatic=int(static.sum());after=record['time']>=8.
        corrected=record['acc']+orientation['correction_acc']
        norm_median=float(np.median(np.linalg.norm(corrected[static],axis=1))) if nstatic else None
        correction_median=float(np.median(np.linalg.norm(orientation['correction_acc'][static],axis=1))) if nstatic else None
        predicted=Rotation.from_quat(orientation['quaternion']).as_euler('xyz')[:,:2]
        expected=record['reference_rpy'][:,:2];delta=np.arctan2(np.sin(predicted-expected),np.cos(predicted-expected))[after]*180/np.pi
        rms=np.sqrt(np.mean(delta**2,axis=0));p95_axes=np.percentile(np.abs(delta),95,axis=0);p95=float(np.max(p95_axes))
        errors,covok=controlled(record,orientation,device)
        finite=all(np.isfinite(v).all() for v in orientation.values())
        unit=float(np.max(np.abs(np.linalg.norm(orientation['quaternion'],axis=1)-1)))
        gates=dict(finite=finite,covariance=bool(covok),quaternion=unit<1e-5,
            static_duration=nstatic*c.DT>=g['static_seconds_min'],
            gravity=nstatic>0 and g['static_force_min']<=norm_median<=g['static_force_max'],
            correction=nstatic>0 and correction_median<g['static_correction_median_max'],
            tilt_rms=bool(np.max(rms)<g['tilt_rms_deg_max']),tilt_p95=p95<g['tilt_p95_deg_max'],
            vertical_velocity=errors['corrected']['z_velocity_rmse']<g['vertical_velocity_rmse_max'],
            vertical_displacement=errors['corrected']['z_displacement_rmse']<g['vertical_position_rmse_max'],
            improves_raw=errors['corrected']['xyz_velocity_rmse']<errors['raw']['xyz_velocity_rmse'])
        reports.append(dict(source=str(path),static_seconds=nstatic*c.DT,corrected_acc_norm_median=norm_median,
            acc_correction_norm_median=correction_median,tilt_rms_deg=rms.tolist(),tilt_abs_p95_deg=p95,tilt_axis_abs_p95_deg=p95_axes.tolist(),
            quaternion_max_norm_error=unit,integration=errors,gates=gates,passed=all(gates.values())))
    return dict(passed=all(r['passed'] for r in reports),records=reports,
        protocol='causal full-record orientation; controlled five-second integrations start at >=8s with identical reference initial states only for diagnostics'),orientations
