"""Past-only model corrections and orientation; inputs are raw six-axis IMU only."""
import json
import numpy as np
import torch
from scipy.spatial.transform import Rotation
import contracts as c


def initial_quat(acc):
    # First sample only: initialization must not peek into a future interval.
    a=np.asarray(acc[0],float)
    roll=np.arctan2(a[1],a[2]);pitch=np.arctan2(-a[0],np.hypot(a[1],a[2]))
    return Rotation.from_euler('xyz',[roll,pitch,0.]).as_quat()


def propagate(runtime,model,bias,device='cpu',batch_size=128):
    """Use only the final output of each trailing window, held for four samples.

    Batched windows are safe: the latest prefix orientation in a batch is
    strictly older than the first prediction in that batch (128*4 < 1000).
    Earlier network outputs are NEVER retrospectively written into the past.
    """
    if batch_size*4>=1000:raise ValueError('Batch overlaps an unavailable prefix orientation')
    acc=np.asarray(runtime['acc']);gyro=np.asarray(runtime['gyro']);n=len(acc)
    correction_acc=np.zeros((n,3));correction_gyro=np.tile(-np.asarray(bias),(n,1))
    acov=np.full((n,3),.08**2);gcov=np.full((n,3),.0032**2)
    q=np.empty((n,4));q[0]=initial_quat(acc);done=0
    def advance(stop):
        nonlocal done
        rotation=Rotation.from_quat(q[done])
        for i in range(done,min(stop,n-1)):
            rotation=rotation*Rotation.from_rotvec((gyro[i]+correction_gyro[i])*c.DT)
            q[i+1]=rotation.as_quat()
        done=min(stop,n-1)
    advance(min(999,n-1));model.eval()
    raw_acc=torch.tensor(acc,device=device,dtype=torch.float32)
    raw_gyro=torch.tensor(gyro,device=device,dtype=torch.float32)
    ends=np.arange(999,n,4);span=torch.arange(1000,device=device)
    with torch.no_grad():
        for begin in range(0,len(ends),batch_size):
            e=ends[begin:begin+batch_size];starts=e-999
            if starts[-1]>done:raise RuntimeError('Noncausal orientation padding')
            gravity=Rotation.from_quat(q[starts]).inv().apply(np.tile([0.,0.,c.G],(len(e),1)))
            pad=torch.tensor(gravity,device=device,dtype=torch.float32)[:,None].expand(-1,9,-1)
            ix=torch.tensor(starts,device=device)[:,None]+span
            pred=model.inference(dict(acc=torch.cat((pad,raw_acc[ix]),1),gyro=torch.cat((torch.zeros_like(pad),raw_gyro[ix]),1)))
            values={k:pred[k][:,-1].cpu().numpy() for k in ('correction_acc','correction_gyro')}
            values.update({k:v[:,-1].cpu().numpy() for k,v in pred['cov_state'].items()})
            if any(not np.isfinite(v).all() for v in values.values()):raise FloatingPointError('Nonfinite AirIMU output')
            if np.any(values['acc_cov']<=0) or np.any(values['gyro_cov']<=0):raise FloatingPointError('Invalid AirIMU variance')
            for j,end in enumerate(e):
                stop=min(end+4,n)
                correction_acc[end:stop]=values['correction_acc'][j]
                correction_gyro[end:stop]=values['correction_gyro'][j] # replace warm-up bias; no double correction
                acov[end:stop]=values['acc_cov'][j];gcov[end:stop]=values['gyro_cov'][j]
            advance(min(e[-1]+4,n-1))
    advance(n-1)
    for i in range(1,n):
        if np.dot(q[i-1],q[i])<0:q[i]*=-1
    return dict(time=runtime['time'],quaternion=q,rotvec=Rotation.from_quat(q).as_rotvec(),
                correction_acc=correction_acc,correction_gyro=correction_gyro,acc_cov=acov,gyro_cov=gcov)


def cache_path(source):return c.HERE/'orientation_cache'/source.parent.name/(source.stem+'.npz')


def cached(source,model,bias,checkpoint,device):
    path=cache_path(source);signature=dict(checkpoint_sha256=c.sha(checkpoint),source_sha256=c.sha(source),
        causal_source_sha256=c.sha(__file__),calibration_sha256=c.sha(c.HERE/'prepared/warmup_calibration.json'))
    if path.exists():
        with np.load(path) as z:
            if any(str(z[k])!=v for k,v in signature.items()):raise RuntimeError('Stale 3D orientation cache')
            return {k:z[k] for k in z.files}
    value=propagate(c.load_runtime(source),model,bias,device)
    c.save_npz(path,**value,**signature);return value


def motion(runtime,orientation,model,device='cpu',teacher_quaternion=None):
    # The original BiGRU can use the entire trailing window, but never a future
    # raw sample. Only its last output is emitted, timestamped by its label.
    acc=runtime['acc'];gyro=runtime['gyro'];ends=np.arange(1003,len(acc)-1,4)
    rotation=orientation['rotvec'] if teacher_quaternion is None else Rotation.from_quat(teacher_quaternion).as_rotvec()
    arrays=[torch.tensor(a,device=device,dtype=torch.float32) for a in (acc,gyro,rotation)]
    span=torch.arange(1000,device=device);means=[];variances=[];model.eval()
    with torch.no_grad():
        for start in range(0,len(ends),128):
            ix=torch.tensor(ends[start:start+128]-999,device=device)[:,None]+span
            output=model(dict(acc=arrays[0][ix],gyro=arrays[1][ix]),arrays[2][ix])
            means.append(output['net_vel'][:,-1].cpu().numpy());variances.append(output['cov'][:,-1].cpu().numpy())
    mean=np.concatenate(means);var=np.concatenate(variances)
    if not np.isfinite(mean).all() or not np.isfinite(var).all() or np.any(var<=0):raise FloatingPointError('Invalid AirIO outputs')
    return ends+1,mean,var
