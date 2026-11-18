"""ReAqROVIO core with explicit, train-calibrated sensor priors.

Network/training code in deepvl/ is kept frozen. This module imports the pinned
15-state mathematical core and changes only explicit sensor calibration/noise
configuration, rather than introducing the DAIO ESKF or truth-driven resets.
"""
from pathlib import Path
import sys
import numpy as np
from scipy.spatial.transform import Rotation
DATA=Path(__file__).resolve().parent
sys.path.insert(0,str(DATA/'deepvl'))
import filter as core


def original_config():
    return dict(initial_covariance_diag=core.INITIAL.tolist(),process_noise_diag=core.NOISE.tolist(),
                initial_gyro_bias=[0.,0.,0.],initial_accel_bias=[0.,0.,0.],
                velocity_gate=core.VELOCITY_GATE,depth_gate=core.DEPTH_GATE,depth_variance=core.DEPTH_VARIANCE)


def replay(record,ends,means,variances,anchor_yaw,anchor_xy,config):
    n=len(record.time);state=np.zeros(16)
    state[6:9]=config['initial_accel_bias'];state[9:12]=config['initial_gyro_bias']
    a=np.median(record.accel[:min(200,n)],axis=0)-state[6:9]
    tilt=[np.arctan2(a[1],a[2]),np.arctan2(-a[0],np.hypot(a[1],a[2])),0.]
    state[12:]=Rotation.from_euler('xyz',tilt).as_quat()
    P=np.diag(config['initial_covariance_diag']);noise=np.asarray(config['process_noise_diag'])
    states=np.empty((n,16));diagonal=np.empty((n,15));events=[];j=0
    for i in range(n):
        if i:
            state,F,G=core.propagation(state,record.accel[i],record.gyro[i],record.time[i]-record.time[i-1])
            P=F@P@F.T+(G*noise)@G.T;P=(P+P.T)*.5
        if i==1600:state,P=core.anchor(state,P,float(anchor_yaw),anchor_xy)
        if j<len(ends) and i==ends[j]:
            h,H=core.heading_observation(state)
            residual=means[j]-h
            state,P,ok,nis=core.measurement_update(state,P,means[j],h,H,np.diag(variances[j]),config['velocity_gate'])
            events.append(dict(time=float(record.time[i]),kind='network_xy',accepted=ok,nis=nis,
                 residual_x=float(residual[0]),residual_y=float(residual[1]),variance_x=float(variances[j,0]),variance_y=float(variances[j,1])))
            j+=1
        if i%4==3:
            H=np.zeros((1,15));H[0,2]=1
            state,P,ok,nis=core.measurement_update(state,P,np.array([-(record.depth[i]-record.depth[0])]),
                      state[2:3],H,np.array([[config['depth_variance']]]),config['depth_gate'])
            events.append(dict(time=float(record.time[i]),kind='depth',accepted=ok,nis=nis))
        if not np.isfinite(state).all() or not np.isfinite(P).all():raise FloatingPointError(f'Filter diverged at {i}')
        if i%200==0 and np.linalg.eigvalsh(P).min() < -1e-10:raise FloatingPointError('Covariance lost PSD')
        states[i]=state;diagonal[i]=np.diag(P)
    rotation=Rotation.from_quat(states[:,12:]);world_velocity=-rotation.apply(states[:,3:6])
    projection=np.column_stack((states[:,:3],world_velocity,rotation.as_euler('xyz')[:,2]))
    return projection,states,diagonal,events
