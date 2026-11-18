"""Unmodified official 15-state AirIO filter and full 3D velocity observation."""
import ast
import numpy as np
import torch
from scipy.spatial.transform import Rotation
import contracts as c
import vendor,causal
import pypose as pp

state_module=vendor.load('airio3d_imu_state',c.BASE/'AirIO/EKF/IMUstate.py')
ekf_module=vendor.load('airio3d_ekf',c.BASE/'AirIO/EKF/ekf.py')
path=c.BASE/'AirIO/EKF/IMUofflinerunner.py'
node=next(n for n in ast.parse(path.read_text(encoding='utf-8')).body if isinstance(n,ast.ClassDef) and n.name=='SingleIMU')
namespace={'torch':torch,'pp':pp,'IMUstate':state_module.IMUstate}
exec(compile(ast.Module(body=[node],type_ignores=[]),str(path),'exec'),namespace)
SingleIMU=namespace['SingleIMU']


def anchor(state,covariance,yaw):
    rotation=pp.so3(state[:3]).Exp().matrix().detach().numpy()
    delta=yaw-np.arctan2(rotation[1,0],rotation[0,0])
    left=pp.SO3(torch.tensor(Rotation.from_euler('z',delta).as_quat(),dtype=state.dtype))
    def transform(x):return torch.cat(((left*pp.so3(x[:3]).Exp()).Log().tensor(),left@x[3:6],left@x[6:9],x[9:]))
    jac=torch.autograd.functional.jacobian(transform,state)
    return transform(state).detach(),(jac@covariance@jac.T).detach()


def replay(runtime,orientation,indices,means,variances,yaw_anchor,progress=True):
    model=SingleIMU().double();model.gravity=torch.tensor([0.,0.,c.G],dtype=torch.float64)
    filt=ekf_module.IMUEKF(model);state=torch.zeros(15,dtype=torch.float64)
    state[:3]=pp.SO3(torch.tensor(causal.initial_quat(runtime['acc']),dtype=torch.float64)).Log().tensor()
    covariance=torch.diag(torch.tensor([np.deg2rad(8)**2]*3+[.3**2]*3+[.5**2]*3+[.05**2]*3+[.2**2]*3,dtype=torch.float64))
    n=len(runtime['time']);states=np.empty((n,15));quaternions=np.empty((n,4));covariances=np.empty((n,15,15));obs_index=0
    def pack(i):
        states[i]=state.detach().numpy();quaternions[i]=pp.so3(state[:3]).Exp().tensor().detach().numpy()
        covariances[i]=covariance.detach().numpy()
    pack(0)
    for i in range(1,n):
        inputs=torch.cat((torch.tensor(runtime['gyro'][i-1]+orientation['correction_gyro'][i-1]),
                          torch.tensor(runtime['acc'][i-1]+orientation['correction_acc'][i-1]),state[9:]))
        q=np.r_[orientation['gyro_cov'][i-1],orientation['acc_cov'][i-1]*100,np.full(6,1e-12)]
        Q=torch.diag(torch.tensor(q,dtype=torch.float64));dt=torch.tensor([c.DT],dtype=torch.float64)
        if obs_index<len(indices) and indices[obs_index]==i:
            obs=torch.tensor(means[obs_index],dtype=torch.float64)
            R=torch.diag(torch.tensor(variances[obs_index]*.1,dtype=torch.float64))
            state,covariance=filt(state=state,input=inputs,obs=obs,P=covariance,dt=dt,Q=Q,R=R);obs_index+=1
        else:state,covariance=filt.state_propogate(state=state,input=inputs,P=covariance,dt=dt,Q=Q)
        state=state.detach();covariance=covariance.detach()
        if i==1600:state,covariance=anchor(state,covariance,yaw_anchor)
        if not torch.isfinite(state).all() or not torch.isfinite(covariance).all():raise FloatingPointError(f'Nonfinite full 3D EKF sample {i}')
        if i%200==0:
            p=covariance.numpy()
            if np.max(np.abs(p-p.T))>1e-6 or np.linalg.eigvalsh((p+p.T)/2).min()<-1e-7:raise FloatingPointError('Invalid full filter covariance')
        pack(i)
        if progress and i%5000==0:print(f'3D AirIO EKF {i}/{n}',flush=True)
    return dict(time=runtime['time'],state=states,quaternion=quaternions,covariance=covariances,
                observation_indices=indices,observation_mean=means,observation_variance=variances)
