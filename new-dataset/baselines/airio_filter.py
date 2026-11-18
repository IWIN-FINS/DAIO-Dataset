"""Official AirIO EKF with a 2D heading-velocity observation and one yaw anchor."""
import ast
import numpy as np
import torch
from scipy.spatial.transform import Rotation
import planar_common as pc

imu_module=pc.load_vendor('airio_imu_state',pc.BASE/'AirIO/EKF/IMUstate.py')
ekf_module=pc.load_vendor('airio_ekf',pc.BASE/'AirIO/EKF/ekf.py')
path=pc.BASE/'AirIO/EKF/IMUofflinerunner.py'
node=next(n for n in ast.parse(path.read_text(encoding='utf-8')).body if isinstance(n,ast.ClassDef) and n.name=='SingleIMU')
namespace={'torch':torch,'pp':pc.pp,'IMUstate':imu_module.IMUstate}
exec(compile(ast.Module(body=[node],type_ignores=[]),str(path),'exec'),namespace)

class PlanarSingleIMU(namespace['SingleIMU']):
    def observation(self,state,input,dt,t=None):
        new=self.state_transition(state,input,dt,t)
        r=pc.pp.so3(new[:3]).Exp().matrix()
        yaw=torch.atan2(r[1,0],r[0,0]);c,s=torch.cos(yaw),torch.sin(yaw)
        return torch.stack((c*new[3]+s*new[4],-s*new[3]+c*new[4]))

def anchor(state,covariance,yaw):
    rotation=pc.pp.so3(state[:3]).Exp().matrix().detach().numpy()
    current=np.arctan2(rotation[1,0],rotation[0,0]);delta=yaw-current
    left=pc.pp.SO3(torch.tensor(Rotation.from_euler('z',delta).as_quat(),dtype=state.dtype))
    def transform(x):
        return torch.cat(((left*pc.pp.so3(x[:3]).Exp()).Log().tensor(),left@x[3:6],left@x[6:9],x[9:]))
    jac=torch.autograd.functional.jacobian(transform,state)
    return transform(state).detach(),(jac@covariance@jac.T).detach()

def replay(seq,cache,observation_times,means,variances,yaw_anchor,progress=True):
    model=PlanarSingleIMU().double();model.gravity=torch.tensor([0.,0.,9.80665],dtype=torch.float64)
    filt=ekf_module.IMUEKF(model)
    state=torch.zeros(15,dtype=torch.float64)
    state[:3]=pc.pp.SO3(torch.tensor(pc.initial_quat(seq.accel),dtype=torch.float64)).Log().tensor()
    p=torch.diag(torch.tensor([.13962634**2]*3+[.3**2]*3+[.5**2]*3+[.05**2]*3+[.2**2]*3,dtype=torch.float64))
    states=np.empty((len(seq.imu_ts),7));covdiag=np.empty((len(seq.imu_ts),15))
    def pack(i):
        r=pc.pp.so3(state[:3]).Exp().matrix().detach().numpy()
        states[i]=[*state[6:9].detach().numpy(),*state[3:6].detach().numpy(),np.arctan2(r[1,0],r[0,0])]
        covdiag[i]=p.diag().detach().numpy()
    pack(0);obs_index=0
    for i in range(1,len(seq.imu_ts)):
        inputs=torch.cat((torch.tensor(seq.gyro[i-1]+cache['correction_gyro'][i-1]),torch.tensor(seq.accel[i-1]+cache['correction_acc'][i-1]),state[9:]))
        q=np.r_[cache['gyro_cov'][i-1],cache['acc_cov'][i-1]*100,np.full(6,1e-12)]
        Q=torch.diag(torch.tensor(q,dtype=torch.float64));dt=torch.tensor([.005],dtype=torch.float64)
        while obs_index<len(observation_times) and observation_times[obs_index]<seq.imu_ts[i]-1e-8:obs_index+=1
        if obs_index<len(observation_times) and abs(observation_times[obs_index]-seq.imu_ts[i])<1e-8:
            obs=torch.tensor(means[obs_index],dtype=torch.float64)
            R=torch.diag(torch.tensor(variances[obs_index]*.1,dtype=torch.float64).clamp_min(1e-8))
            state,p=filt(state=state,input=inputs,obs=obs,P=p,dt=dt,Q=Q,R=R)
            obs_index+=1
        else:state,p=filt.state_propogate(state=state,input=inputs,P=p,dt=dt,Q=Q)
        state=state.detach();p=p.detach()
        if i==1600:state,p=anchor(state,p,yaw_anchor)
        if not torch.isfinite(state).all() or not torch.isfinite(p).all():raise RuntimeError(f'Nonfinite AirIO EKF at sample {i}')
        pack(i)
        if progress and i%5000==0:print(f'AirIO EKF {i}/{len(seq.imu_ts)}',flush=True)
    return states,covdiag
