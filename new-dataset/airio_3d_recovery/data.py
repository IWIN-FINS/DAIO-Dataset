"""Shared window features for training, diagnostics and causal inference."""
import numpy as np
import torch
from scipy.spatial.transform import Rotation
from common import c,causal,CONFIG,HERE,write
from recovery_core import balanced_indices


def features(acc,gyro,rotvec,starts,device,dtype):
    starts=torch.as_tensor(starts,device=device,dtype=torch.long)
    ix=starts[:,None]+torch.arange(1000,device=device)[None]
    arrays=[torch.as_tensor(a,device=device,dtype=dtype) for a in (acc,gyro,rotvec)]
    return dict(acc=arrays[0][ix],gyro=arrays[1][ix]),arrays[2][ix]


class Bank:
    def __init__(self,split,device='cpu',dtype=torch.float32):
        if split not in ('train','val'):raise ValueError('Training bank may not load test trajectories')
        self.split=split;self.device=device;self.dtype=dtype
        self.records=[];self.groups=[];self.starts=[];parts={k:[] for k in ('acc','gyro','rotvec','body_velocity')}
        offset=0;number=0;stride=3 if split=='train' else 10
        for path in c.split_paths()[split]:
            r=c.load_record(path);n=len(r['time'])
            candidate=np.arange(0,max(0,n-1000),stride,dtype=np.int64)
            valid=np.ones(len(candidate),bool)
            end=r['time'][candidate+999] if len(candidate) else np.array([])
            for lo,hi in r['excluded_end_intervals']:valid&=~((end>=lo-1e-9)&(end<=hi+1e-9))
            candidate=candidate[valid]
            self.groups.append(np.arange(number,number+len(candidate),dtype=np.int64))
            self.records.append(dict(source=str(path),samples=n,offset=offset,window_offset=number,windows=len(candidate)))
            self.starts.extend((candidate+offset).tolist());number+=len(candidate)
            r['rotvec']=Rotation.from_quat(r['quaternion']).as_rotvec()
            for k in parts:parts[k].append(r[k])
            offset+=n
        self.starts=np.asarray(self.starts,dtype=np.int64)
        self.arrays={k:torch.as_tensor(np.concatenate(v),device=device,dtype=dtype) for k,v in parts.items()}
        if not len(self.starts):raise ValueError('No eligible windows')

    def __len__(self):return len(self.starts)

    def batch(self,indices):
        starts=self.starts[np.asarray(indices,dtype=np.int64)]
        a=self.arrays;data,rotation=features(a['acc'],a['gyro'],a['rotvec'],starts,self.device,self.dtype)
        ix=torch.as_tensor(starts,device=self.device)[:,None]+torch.arange(1001,device=self.device)[None]
        return data,rotation,a['body_velocity'][ix]

    def selected(self,count,seed=42):
        return balanced_indices(self.groups,count,seed)

    def inventory(self,indices):
        records=[]
        for i in indices:
            row=next(r for r in self.records if r['window_offset']<=i<r['window_offset']+r['windows'])
            start=int(self.starts[i]-row['offset'])
            records.append(dict(bank_index=int(i),source=row['source'],start_sample=start,
                                input_end_sample=start+999,label_end_sample=start+1000))
        return records


def orientation(path):
    with np.load(causal.cache_path(path)) as z:
        return {k:z[k] for k in ('time','quaternion','rotvec','correction_acc','correction_gyro','acc_cov','gyro_cov')}


def causal_motion(runtime,rotation,net,batch_size=128):
    """Only final output from each past-only window; output at input-end + 5ms."""
    n=len(runtime['time']);indices=np.arange(1004,n,4)
    device=next(net.parameters()).device;dt=next(net.parameters()).dtype
    arrays=[torch.as_tensor(a,device=device,dtype=dt) for a in (runtime['acc'],runtime['gyro'],rotation)]
    means=[];variances=[];net.eval()
    with torch.no_grad():
        for offset in range(0,len(indices),batch_size):
            starts=indices[offset:offset+batch_size]-1000
            data,rot=features(*arrays,starts,device,dt)
            out=net(data,rot);means.append(out['net_vel'][:,-1].cpu().double().numpy())
            variances.append(out['cov'][:,-1].cpu().double().numpy())
    if not means:return indices,np.empty((0,3)),np.empty((0,3))
    mean=np.concatenate(means);var=np.concatenate(variances)
    if not np.isfinite(mean).all() or not np.isfinite(var).all() or np.any(var<=0):
        raise FloatingPointError('Invalid causal AirIO output')
    return indices,mean,var
