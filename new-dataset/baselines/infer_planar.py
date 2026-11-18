"""Causal AirIMU orientation cache and AirIO observations (no reference input)."""
import argparse,json
import numpy as np
import torch
from scipy.spatial.transform import Rotation
import planar_common as pc

@torch.no_grad()
def corrections(seq,model,device,batch_size=64):
    n=len(seq.imu_ts);a=np.zeros((n,3));g=np.zeros((n,3));ac=np.full((n,3),.08**2);gc=np.full((n,3),(.0032)**2)
    # Only the last correction of each past-only window is applied going forward.
    ends=np.arange(1011,n,4,dtype=int)
    for start in range(0,len(ends),batch_size):
        ids=ends[start:start+batch_size]
        data={k:torch.tensor(np.stack([v[i-1008:i+1] for i in ids]),dtype=torch.float32,device=device)
              for k,v in [('acc',seq.accel),('gyro',seq.gyro)]}
        output=model.inference(data)
        for j,i in enumerate(ids):
            a[i:min(i+4,n)]=output['correction_acc'][j,-1].cpu().numpy()
            g[i:min(i+4,n)]=output['correction_gyro'][j,-1].cpu().numpy()
            ac[i:min(i+4,n)]=output['cov_state']['acc_cov'][j,-1].clamp(1e-8,1e2).cpu().numpy()
            gc[i:min(i+4,n)]=output['cov_state']['gyro_cov'][j,-1].clamp(1e-9,1e2).cpu().numpy()
    return a,g,ac,gc

def orientation(acc,gyro):
    n=len(acc);r=Rotation.from_quat(pc.initial_quat(acc));quat=np.empty((n,4));quat[0]=r.as_quat()
    increments=Rotation.from_rotvec(gyro*.005)
    for i in range(1,n):r=r*increments[i-1];quat[i]=r.as_quat()
    return quat

def cache_all():
    torch.set_num_threads(4);device=torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    checkpoint=pc.BASE/'runs/airimu/best.pt';payload=torch.load(checkpoint,map_location='cpu',weights_only=False)
    model=pc.official_airimu().to(device).eval();model.load_state_dict(payload['model'],strict=True)
    digest=pc.file_sha(checkpoint)
    for split,paths in pc.resolve_manifest(pc.DATA/'processed/split.yaml',pc.DATA).items():
        for path in paths:
            output=pc.BASE/'orientation_cache'/path.parent.name/(path.stem+'.npz');output.parent.mkdir(exist_ok=True,parents=True)
            if output.exists():
                with np.load(output) as old:
                    if str(old['checkpoint_sha256'])==digest and str(old['source_sha256'])==pc.file_sha(path):continue
                raise RuntimeError('Refuse to overwrite incompatible orientation cache')
            seq=pc.load_sequence(path);a,g,ac,gc=corrections(seq,model,device)
            q=orientation(seq.accel+a,seq.gyro+g)
            np.savez_compressed(output,correction_acc=a,correction_gyro=g,acc_cov=ac,gyro_cov=gc,
                                rotation_vector=Rotation.from_quat(q).as_rotvec(),quaternion=q,
                                checkpoint_sha256=digest,source_sha256=pc.file_sha(path))
            print('orientation_cache',split,path.name,flush=True)

@torch.no_grad()
def motion_observations(seq,cache,model,device,batch_size=64):
    # Last output predicts the next sample endpoint, using only IMU through `end`.
    ends=np.arange(1003,len(seq.imu_ts)-1,4,dtype=int);mean=[];cov=[]
    for start in range(0,len(ends),batch_size):
        ids=ends[start:start+batch_size]
        data={k:torch.tensor(np.stack([v[i-999:i+1] for i in ids]),dtype=torch.float32,device=device)
              for k,v in [('acc',seq.accel),('gyro',seq.gyro)]}
        rot=torch.tensor(np.stack([cache['rotation_vector'][i-999:i+1] for i in ids]),dtype=torch.float32,device=device)
        out=model(data,rot);mean.append(out['net_vel'][:,-1].cpu().numpy());cov.append(out['cov'][:,-1].clamp(1e-8,1e2).cpu().numpy())
    return seq.imu_ts[ends+1],np.concatenate(mean),np.concatenate(cov)

if __name__=='__main__':
    parser=argparse.ArgumentParser();parser.add_argument('action',choices=['cache']);parser.parse_args();cache_all()
