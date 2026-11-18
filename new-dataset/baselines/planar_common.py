"""Pinned official model loading, planar reference, and IMU-only feature caches.

No upstream source is edited. AirIMU absolute imports are namespaced at load time.
"""
from pathlib import Path
import sys,types,importlib.util,hashlib,json,csv
import numpy as np
import torch
from scipy.spatial.transform import Rotation

BASE=Path(__file__).resolve().parent
DATA=BASE.parents[0]
ROOT=BASE.parents[1]
sys.path[:0]=[str(BASE/'site-packages'),str(ROOT/'ROV-IO-Ablation/src'),str(ROOT/'ROV-ESKF-Odometry/src')]
import pypose as pp
from pyhocon import ConfigFactory
from rovio_ablation.data import load_sequence,resolve_manifest,offline_labels

def load_vendor(name, path, replacements=()):
    mod=types.ModuleType(name);mod.__file__=str(path);sys.modules[name]=mod
    text=path.read_text(encoding='utf-8')
    for a,b in replacements:text=text.replace(a,b)
    exec(compile(text,str(path),'exec'),mod.__dict__)
    return mod

def official_airimu():
    package=types.ModuleType('airimu_vendor');package.__path__=[str(BASE/'AirIMU/model')]
    sys.modules['airimu_vendor']=package
    for name in ('net','cnn','code'):
        mod=load_vendor('airimu_vendor.'+name,BASE/f'AirIMU/model/{name}.py',[('from model.','from airimu_vendor.')])
    conf=ConfigFactory.from_dict({'network':'codenet','gtrot':False,'propcov':True,'sampling':50})
    return mod.CodeNet(conf)

def official_airio():
    mod=load_vendor('airio_vendor_code',BASE/'AirIO/model/code.py')
    model=mod.CodeNetMotionwithRot(ConfigFactory.from_dict({'propcov':True}))
    # Preserve encoders/GRUs; replace only the unsupported vertical output dimension.
    for name in ('veldecoder','velcov_decoder'):
        head=getattr(model,name);head[-1]=torch.nn.Linear(head[-1].in_features,2)
    return model

def tilt(acc):
    a=np.median(acc[:min(200,len(acc))],axis=0)
    return np.array([np.arctan2(a[1],a[2]),np.arctan2(-a[0],np.hypot(a[1],a[2]))])

def initial_quat(acc,yaw=0):return Rotation.from_euler('xyz',[*tilt(acc),yaw]).as_quat()

def reference(seq):
    from scipy.signal import savgol_filter
    width=min(101,len(seq.imu_ts) if len(seq.imu_ts)%2 else len(seq.imu_ts)-1)
    if width<5:raise ValueError('Reference too short for smoothing')
    xy=savgol_filter(seq.tag_xy,width,3,axis=0,mode='interp')
    yaw=savgol_filter(seq.tag_yaw,width,3,mode='interp')
    vel=np.gradient(xy,seq.imu_ts,axis=0,edge_order=1)
    c,s=np.cos(yaw),np.sin(yaw)
    heading=np.column_stack((c*vel[:,0]+s*vel[:,1],-s*vel[:,0]+c*vel[:,1]))
    return xy,yaw,vel,heading

def file_sha(p):
    h=hashlib.sha256()
    with Path(p).open('rb') as f:
        for b in iter(lambda:f.read(1048576),b''):h.update(b)
    return h.hexdigest()

def atomic_json(path,obj):
    path=Path(path);path.parent.mkdir(exist_ok=True,parents=True)
    tmp=path.with_suffix('.tmp');tmp.write_text(json.dumps(obj,indent=2,ensure_ascii=False),encoding='utf-8');tmp.replace(path)

def environment_fingerprint():
    from importlib.metadata import version
    return {'torch':torch.__version__,'pypose':pp.__version__,
            'airio_commit':'7214e8a6ddee95ca07a64f01f1fa3e2f8f60df73','airimu_commit':'119fe7ce229e321b3890282909a183aefad61dfb',
            'adapter_sha256':{p.name:file_sha(p) for p in BASE.glob('*.py')},
            'data_manifest':file_sha(DATA/'processed/manifest.csv'),'split':file_sha(DATA/'processed/split.yaml'),
            'data_files':{str(p.relative_to(DATA)):file_sha(p) for p in sorted((DATA/'processed').glob('v*/*.csv'))},
            'upstream_source_sha256':{str(p.relative_to(BASE)):file_sha(p) for repo in ('AirIO','AirIMU') for p in (BASE/repo).rglob('*.py')}}

class PlanarWindows(torch.utils.data.Dataset):
    """1000-sample official windows; references never enter returned IMU features."""
    def __init__(self,split,stage):
        self.stage=stage;self.seqs=[];self.refs=[];self.caches=[];self.entries=[]
        paths=resolve_manifest(DATA/'processed/split.yaml',DATA)[split]
        stride=(10 if stage=='airimu' else 3) if split=='train' else (200 if stage=='airimu' else 10)
        for p in paths:
            seq=load_sequence(p)
            if len(seq.imu_ts)<1010:continue
            idx=len(self.seqs);self.seqs.append(seq);self.refs.append(reference(seq))
            if stage=='airio':
                cache=np.load(BASE/'orientation_cache'/p.parent.name/(p.stem+'.npz'))
                if str(cache['checkpoint_sha256'])!=file_sha(BASE/'runs/airimu/best.pt'):raise RuntimeError('Stale orientation cache')
                self.caches.append({k:cache[k] for k in ('rotation_vector',)})
            for start in range(9 if stage=='airimu' else 0,len(seq.imu_ts)-1000,stride):
                # Shared data-quality exclusion, independent of the model's sensor inputs.
                hist=seq.imu_ts[start+999]-7.675
                sat=seq.rpm_saturation_times_s
                if sat is not None and np.any((sat>=hist)&(sat<=seq.imu_ts[start+999])):continue
                self.entries.append((idx,start))
        if not self.entries:raise ValueError('No planar training windows')
    def __len__(self):return len(self.entries)
    def __getitem__(self,index):
        idx,start=self.entries[index];seq=self.seqs[idx];xy,yaw,v,heading=self.refs[idx]
        lo=start-9 if self.stage=='airimu' else start
        data={'acc':torch.tensor(seq.accel[lo:start+1000],dtype=torch.float32),
              'gyro':torch.tensor(seq.gyro[lo:start+1000],dtype=torch.float32)}
        if self.stage=='airimu':
            data.update(target_xy=torch.tensor(xy[start+1:start+1001]-xy[start],dtype=torch.float32),
                        target_velocity=torch.tensor(v[start+1:start+1001],dtype=torch.float32),
                        target_yaw=torch.tensor(yaw[start+1:start+1001],dtype=torch.float32),
                        initial_rotation=torch.tensor(initial_quat(seq.accel[start:start+200],yaw[start]),dtype=torch.float32),
                        initial_velocity=torch.tensor([*v[start],0],dtype=torch.float32))
        else:
            data.update(rotation=torch.tensor(self.caches[idx]['rotation_vector'][start:start+1000],dtype=torch.float32),
                        target=torch.tensor(heading[start:start+1001],dtype=torch.float32))
        return data

def airimu_loss(model,batch):
    correction=model.inference(batch)
    acc=batch['acc'][:,9:]+correction['correction_acc'];gyro=batch['gyro'][:,9:]+correction['correction_gyro']
    integrator=pp.module.IMUPreintegrator(prop_cov=True,reset=True,gravity=9.80665).to(acc.device)
    init={'pos':torch.zeros((len(acc),1,3),device=acc.device),
          'vel':batch['initial_velocity'][:,None], 'rot':pp.SO3(batch['initial_rotation'][:,None])}
    pos=[];vel=[];yaw=[];cov=[]
    for i in range(0,1000,50):
        out=integrator(init_state=init,dt=torch.full_like(acc[:,i:i+50,:1],.005),gyro=gyro[:,i:i+50],acc=acc[:,i:i+50],
            acc_cov=correction['cov_state']['acc_cov'][:,i:i+50].clamp(1e-8,1e2),
            gyro_cov=correction['cov_state']['gyro_cov'][:,i:i+50].clamp(1e-9,1e2))
        pos.append(out['pos'][:,-1,:2]);vel.append(out['vel'][:,-1,:2])
        r=out['rot'][:,-1].matrix();yaw.append(torch.atan2(r[:,1,0],r[:,0,0]))
        # Project small-angle body rotation covariance onto world yaw derivative.
        r00,r10=r[:,0,0],r[:,1,0];den=(r00*r00+r10*r10).clamp_min(1e-8)
        j=torch.stack((torch.zeros_like(den),(r10*r[:,0,2]-r00*r[:,1,2])/den,(r00*r[:,1,1]-r10*r[:,0,1])/den),-1)
        yawvar=torch.einsum('bi,bij,bj->b',j,out['cov'][:,:3,:3],j).clamp_min(1e-9)
        cov.append(torch.cat((yawvar[:,None],out['cov'].diagonal(dim1=-2,dim2=-1)[:,[3,4,6,7]]),-1).clamp_min(1e-9))
        init={'pos':out['pos'][:,-1:],'vel':out['vel'][:,-1:],'rot':out['rot'][:,-1:],'cov':out['cov'],'Rij':out['Rij']}
    p=torch.stack(pos,1);v=torch.stack(vel,1);y=torch.stack(yaw,1)
    py=batch['target_yaw'][:,49::50];dy=torch.atan2(torch.sin(y-py),torch.cos(y-py))
    dp=p-batch['target_xy'][:,49::50];dv=v-batch['target_velocity'][:,49::50]
    cov=torch.stack(cov,1);dist=torch.cat((dy[:,:,None],dv,dp),-1)
    nll=(dist.square()/cov+cov.log())
    huber=lambda d:torch.nn.functional.huber_loss(d,torch.zeros_like(d),delta=.005)
    loss=1000*(huber(dy)+1e-4*nll[:,:,0].mean())+10*(huber(dv)+1e-4*nll[:,:,1:3].mean())+100*(huber(dp)+1e-4*nll[:,:,3:].mean())
    return loss,torch.mean(dv.square().sum(-1))

def airio_loss(model,batch):
    out=model(batch,batch['rotation']);target=model.get_label(batch['target'])
    error=out['net_vel']-target;cov=out['cov'].clamp(1e-8,1e2)
    loss=100*(torch.nn.functional.huber_loss(error,torch.zeros_like(error),delta=.005)+1e-4*(error.square()/cov+cov.log()).mean())
    return loss,error.square().sum(-1).mean()
