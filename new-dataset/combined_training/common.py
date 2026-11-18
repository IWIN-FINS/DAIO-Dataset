"""Isolated combined experiment: immutable sources and shared native data/model code."""
from pathlib import Path
import sys,os,json,hashlib,csv
import numpy as np
import torch
import yaml
HERE=Path(__file__).resolve().parent;ROOT=HERE.parents[1];DATA=ROOT/'dataset/combined_old_new_v1v2'
RUN=ROOT/'ROV-IO-Ablation/experiments/combined_old_new_daio_vs_no_rpm'
REPORT=ROOT/'ROV-ESKF-Odometry/reports/combined_old_new_daio_vs_no_rpm'
PROJECT=ROOT/'ROV-IO-Ablation'
sys.path[:0]=[str(PROJECT/'src'),str(ROOT/'ROV-ESKF-Odometry/src'),str(ROOT/'new-dataset')]
os.environ['PYTHONIOENCODING']='utf-8'
for stream in (sys.stdout,sys.stderr):
    if stream is not None and hasattr(stream,'reconfigure'):stream.reconfigure(encoding='utf-8',errors='replace')
from reliable_io import atomic_json,torch_save,_temporary,_replace
from rovio_ablation.config import load_config,v20_input_contract,config_hash
from rovio_ablation.data import resolve_manifest,V20WindowDataset
KEYS=('rpm','accel_xy','gyro_z','depth')
SLICES={'rpm':slice(0,8),'accel_xy':slice(8,10),'gyro_z':slice(10,11),'depth':slice(11,12)}

def sha(path):
    h=hashlib.sha256()
    with Path(path).open('rb') as f:
        for part in iter(lambda:f.read(1048576),b''):h.update(part)
    return h.hexdigest()

def read(path):return json.loads(Path(path).read_text(encoding='utf-8'))
def npz(path,**values):
    temporary=_temporary(path)
    try:
        with temporary.open('wb') as f:np.savez_compressed(f,**values)
        _replace(temporary,path,10)
    finally:temporary.unlink(missing_ok=True)

def cfg(model):return load_config(RUN/model/'config.yaml')
def paths():return resolve_manifest(DATA/'split.yaml',DATA)
def fingerprint():
    return dict(torch=str(torch.__version__),scripts={p.name:sha(p) for p in sorted(HERE.glob('*.py'))},
        native={str(p.relative_to(ROOT)):sha(p) for folder in (PROJECT/'src/rovio_ablation',ROOT/'ROV-ESKF-Odometry/src/rov_eskf') for p in sorted(folder.glob('*.py'))},
        io=sha(ROOT/'new-dataset/reliable_io.py'),configs={k:sha(RUN/k/'config.yaml') for k in ('daio','no_rpm')},
        manifest=sha(DATA/'processed/manifest.csv'),split=sha(DATA/'split.yaml'),
        records={str(p.relative_to(DATA)):sha(p) for ps in paths().values() for p in ps})

def cache_path(p):return DATA/'feature_cache'/p.relative_to(DATA/'processed').with_suffix('.npz')
def normalizer(model):
    d=read(DATA/'feature_cache/normalizer.json')
    return {k:np.array(v,dtype=np.float32) for k,v in d.items() if model=='daio' or not k.startswith('rpm_')}

def metric(t,pred,target):
    error=pred.astype(float)-target.astype(float);ticks=np.rint(t/.005).astype(np.int64)
    right=np.searchsorted(ticks,ticks+1000);left=np.flatnonzero(right<len(t));right=right[left]
    ok=ticks[right]-ticks[left]==1000;left=left[ok];right=right[ok]
    if not len(left):raise ValueError('No exact five-second pairs')
    return dict(samples=len(t),ave_mps=float(np.sqrt(np.mean(np.sum(error[:,:2]**2,axis=1)))),
        rve_5s_mps=float(np.sqrt(np.mean(np.sum((error[right,:2]-error[left,:2])**2,axis=1)))),
        vx_rmse_mps=float(np.sqrt(np.mean(error[:,0]**2))),vy_rmse_mps=float(np.sqrt(np.mean(error[:,1]**2))),
        yaw_rate_rmse_radps=float(np.sqrt(np.mean(error[:,2]**2))),rve_pairs=len(left))

class Bank:
    def __init__(self,split,device):
        self.device=device;features=[];starts=[];labels=[];self.records=[];offset=0;count=0
        for path in paths()[split]:
            with np.load(cache_path(path)) as z:
                f=z['features'];s=z['starts'];target=z['target'];time=z['time']
            self.records.append(dict(path=str(path),lo=count,hi=count+len(s),time=time));count+=len(s)
            if len(s):
                features.append(f);starts.append(s+offset);labels.append(target);offset+=len(f)
        self.features=torch.tensor(np.concatenate(features),device=device,dtype=torch.float32)
        self.starts=torch.tensor(np.concatenate(starts),device=device,dtype=torch.long)
        self.target=torch.tensor(np.concatenate(labels),device=device,dtype=torch.float32)
        self.span=torch.arange(256,device=device)*6
        stats=normalizer('daio');self.mean=torch.tensor(np.concatenate([stats[k+'_mean'] for k in KEYS]),device=device)
        self.std=torch.tensor(np.concatenate([stats[k+'_std'] for k in KEYS]),device=device)
    def __len__(self):return len(self.starts)
    def inputs(self,index,keys):
        ix=self.starts[index,None]+self.span[None]
        value=(self.features[ix]-self.mean)/self.std
        return [value[:,:,SLICES[k]].transpose(1,2).contiguous() for k in keys]
