"""Isolated 3D sensor-assisted reference contract; no old experiment mutations."""
from pathlib import Path
import csv,hashlib,json,sys,os
import numpy as np
import torch
import yaml
from scipy.signal import savgol_filter
from scipy.spatial.transform import Rotation

HERE=Path(__file__).resolve().parent;DATA=HERE.parent;ROOT=DATA.parent
BASE=DATA/'baselines';REPORT=ROOT/'ROV-ESKF-Odometry/reports/new_v1v2'
os.environ['PYTHONIOENCODING']='utf-8'
for stream in (sys.stdout,sys.stderr):
    if stream is not None and hasattr(stream,'reconfigure'):stream.reconfigure(encoding='utf-8',errors='replace')
sys.path[:0]=[str(DATA),str(BASE/'site-packages')]
from reliable_io import atomic_json,torch_save,_temporary,_replace
CONFIG=json.loads((HERE/'config.json').read_text(encoding='utf-8'))
G=CONFIG['gravity'];DT=CONFIG['dt']


def sha(path):
    h=hashlib.sha256()
    with Path(path).open('rb') as f:
        for a in iter(lambda:f.read(1048576),b''):h.update(a)
    return h.hexdigest()


def split_paths():
    d=yaml.safe_load((DATA/'processed/split.yaml').read_text(encoding='utf-8'))
    return {k:[DATA/'processed'/p for p in v['files']] for k,v in d['splits'].items()}


def fingerprint():
    return dict(config=sha(HERE/'config.json'),torch=torch.__version__,
        code={p.name:sha(p) for p in sorted(HERE.glob('*.py'))},
        vendor={str(p.relative_to(BASE)):sha(p) for name in ('AirIMU','AirIO') for p in sorted((BASE/name).rglob('*.py'))},
        shared={name:sha(DATA/name) for name in ('reliable_io.py','evaluate_pipeline.py','plot_results.py','figure_io.py','write_eval_report.py','validate_evaluation.py','export_velocity_figures.py')},
        split=sha(DATA/'processed/split.yaml'),manifest=sha(DATA/'processed/manifest.csv'),
        data={str(p.relative_to(DATA)):sha(p) for paths in split_paths().values() for p in paths})


def save_npz(path,**arrays):
    temporary=_temporary(path)
    try:
        with temporary.open('wb') as f:np.savez_compressed(f,**arrays)
        _replace(temporary,path,10)
    finally:temporary.unlink(missing_ok=True)


def read_source(path):
    names=['sample_time_s',*[f'acc_{a}' for a in 'xyz'],*[f'gyro_{a}' for a in 'xyz'],
           'pressure','roll','pitch','tag_x_m','tag_y_m','tag_yaw_rad']
    with Path(path).open(encoding='utf-8-sig') as f:header=next(csv.reader(f))
    a=np.loadtxt(path,delimiter=',',skiprows=1,usecols=[header.index(n) for n in names],encoding='utf-8-sig',ndmin=2)
    if not np.isfinite(a).all():raise ValueError('Nonfinite source: '+str(path))
    np.testing.assert_allclose(np.diff(a[:,0]),DT,atol=1e-10,rtol=0)
    return dict(time=a[:,0],acc=a[:,1:4],gyro=a[:,4:7],depth=a[:,7],rpy=np.c_[a[:,8:10],a[:,12]],xy=a[:,10:12])


def make_reference(raw):
    n=len(raw['time']);width=min(101,n if n%2 else n-1)
    if width<5:raise ValueError('Reference segment too short')
    xyz=savgol_filter(np.c_[raw['xy'],-raw['depth']],width,3,axis=0,mode='interp')
    rpy=savgol_filter(np.unwrap(raw['rpy'],axis=0),width,3,axis=0,mode='interp')
    R=Rotation.from_euler('xyz',rpy);v=np.gradient(xyz,raw['time'],axis=0,edge_order=1)
    canonical=Rotation.from_euler('z',-rpy[0,2])
    pos=canonical.apply(xyz-xyz[0]);vel=canonical.apply(v);rot=canonical*R
    quat=rot.as_quat();world_quat=R.as_quat()
    for q in (quat,world_quat):
        for i in range(1,n):
            if np.dot(q[i],q[i-1])<0:q[i]*=-1
    angular=np.r_[np.zeros((1,3)),(R[:-1].inv()*R[1:]).as_rotvec()/DT]
    static=(np.linalg.norm(v,axis=1)<CONFIG['gates']['static_speed'])&(np.linalg.norm(angular,axis=1)<CONFIG['gates']['static_angular_speed'])
    return dict(time=raw['time'],acc=raw['acc'].astype('float32'),gyro=raw['gyro'].astype('float32'),
        position_world=xyz,velocity_world=v,quaternion_world=world_quat,
        position=pos,velocity=vel,quaternion=quat,body_velocity=R.inv().apply(v),
        gravity_body=rot.inv().apply(np.tile([0.,0.,G],(n,1))),static=static,reference_omega=angular,
        initial_world_yaw=np.array(rpy[0,2]),reference_rpy=rpy)


def cache_path(source):return HERE/'prepared'/source.parent.name/(source.stem+'.npz')


def load_record(source):
    with np.load(cache_path(source)) as z:return {k:z[k] for k in z.files}


def load_runtime(source):
    # This is the only input object allowed into online models/filter/cache.
    with np.load(cache_path(source)) as z:
        return {k:z[k] for k in ('time','acc','gyro')}
