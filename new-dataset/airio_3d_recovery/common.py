"""Isolated dependencies, immutable provenance and experiment configuration."""
from pathlib import Path
import sys,os,json,hashlib,random,contextlib,datetime
import numpy as np
import torch

HERE=Path(__file__).resolve().parent
LEGACY=HERE.parent/'airio_3d'
sys.path.insert(0,str(LEGACY))
import contracts as c
import vendor
import causal
CONFIG=json.loads((HERE/'config.json').read_text(encoding='utf-8'))
REPORT=c.ROOT/'ROV-ESKF-Odometry/reports/airio_3d_recovery'
IMU=LEGACY/'runs/airimu/best.pt'
BASELINE=dict(name='wd_1e3',weight_decay=.001,exempt_norm_bias=False,dtype='float32',endpoint_weight=0.,complexity=0)
CANDIDATES=[BASELINE,
    dict(name='wd_1e4',weight_decay=.0001,exempt_norm_bias=False,dtype='float32',endpoint_weight=0.,complexity=1),
    dict(name='wd_zero',weight_decay=0.,exempt_norm_bias=False,dtype='float32',endpoint_weight=0.,complexity=2)]
SUPPLEMENTAL=[
    dict(name='wd_1e3_exempt_norm_bias',weight_decay=.001,exempt_norm_bias=True,dtype='float32',endpoint_weight=0.,complexity=3),
    dict(name='wd_1e3_fp64',weight_decay=.001,exempt_norm_bias=False,dtype='float64',endpoint_weight=0.,complexity=4)]


def read(path):return json.loads(Path(path).read_text(encoding='utf-8'))
def write(path,value):
    def scalar(obj):
        if isinstance(obj,np.generic):return obj.item()
        if isinstance(obj,np.ndarray):return obj.tolist()
        if isinstance(obj,Path):return str(obj)
        raise TypeError(type(obj).__name__)
    return c.atomic_json(path,json.loads(json.dumps(value,default=scalar)))
def now():return datetime.datetime.now().isoformat(timespec='seconds')
def dtype(spec):return torch.float64 if spec['dtype']=='float64' else torch.float32
def digest(value):return hashlib.sha256(json.dumps(value,sort_keys=True,ensure_ascii=False).encode()).hexdigest()


def setup(seed):
    os.environ.setdefault('CUBLAS_WORKSPACE_CONFIG',':4096:8')
    torch.set_num_threads(4);random.seed(seed);np.random.seed(seed);torch.manual_seed(seed)
    if torch.cuda.is_available():torch.cuda.manual_seed_all(seed)
    torch.backends.cuda.matmul.allow_tf32=False;torch.backends.cudnn.allow_tf32=False
    torch.backends.cudnn.benchmark=False;torch.backends.cudnn.deterministic=True


def rng():
    return dict(python=random.getstate(),numpy=np.random.get_state(),torch=torch.get_rng_state(),
                cuda=torch.cuda.get_rng_state_all() if torch.cuda.is_available() else [])


def restore_rng(state):
    random.setstate(state['python']);np.random.set_state(state['numpy']);torch.set_rng_state(state['torch'])
    if torch.cuda.is_available():torch.cuda.set_rng_state_all(state['cuda'])


@contextlib.contextmanager
def preserve_rng():
    state=rng()
    try:yield
    finally:restore_rng(state)


@contextlib.contextmanager
def process_lock():
    import msvcrt
    HERE.mkdir(parents=True,exist_ok=True)
    with (HERE/'pipeline.lock').open('a+b') as f:
        f.seek(0);f.write(b'0');f.flush();f.seek(0)
        try:msvcrt.locking(f.fileno(),msvcrt.LK_NBLCK,1)
        except OSError:raise RuntimeError('Another recovery command is running')
        try:yield
        finally:f.seek(0);msvcrt.locking(f.fileno(),msvcrt.LK_UNLCK,1)


def context():
    # Fingerprint all existing datasets for provenance, but diagnostics never load test labels.
    legacy=c.fingerprint();prepared=read(LEGACY/'prepared/complete.json')
    if prepared['context']!=legacy:raise RuntimeError('Frozen source/data/reference contract changed')
    frozen=read(LEGACY/'runs/airimu/frozen_physics.json');imu_hash=c.sha(IMU)
    if not frozen['passed'] or frozen['checkpoint_sha256']!=imu_hash:raise RuntimeError('Frozen AirIMU gate/hash mismatch')
    cache_hashes={};orientation_hashes={}
    for row in prepared['records']:
        path=LEGACY/row['cache'];h=c.sha(path)
        if h!=row['cache_sha256']:raise RuntimeError('Reference cache hash mismatch: '+str(path))
        cache_hashes[row['cache']]=h
    for split,paths in c.split_paths().items():
        for path in paths:
            op=causal.cache_path(path)
            with np.load(op) as z:
                expected=dict(checkpoint_sha256=imu_hash,source_sha256=c.sha(path),
                    causal_source_sha256=c.sha(LEGACY/'causal.py'),calibration_sha256=c.sha(LEGACY/'prepared/warmup_calibration.json'))
                if any(str(z[k])!=v for k,v in expected.items()):raise RuntimeError('Frozen orientation provenance mismatch')
            orientation_hashes[str(op.relative_to(LEGACY))]=c.sha(op)
    result=dict(schema=1,config=c.sha(HERE/'config.json'),code={p.name:c.sha(p) for p in sorted(HERE.glob('*.py'))},
        legacy=legacy,reference_caches=cache_hashes,orientation_caches=orientation_hashes,
        airimu=imu_hash,airimu_physics=c.sha(LEGACY/'runs/airimu/frozen_physics.json'),
        torch=torch.__version__,cuda=torch.version.cuda)
    result['signature']=digest(result)
    return result


def require_preflight(ctx):
    path=HERE/'preflight.json'
    if not path.exists():raise RuntimeError('Run preflight before diagnostic or formal training')
    value=read(path)
    if not value['passed'] or value['signature']!=ctx['signature']:raise RuntimeError('Preflight missing, failed or stale')


def model(spec,seed,device):
    setup(seed)
    # Initial FP32 parameters are identical across precision conditions before conversion.
    return vendor.airio(device).to(dtype=dtype(spec))


def parameter_groups(net,spec):
    if not spec['exempt_norm_bias']:return [dict(params=list(net.parameters()),weight_decay=spec['weight_decay'])]
    normalization=set()
    for module in net.modules():
        if isinstance(module,(torch.nn.modules.batchnorm._BatchNorm,torch.nn.LayerNorm,torch.nn.GroupNorm)):
            normalization.update(id(p) for p in module.parameters(recurse=False))
    regular=[];exempt=[]
    for name,p in net.named_parameters():
        (exempt if id(p) in normalization or 'bias' in name else regular).append(p)
    return [dict(params=regular,weight_decay=spec['weight_decay']),dict(params=exempt,weight_decay=0.)]
