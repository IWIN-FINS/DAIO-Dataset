"""Unmodified official 3D models/losses; namespaced optional imports."""
import sys,types
import torch
import contracts as c
import pypose as pp
from pyhocon import ConfigFactory


def load(name,path,replacements=()):
    module=types.ModuleType(name);module.__file__=str(path);sys.modules[name]=module
    s=path.read_text(encoding='utf-8')
    for a,b in replacements:s=s.replace(a,b)
    exec(compile(s,str(path),'exec'),module.__dict__);return module


_ready=False
def setup():
    global _ready
    if _ready:return
    util=types.ModuleType('airio3d_vendor_utils')
    util.report_hasNan=lambda x: not torch.isfinite(x).all()
    sys.modules[util.__name__]=util
    for tag,folder in [('airimu3d_vendor','AirIMU'),('airio3d_vendor','AirIO')]:
        package=types.ModuleType(tag);package.__path__=[str(c.BASE/folder/'model')];sys.modules[tag]=package
        names=('net','cnn','code','loss_func','losses') if folder=='AirIMU' else ('code','loss_func','losses')
        for name in names:
            load(tag+'.'+name,c.BASE/folder/'model'/(name+'.py'),
                 [('from model.',f'from {tag}.'),('from utils import report_hasNan','from airio3d_vendor_utils import report_hasNan')])
    _ready=True


IMU_LOSS=ConfigFactory.from_dict(dict(propcov=True,covaug=True,sampling=50,loss='Huber_loss005',rotloss='Huber_loss005',
    rot_weight=1000.,pos_weight=100.,vel_weight=10.,cov_weight=1e-4))
IO_LOSS=ConfigFactory.from_dict(dict(propcov=True,covaug=True,loss='Huber_loss005',weight=100.,cov_weight=1e-4))


def airimu(device='cpu',fast=True):
    setup();cls=sys.modules['airimu3d_vendor.code'].CodeNet
    if fast:
        class TensorizedCodeNet(cls):
            def _update(self,to_update,feat,frame_len):
                index=((torch.arange(frame_len,device=feat.device)+self.inter_head)//self.interval).clamp_max(feat.shape[1]-1)
                return to_update+feat[:,index,:]
        cls=TensorizedCodeNet
    m=cls(ConfigFactory.from_dict(dict(network='codenet',gtrot=True,propcov=True,sampling=50)))
    # Upstream uses a `gravity` CONFIG KEY as a zero-gravity switch. Do NOT
    # pass that key. Set the physical vector explicitly after construction.
    m.integrator.gravity=torch.tensor([0.,0.,c.G])
    return m.to(device)


def airio(device='cpu'):
    setup()
    return sys.modules['airio3d_vendor.code'].CodeNetMotionwithRot(ConfigFactory.from_dict(dict(propcov=True))).to(device)


def imu_loss(output,label):
    setup();return sys.modules['airimu3d_vendor.losses'].get_loss(output,label,IMU_LOSS)['loss']


def io_loss(output,label):
    setup();return sys.modules['airio3d_vendor.losses'].get_motion_loss(output,label,IO_LOSS)['loss']


def integrate(acc,gyro,initial,rot=None,acc_cov=None,gyro_cov=None):
    """Same 50-point native PyPose integration, for controlled physical checks."""
    integration=pp.module.IMUPreintegrator(prop_cov=True,reset=True,gravity=c.G).to(acc.device)
    if acc_cov is None:acc_cov=torch.full_like(acc,.08**2)
    if gyro_cov is None:gyro_cov=torch.full_like(gyro,.0032**2)
    state=initial;out={k:[] for k in ('pos','vel','rot','cov')}
    for start in range(0,acc.shape[1],50):
        stop=min(start+50,acc.shape[1])
        value=integration(init_state=state,dt=torch.full_like(acc[:,start:stop,:1],c.DT),
            acc=acc[:,start:stop],gyro=gyro[:,start:stop],rot=rot[:,start:stop] if rot is not None else None,
            acc_cov=acc_cov[:,start:stop],gyro_cov=gyro_cov[:,start:stop])
        for k in out:out[k].append(value[k])
        state={k:value[k][:,-1:] for k in ('pos','vel','rot')}
        state.update(cov=value['cov'],Rij=value['Rij'])
    return dict(pos=torch.cat(out['pos'],1),vel=torch.cat(out['vel'],1),rot=torch.cat(out['rot'],1),cov=torch.stack(out['cov'],1))
