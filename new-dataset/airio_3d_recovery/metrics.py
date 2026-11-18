"""No optimizer changes: explicit per-sample, endpoint and upstream metrics."""
import math
import numpy as np
import torch
from scipy.spatial.transform import Rotation
from common import c,vendor,preserve_rng,CONFIG
from data import orientation,causal_motion
from recovery_core import qualification


def loss_parts(out,label,endpoint_weight=0.):
    e=out['net_vel']-label;cov=out['cov']
    huber=torch.nn.functional.huber_loss(e,torch.zeros_like(e),delta=.005)
    nll=(e.square()/cov+cov.log()).mean()
    full=100*(huber+1e-4*nll)
    endpoint=100*(torch.nn.functional.huber_loss(e[:,-1],torch.zeros_like(e[:,-1]),delta=.005)
                  +1e-4*(e[:,-1].square()/cov[:,-1]+cov[:,-1].log()).mean())
    return full+endpoint_weight*endpoint,dict(huber=float(huber.detach()),nll=float(nll.detach()),
        full_loss=float(full.detach()),endpoint_loss=float(endpoint.detach()))


def forward(net,bank,ids,spec):
    data,rot,body=bank.batch(ids);out=net(data,rot);label=net.get_label(body)
    if out['net_vel'].shape!=label.shape:raise RuntimeError('Label alignment shape mismatch')
    loss,parts=loss_parts(out,label,spec['endpoint_weight'])
    return loss,parts,out,label


def encoder_stats(net):
    keys=('feature_encoder','ori_encoder','fcn2','batchnorm2')
    values={k:max(float(p.detach().abs().max()) for n,p in net.named_parameters() if n.startswith(k+'.')) for k in keys}
    return values,all(v>1e-12 for v in values.values())


def parameter_stats(net):
    rows={}
    for name,p in net.named_parameters():
        rows[name]=dict(norm=float(p.detach().double().norm()),max_abs=float(p.detach().abs().max()),
            gradient_norm=None if p.grad is None else float(p.grad.detach().double().norm()),
            gradient_present=p.grad is not None)
    return rows


def window_metrics(net,bank,ids,batch=128):
    net.eval();sse=np.zeros(3);end_sse=np.zeros(3);zero=0.;end_zero=0.;official_sum=0.;official_count=0
    count=end_count=0;pred_sum=np.zeros(3);pred_sq=np.zeros(3);target_sum=np.zeros(3);target_sq=np.zeros(3)
    sensitivity=0.;finite=True;cov_ok=True
    with torch.no_grad():
        for off in range(0,len(ids),batch):
            data,rot,body=bank.batch(ids[off:off+batch]);out=net(data,rot);label=net.get_label(body)
            p=out['net_vel'].double();target=label.double();e=p-target;cv=out['cov']
            finite&=bool(torch.isfinite(p).all() and torch.isfinite(cv).all());cov_ok&=bool((cv>0).all())
            sse+=e.square().sum((0,1)).cpu().numpy();zero+=float(target.square().sum());count+=e.shape[0]*e.shape[1]
            last=p[:,-1];t=target[:,-1]
            end_sse+=(last-t).square().sum(0).cpu().numpy();end_zero+=float(t.square().sum());end_count+=len(p)
            pred_sum+=last.sum(0).cpu().numpy();pred_sq+=last.square().sum(0).cpu().numpy()
            target_sum+=t.sum(0).cpu().numpy();target_sq+=t.square().sum(0).cpu().numpy()
            official_sum+=float(e.mean(1).square().sum());official_count+=len(p)
            if off==0:
                changed=net({k:v.flip(0) for k,v in data.items()},rot.flip(0))['net_vel'][:,-1]
                zeroed=net({k:torch.zeros_like(v) for k,v in data.items()},torch.zeros_like(rot))['net_vel'][:,-1]
                sensitivity=max(float((changed-out['net_vel'][:,-1]).abs().max()),float((zeroed-out['net_vel'][:,-1]).abs().max()))
    pred_std=np.sqrt(np.maximum(0,pred_sq/end_count-(pred_sum/end_count)**2))
    target_std=np.sqrt(np.maximum(0,target_sq/end_count-(target_sum/end_count)**2))
    _,healthy=encoder_stats(net)
    return dict(windows=end_count,all_tokens=count,all_rmse=float(np.sqrt(sse.sum()/count)),
        component_rmse=np.sqrt(sse/count).tolist(),zero_all_rmse=float(np.sqrt(zero/count)),
        endpoint_rmse=float(np.sqrt(end_sse.sum()/end_count)),zero_endpoint_rmse=float(np.sqrt(end_zero/end_count)),
        official_window_mean_residual_rmse=float(np.sqrt(official_sum/official_count)),
        prediction_mean=(pred_sum/end_count).tolist(),prediction_std=pred_std.tolist(),reference_std=target_std.tolist(),
        prediction_std_ratio=float(np.linalg.norm(pred_std)/max(np.linalg.norm(target_std),1e-15)),
        input_sensitivity=sensitivity,encoder_healthy=healthy,finite=bool(finite),covariance_valid=bool(cov_ok))


def validation(net,val_bank):
    windows=window_metrics(net,val_bank,np.arange(len(val_bank)))
    rows=[];sse=zero=teacher_sse=0.;count=0;net.eval()
    for path in c.split_paths()['val']:
        r=c.load_record(path);runtime=c.load_runtime(path);ori=orientation(path)
        ix,p,var=causal_motion(runtime,ori['rotvec'],net)
        tx,tp,tvar=causal_motion(runtime,Rotation.from_quat(r['quaternion']).as_rotvec(),net)
        np.testing.assert_array_equal(ix,tx)
        mask=ix>=1600;ix=ix[mask];p=p[mask];tp=tp[mask]
        q=Rotation.from_quat(ori['quaternion']);refq=Rotation.from_quat(r['quaternion'])
        delta=refq[1600].as_euler('xyz')[2]-q[1600].as_euler('xyz')[2]
        world=Rotation.from_euler('z',delta).apply(q[ix].apply(p));teacher=refq[ix].apply(tp)
        target=r['velocity'][ix];error=world-target
        ss=float(np.sum(error**2));zz=float(np.sum(target**2));tt=float(np.sum((teacher-target)**2))
        sse+=ss;zero+=zz;teacher_sse+=tt;count+=len(ix)
        rows.append(dict(source=str(path),samples=len(ix),deployed_xyz_rmse=float(np.sqrt(ss/len(ix))),
            zero_xyz_rmse=float(np.sqrt(zz/len(ix))),teacher_xyz_rmse=float(np.sqrt(tt/len(ix))),
            component_rmse=np.sqrt(np.mean(error**2,axis=0)).tolist(),
            covariance_min=float(min(var.min(),tvar.min())),covariance_max=float(max(var.max(),tvar.max()))))
    result=dict(xyz_velocity_rmse=float(np.sqrt(sse/count)),zero_xyz_rmse=float(np.sqrt(zero/count)),
        teacher_xyz_rmse=float(np.sqrt(teacher_sse/count)),zero_ratio=float(np.sqrt(sse/zero)),samples=count,
        records=rows,windows=windows,**{k:windows[k] for k in ('input_sensitivity','prediction_std_ratio','encoder_healthy','finite','covariance_valid')})
    result['qualification']=qualification(result)
    return result


def offline_full_record(net):
    """Explicit noncausal diagnostic, matching upstream whole-sequence default."""
    rows=[];device=next(net.parameters()).device;dtype=next(net.parameters()).dtype
    net.eval()
    with torch.no_grad():
        for path in c.split_paths()['val']:
            r=c.load_record(path);n=len(r['time']);orientations=dict(teacher=Rotation.from_quat(r['quaternion']).as_rotvec(),airimu=orientation(path)['rotvec'])
            for mode,rotation in orientations.items():
                out=net({k:torch.as_tensor(r[k][:-1],device=device,dtype=dtype)[None] for k in ('acc','gyro')},
                        torch.as_tensor(rotation[:-1],device=device,dtype=dtype)[None])
                ix=net.get_label(torch.arange(n,device=device,dtype=dtype)[None,:,None]).flatten().long().cpu().numpy()
                keep=ix>=1600;ix=ix[keep];pred=out['net_vel'][0].double().cpu().numpy()[keep]
                # Score in reference body coordinates for teacher mode; convert deployed mode consistently.
                q=Rotation.from_quat(r['quaternion'] if mode=='teacher' else orientation(path)['quaternion'])
                world=q[ix].apply(pred)
                if mode=='airimu':
                    delta=Rotation.from_quat(r['quaternion'][1600]).as_euler('xyz')[2]-q[1600].as_euler('xyz')[2]
                    world=Rotation.from_euler('z',delta).apply(world)
                error=world-r['velocity'][ix]
                rows.append(dict(source=str(path),mode=mode,samples=len(ix),noncausal_xyz_rmse=float(np.sqrt(np.mean(np.sum(error**2,axis=1)))),
                    notice='Whole-sequence bidirectional look-ahead; diagnostic only, never a causal selection metric'))
    return rows
