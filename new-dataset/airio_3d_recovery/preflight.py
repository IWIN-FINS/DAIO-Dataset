"""Real-data forward/backward checks only; no optimizer or training updates."""
import gc
import numpy as np
import torch
from scipy.spatial.transform import Rotation
from common import HERE,LEGACY,IMU,CONFIG,BASELINE,c,vendor,context,model,write,read,now,parameter_groups
from data import Bank,features,causal_motion,orientation
from metrics import loss_parts


def run(device='cuda'):
    ctx=context();checks={};evidence={}
    counts={k:len(v) for k,v in c.split_paths().items()}
    checks['split_49_2_4']=counts==dict(train=49,val=2,demo=4)
    checks['fixed_thresholds_match_config']=(CONFIG['zero_ratio_max']==.9 and CONFIG['sensitivity_min_mps']==1e-6
        and CONFIG['healthy_prediction_std_ratio_min']==.01 and CONFIG['encoder_parameter_floor']==1e-12
        and CONFIG['tie_tolerance_mps']==1e-4 and not CONFIG['auto_start_formal'])
    paths=[p for split in ('train','val') for p in c.split_paths()[split]]
    max_roundtrip=0.;max_rebuild=0.;records=[]
    for path in paths:
        raw=c.read_source(path);r=c.load_record(path);new=c.make_reference(raw)
        error=max(float(np.max(np.abs(r[k]-new[k]))) for k in ('acc','gyro','velocity','body_velocity','quaternion'))
        max_rebuild=max(max_rebuild,error)
        rt=Rotation.from_quat(r['quaternion']).apply(r['body_velocity'])-r['velocity']
        max_roundtrip=max(max_roundtrip,float(np.max(np.abs(rt))))
        records.append(dict(source=str(path),samples=len(r['time']),cache_max_error=error))
    checks['reference_cache_exact']=max_rebuild==0
    checks['body_world_roundtrip']=max_roundtrip<1e-10
    evidence['reference']=dict(records=records,max_roundtrip=max_roundtrip)
    bank=Bank('train',device);val=Bank('val',device)
    checks['window_counts_match_legacy']=len(bank)==442998 and len(val)==3701
    selections={str(n):bank.selected(n).tolist() for n in (128,8192)}
    for n,ids in selections.items():
        write(HERE/'prepared'/f'windows_{n}.json',bank.inventory(ids))
    checks['selection_unique']=all(len(set(v))==int(k) for k,v in selections.items())
    checks['window_boundaries']=all(0<=x['start_sample'] and x['label_end_sample']<next(
        r['samples'] for r in bank.records if r['source']==x['source']) for x in bank.inventory(selections['128']))
    ids=bank.selected(8);data,rot,body=bank.batch(ids)
    net=model(BASELINE,42,device).eval()
    # cuDNN needs the RNN training reserve for backward; these one-layer GRUs
    # have dropout=0. Keep CNN dropout and BatchNorm in eval for numeric parity.
    for module in net.modules():
        if isinstance(module,torch.nn.RNNBase):module.train()
    initial={k:v.detach().clone() for k,v in net.state_dict().items()}
    out=net(data,rot);label=net.get_label(body)
    local,parts=loss_parts(out,label);upstream=vendor.io_loss(out,label)
    checks['loss_matches_upstream']=bool(torch.allclose(local,upstream,atol=1e-9,rtol=1e-6))
    local.backward();g32={n:p.grad.detach().double().cpu().clone() for n,p in net.named_parameters() if p.grad is not None}
    p32=out['net_vel'].detach().double().cpu();net.zero_grad(set_to_none=True)
    unused=[n for n,p in net.named_parameters() if n not in g32]
    checks['live_encoders_have_gradients']=all(any(n.startswith(k) and float(v.norm())>0 for n,v in g32.items()) for k in ('feature_encoder','ori_encoder','fcn2','batchnorm2'))
    precise=model({**BASELINE,'dtype':'float64'},42,device).eval()
    precise.load_state_dict(initial)
    for module in precise.modules():
        if isinstance(module,torch.nn.RNNBase):module.train()
    out64=precise({k:v.double() for k,v in data.items()},rot.double());loss64,_=loss_parts(out64,label.double());loss64.backward()
    abs_forward=float((p32-out64['net_vel'].detach().cpu()).abs().max())
    gnum=gden=0.
    for name,p in precise.named_parameters():
        if p.grad is not None:
            reference=p.grad.detach().cpu();gnum+=float((g32[name]-reference).square().sum());gden+=float(reference.square().sum())
    relative_gradient=float(np.sqrt(gnum/max(gden,1e-30)))
    checks['fp32_fp64_forward']=abs_forward<3e-5
    checks['fp32_fp64_gradient']=relative_gradient<.005
    evidence['precision']=dict(max_output_abs=abs_forward,relative_gradient_l2=relative_gradient,
        comparison='same FP32 initialized weights and batch converted to FP64; dropout/BN eval, dropout-free GRUs train for cuDNN backward',unused_parameters=unused)
    del precise,out64;net.eval();gc.collect();torch.cuda.empty_cache()
    # Compare same sampled windows to the legacy implementation, including quaternion Log convention.
    from banks import Bank as OriginalBank
    old=OriginalBank('val','airio',device);vids=np.linspace(0,len(val)-1,8,dtype=int)
    newdata,newrot,newbody=val.batch(vids);oldd,oldr,oldb=old.batch(torch.tensor(vids,device=device),'airio')
    differences=dict(acc=float((newdata['acc']-oldd['acc']).abs().max()),gyro=float((newdata['gyro']-oldd['gyro']).abs().max()),
                     rotation=float((newrot-oldr).abs().max()),body_velocity=float((newbody-oldb).abs().max()))
    checks['legacy_feature_parity']=max(differences.values())<2e-6;evidence['legacy_feature_parity']=differences
    del old
    labels=net.get_label(torch.arange(1001,device=device,dtype=torch.float32)[None,:,None]).flatten().long().cpu().numpy()
    checks['endpoint_time']=len(labels)==112 and labels[-1]==1000 and labels[0]==14
    evidence['label_indices']=labels.tolist()
    path=c.split_paths()['val'][0];r=c.load_record(path);rt=c.load_runtime(path);q=orientation(path)
    rt={k:v[:2401].copy() for k,v in rt.items()};rotation=q['rotvec'][:2401].copy()
    ix,p,_=causal_motion(rt,rotation,net)
    modified={k:v.copy() for k,v in rt.items()};modified['acc'][1800:]+=100;modified['gyro'][1800:]+=3
    rotated=rotation.copy();rotated[1800:]+=.5
    jx,pp,_=causal_motion(modified,rotated,net)
    checks['future_invariance']=np.array_equal(ix,jx) and np.array_equal(p[ix<=1800],pp[ix<=1800])
    # Single-window use matches the same feature builder used in training.
    starts=ix[:8]-1000
    batchdata,batchrot=features(rt['acc'],rt['gyro'],rotation,starts,device,torch.float32)
    with torch.no_grad():single=net(batchdata,batchrot)['net_vel'][:,-1].cpu().numpy()
    checks['train_online_feature_parity']=bool(np.allclose(single,p[:8],rtol=1e-5,atol=1e-7))
    with torch.no_grad():
        original=net(data,rot)['net_vel'];changed={k:v.clone() for k,v in data.items()};changed['gyro'][:,:,:2]+=.1
        response=float((net(changed,rot)['net_vel']-original).abs().max())
    checks['real_gyro_xy_connected']=response>1e-6;evidence['gyro_xy_response']=response
    checks['no_weights_updated']=all(torch.equal(v,initial[k]) for k,v in net.state_dict().items())
    del net,bank,val;gc.collect();torch.cuda.empty_cache()
    # Recheck frozen AirIMU using current validation data; save only to this experiment.
    import physics
    imu=vendor.airimu(device);imu.load_state_dict(torch.load(IMU,map_location=device,weights_only=False)['model']);imu.eval()
    bias=read(LEGACY/'prepared/warmup_calibration.json')['gyro_bias']
    physical,_=physics.validate(imu,bias,device);checks['frozen_airimu_physics']=physical['passed']
    write(HERE/'frozen_airimu_recheck.json',physical)
    result=dict(passed=all(checks.values()),checks=checks,evidence=evidence,signature=ctx['signature'],
        time=now(),no_optimizer_step=True,no_test_labels_used=True)
    write(HERE/'context.json',ctx);write(HERE/'preflight.json',result)
    print('PREFLIGHT',result['passed'],checks,flush=True)
    if not result['passed']:raise RuntimeError('Preflight failed: '+str([k for k,v in checks.items() if not v]))
    return result
