"""Gated, isolated Test1-4 replay; no changes to any existing method/results."""
import csv
import numpy as np
import torch
from scipy.spatial.transform import Rotation
from common import HERE,REPORT,LEGACY,IMU,c,read,write,model,now,digest
from diagnosis import formal_lock
from trainer import run_path
from data import Bank,orientation,causal_motion
from metrics import validation
import runtime_filter


def trajectory_metrics(time,position,velocity,yaw,ref_position,ref_velocity,ref_yaw,mask):
    ev=velocity-ref_velocity;ep=position-ref_position
    if not mask.any():raise ValueError('Empty scoring mask')
    angle=np.arctan2(np.sin(yaw-ref_yaw),np.cos(yaw-ref_yaw))
    ticks=np.rint(np.asarray(time)/.005).astype(np.int64)
    lookup={int(t):i for i,t in enumerate(ticks)}
    starts=np.asarray([i for i,t in enumerate(ticks) if int(t+1000) in lookup],dtype=int)
    ends=np.asarray([lookup[int(ticks[i]+1000)] for i in starts],dtype=int)
    take=mask[starts]&mask[ends];starts=starts[take];ends=ends[take]
    if not len(starts):raise ValueError('No exact five-second pairs')
    relative=ep[ends,:2]-ep[starts,:2]
    return dict(samples=int(mask.sum()),velocity_x_rmse=float(np.sqrt(np.mean(ev[mask,0]**2))),
        velocity_y_rmse=float(np.sqrt(np.mean(ev[mask,1]**2))),velocity_xy_rmse=float(np.sqrt(np.mean(np.sum(ev[mask,:2]**2,axis=1)))),
        position_x_rmse=float(np.sqrt(np.mean(ep[mask,0]**2))),position_y_rmse=float(np.sqrt(np.mean(ep[mask,1]**2))),
        position_xy_rmse=float(np.sqrt(np.mean(np.sum(ep[mask,:2]**2,axis=1)))),
        final_position_error=float(np.linalg.norm(ep[np.flatnonzero(mask)[-1],:2])),
        relative_displacement_5s_rmse=float(np.sqrt(np.mean(np.sum(relative**2,axis=1)))),relative_pairs=len(starts),
        yaw_rmse_rad=float(np.sqrt(np.mean(angle[mask]**2))),
        velocity_z_rmse=float(np.sqrt(np.mean(ev[mask,2]**2))),position_z_rmse=float(np.sqrt(np.mean(ep[mask,2]**2))),
        velocity_xyz_rmse=float(np.sqrt(np.mean(np.sum(ev[mask]**2,axis=1)))),
        position_xyz_rmse=float(np.sqrt(np.mean(np.sum(ep[mask]**2,axis=1)))))


def qualified_model(ctx,device):
    lock=formal_lock(ctx);folder=run_path('formal',lock['spec'],42);complete=read(folder/'complete.json')
    if complete['best_qualified'] is None:raise RuntimeError('Formal model never passed validation')
    ckpt=folder/'best_qualified.pt'
    if c.sha(ckpt)!=complete['checkpoint_hashes']['best_qualified.pt']:raise RuntimeError('Formal checkpoint hash changed')
    payload=torch.load(ckpt,map_location='cpu',weights_only=False)
    if payload['context']!=ctx['signature']:raise RuntimeError('Formal model source/config changed')
    net=model(lock['spec'],42,device);net.load_state_dict(payload['model']);net.eval()
    val=Bank('val',device,next(net.parameters()).dtype);gate=validation(net,val)
    if not gate['qualification']['passed']:raise RuntimeError('Frozen best model failed repeated validation')
    write(HERE/'formal_frozen_validation.json',dict(checkpoint_sha256=c.sha(ckpt),gate=gate,signature=ctx['signature']))
    del val
    return net,ckpt,lock


def run(ctx,device='cuda'):
    net,checkpoint,lock=qualified_model(ctx,device)
    out=REPORT/'evaluation';out.mkdir(parents=True,exist_ok=True)
    signature=digest(dict(context=ctx['signature'],checkpoint=c.sha(checkpoint),imu=c.sha(IMU)))
    rows=[];artifacts={}
    manifest={r['processed_relative_path']:r for r in csv.DictReader((c.DATA/'processed/manifest.csv').open(encoding='utf-8'))}
    for number,path in enumerate(c.split_paths()['demo'],1):
        reference=c.load_record(path);runtime=c.load_runtime(path);ori=orientation(path)
        full_path=out/f'test{number}_full_state.npz';sample_path=out/f'test{number}_samples.npz'
        if full_path.exists():
            with np.load(full_path) as z:
                if str(z['method_signature'])!=signature or str(z['source_sha256'])!=c.sha(path):raise RuntimeError('Stale evaluation cache')
                full={k:z[k] for k in z.files}
        else:
            ix,means,variance=causal_motion(runtime,ori['rotvec'],net)
            full=runtime_filter.replay(runtime,ori,ix,means,variance,float(reference['reference_rpy'][1600,2]))
            full['state'][1600:,6:9]+=reference['position_world'][1600]-full['state'][1600,6:9]
            c.save_npz(full_path,**full,method_signature=signature,source_sha256=c.sha(path),checkpoint_sha256=c.sha(checkpoint))
        for key in ('state','quaternion','covariance'):
            if not np.isfinite(full[key]).all():raise FloatingPointError('Nonfinite '+key)
        covariance=full['covariance']
        np.testing.assert_allclose(covariance,covariance.swapaxes(-1,-2),atol=1e-6,rtol=0)
        for i in range(0,len(covariance),1024):
            cov=covariance[i:i+1024]
            if np.linalg.eigvalsh((cov+cov.swapaxes(-1,-2))*.5).min()<-1e-7:raise FloatingPointError('Non-PSD covariance')
        np.testing.assert_allclose(np.linalg.norm(full['quaternion'],axis=1),1,atol=1e-10,rtol=0)
        np.testing.assert_allclose(full['state'][0,3:6],0,atol=0,rtol=0)
        np.testing.assert_allclose(full['state'][1600,6:9],reference['position_world'][1600],atol=1e-10,rtol=0)
        yaw=Rotation.from_quat(full['quaternion']).as_euler('xyz')[:,2]
        t=runtime['time'][1600:];pos=full['state'][1600:,6:9];vel=full['state'][1600:,3:6];y=yaw[1600:]
        pgt=reference['position_world'][1600:];vgt=reference['velocity_world'][1600:];ygt=reference['reference_rpy'][1600:,2]
        c.save_npz(sample_path,time=t,position=pos[:,:2],velocity=vel[:,:2],yaw=y,position_xyz=pos,velocity_xyz=vel,
            reference_position=pgt,reference_velocity=vgt,reference_yaw=ygt,quaternion=full['quaternion'][1600:],
            source_sha256=c.sha(path),checkpoint_sha256=c.sha(checkpoint),method_signature=signature)
        matrix=np.c_[t,t-8,pgt,vgt,ygt,pos,vel,y]
        csv_path=out/f'test{number}_samples.csv'
        np.savetxt(csv_path,matrix,delimiter=',',fmt='%.17g',comments='',encoding='utf-8',
            header='time_s,time_after_initialization_s,gt_x,gt_y,gt_z,gt_vx,gt_vy,gt_vz,gt_yaw,airio_x,airio_y,airio_z,airio_vx,airio_vy,airio_vz,airio_yaw')
        # Independently reload exported CSV and verify full precision and actual timestamp pairs.
        reload=np.loadtxt(csv_path,delimiter=',',skiprows=1)
        np.testing.assert_array_equal(reload,matrix)
        all_mask=np.ones(len(t),bool);clean=all_mask.copy()
        record=manifest[path.relative_to(c.DATA/'processed').as_posix()]
        import json
        repaired=np.asarray(json.loads(record['repaired_sample_indices']),int)-1600
        clean[repaired[(repaired>=0)&(repaired<len(t))]]=False
        for kind,mask in [('all_samples',all_mask),('without_repaired_samples',clean)]:
            m=trajectory_metrics(reload[:,0],reload[:,9:12],reload[:,12:15],reload[:,15],reload[:,2:5],reload[:,5:8],reload[:,8],mask)
            er=(Rotation.from_quat(full['quaternion'][1600:]).inv()*Rotation.from_quat(reference['quaternion_world'][1600:])).as_rotvec()
            m['attitude_so3_rmse_rad']=float(np.sqrt(np.mean(np.sum(er[mask]**2,axis=1))))
            rows.append(dict(test=f'Test {number}',scoring=kind,**m))
        for f in (full_path,sample_path,csv_path):artifacts[f.name]=c.sha(f)
        print('TEST',number,'validated',len(t),'samples',flush=True)
    for kind in ('all_samples','without_repaired_samples'):
        selected=[r for r in rows if r['scoring']==kind]
        rows.append(dict(test='Mean',scoring=kind,**{k:float(np.mean([r[k] for r in selected])) for k in selected[0] if k not in ('test','scoring')}))
    with (out/'metrics.csv').open('w',encoding='utf-8-sig',newline='') as f:
        writer=csv.DictWriter(f,fieldnames=list(rows[0]));writer.writeheader();writer.writerows(rows)
    provenance=dict(accepted=True,signature=signature,context=ctx['signature'],checkpoint=str(checkpoint),checkpoint_sha256=c.sha(checkpoint),
        imu_checkpoint=str(IMU),imu_sha256=c.sha(IMU),spec=lock['spec'],artifacts=artifacts,time=now(),
        reference='mixed AprilTag/pressure/IMU reference; not independent 6DoF mocap',
        online='six-axis raw IMU only, causal trailing-window velocity, zero initial velocity; one yaw/XYZ anchor at8s',
        variance='Original 15-state filter; acc covariance x100, velocity covariance x0.1, original remaining settings',
        rve='5s XY displacement error between exact timestamp endpoints; no new rotational re-alignment',
        differences='See recovery README; endpoint-only causal deployment differs from official whole-sequence offline inference')
    write(out/'provenance.json',provenance)
    return provenance
