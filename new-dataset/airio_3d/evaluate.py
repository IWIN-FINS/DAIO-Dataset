"""Test only after validation gates; write full 3D and common horizontal interfaces."""
import csv,json,time
import numpy as np
import torch
from scipy.spatial.transform import Rotation
import contracts as c
import vendor,causal,runtime_filter


def main(device='cuda'):
    import train
    train.setup()
    context=c.fingerprint();run=c.HERE/'runs/airio';checkpoint=run/'best.pt';imu_checkpoint=c.HERE/'runs/airimu/best.pt'
    complete=json.loads((run/'complete.json').read_text(encoding='utf-8'));frozen=json.loads((c.HERE/'runs/airimu/frozen_physics.json').read_text(encoding='utf-8'))
    if complete['context']!=context or complete['checkpoint_sha256']!=c.sha(checkpoint) or not complete['best']['gate']['passed']:raise RuntimeError('AirIO is not eligible')
    if not frozen['passed'] or frozen['checkpoint_sha256']!=c.sha(imu_checkpoint):raise RuntimeError('AirIMU gate changed')
    bias=json.loads((c.HERE/'prepared/warmup_calibration.json').read_text(encoding='utf-8'))['gyro_bias']
    imu=vendor.airimu(device);imu.load_state_dict(torch.load(imu_checkpoint,map_location=device,weights_only=False)['model']);imu.eval()
    io=vendor.airio(device);io.load_state_dict(torch.load(checkpoint,map_location=device,weights_only=False)['model']);io.eval()
    signature=json.dumps(dict(context=context,imu=c.sha(imu_checkpoint),airio=c.sha(checkpoint)),sort_keys=True)
    import hashlib
    signature=hashlib.sha256(signature.encode()).hexdigest();out=c.REPORT/'airio_3d';out.mkdir(exist_ok=True,parents=True);rows=[]
    for number,path in enumerate(c.split_paths()['demo'],1):
        sample_path=out/f'test{number}_samples.npz';full_path=out/f'test{number}_full_state.npz'
        reference=c.load_record(path);runtime=c.load_runtime(path)
        if sample_path.exists():
            with np.load(sample_path) as z:
                if str(z['method_signature'])!=signature or str(z['source_sha256'])!=c.sha(path):raise RuntimeError('Stale 3D test output')
            with np.load(full_path) as z:full={k:z[k] for k in z.files}
        else:
            orientation=causal.cached(path,imu,bias,imu_checkpoint,device);indices,means,variances=causal.motion(runtime,orientation,io,device)
            full=runtime_filter.replay(runtime,orientation,indices,means,variances,float(reference['reference_rpy'][1600,2]))
            # Coordinate translation is applied once, preserving the mixed-reference world origin.
            full['state'][1600:,6:9]+=reference['position_world'][1600]-full['state'][1600,6:9]
            full['reference_position_world']=reference['position_world'];full['reference_velocity_world']=reference['velocity_world'];full['reference_quaternion_world']=reference['quaternion_world']
            c.save_npz(full_path,**full,checkpoint_sha256=c.sha(checkpoint),imu_checkpoint_sha256=c.sha(imu_checkpoint),method_signature=signature)
            yaw=Rotation.from_quat(full['quaternion']).as_euler('xyz')[:,2]
            c.save_npz(sample_path,time=runtime['time'][1600:],position=full['state'][1600:,6:8],velocity=full['state'][1600:,3:5],yaw=yaw[1600:],
                source_sha256=c.sha(path),checkpoint_sha256=c.sha(checkpoint),method_signature=signature)
        e_v=full['state'][1600:,3:6]-reference['velocity_world'][1600:];e_p=full['state'][1600:,6:9]-reference['position_world'][1600:]
        er=(Rotation.from_quat(full['quaternion'][1600:]).inv()*Rotation.from_quat(reference['quaternion_world'][1600:])).as_rotvec()
        manifest={r['processed_relative_path']:r for r in csv.DictReader((c.DATA/'processed/manifest.csv').open(encoding='utf-8'))}
        without=np.ones(len(e_v),bool);rec=manifest[path.relative_to(c.DATA/'processed').as_posix()]
        repaired=np.asarray(json.loads(rec['repaired_sample_indices']),dtype=int)-1600
        without[repaired[(repaired>=0)&(repaired<len(without))]]=False
        rpy=Rotation.from_quat(full['quaternion'][1600:]).as_euler('xyz')-reference['reference_rpy'][1600:]
        rpy=np.arctan2(np.sin(rpy),np.cos(rpy))
        for score,mask in [('all_samples',np.ones(len(without),bool)),('without_repaired_samples',without)]:
            row=dict(test=f'Test {number}',scoring=score,samples=int(mask.sum()),z_velocity_rmse=float(np.sqrt(np.mean(e_v[mask,2]**2))),z_position_rmse=float(np.sqrt(np.mean(e_p[mask,2]**2))),
                xyz_velocity_rmse=float(np.sqrt(np.mean(np.sum(e_v[mask]**2,axis=1)))),xyz_position_rmse=float(np.sqrt(np.mean(np.sum(e_p[mask]**2,axis=1)))),
                attitude_so3_rmse_rad=float(np.sqrt(np.mean(np.sum(er[mask]**2,axis=1)))),
                roll_rmse_rad=float(np.sqrt(np.mean(rpy[mask,0]**2))),pitch_rmse_rad=float(np.sqrt(np.mean(rpy[mask,1]**2))),yaw_rmse_rad=float(np.sqrt(np.mean(rpy[mask,2]**2))))
            rows.append(row);print('3D TEST',row,flush=True)
    for score in ('all_samples','without_repaired_samples'):
        selected=[r for r in rows if r['scoring']==score]
        rows.append(dict(test='Mean',scoring=score,**{k:float(np.mean([r[k] for r in selected])) for k in selected[0] if k.endswith(('_rmse','_rad'))}))
    with (out/'metrics_3d.csv').open('w',encoding='utf-8-sig',newline='') as f:
        w=csv.DictWriter(f,fieldnames=list(rows[0]));w.writeheader();w.writerows(rows)
    result=dict(checkpoint=str(checkpoint),sha256=c.sha(checkpoint),imu_checkpoint=str(imu_checkpoint),imu_sha256=c.sha(imu_checkpoint),
        context=context,reference='AprilTag XY/yaw + pressure Z + IMU solved roll/pitch, sensor-assisted reference',
        online='six-axis IMU only; initial velocity zero; single yaw/XYZ coordinate alignment at 8s; no continuous reference updates',method_signature=signature)
    c.atomic_json(out/'provenance.json',result);return result


if __name__=='__main__':main()
