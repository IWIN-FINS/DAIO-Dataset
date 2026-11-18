"""Held-out network evaluation on8 tracks, and 200Hz ESKF replay on new4 only."""
import csv,copy
import numpy as np
import torch
import common as c
import train
from rovio_ablation.model import build_model
from rov_eskf.model import load_checkpoint
from rov_eskf.data import load_sequence
from rov_eskf.windows import network_observations
from rov_eskf.replay import replay_attitude_aided_11,save_replay11
from rov_eskf.evaluate import offline_reference,state_projection

def csv_write(path,rows):
    path.parent.mkdir(parents=True,exist_ok=True)
    keys=list(dict.fromkeys(k for r in rows for k in r))
    with path.open('w',encoding='utf-8-sig',newline='') as f:
        writer=csv.DictWriter(f,fieldnames=keys);writer.writeheader();writer.writerows(rows)

def odom_metrics(p,v,y,gt,gv,gy,mask):
    e=p-gt;ev=v-gv;ey=np.arctan2(np.sin(y-gy),np.cos(y-gy));lag=1000;pair=mask[lag:]&mask[:-lag]
    relative=(p[lag:]-p[:-lag])-(gt[lag:]-gt[:-lag]);rmse=lambda a:float(np.sqrt(np.mean(a*a)))
    return dict(samples=int(mask.sum()),vx_rmse_mps=rmse(ev[mask,0]),vy_rmse_mps=rmse(ev[mask,1]),
        velocity_xy_rmse_mps=float(np.sqrt(np.mean(np.sum(ev[mask]**2,axis=1)))),
        px_rmse_m=rmse(e[mask,0]),py_rmse_m=rmse(e[mask,1]),position_xy_rmse_m=float(np.sqrt(np.mean(np.sum(e[mask]**2,axis=1)))),
        final_position_error_m=float(np.linalg.norm(e[-1])),rpe_5s_rmse_m=float(np.sqrt(np.mean(np.sum(relative[pair]**2,axis=1)))),yaw_rmse_rad=rmse(ey[mask]))

def main():
    train.setup();c.REPORT.mkdir(parents=True,exist_ok=True);context=c.fingerprint();bank=c.Bank('demo','cuda');network_rows=[];odom_rows=[];models={}
    manifest={r['processed_relative_path']:r for r in csv.DictReader((c.DATA/'processed/manifest.csv').open(encoding='utf-8'))}
    for name in ('daio','no_rpm'):
        ckpt=c.RUN/name/'best.pt';done=c.read(c.RUN/name/'complete.json')
        if done['context']!=context or done['checkpoint_sha256']!=c.sha(ckpt):raise RuntimeError('Model provenance mismatch')
        model,payload=load_checkpoint(ckpt,'cuda');mean,std=train.predictions(model,bank);target=bank.target.cpu().numpy()
        models[name]=dict(checkpoint=str(ckpt),sha256=c.sha(ckpt),epoch=payload['epoch'])
        for record in bank.records:
            path=c.Path(record['path']);relative=path.relative_to(c.DATA/'processed');domain=relative.parts[0];number=('025','028','029','030').index(path.name[:3])+1
            sl=slice(record['lo'],record['hi']);p=mean[sl];s=std[sl];gt=target[sl];time=record['time']
            folder=c.REPORT/'network'/domain/name;folder.mkdir(parents=True,exist_ok=True)
            c.npz(folder/f'test{number}.npz',time=time,prediction=p,std=s,target=gt,source_sha256=c.sha(path),checkpoint_sha256=c.sha(ckpt))
            np.savetxt(folder/f'test{number}.csv',np.c_[time,p,s,gt],delimiter=',',header='time,vx,vy,yaw_rate,std_vx,std_vy,std_yaw_rate,gt_vx,gt_vy,gt_yaw_rate',comments='')
            network_rows.append(dict(model=name,domain=domain,test=f'Test {number}',**c.metric(time,p,gt)))
        for domain in ('old','new','all'):
            selected=[r for r in network_rows if r['model']==name and r['test']!='Mean' and (domain=='all' or r['domain']==domain)]
            network_rows.append(dict(model=name,domain=domain,test='Mean',**{k:float(np.mean([r[k] for r in selected])) for k in ('ave_mps','rve_5s_mps','vx_rmse_mps','vy_rmse_mps','yaw_rate_rmse_radps')}))
        for number,path in enumerate([p for p in c.paths()['demo'] if '/new/' in p.as_posix()],1):
            directory=c.REPORT/'odometry'/name/f'test{number}';saved=directory/'samples.npz'
            seq=load_sequence(path,require_reference=True);gt,gy,gv=offline_reference(seq,seq.imu_ts,101,3)
            config=c.yaml.safe_load((c.ROOT/'ROV-ESKF-Odometry/configs/benchmark_paper_test1_4_current_model.yaml').read_text(encoding='utf-8'))
            config['model'].update(checkpoint=str(ckpt),device='cuda')
            config['data'].update(root=str(c.DATA),manifest=str(c.DATA/'split.yaml'),records=[])
            signature=c.sha(ckpt)+c.sha(path)+c.sha(__file__)
            if saved.exists():
                with np.load(saved) as z:
                    if str(z['signature'])!=signature:raise RuntimeError('Stale odometry output')
                    ts=z['time'];p=z['position'];v=z['velocity'];y=z['yaw']
            else:
                network=network_observations(seq,model,payload,'cuda',batch_size=128)
                full=replay_attitude_aided_11(seq,config,network=network,reanchor_yaw=float(gy[1600]),reanchor_time=8.)
                if not np.isfinite(full.state).all() or not np.isfinite(full.covariance_diag).all() or np.min(full.covariance_diag)<-1e-9:raise FloatingPointError('Invalid ESKF states/covariance')
                save_replay11(full,directory,save_covariance=True,save_event_trace=True,suffix='_full')
                state=state_projection(full.state);ts=seq.imu_ts[1600:];p=state[1600:,:2].copy();p+=gt[1600]-p[0];v=state[1600:,3:5];y=state[1600:,6]
                c.npz(saved,time=ts,position=p,velocity=v,yaw=y,reference_position=gt[1600:],reference_velocity=gv[1600:],reference_yaw=gy[1600:],
                    source_sha256=c.sha(path),checkpoint_sha256=c.sha(ckpt),signature=signature)
                np.savetxt(directory/'samples.csv',np.c_[ts,p,v,y,gt[1600:],gv[1600:],gy[1600:]],delimiter=',',
                    header='time,x,y,vx,vy,yaw,gt_x,gt_y,gt_vx,gt_vy,gt_yaw',comments='')
                (directory/'config.yaml').write_text(c.yaml.safe_dump(config,sort_keys=False),encoding='utf-8')
            mask=np.ones(len(ts),bool);rec=manifest[path.relative_to(c.DATA/'processed').as_posix()];repair=np.array(c.json.loads(rec['repaired_sample_indices']),int)-1600
            excluded=mask.copy();excluded[repair[(repair>=0)&(repair<len(mask))]]=False
            for score,m in [('all_samples',mask),('without_repaired_samples',excluded)]:
                odom_rows.append(dict(model=name,test=f'Test {number}',scoring=score,**odom_metrics(p,v,y,gt[1600:],gv[1600:],gy[1600:],m)))
            print('ESKF',name,'Test',number,odom_rows[-2],flush=True)
        for score in ('all_samples','without_repaired_samples'):
            selected=[r for r in odom_rows if r['model']==name and r['scoring']==score]
            odom_rows.append(dict(model=name,test='Mean',scoring=score,**{k:float(np.mean([r[k] for r in selected])) for k in selected[0] if k not in ('model','test','scoring','samples')}))
    csv_write(c.REPORT/'network_metrics.csv',network_rows);csv_write(c.REPORT/'odometry_metrics.csv',odom_rows)
    c.atomic_json(c.REPORT/'provenance.json',dict(context=context,models=models,selection='four validation record AVE macro mean, epochs41--80',
        odometry='new tests only;200Hz;0--8s warmup;zero initial velocity;one yaw/XY alignment at8s',network='common native50Hz valid windows, separate old/new domains'))

if __name__=='__main__':main()
