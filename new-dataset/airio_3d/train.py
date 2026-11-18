"""Fresh FP32 training of the official full-3D networks, with resumable ordering."""
import argparse,json,random,time,datetime
import numpy as np
import torch
from scipy.spatial.transform import Rotation
import contracts as c
import vendor,causal,physics
from banks import Bank


def setup():
    torch.set_num_threads(4);random.seed(42);np.random.seed(42);torch.manual_seed(42)
    if torch.cuda.is_available():torch.cuda.manual_seed_all(42)
    torch.backends.cuda.matmul.allow_tf32=False;torch.backends.cudnn.allow_tf32=False
    torch.backends.cudnn.benchmark=False;torch.backends.cudnn.deterministic=True


def rng_state():
    return dict(python=random.getstate(),numpy=np.random.get_state(),torch=torch.get_rng_state(),
                cuda=torch.cuda.get_rng_state_all() if torch.cuda.is_available() else [])


def restore_rng(value):
    random.setstate(value['python']);np.random.set_state(value['numpy']);torch.set_rng_state(value['torch'])
    if torch.cuda.is_available():torch.cuda.set_rng_state_all(value['cuda'])


def early_state(state,metric,eligible,old_lr,new_lr,config):
    """LR changes reset patience; failed physics cannot create a best model."""
    improved=eligible and metric<state['best']-config['min_delta']
    if improved:state['best']=metric;state['bad_epochs']=0
    if new_lr<old_lr:
        state['lr_reductions']+=1;state['bad_epochs']=0
    elif state['lr_reductions'] and not improved:state['bad_epochs']+=1
    stop=state['lr_reductions']>0 and np.isfinite(state['best']) and state['bad_epochs']>=config['early_patience']
    return improved,stop


def forward(model,bank,indices,stage):
    if stage=='airimu':
        data,initial,label=bank.batch(indices,stage);out=model(data,initial)
        return vendor.imu_loss(out,label),out['vel'][:,-1]-label['gt_vel'][:,-1]
    data,rotation,body=bank.batch(indices,stage);out=model(data,rotation);label=model.get_label(body)
    if out['net_vel'].shape!=label.shape:raise RuntimeError('Official endpoint label alignment failed')
    return vendor.io_loss(out,label),out['net_vel'][:,-1]-label[:,-1]


def teacher_validation(model,bank,stage):
    model.eval();sse=np.zeros(3);losses=0.;count=0
    with torch.no_grad():
        for start in range(0,len(bank),128):
            index=torch.arange(start,min(start+128,len(bank)),device=bank.device)
            loss,error=forward(model,bank,index,stage);n=len(index)
            sse+=(error**2).sum(0).cpu().numpy();losses+=float(loss)*n;count+=n
    return dict(loss=losses/count,component_rmse=np.sqrt(sse/count).tolist(),xyz_velocity_rmse=float(np.sqrt(sse.sum()/count)),windows=count)


def deploy_validation(model,imu,bias,imu_checkpoint,device):
    records=[];error_sum=zero_sum=0.;count=0
    for path in c.split_paths()['val']:
        r=c.load_record(path);runtime=c.load_runtime(path);orientation=causal.cached(path,imu,bias,imu_checkpoint,device)
        index,mean,var=causal.motion(runtime,orientation,model,device)
        keep=index>=1600;index=index[keep];mean=mean[keep]
        q=Rotation.from_quat(orientation['quaternion']);delta=Rotation.from_quat(r['quaternion'][1600]).as_euler('xyz')[2]-q[1600].as_euler('xyz')[2]
        world=Rotation.from_euler('z',delta).apply(q[index].apply(mean))
        e=world-r['velocity'][index];z=r['velocity'][index]
        s=float(np.sum(e*e));b=float(np.sum(z*z));error_sum+=s;zero_sum+=b;count+=len(index)
        ti,tm,_=causal.motion(runtime,orientation,model,device,teacher_quaternion=r['quaternion'])
        teacher_world=Rotation.from_quat(r['quaternion'][ti]).apply(tm);te=teacher_world-r['velocity'][ti]
        records.append(dict(source=str(path),samples=len(index),deployed_xyz_rmse=float(np.sqrt(s/len(index))),
            zero_xyz_rmse=float(np.sqrt(b/len(index))),teacher_xyz_rmse=float(np.sqrt(np.mean(np.sum(te[ti>=1600]**2,axis=1))))))
    ratio=float(np.sqrt(error_sum/zero_sum));return dict(records=records,xyz_velocity_rmse=float(np.sqrt(error_sum/count)),
        zero_xyz_rmse=float(np.sqrt(zero_sum/count)),zero_ratio=ratio,
        passed=ratio<=c.CONFIG['gates']['network_zero_baseline_ratio_max'],samples=count)


def main(stage,device='cuda'):
    setup();context=c.fingerprint();prepared=json.loads((c.HERE/'prepared/complete.json').read_text(encoding='utf-8'))
    if prepared['context']!=context:raise RuntimeError('Source/data/config changed after reference preparation')
    if device=='cuda' and not torch.cuda.is_available():raise RuntimeError('CUDA required for formal training')
    config=c.CONFIG[stage];run=c.HERE/'runs'/stage;run.mkdir(parents=True,exist_ok=True)
    complete=run/'complete.json'
    if complete.exists():
        result=json.loads(complete.read_text(encoding='utf-8'));best=run/'best.pt'
        if result['context']!=context or result['checkpoint_sha256']!=c.sha(best):raise RuntimeError('Completed run changed')
        return result
    model=getattr(vendor,stage)(device);optimizer=torch.optim.Adam(model.parameters(),lr=config['lr'],weight_decay=config['weight_decay'])
    scheduler=torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer,mode='min',factor=config['lr_factor'],patience=config['lr_patience'],min_lr=config['min_lr'])
    train=Bank('train',stage,device);val=Bank('val',stage,device)
    c.atomic_json(run/'window_inventory.json',dict(train=train.records,val=val.records))
    bias=prepared['calibration']['gyro_bias'];imu=imu_checkpoint=None
    if stage=='airio':
        imu_checkpoint=c.HERE/'runs/airimu/best.pt';frozen=json.loads((c.HERE/'runs/airimu/frozen_physics.json').read_text(encoding='utf-8'))
        if not frozen['passed'] or frozen['checkpoint_sha256']!=c.sha(imu_checkpoint):raise RuntimeError('AirIMU has not passed final physical validation')
        imu=vendor.airimu(device);imu.load_state_dict(torch.load(imu_checkpoint,map_location=device,weights_only=False)['model']);imu.eval()
        for p in imu.parameters():p.requires_grad_(False)
    state=dict(epoch=1,offset=0,order=None,total_loss=0.,samples=0,updates=0,history=[],
        early=dict(best=float('inf'),bad_epochs=0,lr_reductions=0),best=float('inf'))
    if (run/'last.pt').exists():
        saved=torch.load(run/'last.pt',map_location='cpu',weights_only=False)
        if saved['context']!=context:raise RuntimeError('Refusing resume with changed source/data/config')
        model.load_state_dict(saved['model']);optimizer.load_state_dict(saved['optimizer']);scheduler.load_state_dict(saved['scheduler'])
        state=saved['state'];restore_rng(saved['rng'])
        print('Exact resume',stage,'epoch',state['epoch'],'offset',state['offset'],flush=True)
    def checkpoint(destination):
        c.torch_save(destination,dict(model=model.state_dict(),optimizer=optimizer.state_dict(),scheduler=scheduler.state_dict(),
            state=state,rng=rng_state(),context=context,stage=stage,epoch=state['epoch'],seed=42))
    while state['epoch']<=config['epochs']:
        epoch=state['epoch'];started=time.time();model.train()
        if state['order'] is None:state['order']=torch.randperm(len(train)).tolist()
        for offset in range(state['offset'],len(train),128):
            index=torch.tensor(state['order'][offset:offset+128],device=device);optimizer.zero_grad(set_to_none=True)
            loss,_=forward(model,train,index,stage)
            if not torch.isfinite(loss):raise FloatingPointError('Nonfinite full 3D training loss')
            loss.backward()
            if any(p.grad is not None and not torch.isfinite(p.grad).all() for p in model.parameters()):raise FloatingPointError('Nonfinite gradient')
            optimizer.step();n=len(index);state['total_loss']+=float(loss.detach())*n;state['samples']+=n
            state['offset']=offset+n;state['updates']+=1
            if state['updates']%25==0:
                progress=dict(stage=stage,epoch=epoch,epochs=config['epochs'],batch_end=state['offset'],windows=len(train),
                    updates=state['updates'],train_loss=state['total_loss']/state['samples'],lr=optimizer.param_groups[0]['lr'],
                    timestamp=datetime.datetime.now().isoformat(),epoch_seconds_so_far=time.time()-started)
                c.atomic_json(run/'progress.json',progress);print(json.dumps(progress),flush=True)
            if state['updates']%100==0:checkpoint(run/'last.pt')
        teacher=teacher_validation(model,val,stage)
        if stage=='airimu':
            try:gate,_=physics.validate(model,bias,device)
            except (FloatingPointError,ValueError) as exc:gate=dict(passed=False,error=repr(exc))
            metric=teacher['xyz_velocity_rmse']
        else:
            gate=deploy_validation(model,imu,bias,imu_checkpoint,device);metric=gate['xyz_velocity_rmse']
        c.atomic_json(run/'checks'/f'epoch_{epoch:03d}.json',dict(epoch=epoch,teacher=teacher,gate=gate))
        old_lr=optimizer.param_groups[0]['lr'];scheduler.step(metric);new_lr=optimizer.param_groups[0]['lr']
        if stage=='airimu':improved,stop=early_state(state['early'],metric,gate['passed'],old_lr,new_lr,config)
        else:improved=gate['passed'] and metric<state['best'];stop=False
        row=dict(epoch=epoch,train_loss=state['total_loss']/state['samples'],teacher=teacher,gate=gate,
            metric=metric,eligible=gate['passed'],lr=new_lr,seconds=time.time()-started,updates=state['updates'])
        state['history'].append(row)
        if improved:
            state['best']=metric;checkpoint(run/'best.pt');c.atomic_json(run/'best.json',dict(epoch=epoch,metric=metric,gate=gate,checkpoint_sha256=c.sha(run/'best.pt')))
        c.atomic_json(run/'history.json',state['history']);print(stage,'EPOCH',epoch,'XYZ_RMSE',metric,'PHYSICS/BASELINE_PASS',gate['passed'],'LR',new_lr,flush=True)
        state.update(epoch=epoch+1,offset=0,order=None,total_loss=0.,samples=0);checkpoint(run/'last.pt')
        if stop:break
    if not (run/'best.pt').exists():raise RuntimeError(stage+' exhausted its budget without an eligible model; downstream training/promotion is blocked')
    result=dict(stage=stage,epochs_completed=state['epoch']-1,reason='early_stop' if stop else 'epoch_budget',
        checkpoint_sha256=c.sha(run/'best.pt'),context=context,best=json.loads((run/'best.json').read_text(encoding='utf-8')))
    c.atomic_json(complete,result);return result


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('stage',choices=['airimu','airio']);p.add_argument('--device',default='cuda');a=p.parse_args();main(a.stage,a.device)
