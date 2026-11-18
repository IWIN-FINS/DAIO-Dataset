"""Batch-128 planar training with exact optimizer-boundary resume checkpoints."""
import json,random,time,math
import numpy as np
import torch
from torch.utils.data import DataLoader
import planar_common as pc

def save(path,payload):
    tmp=path.with_suffix('.tmp');torch.save(payload,tmp);tmp.replace(path)

class OrderedBatches:
    """Explicit epoch permutation; checkpoint offsets do not depend on prefetch."""
    def __init__(self,order,start,batch):self.order,self.start,self.batch=order,start,batch
    def __iter__(self):
        for i in range(self.start,len(self.order),self.batch):yield self.order[i:i+self.batch].tolist()
    def __len__(self):return math.ceil((len(self.order)-self.start)/self.batch)

def train(stage,smoke=False,micro_batch=128,checkpoint_every_updates=100):
    if micro_batch<1 or micro_batch>128 or 128%micro_batch:raise ValueError('micro-batch must divide effective batch 128')
    torch.set_num_threads(4);random.seed(42);np.random.seed(42);torch.manual_seed(42);torch.cuda.manual_seed_all(42)
    torch.backends.cuda.matmul.allow_tf32=False;torch.backends.cudnn.allow_tf32=False
    device=torch.device('cuda' if torch.cuda.is_available() and not smoke else 'cpu')
    directory=pc.BASE/'runs'/stage;directory.mkdir(exist_ok=True,parents=True)
    settings={'stage':stage,'seed':42,'epochs':100 if stage=='airimu' else 200,'micro_batch':micro_batch,
              'effective_batch':128,'lr':.001,'weight_decay':.0001 if stage=='airimu' else .001,
              'window_samples':1000,'training_stride':10 if stage=='airimu' else 3,'gravity':9.80665,
              'supervision':'world XY position/velocity and yaw only; no vertical or tilt targets',
              'test_online_inputs':['acc_xyz','gyro_xyz'],'label_orientation_input':False,
              'throughput_revision':'20261031_batch128','sampler':'explicit_epoch_permutation',
              'checkpoint_every_updates':checkpoint_every_updates,'precision':'fp32_no_tf32'}
    old=None;context={'settings':settings,'fingerprint':pc.environment_fingerprint()}
    if not smoke:
        if (directory/'last.pt').exists():
            old=torch.load(directory/'last.pt',map_location='cpu',weights_only=False)
            if old['context']!=context:raise RuntimeError('Data, source, dependencies or settings changed: refuse resume')
        else:pc.atomic_json(directory/'context.json',context)
    training=pc.PlanarWindows('val' if smoke else 'train',stage);validation=pc.PlanarWindows('val',stage)
    if smoke:
        training=torch.utils.data.Subset(training,range(2));validation=torch.utils.data.Subset(validation,range(2))
    generator=torch.Generator().manual_seed(42)
    model=(pc.official_airimu() if stage=='airimu' else pc.official_airio()).to(device)
    optimizer=torch.optim.Adam(model.parameters(),lr=.001,weight_decay=settings['weight_decay'])
    scheduler=torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer,factor=.1 if stage=='airimu' else .2,patience=10 if stage=='airimu' else 5,min_lr=1e-5)
    loss_fn=pc.airimu_loss if stage=='airimu' else pc.airio_loss
    first=1;best=float('inf');updates=0
    if old:
        model.load_state_dict(old['model']);optimizer.load_state_dict(old['optimizer']);scheduler.load_state_dict(old['scheduler'])
        first=old['epoch'] if old['in_epoch'] else old['epoch']+1;best=old['best'];updates=old['updates']
        generator.set_state(old['shuffle_rng']);random.setstate(old['python_rng']);np.random.set_state(old['numpy_rng']);torch.set_rng_state(old['torch_rng'])
        if device.type=='cuda':torch.cuda.set_rng_state_all(old['cuda_rng'])
        committed=[];history=directory/'metrics.jsonl';completed=old['epoch']-int(old['in_epoch'])
        if history.exists():
            for line in history.read_text(encoding='utf-8').splitlines():
                try:
                    row=json.loads(line)
                    if row['epoch']<completed or (old['in_epoch'] and row['epoch']==completed):committed.append(row)
                except (ValueError,KeyError):pass
        if not old['in_epoch']:committed.append(old['metrics'])
        history.write_text(''.join(json.dumps(r)+'\n' for r in committed),encoding='utf-8')
        if not old['in_epoch'] and old['metrics']['best']:save(directory/'best.pt',old)
    def checkpoint(epoch,in_epoch,order,next_index,total,count,seconds,metrics=None):
        payload={'model':model.state_dict(),'optimizer':optimizer.state_dict(),'scheduler':scheduler.state_dict(),
                 'epoch':epoch,'in_epoch':in_epoch,'next_index':next_index,'order':order,'updates':updates,
                 'total_loss':total,'count':count,'elapsed_seconds':seconds,'best':best,'context':context,
                 'shuffle_rng':generator.get_state(),'python_rng':random.getstate(),'numpy_rng':np.random.get_state(),
                 'torch_rng':torch.get_rng_state(),'cuda_rng':torch.cuda.get_rng_state_all() if device.type=='cuda' else [],'metrics':metrics}
        save(directory/'last.pt',payload)
        return payload
    for epoch in range(first,2 if smoke else settings['epochs']+1):
        resumed=bool(old and old['in_epoch'] and epoch==first)
        order=old['order'] if resumed else torch.randperm(len(training),generator=generator)
        offset=old['next_index'] if resumed else 0
        total=old['total_loss'] if resumed else 0.;count=old['count'] if resumed else 0
        carried=old['elapsed_seconds'] if resumed else 0.;start=time.perf_counter()
        loader=DataLoader(training,batch_sampler=OrderedBatches(order,offset,2 if smoke else micro_batch),num_workers=0,
                          generator=torch.Generator().manual_seed(42),pin_memory=device.type=='cuda')
        model.train();optimizer.zero_grad(set_to_none=True);accum=0;seen=offset
        for i,batch in enumerate(loader,1):
            batch={k:v.to(device,non_blocking=device.type=='cuda') for k,v in batch.items()};n=len(batch['acc'])
            loss,metric=loss_fn(model,batch)
            if not torch.isfinite(loss):raise RuntimeError('Nonfinite planar loss')
            (loss*n/128).backward();accum+=n;seen+=n;total+=float(loss.detach())*n;count+=n
            boundary=accum>=128 or i==len(loader)
            if boundary:
                if accum<128:
                    for param in model.parameters():
                        if param.grad is not None:param.grad.mul_(128/accum)
                torch.nn.utils.clip_grad_norm_(model.parameters(),5,error_if_nonfinite=True);optimizer.step();optimizer.zero_grad(set_to_none=True);accum=0;updates+=1
                seconds=carried+time.perf_counter()-start
                if not smoke and updates%checkpoint_every_updates==0:checkpoint(epoch,True,order,seen,total,count,seconds)
                if not smoke and (updates%25==0 or i==len(loader)):
                    pc.atomic_json(directory/'progress.json',{'epoch':epoch,'epochs':settings['epochs'],'windows_done':seen,'windows_total':len(training),
                        'epoch_percent':100*seen/len(training),'micro_batch':micro_batch,'effective_batch':128,'optimizer_updates':updates,
                        'windows_per_second':(seen-offset)/max(time.perf_counter()-start,1e-9),'epoch_elapsed_seconds':seconds})
                if updates%100==0 or i==len(loader):
                    print(f'{stage} epoch={epoch} windows={seen}/{len(training)} micro_batch={micro_batch} loss={total/count:.6f} windows_per_s={(seen-offset)/max(time.perf_counter()-start,1e-9):.1f}',flush=True)
        model.eval();square=0;val_count=0
        val_loader=DataLoader(validation,batch_size=2 if smoke else micro_batch,shuffle=False,num_workers=0,pin_memory=device.type=='cuda',generator=torch.Generator().manual_seed(42))
        with torch.no_grad():
            for batch in val_loader:
                batch={k:v.to(device,non_blocking=device.type=='cuda') for k,v in batch.items()};_,metric=loss_fn(model,batch)
                square+=float(metric)*len(batch['acc']);val_count+=len(batch['acc'])
        rmse=math.sqrt(square/val_count);scheduler.step(rmse);improved=rmse<best;best=min(best,rmse)
        row={'epoch':epoch,'train_loss':total/count,'val_xy_velocity_vector_rmse':rmse,'seconds':carried+time.perf_counter()-start,'best':improved}
        print(json.dumps(row),flush=True)
        if smoke:return row
        payload=checkpoint(epoch,False,None,0,0,0,0,metrics=row)
        if improved:save(directory/'best.pt',payload)
        with (directory/'metrics.jsonl').open('a',encoding='utf-8') as f:f.write(json.dumps(row)+'\n')
        pc.atomic_json(directory/'progress.json',{'epoch':epoch,'epochs':settings['epochs'],'epoch_complete':True,'best_validation_vector_rmse':best})
        old=None
    pc.atomic_json(directory/'complete.json',{'best_validation_vector_rmse':best,'epochs':settings['epochs']})
