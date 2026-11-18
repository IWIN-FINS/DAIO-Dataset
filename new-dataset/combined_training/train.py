"""FP32 cached two-model trainer, macro-validation selection and exact sampler resume."""
import argparse,time,datetime
import numpy as np
import torch
import common as c
from rovio_ablation.model import build_model
from rovio_ablation.metrics import hybrid_loss
from rovio_ablation.utils import set_seed,rng_state,restore_rng_state

def setup():
    torch.set_num_threads(4);set_seed(42,False)
    torch.backends.cuda.matmul.allow_tf32=False;torch.backends.cudnn.allow_tf32=False
    torch.backends.cudnn.benchmark=False;torch.backends.cudnn.deterministic=True

def permutation(n,epoch):return torch.randperm(n,generator=torch.Generator().manual_seed(42+epoch))

def predictions(model,bank):
    model.eval();mean=[];std=[]
    with torch.inference_mode():
        for start in range(0,len(bank),128):
            index=torch.arange(start,min(start+128,len(bank)),device=bank.device)
            m,s=model(*bank.inputs(index,model.input_keys));mean.append(m.cpu().numpy());std.append(s.exp().cpu().numpy())
    return np.concatenate(mean),np.concatenate(std)

def validation(model,bank):
    mean,std=predictions(model,bank);target=bank.target.cpu().numpy();records=[]
    if not np.isfinite(mean).all() or not np.isfinite(std).all() or np.any(std<=0):raise FloatingPointError('Invalid validation output')
    for r in bank.records:
        if r['hi']==r['lo']:continue
        sl=slice(r['lo'],r['hi']);records.append(dict(path=r['path'],**c.metric(r['time'],mean[sl],target[sl])))
    return dict(ave_mps=float(np.mean([r['ave_mps'] for r in records])),
        old_ave_mps=float(np.mean([r['ave_mps'] for r in records if '/old/' in r['path'].replace('\\','/')])),
        new_ave_mps=float(np.mean([r['ave_mps'] for r in records if '/new/' in r['path'].replace('\\','/')])),records=records)

def main(name):
    setup();device='cuda'
    if not torch.cuda.is_available():raise RuntimeError('CUDA unavailable')
    cfg=c.cfg(name);context=c.fingerprint();prepared=c.read(c.DATA/'feature_cache/complete.json')
    if prepared['context']!=context:raise RuntimeError('Prepared context differs')
    run=c.RUN/name;complete=run/'complete.json'
    if complete.exists():
        done=c.read(complete)
        if done['context']!=context or done['checkpoint_sha256']!=c.sha(run/'best.pt'):raise RuntimeError('Completed experiment changed')
        return
    model=build_model(cfg).to(device);optimizer=torch.optim.AdamW(model.parameters(),lr=.001,weight_decay=.0001)
    scheduler=torch.optim.lr_scheduler.CosineAnnealingLR(optimizer,T_max=80)
    train=c.Bank('train',device);val=c.Bank('val',device);normalizer=c.normalizer(name)
    state=dict(epoch=1,offset=0,updates=0,best=float('inf'),best_epoch=None,loss_sum=0.,samples=0,history=[])
    if (run/'last.pt').exists():
        saved=torch.load(run/'last.pt',map_location='cpu',weights_only=False)
        if saved['context']!=context:raise RuntimeError('Resume source/data/config differs')
        model.load_state_dict(saved['model']);optimizer.load_state_dict(saved['optimizer']);scheduler.load_state_dict(saved['scheduler'])
        state=saved['training_state'];restore_rng_state(saved['rng_state']);print('RESUME',name,state['epoch'],state['offset'],flush=True)
    def checkpoint(path):
        c.torch_save(path,dict(format_version=1,epoch=state['epoch'],model=model.state_dict(),optimizer=optimizer.state_dict(),
            scheduler=scheduler.state_dict(),best_metric=state['best'],best_epoch=state['best_epoch'],normalizer=normalizer,
            config=cfg,config_sha256=c.config_hash(cfg),manifest_sha256=c.sha(c.DATA/'split.yaml'),input_contract=c.v20_input_contract(cfg),
            rng_state=rng_state(),training_state=state,context=context,sampler='torch.randperm(generator seed=42+epoch)',
            sampler_sha256=__import__('hashlib').sha256(permutation(len(train),min(state['epoch'],80)).numpy().tobytes()).hexdigest()))
    while state['epoch']<=80:
        epoch=state['epoch'];phase=1 if epoch<=40 else 2;order=permutation(len(train),epoch);start_time=time.time();model.train()
        for offset in range(state['offset'],len(train),128):
            index=order[offset:offset+128].to(device);optimizer.zero_grad(set_to_none=True)
            mean,logstd=model(*train.inputs(index,model.input_keys));loss,_=hybrid_loss(mean,logstd,train.target[index],phase,.1,.05)
            if not torch.isfinite(loss):raise FloatingPointError('Nonfinite training loss')
            loss.backward();torch.nn.utils.clip_grad_norm_(model.parameters(),5.,error_if_nonfinite=True);optimizer.step()
            n=len(index);state['offset']=offset+n;state['updates']+=1;state['samples']+=n;state['loss_sum']+=float(loss.detach())*n
            if state['updates']%25==0:
                p=dict(model=name,epoch=epoch,epochs=80,phase=phase,windows_done=state['offset'],windows=len(train),updates=state['updates'],
                    train_loss=state['loss_sum']/state['samples'],lr=optimizer.param_groups[0]['lr'],updated=datetime.datetime.now().isoformat())
                c.atomic_json(run/'progress.json',p);print(c.json.dumps(p),flush=True)
            if state['updates']%250==0:checkpoint(run/'last.pt')
        metrics=validation(model,val);scheduler.step()
        improved=phase==2 and metrics['ave_mps']<state['best']
        row=dict(epoch=epoch,phase=phase,training_loss=state['loss_sum']/state['samples'],validation=metrics,
            seconds=time.time()-start_time,lr=optimizer.param_groups[0]['lr'],is_best=improved)
        state['history'].append(row)
        if improved:
            state['best']=metrics['ave_mps'];state['best_epoch']=epoch;checkpoint(run/'best.pt')
            c.atomic_json(run/'best.json',dict(epoch=epoch,validation=metrics,checkpoint_sha256=c.sha(run/'best.pt')))
        c.atomic_json(run/'history.json',state['history']);print(name,'EPOCH',epoch,'VAL_MACRO_AVE',metrics['ave_mps'],'BEST',state['best_epoch'],flush=True)
        state.update(epoch=epoch+1,offset=0,loss_sum=0.,samples=0);checkpoint(run/'last.pt')
    if not (run/'best.pt').exists():raise RuntimeError('No phase2 checkpoint selected')
    c.atomic_json(complete,dict(model=name,epochs=80,best_epoch=state['best_epoch'],checkpoint_sha256=c.sha(run/'best.pt'),context=context))

if __name__=='__main__':
    parser=argparse.ArgumentParser();parser.add_argument('model',choices=['daio','no_rpm']);main(parser.parse_args().model)
