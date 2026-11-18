"""Eight official DeepVL GRUs; explicit optimizer updates and exact resume."""
import argparse,json,random,time
import numpy as np
import torch
import common as c
from prepare import prepare


def rng_state(generator):
    return dict(python=random.getstate(),numpy=np.random.get_state(),torch=torch.get_rng_state(),
                cuda=torch.cuda.get_rng_state_all() if torch.cuda.is_available() else [],sampler=generator.get_state())


def restore(state,generator):
    random.setstate(state['python']);np.random.set_state(state['numpy']);torch.set_rng_state(state['torch'])
    if state['cuda']:torch.cuda.set_rng_state_all(state['cuda'])
    generator.set_state(state['sampler'])


def main(smoke=False):
    torch.set_num_threads(4);torch.backends.cuda.matmul.allow_tf32=False;torch.backends.cudnn.allow_tf32=False
    random.seed(42);np.random.seed(42);torch.manual_seed(42)
    device='cuda' if torch.cuda.is_available() else 'cpu'
    d=prepare(); arrays=np.load(d/'arrays.npz');stats=json.loads((d/'normalization.json').read_text())
    run=c.HERE/('smoke' if smoke else 'runs');run.mkdir(exist_ok=True)
    context=c.fingerprint(); models=[];opts=[];schedulers=[]
    for seed in range(42,50):
        torch.manual_seed(seed);m=c.model(device);models.append(m)
        op=torch.optim.Adam(m.parameters(),lr=.001);opts.append(op)
        schedulers.append(torch.optim.lr_scheduler.MultiStepLR(op,[1000,2000],gamma=.1))
    generator=torch.Generator().manual_seed(42);first=1;best=float('inf');history=[]
    if (run/'last.pt').exists() and not smoke:
        old=torch.load(run/'last.pt',map_location='cpu',weights_only=False)
        if old['context']!=context:raise RuntimeError('Checkpoint data/config/source mismatch')
        for i in range(8):
            models[i].load_state_dict(old['models'][i]);opts[i].load_state_dict(old['optimizers'][i]);schedulers[i].load_state_dict(old['schedulers'][i])
        restore(old['rng'],generator);first=old['update']+1;best=old['best'];history=old['history']
        if old.get('selected_best'):c.torch_save(run/'best.pt',old)
    x=torch.tensor(arrays['x'],device=device);y=torch.tensor(arrays['y'],device=device)
    candidates=torch.tensor(arrays['windows'],device=device);span=torch.arange(300,device=device)
    parameters=sum(p.numel() for p in models[0].parameters());start=time.perf_counter();start_update=first-1
    c.atomic_json(run/'context.json',context)
    print('DeepVL device',device,'parameters/member',parameters,'ensemble',parameters*8,flush=True)
    limit=2 if smoke else 2500
    for update in range(first,limit+1):
        selected=torch.randint(len(candidates),(2 if smoke else 128,),generator=generator).to(device)
        ix=candidates[selected,None]+span[None];batch=x[ix];target=y[ix[:,-1]];losses=[]
        for m,op,sch in zip(models,opts,schedulers):
            m.train();mu,log,_=m(batch);mu=mu[:,-1];log=log[:,-1].clamp(min=float(np.log(1e-6)))
            loss=(mu-target).square().mean() if update<=1250 else ((mu-target).square()/(2*torch.exp(2*log))+log).mean()
            if not torch.isfinite(loss):raise FloatingPointError('Nonfinite DeepVL loss')
            op.zero_grad(set_to_none=True);loss.backward();op.step();sch.step();losses.append(float(loss.detach()))
        selected_best=False
        if update%20==0 or smoke or update==2500:
            squared=0.;count=0
            for i in range(int(arrays['validation_count'])):
                vx=arrays[f'val{i}_x'];truth=arrays[f'val{i}_y']
                if smoke:vx=vx[:40];truth=truth[:40]
                pred,_=c.predict(models,vx,stats,device)
                mask=arrays[f'val{i}_time'][:len(pred)]>=8
                if smoke:mask=np.ones(len(pred),bool)
                squared+=float(np.sum((pred[mask]-truth[mask])**2));count+=int(mask.sum())
            rmse=float(np.sqrt(squared/count))
            selected_best=update>1250 and rmse<best
            if selected_best:best=rmse
            row=dict(update=update,loss=float(np.mean(losses)),val_xy_velocity_vector_rmse=rmse,
                     validation_samples=count,selected_best=selected_best,lr=opts[0].param_groups[0]['lr'])
            history.append(row)
            print(json.dumps(row),flush=True)
        elapsed=time.perf_counter()-start
        if update%10==0 or smoke:
            c.atomic_json(run/'progress.json',dict(update=update,updates=limit,percent=100*update/limit,
                seconds_per_update=elapsed/(update-start_update),best_validation_vector_rmse=best if np.isfinite(best) else None,
                phase='MSE' if update<=1250 else 'GNLL',time_unix=time.time()))
        if update%20==0 or smoke or update==2500:
            payload=dict(update=update,models=[m.state_dict() for m in models],optimizers=[o.state_dict() for o in opts],
                         schedulers=[s.state_dict() for s in schedulers],rng=rng_state(generator),context=context,
                         normalization=stats,best=best,history=history,selected_best=selected_best,
                         parameters_per_member=parameters,ensemble_parameters=parameters*8)
            c.torch_save(run/'last.pt',payload)
            if selected_best:c.torch_save(run/'best.pt',payload)
            (run/'metrics.jsonl').write_text(''.join(json.dumps(r)+'\n' for r in history),encoding='utf-8')
    if not smoke:
        if not (run/'best.pt').exists():raise RuntimeError('No GNLL-stage best checkpoint')
        best_payload=torch.load(run/'best.pt',map_location='cpu',weights_only=False)
        c.atomic_json(run/'complete.json',dict(updates=2500,best_update=best_payload['update'],best_validation_vector_rmse=best,
                      checkpoint_sha256=c.sha(run/'best.pt'),time_unix=time.time()))
    print('DeepVL training completed',flush=True)


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--smoke',action='store_true');main(p.parse_args().smoke)
