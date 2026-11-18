"""Real-data FP32 batch-128 backward checks; never saves training weights."""
import time,json
import numpy as np
import torch
import contracts as c
import vendor,causal,train,runtime_filter,physics
from banks import Bank


def main(device='cuda'):
    train.setup();result=dict(device=device,torch=torch.__version__,context=c.fingerprint(),stages={})
    inventory=json.loads((c.HERE/'prepared/complete.json').read_text(encoding='utf-8'))
    result['split_counts']={split:sum(r['split']==split for r in inventory['records']) for split in ('train','val','demo')}
    result['validation_static_seconds']=[r['static_seconds'] for r in inventory['records'] if r['split']=='val']
    for stage in ('airimu','airio'):
        bank=Bank('train',stage,device);model=getattr(vendor,stage)(device).train()
        if stage=='airimu':
            import pypose as pp
            data,initial,label=bank.batch(torch.arange(2,device=device),stage)
            expected=initial['rot'].Inv()@torch.tensor([0.,0.,c.G],device=device)
            torch.testing.assert_close(data['acc'][:,:9],expected.expand(-1,9,-1),atol=1e-5,rtol=1e-5)
            assert data['gyro'][:,:9].abs().max()==0 and label['gt_pos'].shape==(2,1000,3)
        optimizer=torch.optim.Adam(model.parameters(),lr=.001)
        if device=='cuda':torch.cuda.reset_peak_memory_stats();torch.cuda.synchronize()
        started=time.time();loss,error=train.forward(model,bank,torch.arange(128,device=device),stage);loss.backward()
        assert torch.isfinite(loss) and all(p.grad is None or torch.isfinite(p.grad).all() for p in model.parameters())
        optimizer.step()
        if device=='cuda':torch.cuda.synchronize()
        result['stages'][stage]=dict(loss=float(loss.detach()),seconds=time.time()-started,windows=len(bank),
            parameters=sum(p.numel() for p in model.parameters()),max_allocated_mib=torch.cuda.max_memory_allocated()/1024**2 if device=='cuda' else None)
        print('SMOKE',stage,result['stages'][stage],flush=True)
        del bank,model,optimizer,loss,error
        if device=='cuda':torch.cuda.empty_cache()
    # The untrained check is evidence only, not a passing-check requirement.
    model=vendor.airimu(device);model.eval()
    gate,_=physics.validate(model,inventory['calibration']['gyro_bias'],device)
    c.atomic_json(c.HERE/'initial_untrained_physics.json',gate);del model
    n=40;acc=np.tile([0,0,c.G],(n,1));gyro=np.zeros((n,3));time_s=np.arange(n)*c.DT
    runtime=dict(time=time_s,acc=acc,gyro=gyro)
    orientation=dict(correction_acc=acc*0,correction_gyro=gyro,acc_cov=np.full((n,3),.01),gyro_cov=np.full((n,3),.001))
    full=runtime_filter.replay(runtime,orientation,np.array([20]),np.zeros((1,3)),np.full((1,3),.01),0.,False)
    assert np.max(np.abs(full['state']))<1e-8
    assert np.max(np.abs(np.linalg.norm(full['quaternion'],axis=1)-1))<1e-10
    assert np.linalg.eigvalsh(full['covariance']).min()>-1e-9
    result['static_full_filter_passed']=True;c.atomic_json(c.HERE/'smoke_report.json',result);return result


if __name__=='__main__':main()
