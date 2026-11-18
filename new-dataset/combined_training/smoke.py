"""Preflight: native feature parity, input isolation, batch128, and resumable dropout."""
from dataclasses import replace
import copy,time
import numpy as np
import torch
import common as c
import train
from rovio_ablation.model import build_model
from rovio_ablation.data import load_sequence as training_sequence
from rovio_ablation.utils import rng_state,restore_rng_state
from rov_eskf.data import load_sequence
from rov_eskf.model import load_checkpoint
from rov_eskf.windows import V20WindowBuilder,network_observations
from rov_eskf.replay import replay_attitude_aided_11

def main():
    train.setup();context=c.fingerprint();bank=c.Bank('train','cuda');out=c.RUN/'smoke';out.mkdir(parents=True,exist_ok=True);results={}
    for name in ('daio','no_rpm'):
        cfg=c.cfg(name);model=build_model(cfg).to('cuda').train();optimizer=torch.optim.AdamW(model.parameters(),lr=.001,weight_decay=.0001)
        index=torch.arange(128,device='cuda');inputs=bank.inputs(index,model.input_keys)
        before=time.time();mean,log=model(*inputs);loss=(mean-bank.target[index]).square().mean();optimizer.zero_grad();loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(),5.,error_if_nonfinite=True);optimizer.step();torch.cuda.synchronize()
        assert torch.isfinite(loss);seconds=time.time()-before
        payload=dict(format_version=1,config=cfg,model=model.state_dict(),normalizer=c.normalizer(name),optimizer=optimizer.state_dict(),rng_state=rng_state())
        c.torch_save(out/(name+'.pt'),payload)
        # The next stochastic optimizer step must be repeatable from disk state.
        def step(m,o):
            m.train();o.zero_grad();a,b=m(*inputs);loss=(a-bank.target[index]).square().mean()+b.square().mean()*.0001;loss.backward();o.step()
        step(model,optimizer);expected={k:v.clone() for k,v in model.state_dict().items()}
        saved=torch.load(out/(name+'.pt'),map_location='cpu',weights_only=False);restored=build_model(cfg).to('cuda');restored.load_state_dict(saved['model'])
        opt=torch.optim.AdamW(restored.parameters(),lr=.001,weight_decay=.0001);opt.load_state_dict(saved['optimizer']);restore_rng_state(saved['rng_state']);step(restored,opt)
        for k,v in restored.state_dict().items():torch.testing.assert_close(v,expected[k],atol=1e-6,rtol=1e-5)
        runtime,_=load_checkpoint(out/(name+'.pt'),'cuda');model.load_state_dict(saved['model']);model.eval()
        with torch.no_grad():
            expected_out=model(*inputs);actual=runtime(*inputs)
            for a,b in zip(expected_out,actual):torch.testing.assert_close(a,b,atol=1e-6,rtol=1e-5)
        worst=0.
        for path in c.paths()['demo']:
            ds=c.V20WindowDataset([path],cfg,c.normalizer(name));native=V20WindowBuilder(load_sequence(path),saved);arrays=native.feature_arrays();lookup={round(t*200):i for i,t in enumerate(native.ts)}
            with np.load(c.cache_path(path)) as z:features=z['features'];starts=z['starts']
            for i in np.unique(np.linspace(0,len(ds)-1,7,dtype=int)):
                item=ds[int(i)];row=lookup[round(float(item['time'])*200)];raw=features[starts[i]+6*np.arange(256)]
                for j,key in enumerate(runtime.input_keys):
                    value=(raw[:,c.SLICES[key]]-saved['normalizer'][key+'_mean'])/saved['normalizer'][key+'_std'];value=value.T
                    np.testing.assert_allclose(value,item[key].numpy(),atol=3e-5,rtol=1e-5)
                    np.testing.assert_allclose(value,arrays[j][row],atol=3e-5,rtol=1e-5);worst=max(worst,float(np.max(np.abs(value-arrays[j][row]))))
            if name=='no_rpm':
                seq=load_sequence(path);changed=replace(seq,rpm=np.full_like(seq.rpm,32767.))
                b=V20WindowBuilder(changed,saved);np.testing.assert_array_equal(native.ts,b.ts)
                for a,v in zip(arrays,b.feature_arrays()):np.testing.assert_array_equal(a,v)
                assert 'rpm' not in runtime.input_keys and not any(k.startswith('rpm_') for k in saved['normalizer'])
        path=next(p for p in c.paths()['demo'] if '/new/' in p.as_posix());seq=load_sequence(path)
        net=network_observations(seq,runtime,saved,'cuda',128)
        config=c.yaml.safe_load((c.ROOT/'ROV-ESKF-Odometry/configs/benchmark_paper_test1_4_current_model.yaml').read_text(encoding='utf-8'))
        full=replay_attitude_aided_11(seq,config,network=net,end_time=9.)
        assert np.isfinite(full.state).all() and np.isfinite(full.covariance_diag).all() and np.min(full.covariance_diag)>-1e-9
        results[name]=dict(batch128_seconds=seconds,parameters=sum(p.numel() for p in model.parameters()),feature_max_abs_difference=worst,
            resume_optimizer_rng_passed=True,runtime_forward_parity_passed=True,short_eskf_passed=True)
        print('SMOKE',name,results[name],flush=True)
    c.atomic_json(c.RUN/'smoke_report.json',dict(ok=True,context=context,results=results))

if __name__=='__main__':main()
