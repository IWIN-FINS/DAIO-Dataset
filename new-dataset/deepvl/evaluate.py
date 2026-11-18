"""Produce the common 200Hz sample interface without touching other methods."""
from pathlib import Path
import csv,json,time
import numpy as np
import torch
import common as c
from filter import replay


def evaluate():
    checkpoint=c.HERE/'runs/best.pt'
    device='cuda' if torch.cuda.is_available() else 'cpu';torch.set_num_threads(4)
    models,payload=c.load_ensemble(checkpoint,device)
    if payload['context']!=c.fingerprint():raise RuntimeError('DeepVL source/data differ from trained checkpoint')
    out=c.ROOT/'ROV-ESKF-Odometry/reports/new_v1v2/deepvl_planar';out.mkdir(parents=True,exist_ok=True)
    signature=c.sha(checkpoint)+c.sha(c.HERE/'filter.py')+c.sha(c.HERE/'common.py')
    sources=dict(checkpoint=str(checkpoint),sha256=c.sha(checkpoint),update=payload['update'],
         parameters_per_member=payload['parameters_per_member'],ensemble_parameters=payload['ensemble_parameters'],
         adaptation='Official DeepVL 8-GRU ensemble + ReAqROVIO no-vision inertial core; heading XY supervision/update',
         online_sensors=['six-axis IMU','8 thruster commands','battery voltage','relative depth'],network_hz=20,
         source_fingerprint=payload['context'])
    timings=[]
    for test,path in enumerate(c.splits()['demo'],1):
        rec=c.load_record(path,reference=False);offline=c.load_record(path,reference=True)
        xy,yaw,_,_=c.reference(offline);dest=out/f'test{test}_samples.npz'
        if dest.exists():
            with np.load(dest) as z:
                if str(z['method_signature'])!=signature or str(z['source_sha256'])!=c.sha(path):raise RuntimeError('Stale DeepVL replay')
            continue
        begin=time.perf_counter();ends,mu,var=c.observations(rec,models,payload['normalization'],device)
        inference=time.perf_counter()-begin;begin=time.perf_counter()
        projection,state,cov,events=replay(rec,ends,mu,var,yaw[1600],xy[1600])
        np.savez_compressed(out/f'test{test}_full_state.npz',time=rec.time,state=state,covariance_diag=cov,
                 observation_times=rec.time[ends],observation_mean=mu,observation_variance=var)
        np.savez_compressed(dest,time=rec.time[1600:],position=projection[1600:,:2],velocity=projection[1600:,3:5],yaw=projection[1600:,6],
                 checkpoint_sha256=sources['sha256'],source_sha256=c.sha(path),method_signature=signature)
        with (out/f'test{test}_updates.csv').open('w',encoding='utf-8-sig',newline='') as f:
            w=csv.DictWriter(f,fieldnames=['time','kind','accepted','nis']);w.writeheader();w.writerows(events)
        timings.append(dict(test=test,network_seconds=inference,network_updates=len(ends),
                            milliseconds_per_ensemble_step=1000*inference/len(ends),filter_seconds=time.perf_counter()-begin))
        print('DeepVL Test',test,'complete',timings[-1],flush=True)
    # Stable steady-state batch1 latency including all 8 networks (no file/feature I/O).
    packet=torch.zeros((1,1,16),device=device);hidden=[None]*8
    with torch.no_grad():
        for k in range(220):
            if k==20:
                if device=='cuda':torch.cuda.synchronize()
                begin=time.perf_counter()
            for j,m in enumerate(models):_,_,hidden[j]=m(packet,hidden[j])
        if device=='cuda':torch.cuda.synchronize()
    sources['batch1_ensemble_inference_ms']=(time.perf_counter()-begin)/200*1000
    sources['timings']=timings;c.atomic_json(out/'provenance.json',sources)
    return sources


if __name__=='__main__':evaluate()
