"""Independent endpoint checks for exported network and fused odometry metrics."""
import csv
import numpy as np
import common as c

def main():
    rows=list(csv.DictReader((c.REPORT/'network_metrics.csv').open(encoding='utf-8-sig')));checked=0
    for domain in ('old','new'):
        for name in ('daio','no_rpm'):
            for number in range(1,5):
                with np.load(c.REPORT/'network'/domain/name/f'test{number}.npz') as z:
                    err=z['prediction'].astype(float)-z['target'];time=z['time'];ticks=np.rint(time*200).astype(int);lookup={int(t):i for i,t in enumerate(ticks)}
                    pairs=np.array([(i,lookup[int(t)+1000]) for i,t in enumerate(ticks) if int(t)+1000 in lookup])
                    ave=np.sqrt(np.mean(np.sum(err[:,:2]**2,axis=1)));rve=np.sqrt(np.mean(np.sum((err[pairs[:,1],:2]-err[pairs[:,0],:2])**2,axis=1)))
                    r=next(r for r in rows if r['domain']==domain and r['model']==name and r['test']==f'Test {number}')
                    np.testing.assert_allclose([float(r['ave_mps']),float(r['rve_5s_mps'])],[ave,rve],atol=1e-7,rtol=1e-5);checked+=1
    odom=list(csv.DictReader((c.REPORT/'odometry_metrics.csv').open(encoding='utf-8-sig')))
    for name in ('daio','no_rpm'):
        for number in range(1,5):
            with np.load(c.REPORT/'odometry'/name/f'test{number}/samples.npz') as z:
                np.testing.assert_allclose(np.diff(z['time']),.005,atol=1e-9,rtol=0)
                np.testing.assert_allclose(z['position'][0],z['reference_position'][0],atol=1e-12,rtol=0)
                ev=z['velocity']-z['reference_velocity'];ep=z['position']-z['reference_position']
                value=[np.sqrt(np.mean(np.sum(ev*ev,axis=1))),np.sqrt(np.mean(np.sum(ep*ep,axis=1)))]
                r=next(r for r in odom if r['model']==name and r['test']==f'Test {number}' and r['scoring']=='all_samples')
                np.testing.assert_allclose([float(r['velocity_xy_rmse_mps']),float(r['position_xy_rmse_m'])],value,rtol=1e-10,atol=1e-12);checked+=1
    c.atomic_json(c.REPORT/'metric_validation.json',dict(ok=True,independently_checked=checked));print('Validated',checked,'trials',flush=True)

if __name__=='__main__':main()
