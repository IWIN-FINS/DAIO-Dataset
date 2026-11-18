"""Validate new results and prove that original methods were not changed."""
import csv,json
from pathlib import Path
import numpy as np
import common as c


def main():
    out=c.ROOT/'ROV-ESKF-Odometry/reports/new_v1v2'
    before=c.HERE/'previous_evaluation'
    old=list(csv.DictReader((before/'ROV-ESKF-Odometry/reports/new_v1v2/metrics.csv').open(encoding='utf-8-sig')))
    new=list(csv.DictReader((out/'metrics.csv').open(encoding='utf-8-sig')))
    for row in old:
        current=next(r for r in new if all(r[k]==row[k] for k in ('method','test','scoring')))
        for key,value in row.items():
            if key in ('method','test','scoring') or not value:continue
            np.testing.assert_allclose(float(current[key]),float(value),rtol=0,atol=1e-12)
    inventory=json.loads((before/'inventory.json').read_text(encoding='utf-8'))
    preserved=[]
    for item in inventory:
        if item['file'].endswith('_samples.npz'):
            assert c.sha(c.ROOT/item['file'])==item['sha256'];preserved.append(item['file'])
    manifest={r['processed_relative_path']:r for r in csv.DictReader((c.DATA/'processed/manifest.csv').open(encoding='utf-8'))}
    checked=0
    for i,path in enumerate(c.splits()['demo'],1):
        rec=c.load_record(path,reference=True);xy,yaw,vel,_=c.reference(rec)
        saved=np.load(out/'deepvl_planar'/f'test{i}_samples.npz')
        np.testing.assert_allclose(saved['time'],rec.time[1600:],rtol=0,atol=1e-10)
        np.testing.assert_allclose(saved['position'][0],xy[1600],rtol=0,atol=1e-12)
        repaired=np.asarray(json.loads(manifest[path.relative_to(c.DATA/'processed').as_posix()]['repaired_sample_indices']),int)-1600
        for scoring in ('all_samples','without_repaired_samples'):
            mask=np.ones(len(saved['time']),bool)
            if scoring!='all_samples':mask[repaired[(repaired>=0)&(repaired<len(mask))]]=False
            ep=saved['position']-xy[1600:];ev=saved['velocity']-vel[1600:]
            er=(saved['position'][1000:]-saved['position'][:-1000])-(xy[2600:]-xy[1600:-1000])
            both=mask[1000:]&mask[:-1000]
            ey=np.arctan2(np.sin(saved['yaw']-yaw[1600:]),np.cos(saved['yaw']-yaw[1600:]))
            values=dict(velocity_x_rmse_mps=np.sqrt(np.mean(ev[mask,0]**2)),velocity_y_rmse_mps=np.sqrt(np.mean(ev[mask,1]**2)),
                velocity_xy_vector_rmse_mps=np.sqrt(np.mean(np.sum(ev[mask]**2,axis=1))),
                position_x_rmse_m=np.sqrt(np.mean(ep[mask,0]**2)),position_y_rmse_m=np.sqrt(np.mean(ep[mask,1]**2)),
                position_xy_vector_rmse_m=np.sqrt(np.mean(np.sum(ep[mask]**2,axis=1))),final_position_error_m=np.linalg.norm(ep[-1]),
                rpe_5s_xy_vector_rmse_m=np.sqrt(np.mean(np.sum(er[both]**2,axis=1))),yaw_rmse_rad=np.sqrt(np.mean(ey[mask]**2)))
            row=next(r for r in new if r['method']=='deepvl_planar' and r['test']==f'Test {i}' and r['scoring']==scoring)
            for k,v in values.items():np.testing.assert_allclose(float(row[k]),v,rtol=1e-9,atol=1e-10)
            assert int(row['sample_count'])==mask.sum();checked+=1
        full=np.load(out/'deepvl_planar'/f'test{i}_full_state.npz')
        assert np.all(full['covariance_diag']>=-1e-10)
        assert np.isfinite(full['observation_variance']).all() and np.all(full['observation_variance']>0)
    c.atomic_json(out/'deepvl_validation.json',dict(ok=True,independent_scores=checked,
                   unchanged_prior_metrics=len(old),unchanged_prior_sample_files=preserved))
    print('DeepVL validation passed; all 3 prior methods unchanged',flush=True)


if __name__=='__main__':main()
