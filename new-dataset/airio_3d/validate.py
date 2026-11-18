"""Independent full-state checks plus unchanged-method evidence after publication."""
import csv,json
import numpy as np
from scipy.spatial.transform import Rotation
import contracts as c
import promote


def main():
    out=c.REPORT/'airio_3d';rows=list(csv.DictReader((out/'metrics_3d.csv').open(encoding='utf-8-sig')));checked=[]
    source=json.loads((out/'provenance.json').read_text(encoding='utf-8'))
    if source['context']!=c.fingerprint():raise RuntimeError('3D publication context changed')
    for number,path in enumerate(c.split_paths()['demo'],1):
        ref=c.load_record(path)
        with np.load(out/f'test{number}_full_state.npz') as z:
            for key in ('state','quaternion','covariance'):
                if not np.isfinite(z[key]).all():raise FloatingPointError('Nonfinite full state')
            if str(z['checkpoint_sha256'])!=source['sha256']:raise RuntimeError('Checkpoint mismatch')
            np.testing.assert_allclose(np.linalg.norm(z['quaternion'],axis=1),1,atol=1e-10,rtol=0)
            p=z['covariance'];np.testing.assert_allclose(p,p.swapaxes(-1,-2),atol=1e-6,rtol=0)
            if np.linalg.eigvalsh((p[::200]+p[::200].swapaxes(-1,-2))*.5).min()<-1e-7:raise FloatingPointError('Non-PSD covariance')
            v=z['state'][1600:,3:6]-ref['velocity_world'][1600:];e=z['state'][1600:,6:9]-ref['position_world'][1600:]
            rot=(Rotation.from_quat(z['quaternion'][1600:]).inv()*Rotation.from_quat(ref['quaternion_world'][1600:])).as_rotvec()
            expected=dict(z_velocity_rmse=np.sqrt(np.mean(v[:,2]**2)),z_position_rmse=np.sqrt(np.mean(e[:,2]**2)),
                xyz_velocity_rmse=np.sqrt(np.mean(np.sum(v*v,axis=1))),xyz_position_rmse=np.sqrt(np.mean(np.sum(e*e,axis=1))),
                attitude_so3_rmse_rad=np.sqrt(np.mean(np.sum(rot*rot,axis=1))))
            row=next(r for r in rows if r['test']==f'Test {number}' and r['scoring']=='all_samples')
            for k,value in expected.items():np.testing.assert_allclose(float(row[k]),value,rtol=1e-10,atol=1e-12)
            with np.load(out/f'test{number}_samples.npz') as common:
                np.testing.assert_allclose(common['velocity'],z['state'][1600:,3:5],rtol=0,atol=0)
                np.testing.assert_allclose(common['position'],z['state'][1600:,6:8],rtol=0,atol=0)
            checked.append(number)
    before=json.loads((c.HERE/'previous_main_comparison/snapshot.json').read_text(encoding='utf-8'));promote.verify_unchanged(before)
    c.atomic_json(c.HERE/'full_state_validation.json',dict(ok=True,tests=checked,context=source['context']))
    print('Validated full 3D states, covariance and recomputed metrics',flush=True)


if __name__=='__main__':main()
