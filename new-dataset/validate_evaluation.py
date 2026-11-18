"""Independent score recomputation and PNG metadata validation."""
from pathlib import Path
import sys,csv,json,hashlib
import numpy as np
from PIL import Image
ROOT=Path(__file__).resolve().parents[1];DATA=ROOT/'new-dataset';OUT=ROOT/'ROV-ESKF-Odometry/reports/new_v1v2'
sys.path.insert(0,str(DATA/'baselines'))
import planar_common as pc

def main():
    provenance=json.loads((OUT/'provenance.json').read_text(encoding='utf-8'))
    rows=list(csv.DictReader((OUT/'metrics.csv').open(encoding='utf-8-sig')))
    paths=pc.resolve_manifest(DATA/'processed/split.yaml',DATA)['demo'];checked=0
    for method,source in provenance['models'].items():
        assert pc.file_sha(Path(source['checkpoint']))==source['sha256']
        for test,path in enumerate(paths,1):
            seq=pc.load_sequence(path);xy,yaw,velocity,_=pc.reference(seq);xy=xy[1600:];velocity=velocity[1600:]
            with np.load(OUT/method/f'test{test}_samples.npz') as saved:
                np.testing.assert_allclose(saved['time'],seq.imu_ts[1600:],rtol=0,atol=1e-9)
                np.testing.assert_allclose(saved['position'][0],xy[0],rtol=0,atol=1e-12)
                assert 'method_signature' in saved
                error=saved['position']-xy;v_error=saved['velocity']-velocity
                relative=(saved['position'][1000:]-saved['position'][:-1000])-(xy[1000:]-xy[:-1000])
                expected={'velocity_xy_vector_rmse_mps':np.sqrt(np.mean(np.sum(v_error**2,axis=1))),
                          'position_xy_vector_rmse_m':np.sqrt(np.mean(np.sum(error**2,axis=1))),
                          'rpe_5s_xy_vector_rmse_m':np.sqrt(np.mean(np.sum(relative**2,axis=1))),
                          'final_position_error_m':np.linalg.norm(error[-1])}
                row=next(r for r in rows if r['method']==method and r['test']==f'Test {test}' and r['scoring']=='all_samples')
                for key,value in expected.items():np.testing.assert_allclose(float(row[key]),value,rtol=1e-9,atol=1e-10)
                assert int(row['sample_count'])==len(xy);checked+=1
    Image.MAX_IMAGE_PIXELS=None;images=[]
    for path in sorted([*OUT.glob('*.png'),*(OUT/'network_ablation').glob('*.png')]):
        with Image.open(path) as im:
            dpi=im.info.get('dpi');assert dpi and abs(dpi[0]-1200)<.1,(path,dpi)
            if path.name.endswith(('_world_velocity.png','_xy_trajectory.png','_body_velocity.png','_velocity_error.png')):
                assert im.size==(9600,4800),(path,im.size)
            if path.name.endswith('_error_boxplot.png'):assert im.size==(16200,5400),(path,im.size)
            images.append({'file':str(path.relative_to(OUT)),'size':im.size,'dpi':dpi,'sha256':pc.file_sha(path)})
    assert len(images)==23,len(images)
    pc.atomic_json(OUT/'artifact_validation.json',{'ok':True,'independently_recomputed_trials':checked,'png_count':len(images),'pngs':images})
    print('Validated:',checked,'odometry trials;',len(images),'1200-dpi PNGs',flush=True)

if __name__=='__main__':main()
