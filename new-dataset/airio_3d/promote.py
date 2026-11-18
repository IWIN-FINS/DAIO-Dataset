"""Replace only the failed planar AirIO after gates; preserve three other methods."""
import csv,json,shutil
import numpy as np
import contracts as c

METHODS=('horizontal_rpm','horizontal_command','airio_3d','deepvl_planar')
UNCHANGED=('horizontal_rpm','horizontal_command','deepvl_planar')


def snapshot():
    out=c.HERE/'previous_main_comparison';path=out/'snapshot.json'
    if path.exists():
        old=json.loads(path.read_text(encoding='utf-8'));verify_unchanged(old);return old
    out.mkdir(parents=True,exist_ok=True)
    provenance=json.loads((c.REPORT/'provenance.json').read_text(encoding='utf-8'))
    files={}
    for method in UNCHANGED:
        checkpoint=provenance['models'][method]['checkpoint'];files[checkpoint]=c.sha(checkpoint)
        for p in (c.REPORT/method).glob('test*_samples.npz'):files[str(p)]=c.sha(p)
    rows=list(csv.DictReader((c.REPORT/'metrics.csv').open(encoding='utf-8-sig')))
    value=dict(files=files,rows=[r for r in rows if r['method'] in UNCHANGED])
    for p in [*(c.REPORT.glob('*.png')),*(c.REPORT.glob('*.csv')),c.REPORT/'provenance.json',c.REPORT/'README.md',c.REPORT/'metric_definitions.md']:
        if p.exists():shutil.copy2(p,out/p.name)
    c.atomic_json(path,value);return value


def verify_unchanged(snapshot):
    for path,digest in snapshot['files'].items():
        if c.sha(path)!=digest:raise RuntimeError('Another method changed: '+path)


def main():
    before=snapshot();verify_unchanged(before)
    source=json.loads((c.REPORT/'airio_3d/provenance.json').read_text(encoding='utf-8'))
    if source['context']!=c.fingerprint():raise RuntimeError('3D evaluation source changed')
    provenance=json.loads((c.REPORT/'provenance.json').read_text(encoding='utf-8'))
    import evaluate_pipeline as ep
    manifest={r['processed_relative_path']:r for r in csv.DictReader((c.DATA/'processed/manifest.csv').open(encoding='utf-8'))}
    new=[]
    for number,path in enumerate(c.split_paths()['demo'],1):
        r=c.load_record(path)
        with np.load(c.REPORT/'airio_3d'/f'test{number}_samples.npz') as z:
            np.testing.assert_allclose(z['time'],r['time'][1600:],atol=1e-9,rtol=0)
            without=np.ones(len(z['time']),bool);rec=manifest[path.relative_to(c.DATA/'processed').as_posix()]
            excluded=np.asarray(json.loads(rec['repaired_sample_indices']),dtype=int)-1600
            without[excluded[(excluded>=0)&(excluded<len(without))]]=False
            for score,mask in [('all_samples',np.ones(len(without),bool)),('without_repaired_samples',without)]:
                metrics=ep.metrics(z['position'],z['velocity'],z['yaw'],r['position_world'][1600:,:2],r['velocity_world'][1600:,:2],r['reference_rpy'][1600:,2],mask)
                new.append(dict(method='airio_3d',test=f'Test {number}',scoring=score,start_s=8.,end_s=float(z['time'][-1]),**metrics))
    for score in ('all_samples','without_repaired_samples'):
        chosen=[r for r in new if r['scoring']==score]
        new.append(dict(method='airio_3d',test='Mean',scoring=score,**{k:float(np.mean([r[k] for r in chosen])) for k in chosen[0] if k.endswith(('_m','_mps','_rad'))}))
    all_rows=[]
    for method in METHODS:all_rows.extend(new if method=='airio_3d' else [r for r in before['rows'] if r['method']==method])
    with (c.REPORT/'metrics.csv').open('w',encoding='utf-8-sig',newline='') as f:
        w=csv.DictWriter(f,fieldnames=list(before['rows'][0]));w.writeheader();w.writerows(all_rows)
    provenance['models']={method:source if method=='airio_3d' else provenance['models'][method] for method in METHODS}
    provenance.update(evaluated_methods=list(METHODS),evaluation_source=c.sha(__file__),airio_3d_context=c.fingerprint(),
        alignment='once at 8s, yaw and XY position; AirIO additionally saves pressure-referenced Z for offline scoring, not online updates',
        previous_planar_airio_archive=str(c.HERE/'previous_main_comparison'))
    c.atomic_json(c.REPORT/'provenance.json',provenance)
    old=(c.REPORT/'metric_definitions.md').read_text(encoding='utf-8')
    note='\n## AirIO full-3D mixed-reference retraining\nAirIO (3D ROV) uses official full-3D AirIMU/CodeNetMotionwithRot losses, nine-dimensional preintegration covariance and the full 15-state official filter. Reference XYZ uses AprilTag XY and negative pressure depth; reference roll/pitch are IMU solved angles and yaw is AprilTag. These are sensor-assisted reference labels, not independent motion capture. Testing uses six-axis IMU only, zero initial velocity and one yaw/coordinate alignment at 8s. Teacher-pose and causal deployed-pose validation results are reported separately. Full XYZ/quaternion/covariance outputs and extra 3D metrics are in airio_3d/. The former planar AirIO is retained as a failed diagnostic in previous_main_comparison and airio_planar/.\n'
    if '## AirIO full-3D mixed-reference retraining' not in old:(c.REPORT/'metric_definitions.md').write_text(old+note,encoding='utf-8')
    verify_unchanged(before);c.atomic_json(c.HERE/'unchanged_methods_validation.json',dict(ok=True,files=before['files'],rows=len(before['rows'])))
    print('Promoted airio_3d; three other model/prediction files and metrics unchanged',flush=True)


if __name__=='__main__':main()
