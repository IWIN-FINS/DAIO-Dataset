"""Build all references; estimate warm-up gyro calibration on TRAIN only."""
import csv,json
import numpy as np
from scipy.spatial.transform import Rotation
import contracts as c


def main():
    context=c.fingerprint();destination=c.HERE/'prepared'
    if (destination/'complete.json').exists():
        old=json.loads((destination/'complete.json').read_text(encoding='utf-8'))
        if old['context']!=context:raise RuntimeError('Prepared context changed; use a new experiment')
        for r in old['records']:
            if c.sha(c.HERE/r['cache'])!=r['cache_sha256']:raise RuntimeError('Prepared cache changed')
        return old
    manifest={r['processed_relative_path']:r for r in csv.DictReader((c.DATA/'processed/manifest.csv').open(encoding='utf-8'))}
    records=[];biases=[];source_list=[]
    paths_by_split=c.split_paths()
    if {k:len(v) for k,v in paths_by_split.items()}!={'train':49,'val':2,'demo':4}:raise ValueError('Split changed')
    flat=[str(p) for v in paths_by_split.values() for p in v]
    if len(set(flat))!=55:raise ValueError('Split leakage or duplicate segments')
    for split,paths in c.split_paths().items():
        for path in paths:
            raw=c.read_source(path);ref=c.make_reference(raw)
            info=manifest[path.relative_to(c.DATA/'processed').as_posix()]
            ref['excluded_end_intervals']=np.array(json.loads(info['network_window_exclusion_end_intervals_s']),dtype=float).reshape(-1,2)
            c.save_npz(c.cache_path(path),**ref)
            records.append(dict(split=split,source=str(path),source_sha256=c.sha(path),samples=len(raw['time']),
                cache=str(c.cache_path(path).relative_to(c.HERE)),cache_sha256=c.sha(c.cache_path(path)),
                usable_window=len(raw['time'])>1000,static_seconds=float(ref['static'].sum()*c.DT)))
            if split=='val' and ref['static'].sum()*c.DT<c.CONFIG['gates']['static_seconds_min']:raise ValueError('Insufficient validation static reference')
            if split=='train' and len(raw['time'])>=1001:
                rot=Rotation.from_quat(ref['quaternion'])
                expected=(rot[:-1].inv()*rot[1:]).as_rotvec()/c.DT
                biases.append(np.median(raw['gyro'][:-1]-expected,axis=0));source_list.append(str(path))
            print('Prepared 3D reference',split,path.name,flush=True)
    calibration=dict(gyro_bias=np.median(biases,axis=0).tolist(),sources=source_list,fit_split='train',
        rule='median of per-record raw gyro minus mixed-reference angular-rate residual; warm-up only, never added to network correction')
    c.atomic_json(destination/'warmup_calibration.json',calibration)
    result=dict(context=context,records=records,calibration=calibration,
                reference='sensor-assisted mixed reference, not independent 6DoF motion capture')
    c.atomic_json(destination/'complete.json',result)
    return result


if __name__=='__main__':main()
