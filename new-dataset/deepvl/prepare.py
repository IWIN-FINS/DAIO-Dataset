"""Train-only statistics and immutable 15s sampling manifest."""
import json
import numpy as np
import common as c


def prepare():
    directory=c.HERE/'prepared'; directory.mkdir(exist_ok=True)
    context=c.fingerprint()
    if (directory/'context.json').exists():
        if json.loads((directory/'context.json').read_text(encoding='utf-8')) != context:
            raise RuntimeError('Prepared data source/settings mismatch; do not silently overwrite')
        hashes=json.loads((directory/'hashes.json').read_text(encoding='utf-8'))
        for name,digest in hashes.items():
            if c.sha(directory/name)!=digest:raise RuntimeError('Prepared artifact corrupted: '+name)
        return directory
    rng=np.random.default_rng(42); xs=[]; ys=[]; windows=[]; manifest=[]; offset=0; validations=[]
    for split, paths in c.splits().items():
        if split not in ('train','val'):continue
        for path in paths:
            rec=c.load_record(path,reference=True)
            if len(rec.time)<3000:
                manifest.append(dict(split=split, file=str(path), reason='shorter than 15s', windows=0));continue
            ends,x=c.feature_points(rec);xy,yaw,world,y=c.reference(rec); y=y[ends]
            if split=='train':
                candidates=len(x)-299
                count=min(candidates, int(np.floor(candidates/20*7.5)))
                starts=np.sort(rng.choice(candidates,size=count,replace=False))
                windows.extend((starts+offset).tolist());xs.append(x);ys.append(y);offset+=len(x)
                manifest.append(dict(split=split,file=str(path),windows=count,starts_20hz=starts.tolist()))
            else:
                validations.append((x,y,ends,rec.time[ends],world[ends],yaw[ends],str(path)))
    x=np.concatenate(xs); y=np.concatenate(ys)
    im=x.mean(axis=0); sd=x.std(axis=0); im[14]=0.;sd[14]=1.
    stats=dict(input_mean=im.tolist(),input_std=np.maximum(sd,1e-6).tolist(),
               output_mean=y.mean(axis=0).tolist(),output_std=np.maximum(y.std(axis=0),1e-6).tolist(),
               fit_split='train only', train_points=len(x))
    arrays=dict(x=c.normalize(x,stats), y=((y-stats['output_mean'])/stats['output_std']).astype(np.float32),
                windows=np.asarray(windows,dtype=np.int64),validation_count=np.array(len(validations)))
    for i,(vx,vy,ends,t,w,angle,path) in enumerate(validations):
        arrays.update({f'val{i}_x':c.normalize(vx,stats),f'val{i}_y':vy,
                       f'val{i}_time':t,f'val{i}_world':w,f'val{i}_yaw':angle})
    np.savez_compressed(directory/'arrays.npz',**arrays)
    c.atomic_json(directory/'normalization.json',stats)
    c.atomic_json(directory/'windows.json',manifest)
    c.atomic_json(directory/'hashes.json',{n:c.sha(directory/n) for n in ('arrays.npz','normalization.json','windows.json')})
    c.atomic_json(directory/'context.json',context)
    print('Prepared',len(windows),'fixed training windows;',len(validations),'validation trajectories',flush=True)
    return directory


if __name__=='__main__':prepare()
