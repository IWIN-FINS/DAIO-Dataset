"""Create byte-identical namespaced copies and preserve all anomaly metadata."""
import csv,json,shutil
import numpy as np
import common as c

def snapshot():
    c.DATA.mkdir(parents=True,exist_ok=True);destination=c.DATA/'processed';rows=[];splits={k:[] for k in ('train','val','demo')};sources=[];source_splits={};digests=set()
    inputs=[('old',c.ROOT/'dataset',c.ROOT/'ROV-IO-Training/manifests/v24_val26_27_demo4.yaml'),
            ('new',c.ROOT/'new-dataset',c.ROOT/'new-dataset/processed/split.yaml')]
    for domain,root,manifest in inputs:
        records={r['processed_relative_path']:r for r in csv.DictReader((root/'processed/manifest.csv').open(encoding='utf-8-sig'))}
        for split,paths in c.resolve_manifest(manifest,root).items():
            for path in paths:
                relative=path.relative_to(root/'processed').as_posix();record=records[relative].copy();uid=domain+'/'+record['source_relative_path']
                if uid in source_splits and source_splits[uid]!=split:raise ValueError('Source split leakage')
                source_splits[uid]=split;digest=c.sha(path)
                if digest in digests:raise ValueError('Duplicated source contents')
                digests.add(digest)
                with path.open(encoding='utf-8-sig') as f:header=next(csv.reader(f))
                names=['sample_time_s','pressure','acc_x','acc_y','gyro_z',*[f'int_rpm_{i}' for i in range(1,9)]]
                a=np.loadtxt(path,delimiter=',',skiprows=1,usecols=[header.index(k) for k in names],encoding='utf-8-sig')
                if not np.isfinite(a).all():raise ValueError('Nonfinite required inputs')
                np.testing.assert_allclose(np.diff(a[:,0]),.005,atol=1e-9,rtol=0)
                if 'rpm_saturation_times_s' not in record:
                    indices=np.array(json.loads(record['rpm_saturation_source_rows']),dtype=int)-int(record['source_row_start'])
                    record['rpm_saturation_times_s']=json.dumps(a[indices,0].tolist())
                declared=np.array(json.loads(record['rpm_saturation_times_s']),float)
                measured=a[np.any(np.abs(a[:,5:])>=32700,axis=1),0]
                np.testing.assert_allclose(np.sort(declared),np.sort(measured),atol=1e-9,rtol=0)
                relative=domain+'/'+relative;target=destination/relative;target.parent.mkdir(parents=True,exist_ok=True)
                if target.exists():
                    if c.sha(target)!=digest:raise RuntimeError('Existing snapshot differs')
                else:shutil.copy2(path,target)
                record.update(processed_relative_path=relative,source_relative_path=uid,dataset_domain=domain,
                    original_processed_path=str(path),sha256=digest,repaired_sample_indices=record.get('repaired_sample_indices','[]'),
                    repair_metadata_available=str(domain=='new'),raw_source_identity=uid)
                # Namespaced row-map paths remain traceable to their original source.
                if record.get('row_map_relative_path'):record['original_row_map_path']=str(root/'processed'/record['row_map_relative_path'])
                rows.append(record);splits[split].append(relative)
                sources.append(dict(source=str(path),copy=str(target),sha256=digest,split=split,source_uid=uid,
                    received_start=record['received_utc_start_s'],received_end=record['received_utc_end_s']))
    assert {k:len(v) for k,v in splits.items()}=={'train':125,'val':4,'demo':8}
    # UTC identity detects processed fragments from the same physical recording.
    for i,left in enumerate(sources):
        for right in sources[i+1:]:
            if left['split']==right['split']:continue
            lo=max(float(left['received_start']),float(right['received_start']));hi=min(float(left['received_end']),float(right['received_end']))
            if hi-lo>.1:raise ValueError('Overlapping recording intervals across splits')
    columns=sorted(set().union(*(set(r) for r in rows)))
    with (destination/'manifest.csv').open('w',encoding='utf-8',newline='') as f:
        writer=csv.DictWriter(f,fieldnames=columns);writer.writeheader();writer.writerows(rows)
    manifest=dict(schema_version=3,data_format='processed_csv_200hz',coordinate_frame='FLU',offline_apriltag_only=True,
        processed_root='processed',splits={k:dict(files=v,expected_count=len(v)) for k,v in splits.items()})
    (c.DATA/'split.yaml').write_text(c.yaml.safe_dump(manifest,sort_keys=False),encoding='utf-8')
    c.atomic_json(c.DATA/'source_inventory.json',dict(records=sources,counts={k:len(v) for k,v in splits.items()},exact_duplicates=0,utc_overlap_across_splits=False))
    base=c.yaml.safe_load((c.PROJECT/'experiments/new_v1v2/horizontal_rpm/config.yaml').read_text(encoding='utf-8'))
    for model in ('daio','no_rpm'):
        cfg=json.loads(json.dumps(base));cfg['experiment']['name']='combined_'+model;cfg['experiment']['device']='cuda'
        cfg['model']['name']=model;cfg['model']['modalities']=list(c.KEYS if model=='daio' else c.KEYS[1:])
        cfg['evaluation']['selection_aggregation']='four_record_macro_AVE'
        directory=c.RUN/model;directory.mkdir(parents=True,exist_ok=True)
        text=c.yaml.safe_dump(cfg,sort_keys=False);path=directory/'config.yaml'
        if path.exists() and path.read_text(encoding='utf-8')!=text:raise RuntimeError('Config changed')
        path.write_text(text,encoding='utf-8')

def features():
    complete=c.DATA/'feature_cache/complete.json';context=c.fingerprint()
    if complete.exists():
        old=c.read(complete)
        if old['context']!=context:raise RuntimeError('Prepared experiment context changed')
        for name,digest in old['cache_hashes'].items():
            if c.sha(c.DATA/name)!=digest:raise RuntimeError('Feature cache changed')
        return
    cfg=c.cfg('daio');inventory=[]
    for split,paths in c.paths().items():
        for path in paths:
            # A single native Dataset defines the exact allowed windows/labels.
            from rovio_ablation.data import load_sequence
            sequence=load_sequence(path);n=len(sequence.imu_ts)
            if n>=1536:
                ds=c.V20WindowDataset([path],cfg);starts=np.array([e[1] for e in ds.entries],dtype=np.int64)
                target=np.array([e[3] for e in ds.entries],dtype=np.float32);time=np.array([e[2] for e in ds.entries])
            else:starts=np.empty(0,dtype=np.int64);target=np.empty((0,3),dtype=np.float32);time=np.empty(0)
            raw=np.c_[sequence.rpm,sequence.accel[:,:2],sequence.gyro[:,2],sequence.depth_m]
            if n>=6:
                cumulative=np.r_[np.zeros((1,12)),np.cumsum(raw,axis=0)]
                mean=((cumulative[6:]-cumulative[:-6])/6).astype('float32')
            else:mean=np.empty((0,12),dtype=np.float32)
            if len(starts):
                for index in np.unique(np.linspace(0,len(starts)-1,min(7,len(starts)),dtype=int)):
                    cached=mean[starts[index]+6*np.arange(256)]
                    native=np.concatenate([ds._feature_window(0,int(starts[index]),key).T for key in c.KEYS],axis=1)
                    np.testing.assert_allclose(cached,native,rtol=1e-6,atol=5e-5)
            c.npz(c.cache_path(path),features=mean,starts=starts,target=target,time=time)
            inventory.append(dict(path=str(path),split=split,windows=len(starts),samples=n))
            print('Cached',split,path.relative_to(c.DATA/'processed'),len(starts),flush=True)
    train=[r for r in inventory if r['split']=='train'];counts=np.cumsum([r['windows'] for r in train]);total=int(counts[-1])
    selected=np.linspace(0,total-1,min(4096,total),dtype=int);sums=np.zeros(12);squares=np.zeros(12);number=0
    previous=0
    for record,stop in zip(train,counts):
        local=selected[(selected>=previous)&(selected<stop)]-previous;previous=stop
        if not len(local):continue
        with np.load(c.cache_path(c.Path(record['path']))) as z:
            blocks=z['features'][z['starts'][local,None]+6*np.arange(256)].astype(float)
        sums+=blocks.sum((0,1));squares+=(blocks*blocks).sum((0,1));number+=blocks.shape[0]*256
    mean=sums/number;std=np.maximum(np.sqrt(np.maximum(squares/number-mean*mean,0)),.001);mean[:8]=0;std[:8]=1000
    normalizer={}
    for k in c.KEYS:normalizer.update({k+'_mean':mean[c.SLICES[k]].astype('float32').tolist(),k+'_std':std[c.SLICES[k]].astype('float32').tolist()})
    c.atomic_json(c.DATA/'feature_cache/normalizer.json',normalizer)
    caches={str(p.relative_to(c.DATA)):c.sha(p) for p in (c.DATA/'feature_cache').rglob('*.npz')}
    caches['feature_cache/normalizer.json']=c.sha(c.DATA/'feature_cache/normalizer.json')
    c.atomic_json(complete,dict(context=context,records=inventory,normalizer_fit_split='train',sampled_training_windows=len(selected),cache_hashes=caches))

if __name__=='__main__':snapshot();features()
