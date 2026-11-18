"""Clean the extended 33-column logs into an immutable legacy-compatible snapshot.

Firmware increments locate missing slots only. All physical time is sample_index/200.
Never edits raw data. Offline repairs are explicitly marked, not claimed causal.
"""
from __future__ import annotations
import argparse
import csv
from datetime import datetime, timezone
import hashlib
import json
import math
from pathlib import Path
import shutil
import tempfile
import numpy as np

HEADER=['received_utc','pressure','acc_x','acc_y','acc_z','gyro_x','gyro_y','gyro_z',
        'roll','pitch','yaw',*[f'int_rpm_{i}' for i in range(1,9)],
        *[f'throttle_{i}' for i in range(1,9)],'battery_voltage','firmware_timestamp',
        'tag_timestamp_utc','tag_x_m','tag_y_m','tag_yaw_rad']
TELEMETRY=HEADER[1:28]
ANGLES=['roll','pitch','yaw','tag_yaw_rad']
LIMITS={'pressure':(-1,100),**{f'acc_{a}':(-100,100) for a in 'xyz'},
        **{f'gyro_{a}':(-20,20) for a in 'xyz'},'roll':(-math.pi/2,math.pi/2),
        'pitch':(-math.pi/2,math.pi/2),**{f'throttle_{i}':(-1,1) for i in range(1,9)},
        'battery_voltage':(0,60)}

def number(s):
    try:return float(s)
    except (TypeError,ValueError):return float('nan')

def utc(s):
    try:return datetime.fromisoformat(s.replace('Z','+00:00')).timestamp()
    except (ValueError,TypeError,AttributeError):return float('nan')

def runs(mask):
    e=np.diff(np.r_[False,mask,False].astype(int))
    return zip(np.flatnonzero(e==1),np.flatnonzero(e==-1))

def clean_records(source):
    """Return segments and issues; source rows are CSV physical line numbers."""
    rows=[dict(r,_source_row=i+2,_action='original') for i,r in enumerate(source)]
    if not rows:return [],[]
    issues=[]; bad=np.zeros(len(rows),bool)
    for i,r in enumerate(rows):
        reason=[]
        if None in r or any(c not in r for c in HEADER):reason.append('row_width')
        for c in TELEMETRY:
            v=number(r.get(c));lim=LIMITS.get(c)
            if not math.isfinite(v) or (lim and not lim[0]<=v<=lim[1]):reason.append(c)
        if not math.isfinite(utc(r.get('received_utc'))):reason.append('received_utc')
        if reason:bad[i]=True;issues.append({'source_row':i+2,'kind':'invalid_packet','detail':','.join(reason)})
    # Reject isolated yaw spikes without rejecting legitimate wrapped/multi-turn yaw.
    yaw=np.array([number(r.get('yaw')) for r in rows]);valid=np.flatnonzero(~bad & np.isfinite(yaw))
    for j in range(1,len(valid)-1):
        a,i,b=valid[j-1:j+2]
        angle=lambda x:math.atan2(math.sin(x),math.cos(x))
        if abs(angle(yaw[i]-yaw[a]))>.2 and abs(angle(yaw[b]-yaw[i]))>.2 and abs(angle(yaw[b]-yaw[a]))<.1:
            bad[i]=True;issues.append({'source_row':i+2,'kind':'isolated_yaw_spike','detail':'whole telemetry repair'})
    valid=np.flatnonzero(~bad)
    if len(valid)<2:raise ValueError('Fewer than two valid telemetry packets')
    lo,hi=valid[0],valid[-1]
    for i in list(range(lo))+list(range(hi+1,len(rows))):
        issues.append({'source_row':i+2,'kind':'edge_trim','detail':'invalid edge packet'})
    rows=rows[lo:hi+1];bad=bad[lo:hi+1]
    fw=np.array([number(r.get('firmware_timestamp')) for r in rows]);host=np.array([utc(r.get('received_utc')) for r in rows])
    dif=np.diff(fw);normal=dif[np.isfinite(dif)&(dif>0)]
    if not len(normal):raise ValueError('Cannot identify firmware slot increments')
    step=float(np.median(normal))
    # Long damaged runs and bad clocks are segment boundaries, never interpolated over.
    drop=np.zeros(len(rows),bool)
    for a,b in runs(bad):
        if (b-a+1)*.005>.0500001:drop[a:b]=True
    expanded=[];segments=[]
    def flush():
        nonlocal expanded
        if expanded:segments.append(expanded);expanded=[]
    for i,r in enumerate(rows):
        if drop[i]:
            issues.append({'source_row':r['_source_row'],'kind':'long_invalid_run','detail':'segment boundary'})
            flush();continue
        if i and not drop[i-1]:
            df=fw[i]-fw[i-1];dh=host[i]-host[i-1]
            ratio=df/step if math.isfinite(df) else float('nan')
            k=int(round(ratio)) if math.isfinite(ratio) else -1
            uncertain=k<1 or abs(ratio-k)>.25
            boundary=uncertain or k*.005>.0500001 or (math.isfinite(dh) and (dh<0 or dh>.050001))
            if boundary:
                issues.append({'source_row':r['_source_row'],'kind':'time_boundary','detail':f'host={dh};firmware={df};slots={k}'})
                flush()
            elif k>1:
                issues.append({'source_row':r['_source_row'],'kind':'missing_slots','detail':str(k-1)})
                for j in range(1,k):
                    ins={c:'' for c in HEADER}
                    ins.update(_source_row='',_action='inserted',_before_source_row=rows[i-1]['_source_row'],_after_source_row=r['_source_row'])
                    for c in ('received_utc','tag_timestamp_utc'):
                        l,h=utc(rows[i-1].get(c)),utc(r.get(c))
                        if math.isfinite(l) and math.isfinite(h):
                            ins[c]=datetime.fromtimestamp(l+(h-l)*j/k,timezone.utc).isoformat(timespec='milliseconds')
                    ins['firmware_timestamp']=str(fw[i-1]+df*j/k)
                    expanded.append(ins)
        if bad[i]:
            for c in TELEMETRY:r[c]=''
            r['_action']='repaired'
        expanded.append(r)
    flush()
    output=[]
    for seg in segments:
        # Trim any damaged boundaries created by segmentation: no extrapolation.
        good=lambda r:all(math.isfinite(number(r.get(c))) for c in TELEMETRY)
        while seg and not good(seg[0]):seg=seg[1:]
        while seg and not good(seg[-1]):seg=seg[:-1]
        if len(seg)<2:continue
        columns=TELEMETRY+['tag_x_m','tag_y_m','tag_yaw_rad']
        for c in columns:
            a=np.array([number(r.get(c)) for r in seg]);v=np.isfinite(a);ids=np.flatnonzero(v)
            if len(ids)<2 or not v[0] or not v[-1]:
                raise ValueError(f'{c}: no bounded interpolation anchors; investigate labels')
            if c in ANGLES:
                # Keep original strings if no wrapping or repair was needed.
                unwrapped=np.unwrap(a[v]); changed=~np.isclose(unwrapped,a[v],atol=1e-12,rtol=0)
                for ix,value,change in zip(ids,unwrapped,changed):
                    if change:seg[ix][c]=format(value,'.17g')
                a[v]=unwrapped
            missing=~v
            if missing.any():
                a[missing]=np.interp(np.flatnonzero(missing),ids,a[v])
                for ix in np.flatnonzero(missing):seg[ix][c]=format(a[ix],'.17g')
        output.append(seg)
    return output,issues

def sha(p):
    h=hashlib.sha256()
    with p.open('rb') as f:
        for b in iter(lambda:f.read(1024*1024),b''):h.update(b)
    return h.hexdigest()

def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--raw-root',type=Path,default=Path(__file__).parents[1])
    parser.add_argument('--output-root',type=Path)
    args=parser.parse_args();root=args.raw_root.resolve();out=(args.output_root or root/'processed').resolve()
    if out.exists():raise FileExistsError(f'Immutable snapshot already exists: {out}')
    staging=Path(tempfile.mkdtemp(prefix='processed_staging_',dir=out.parent))
    manifest=[];issues=[];sources={};split={'train':[],'val':[],'demo':[]};splitrows=[]
    try:
        for version in ('v1','v2'):
            for p in sorted((root/version).rglob('*.csv')):
                rel=f'{version}/{p.name}'
                if rel in sources:raise ValueError('Duplicate raw basename: '+rel)
                sources[rel]=sha(p)
                with p.open(encoding='utf-8-sig',newline='') as f:
                    reader=csv.DictReader(f)
                    if reader.fieldnames!=HEADER:raise ValueError(f'Schema mismatch: {p}')
                    segs,found=clean_records(list(reader))
                issues.extend(dict(source_relative_path=rel,**r) for r in found)
                subset='val' if version=='v2' and p.stem[:3] in ('026','027') else 'demo' if version=='v2' and p.stem[:3] in ('025','028','029','030') else 'train'
                for s,seg in enumerate(segs,1):
                    target=f'{version}/{p.stem}__segment_{s:03d}.csv';dest=staging/target;dest.parent.mkdir(exist_ok=True)
                    sat=[j for j,r in enumerate(seg) if any(abs(number(r[f'int_rpm_{i}']))>=32700 for i in range(1,9))]
                    repaired=[j for j,r in enumerate(seg) if r['_action']!='original']
                    with dest.open('w',encoding='utf-8',newline='') as f:
                        w=csv.writer(f);w.writerow(HEADER+['sample_time_s'])
                        for j,r in enumerate(seg):w.writerow([r[c] for c in HEADER]+[f'{j*.005:.9f}'])
                    maprel=f'row_maps/{version}/{p.stem}__segment_{s:03d}.csv'
                    mp=staging/maprel;mp.parent.mkdir(exist_ok=True,parents=True)
                    with mp.open('w',encoding='utf-8',newline='') as f:
                        w=csv.writer(f);w.writerow(['sample_index','sample_time_s','source_csv_line','action','before_source_csv_line','after_source_csv_line'])
                        for j,r in enumerate(seg):w.writerow([j,f'{j*.005:.9f}',r['_source_row'],r['_action'],r.get('_before_source_row',''),r.get('_after_source_row','')])
                    orig=[r['_source_row'] for r in seg if r['_source_row']!='']
                    entry={'segment_id':target[:-4],'source_relative_path':rel,'processed_relative_path':target,
                           'source_row_start':min(orig),'source_row_end':max(orig),'frame_count':len(seg),
                           'sample_time_start_s':0,'sample_time_end_s':(len(seg)-1)*.005,'duration_s':(len(seg)-1)*.005,'sample_rate_hz':200,
                           'received_utc_start_s':utc(seg[0]['received_utc']),'received_utc_end_s':utc(seg[-1]['received_utc']),
                           'utc_gap_before_s':'','rpm_saturation_source_rows':json.dumps([seg[j]['_source_row'] for j in sat]),
                           'rpm_saturation_sample_indices':json.dumps(sat),'rpm_saturation_times_s':json.dumps([j*.005 for j in sat]),
                           'network_window_exclusion_end_intervals_s':json.dumps([[j*.005,min((j+1535)*.005,(len(seg)-1)*.005)] for j in sat]),
                           'repaired_sample_indices':json.dumps(repaired),'row_map_relative_path':maprel,'sha256':sha(dest)}
                    manifest.append(entry);split[subset].append(target);splitrows.append({'split':'test' if subset=='demo' else subset,'source':rel,'processed':target,'sha256':entry['sha256']})
                print(f'{rel}: {len(segs)} segment(s), {sum(len(x) for x in segs)} rows, {len(found)} issues',flush=True)
        for name,rs in [('manifest',manifest),('row_issues',issues),('split_inventory',splitrows)]:
            with (staging/f'{name}.csv').open('w',encoding='utf-8',newline='') as f:
                w=csv.DictWriter(f,fieldnames=list(rs[0]) if rs else ['none']);w.writeheader();w.writerows(rs)
        audit={'schema_version':'extended_33_columns_200hz_v1','source_csv_count':len(sources),'raw_file_sha256':sources,
               'processed_segment_count':len(manifest),'frame_count':sum(r['frame_count'] for r in manifest),
               'split_counts':{k:len(v) for k,v in split.items()},'source_time_base':'uniform_5ms_with_confirmed_missing_slots',
               'inserted_rows':sum(int(r['detail']) for r in issues if r['kind']=='missing_slots'),
               'repaired_existing_rows':sum(r['kind']=='invalid_packet' for r in issues),
               'issue_counts':{k:sum(r['kind']==k for r in issues) for k in sorted({r['kind'] for r in issues})}}
        (staging/'audit_summary.json').write_text(json.dumps(audit,indent=2,ensure_ascii=False),encoding='utf-8')
        import yaml
        manifest_config={'schema_version':3,'data_format':'processed_csv_200hz','coordinate_frame':'FLU; x forward y left z up',
                         'offline_apriltag_only':True,'processed_root':'processed',
                         'splits':{k:{'files':v,'expected_count':len(v)} for k,v in split.items()}}
        (staging/'split.yaml').write_text(yaml.safe_dump(manifest_config,allow_unicode=True,sort_keys=False),encoding='utf-8')
        staging.rename(out)
        print(json.dumps({k:v for k,v in audit.items() if k!='raw_file_sha256'},indent=2))
    except BaseException:
        # Preserve staging for diagnosis; never remove a source or an existing snapshot.
        print(f'Incomplete staging retained: {staging}',flush=True)
        raise

if __name__=='__main__':main()
