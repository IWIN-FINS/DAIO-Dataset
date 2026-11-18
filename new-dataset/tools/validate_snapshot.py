"""Independently validate published CSVs, source hashes, maps, and disjoint splits."""
from pathlib import Path
import sys,csv,json,hashlib,math
import numpy as np
import yaml
ROOT=Path(__file__).resolve().parents[1]

def sha(path):
    h=hashlib.sha256()
    with path.open('rb') as f:
        for b in iter(lambda:f.read(1048576),b''):h.update(b)
    return h.hexdigest()

def main():
    p=ROOT/'processed';audit=json.loads((p/'audit_summary.json').read_text(encoding='utf-8'))
    manifests=list(csv.DictReader((p/'manifest.csv').open(encoding='utf-8')))
    split=yaml.safe_load((p/'split.yaml').read_text(encoding='utf-8'))['splits']
    assert len(manifests)==55
    sources={};membership={};cells=0;inserted=repaired=0;frames=0
    for group,spec in split.items():
        for rel in spec['files']:
            assert rel not in membership;membership[rel]=group
    for m in manifests:
        rel=m['processed_relative_path'];source=m['source_relative_path'];dest=p/rel
        assert sha(dest)==m['sha256']
        files=list((ROOT/source.split('/')[0]).rglob(Path(source).name));assert len(files)==1
        assert sha(files[0])==audit['raw_file_sha256'][source]
        if source not in sources:
            with files[0].open(encoding='utf-8-sig',newline='') as f:sources[source]=list(csv.DictReader(f))
        source_groups={membership[r['processed_relative_path']] for r in manifests if r['source_relative_path']==source}
        assert len(source_groups)==1
        with dest.open(encoding='utf-8',newline='') as f:rows=list(csv.DictReader(f))
        with (p/m['row_map_relative_path']).open(encoding='utf-8',newline='') as f:maps=list(csv.DictReader(f))
        assert len(rows)==len(maps)==int(m['frame_count']);frames+=len(rows)
        for i,(row,maprow) in enumerate(zip(rows,maps)):
            assert abs(float(row['sample_time_s'])-i*.005)<1e-8
            for c,v in row.items():
                if c not in ('received_utc','tag_timestamp_utc'):assert math.isfinite(float(v)),(rel,i,c)
            if maprow['action']=='original':
                old=sources[source][int(maprow['source_csv_line'])-2]
                for c in old:
                    if c in ('received_utc','tag_timestamp_utc'):assert old[c]==row[c]
                    elif c in ('roll','pitch','yaw','tag_yaw_rad'):
                        assert abs(math.remainder(float(old[c])-float(row[c]),2*math.pi))<1e-10
                    else:assert float(old[c])==float(row[c]),(rel,i,c)
                    cells+=1
            elif maprow['action']=='inserted':inserted+=1
            elif maprow['action']=='repaired':
                repaired+=1;old=sources[source][int(maprow['source_csv_line'])-2]
                for c in ('tag_x_m','tag_y_m'):assert float(row[c])==float(old[c])
        for i,maprow in enumerate(maps):
            if maprow['action']=='original':continue
            lo=i-1;hi=i+1
            while maps[lo]['action']!='original':lo-=1
            while maps[hi]['action']!='original':hi+=1
            fraction=(i-lo)/(hi-lo)
            columns=list(rows[i])[1:28]
            if maprow['action']=='inserted':columns+=['tag_x_m','tag_y_m','tag_yaw_rad']
            for c in columns:
                a,b,value=map(float,(rows[lo][c],rows[hi][c],rows[i][c]))
                delta=math.remainder(b-a,2*math.pi) if c in ('roll','pitch','yaw','tag_yaw_rad') else b-a
                expected=a+fraction*delta
                error=math.remainder(value-expected,2*math.pi) if c in ('roll','pitch','yaw','tag_yaw_rad') else value-expected
                assert abs(error)<=1e-10*max(1,abs(expected)),(rel,i,c,value,expected)
        print('verified',rel,flush=True)
    assert inserted==27 and repaired==10 and frames==1483117
    result={'ok':True,'source_hashes_verified':54,'segments':55,'frame_count':frames,'original_cells_verified':cells,
            'inserted_rows':inserted,'repaired_packets':repaired,'split_counts':{k:len(v['files']) for k,v in split.items()}}
    (ROOT/'snapshot_validation.json').write_text(json.dumps(result,indent=2),encoding='utf-8');print(json.dumps(result,indent=2))

if __name__=='__main__':main()
