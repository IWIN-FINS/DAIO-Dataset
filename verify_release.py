"""Verify the distributed files with Python's standard library."""
from collections import Counter
from pathlib import Path
import csv
import hashlib

ROOT = Path(__file__).resolve().parent


def digest(path):
    h = hashlib.sha256()
    with path.open('rb') as stream:
        for part in iter(lambda: stream.read(1024 * 1024), b''):
            h.update(part)
    return h.hexdigest()


def main():
    assert not (ROOT/'dataset').exists(), 'Old dataset must not be included'
    for dataset in ('new-dataset',):
        for name in ('v1', 'v2', 'raw_files.csv'):
            if (ROOT / dataset / name).exists():
                raise ValueError(f'Raw dataset entry in processed-only release: {dataset}/{name}')
    inventory = (ROOT/'SHA256SUMS').read_text(encoding='utf8').splitlines()
    hashes = {}
    for line in inventory:
        expected, name = line.split('  ', 1)
        path = (ROOT/name).resolve()
        if not path.is_relative_to(ROOT) or name in hashes or digest(path) != expected:
            raise ValueError(f'Checksum mismatch or invalid entry: {name}')
        hashes[name] = expected
    with (ROOT/'datasets.csv').open(encoding='utf8') as stream:
        rows = list(csv.DictReader(stream))
    counts = Counter((r['domain'],r['split']) for r in rows)
    expected = {(d,s):n for d,ns in [('new',(49,2,4))] for s,n in zip(('train','val','test'),ns)}
    assert counts == expected and len({r['sha256'] for r in rows}) == len(rows)
    assert all(hashes[r['file']] == r['sha256'] for r in rows)
    assert len({(r['recording'],r['split']) for r in rows}) == len({r['recording'] for r in rows})
    for d in ('new',):
        tests=sorted((r for r in rows if r['domain']==d and r['split']=='test'),key=lambda r:r['test'])
        assert [Path(r['file']).name[:3] for r in tests]==['025','028','029','030']
    with (ROOT/'models/index.csv').open(encoding='utf8') as stream:
        models=list(csv.DictReader(stream))
    assert all(hashes[r['file']]==r['sha256'] and (ROOT/r['training']).is_file() for r in models)
    print(f'OK: {len(hashes)} files, {len(rows)} segments, {len(models)} checkpoints.')


if __name__ == '__main__':
    main()
