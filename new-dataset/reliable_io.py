"""Windows-safe artifact writes. Training state is never ignored on write failure."""
from pathlib import Path
import os,json,time,tempfile,sys

def _temporary(path):
    path=Path(path);path.parent.mkdir(parents=True,exist_ok=True)
    handle,name=tempfile.mkstemp(prefix=path.name+'.',suffix='.tmp',dir=path.parent)
    os.close(handle)
    return Path(name)

def _replace(source,target,attempts):
    for i in range(attempts):
        try:
            os.replace(source,target)
            return
        except PermissionError:
            if i==attempts-1:raise
            time.sleep(min(.05*2**i,1.0))

def atomic_json(path,obj):
    path=Path(path);temporary=_temporary(path)
    try:
        temporary.write_text(json.dumps(obj,indent=2,ensure_ascii=False),encoding='utf-8')
        try:_replace(temporary,path,5 if path.name=='progress.json' else 10)
        except PermissionError:
            if path.name!='progress.json':raise
            # Progress is telemetry, not a commit point. Keep the last good JSON,
            # append the current observation, and permit optimizer work to continue.
            try:
                with path.with_name('progress_fallback.jsonl').open('a',encoding='utf-8') as f:
                    f.write(json.dumps({'time_unix':time.time(),'progress':obj},ensure_ascii=False)+'\n')
            except OSError:pass
            print(f'Nonfatal progress-file lock: {path}; checkpoint writes remain mandatory.',file=sys.stderr,flush=True)
            return False
        return True
    finally:
        try:temporary.unlink(missing_ok=True)
        except OSError:pass

def torch_save(path,payload):
    import torch
    path=Path(path);temporary=_temporary(path)
    try:
        torch.save(payload,temporary)
        _replace(temporary,path,10)
    finally:
        try:temporary.unlink(missing_ok=True)
        except OSError:pass
