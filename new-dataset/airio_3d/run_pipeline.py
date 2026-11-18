"""Single-GPU sequential controller; failures never bypass validation gates."""
import argparse,datetime,json,os,subprocess,sys,time,traceback
import contracts as c


def final_gate(device):
    import torch
    import vendor,physics,causal
    import train
    train.setup()
    checkpoint=c.HERE/'runs/airimu/best.pt';model=vendor.airimu(device)
    payload=torch.load(checkpoint,map_location=device,weights_only=False)
    if payload['context']!=c.fingerprint():raise RuntimeError('Frozen AirIMU source changed')
    model.load_state_dict(payload['model']);model.eval()
    bias=json.loads((c.HERE/'prepared/warmup_calibration.json').read_text(encoding='utf-8'))['gyro_bias']
    report,_=physics.validate(model,bias,device);report['checkpoint_sha256']=c.sha(checkpoint)
    c.atomic_json(c.HERE/'runs/airimu/frozen_physics.json',report)
    if not report['passed']:raise RuntimeError('Best AirIMU failed repeated causal physics check; AirIO will not start')
    for paths in c.split_paths().values():
        for path in paths:
            causal.cached(path,model,bias,checkpoint,device);print('Frozen causal pose cache',path.name,flush=True)
    c.atomic_json(c.HERE/'orientation_cache/complete.json',dict(checkpoint_sha256=c.sha(checkpoint),records=55,context=c.fingerprint()))


def main(through='all',device='cuda'):
    # OS-level lock is automatically released on crash; stale PID files are not locks.
    lock=(c.HERE/'pipeline.lock').open('a+b');lock.seek(0);lock.write(b'0');lock.flush();lock.seek(0)
    import msvcrt
    try:msvcrt.locking(lock.fileno(),msvcrt.LK_NBLCK,1)
    except OSError:raise RuntimeError('Another airio_3d controller is already running')
    context=c.fingerprint();status=dict(pid=os.getpid(),started=datetime.datetime.now().isoformat(),through=through,context=context)
    c.atomic_json(c.HERE/'process.json',status)
    def update(stage,**extra):
        status.update(stage=stage,updated=datetime.datetime.now().isoformat(),**extra);c.atomic_json(c.HERE/'status.json',status);print('STAGE',stage,flush=True)
    def call(script,*args):
        child=subprocess.Popen([sys.executable,'-B',str(script),*args],cwd=c.ROOT,stdout=sys.stdout,stderr=sys.stderr,creationflags=subprocess.CREATE_NO_WINDOW)
        update(status['stage'],child_pid=child.pid)
        code=child.wait()
        if code:raise RuntimeError(f'{script.name} failed with exit {code}')
    try:
        import promote
        promote.snapshot();update('prepare');call(c.HERE/'prepare.py')
        if through=='prepare':update('complete_prepare');return
        if through!='evaluate':
            smoke=c.HERE/'smoke_report.json'
            if not smoke.exists() or json.loads(smoke.read_text(encoding='utf-8')).get('context')!=context:
                update('smoke');call(c.HERE/'smoke.py')
            update('airimu_training');call(c.HERE/'train.py','airimu','--device',device)
            update('freeze_and_recheck_airimu');call(c.HERE/'run_pipeline.py','--internal-final-gate','--device',device)
            if through=='airimu':update('complete_airimu');return
            update('airio_training');call(c.HERE/'train.py','airio','--device',device)
            if through=='airio':update('complete_airio');return
        update('test_evaluation');call(c.HERE/'evaluate.py')
        update('full_state_validation');call(c.HERE/'validate.py')
        update('promote');call(c.HERE/'promote.py')
        update('publication_plots');call(c.DATA/'plot_results.py','--methods',*promote.METHODS)
        update('export_velocity_figures');call(c.DATA/'export_velocity_figures.py')
        promote.verify_unchanged(json.loads((c.HERE/'previous_main_comparison/snapshot.json').read_text(encoding='utf-8')))
        update('complete',completed=datetime.datetime.now().isoformat())
    except BaseException as exc:
        update('failed',error=repr(exc),traceback=traceback.format_exc());raise
    finally:
        lock.seek(0);msvcrt.locking(lock.fileno(),msvcrt.LK_UNLCK,1);lock.close()


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--through',choices=['prepare','airimu','airio','evaluate','all'],default='all')
    p.add_argument('--device',default='cuda');p.add_argument('--internal-final-gate',action='store_true');args=p.parse_args()
    if args.internal_final_gate:final_gate(args.device)
    else:main(args.through,args.device)
