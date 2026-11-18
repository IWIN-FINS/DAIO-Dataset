"""Sequential persistent pipeline, protected by an OS lock; same command resumes."""
import argparse,datetime,subprocess,sys,os,traceback
import common as c

def main(prepare_only=False):
    import msvcrt
    c.RUN.mkdir(parents=True,exist_ok=True);lock=(c.RUN/'pipeline.lock').open('a+b');lock.seek(0);lock.write(b'0');lock.flush();lock.seek(0)
    try:msvcrt.locking(lock.fileno(),msvcrt.LK_NBLCK,1)
    except OSError:raise RuntimeError('Combined experiment already running')
    state=dict(pid=os.getpid(),started=datetime.datetime.now().isoformat())
    def status(stage,**fields):
        state.update(stage=stage,updated=datetime.datetime.now().isoformat(),**fields);c.atomic_json(c.RUN/'status.json',state);print('STAGE',stage,flush=True)
    def call(script,*args):
        p=subprocess.Popen([sys.executable,'-B',str(c.HERE/script),*args],cwd=c.ROOT,stdout=sys.stdout,stderr=sys.stderr,creationflags=subprocess.CREATE_NO_WINDOW)
        status(state['stage'],child_pid=p.pid);code=p.wait()
        if code:raise RuntimeError(f'{script} exited {code}')
    try:
        status('prepare');call('prepare.py')
        if prepare_only:status('prepared');return
        context=c.fingerprint();smoke=c.RUN/'smoke_report.json'
        if not smoke.exists() or c.read(smoke)['context']!=context:status('smoke');call('smoke.py')
        for name in ('daio','no_rpm'):status('train_'+name);call('train.py',name)
        status('network_and_odometry_evaluation');call('evaluate.py')
        status('validate_metrics');call('validate.py')
        status('publication_plots');call('plots.py')
        c.REPORT.mkdir(parents=True,exist_ok=True)
        (c.REPORT/'README.md').write_text((c.HERE/'README.md').read_text(encoding='utf-8'),encoding='utf-8')
        status('complete')
    except BaseException as e:status('failed',error=repr(e),traceback=traceback.format_exc());raise
    finally:lock.seek(0);msvcrt.locking(lock.fileno(),msvcrt.LK_UNLCK,1);lock.close()

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--prepare-only',action='store_true');main(p.parse_args().prepare_only)
