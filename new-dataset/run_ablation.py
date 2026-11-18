"""Sequential, restartable seed-42 training queue; never selects by test score."""
from pathlib import Path
import sys,json,time,traceback
ROOT=Path(__file__).resolve().parents[1]
PROJECT=ROOT/'ROV-IO-Ablation'
sys.path.insert(0,str(PROJECT/'src'))
from rovio_ablation import engine
from rovio_ablation.config import load_config
from rovio_ablation.utils import atomic_json

def main():
    data=ROOT/'new-dataset';status_path=data/'training_status.json'
    status=json.loads(status_path.read_text(encoding='utf-8')) if status_path.exists() else {}
    for name in ('horizontal_rpm','full_imu_rpm','horizontal_command'):
        d=PROJECT/'experiments/new_v1v2'/name;cfg=load_config(d/'config.yaml')
        old=status.get(name,{})
        if old.get('state')=='complete':continue
        if old.get('run_dir'):
            run=Path(old['run_dir'])
            if not (run/'last.pt').exists():raise RuntimeError(f'Incomplete initial run {run}; inspect before retry')
            resume=True
        else:
            run=engine.create_run_dir(d/'runs',cfg['experiment']['name'],42)
            engine._initialize_new_run(run,cfg,d/'config.yaml',data/'processed/split.yaml',data,PROJECT,sys.argv)
            resume=False
        status[name]={'state':'running','run_dir':str(run),'started_unix':time.time()};atomic_json(status_path,status)
        try:
            if resume:engine.resume_run(run,data,PROJECT,sys.argv)
            else:engine._train(run,cfg,data/'processed/split.yaml',data,PROJECT,resume=False)
            status[name]['state']='complete';status[name]['finished_unix']=time.time();atomic_json(status_path,status)
        except BaseException:
            status[name]['state']='failed';status[name]['error']=traceback.format_exc();atomic_json(status_path,status);raise

if __name__=='__main__':main()
