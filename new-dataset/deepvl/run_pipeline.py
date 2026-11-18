"""Sequential reproducible DeepVL train/evaluate/figure controller."""
from pathlib import Path
import json,subprocess,sys,time,traceback
HERE=Path(__file__).resolve().parent;DATA=HERE.parent;ROOT=DATA.parent
sys.path.insert(0,str(DATA))
from reliable_io import atomic_json


def main():
    def run(script,*args):
        command=[sys.executable,'-u','-B',str(script),*args]
        atomic_json(HERE/'pipeline_status.json',dict(state='running',command=command,time_unix=time.time()))
        subprocess.run(command,cwd=ROOT,check=True)
    try:
        if not (HERE/'runs/complete.json').exists():run(HERE/'train.py')
        run(DATA/'evaluate_pipeline.py','--methods','horizontal_rpm','horizontal_command','airio_planar','deepvl_planar')
        run(DATA/'plot_network_ablation.py')
        run(DATA/'plot_results.py','--methods','horizontal_rpm','horizontal_command','airio_planar','deepvl_planar')
        run(HERE/'validate.py')
        atomic_json(HERE/'pipeline_status.json',dict(state='complete',time_unix=time.time()))
    except BaseException:
        atomic_json(HERE/'pipeline_status.json',dict(state='failed',error=traceback.format_exc(),time_unix=time.time()));raise


if __name__=='__main__':main()
