"""Train the standardized three-dimensional AirIO using the frozen AirIMU."""
import argparse
import sys
from pathlib import Path
import torch

ROOT = Path(__file__).resolve().parent
sys.path.insert(0,str(ROOT/'new-dataset'))
import airio_six_probes as probes
from airio_six_probes import common
import prepare


def main(checkpoint,device):
    prepared = prepare.main()
    model = common.vendor.airimu(device)
    payload = torch.load(checkpoint,map_location='cpu',weights_only=False)
    model.load_state_dict(payload['model'])
    for path in common.c.split_paths()['val']:
        common.causal.cached(path,model.eval(),prepared['calibration']['gyro_bias'],checkpoint,device)
    scaling = probes.fit_scaling()
    spec = dict(probes.BASE,name='imu_scaling',arm='imu_scaling',mean=scaling['mean'],scale=scaling['scale'])
    provenance = dict(stage='formal',data=prepared['context'],airimu=common.c.sha(checkpoint),spec=spec,
                      trainer={p.name:common.c.sha(p) for p in common.HERE.glob('*.py')},
                      config=common.CONFIG,entry=common.c.sha(__file__),model_source=common.c.sha(probes.__file__))
    ctx = dict(signature=common.digest(provenance))
    probes.trainer.HERE = ROOT/'outputs/airio_training'
    probes.trainer.model = probes.make_model
    common.write(probes.trainer.HERE/'provenance.json',provenance)
    probes.trainer.train_run(ctx,spec,'formal',42,device)


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--airimu',type=Path,default=ROOT/'models/airio/airimu.pt')
    p.add_argument('--device',default='cuda')
    a=p.parse_args()
    main(a.airimu,a.device)
