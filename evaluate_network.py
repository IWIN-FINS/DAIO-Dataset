"""Evaluate a DAIO-family or ablation checkpoint on one old/new test record."""
import argparse
import csv
import sys
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader

ROOT=Path(__file__).resolve().parent
sys.path[:0]=[str(ROOT/'ROV-IO-Ablation/src'),str(ROOT/'ROV-IO-Training/src')]


def main():
    with (ROOT/'models/index.csv').open(encoding='utf8') as f:
        options={r['model']:r['file'] for r in csv.DictReader(f) if r['model'].split('/')[0] in ('old','new','combined','onboard')}
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--model',choices=options,required=True)
    parser.add_argument('--domain',choices=['new'],default='new')
    parser.add_argument('--test',type=int,choices=range(1,5),required=True)
    parser.add_argument('--device',choices=['cpu','cuda'],default='cpu')
    parser.add_argument('--output',type=Path)
    args=parser.parse_args()
    torch.set_num_threads(4)
    if args.domain=='old' and args.model in ('new/full_imu_rpm','new/horizontal_command'):
        parser.error('The old recordings lack the requested gyro or thruster command channels.')
    ckpt=torch.load(ROOT/options[args.model],map_location='cpu',weights_only=False)
    cfg=ckpt['config']
    if 'modalities' in cfg['model']:
        from rovio_ablation.model import build_model
        from rovio_ablation.data import V20WindowDataset
        keys=cfg['model']['modalities']
    else:
        from rovio_training.model import build_model
        from rovio_training.data import V20WindowDataset
        keys=('rpm','accel_xy','gyro_z','depth')
    with (ROOT/'datasets.csv').open(encoding='utf8') as f:
        row=next(r for r in csv.DictReader(f) if (r['domain'],r['test'])==(args.domain,f'Test {args.test}'))
    dataset=V20WindowDataset([ROOT/row['file']],cfg,ckpt['normalizer'])
    model=build_model(cfg).to(args.device)
    model.load_state_dict(ckpt['model'])
    model.eval()
    rows=[]
    with torch.inference_mode():
        for batch in DataLoader(dataset,batch_size=128,shuffle=False):
            means,log_std=model(*(batch[key].to(args.device) for key in keys))
            rows.append(np.column_stack([batch['time'].numpy(),means.cpu().numpy(),log_std.exp().cpu().numpy(),batch['target'].numpy()]))
    target=args.output or ROOT/'outputs/network'/args.model/f'{args.domain}_test{args.test}.csv'
    target.parent.mkdir(parents=True,exist_ok=True)
    np.savetxt(target,np.vstack(rows),delimiter=',',fmt='%.17g',comments='',
               header='time_s,vx,vy,yaw_rate,std_vx,std_vy,std_yaw_rate,gt_vx,gt_vy,gt_yaw_rate')
    print(f'{len(dataset)} network samples -> {target}')


if __name__=='__main__':main()
