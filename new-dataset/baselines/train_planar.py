"""Official planar training entry point; batch128 throughput revision."""
import argparse
from train_planar_fast import train

if __name__=='__main__':
    parser=argparse.ArgumentParser()
    parser.add_argument('stage',choices=['airimu','airio'])
    parser.add_argument('--smoke',action='store_true')
    parser.add_argument('--micro-batch',type=int,default=128)
    parser.add_argument('--checkpoint-every-updates',type=int,default=100)
    args=parser.parse_args()
    if args.checkpoint_every_updates<1:parser.error('checkpoint interval must be positive')
    train(args.stage,args.smoke,args.micro_batch,args.checkpoint_every_updates)
