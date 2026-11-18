"""Explicit commands: preflight, diagnose, train, evaluate, report, status."""
import argparse,os,traceback
from common import HERE,REPORT,read,write,context,require_preflight,process_lock,now


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('command',choices=['preflight','diagnose','train','evaluate','report','status'])
    parser.add_argument('--through',choices=['fixed','representative','confirmation','all'],default='all',help='diagnose only')
    parser.add_argument('--device',default='cuda')
    args=parser.parse_args()
    if args.command=='status':
        for name in ('status.json','progress.json','diagnosis_result.json'):
            if (HERE/name).exists():print(name,read(HERE/name),flush=True)
        if not (HERE/'status.json').exists():print('No recovery command has been run.')
        return
    with process_lock():
        write(HERE/'status.json',dict(command=args.command,stage='running',pid=os.getpid(),time=now()))
        try:
            if args.command=='preflight':
                from preflight import run
                result=run(args.device)
            elif args.command=='report':
                from reporting import diagnostic_report
                diagnostic_report();result=dict(report=str(REPORT))
            else:
                ctx=context();require_preflight(ctx)
                if args.command=='diagnose':
                    from diagnosis import execute
                    from reporting import diagnostic_report
                    result=execute(ctx,args.through,args.device)
                    write(HERE/'diagnosis_result.json',result);diagnostic_report()
                elif args.command=='train':
                    from diagnosis import formal_lock
                    from trainer import train_run
                    from reporting import diagnostic_report
                    lock=formal_lock(ctx);result=train_run(ctx,lock['spec'],'formal',42,args.device)
                    diagnostic_report()
                    if result['best_qualified'] is None:raise RuntimeError('No qualified formal model; evaluation blocked')
                else:
                    from evaluation import run
                    from reporting import comparison_figures
                    result=run(ctx,args.device);comparison_figures()
            write(HERE/'status.json',dict(command=args.command,stage='complete',pid=os.getpid(),time=now(),result=result))
            print('COMPLETE',args.command,flush=True)
            if args.command=='diagnose':print('Formal training has not started. Run train explicitly only after diagnosis passes.',flush=True)
        except BaseException as exc:
            write(HERE/'status.json',dict(command=args.command,stage='failed',pid=os.getpid(),time=now(),error=repr(exc),traceback=traceback.format_exc()))
            raise


if __name__=='__main__':main()
