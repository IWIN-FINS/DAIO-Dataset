"""Explicitly invoked training only; exact sampling/RNG resume and separate gates."""
import gc,json,time,math
import numpy as np
import torch
from common import HERE,CONFIG,c,write,read,dtype,model,parameter_groups,rng,restore_rng,preserve_rng,digest,now
from data import Bank
from metrics import forward,window_metrics,validation,parameter_stats,encoder_stats,offline_full_record
from recovery_core import epoch_order,collapse


def run_path(phase,spec,seed):return HERE/'runs'/phase/spec['name']/f'seed_{seed}'


def train_run(ctx,spec,phase,seed,device='cuda'):
    if phase not in ('fixed','representative','pilot','formal'):raise ValueError(phase)
    if not torch.cuda.is_available() and device.startswith('cuda'):raise RuntimeError('CUDA unavailable')
    destination=run_path(phase,spec,seed);destination.mkdir(parents=True,exist_ok=True)
    signature=digest(dict(context=ctx['signature'],spec=spec,phase=phase,seed=seed))
    if (destination/'complete.json').exists():
        done=read(destination/'complete.json')
        if done['signature']!=signature:raise RuntimeError('Refusing completed run with changed provenance')
        for label,sha in done['checkpoint_hashes'].items():
            if c.sha(destination/label)!=sha:raise RuntimeError('Completed checkpoint was changed')
        return done
    net=model(spec,seed,device);train=Bank('train',device,dtype(spec));val=Bank('val',device,dtype(spec))
    count=CONFIG['fixed_windows'] if phase=='fixed' else CONFIG['representative_windows']
    ids=train.selected(count,42) if phase in ('fixed','representative') else np.arange(len(train))
    # Identical selection across seeds as well as models; only sampling and initialization seed change.
    inventory_signature=digest(dict(ids=ids.tolist(),starts=train.starts[ids].tolist(),records=train.records))
    write(destination/'config.json',dict(signature=signature,context=ctx['signature'],spec=spec,seed=seed,phase=phase,
        windows=len(ids),inventory_signature=inventory_signature,settings=CONFIG))
    if phase in ('fixed','representative'):
        write(destination/'window_inventory.json',train.inventory(ids))
    else:
        # Complete index list stores record boundaries without expanding hundreds of thousands of JSON objects.
        c.save_npz(destination/'window_indices.npz',indices=ids,starts=train.starts[ids])
        write(destination/'window_records.json',train.records)
    optimizer=torch.optim.Adam(parameter_groups(net,spec),lr=CONFIG['lr'])
    scheduled=phase in ('pilot','formal')
    scheduler=torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer,mode='min',factor=CONFIG['lr_factor'],
        patience=CONFIG['lr_patience'],min_lr=CONFIG['min_lr']) if scheduled else None
    max_epochs=CONFIG['formal_epochs'] if phase=='formal' else CONFIG['pilot_epochs'] if phase=='pilot' else 10**9
    max_updates=CONFIG['fixed_updates'] if phase=='fixed' else CONFIG['representative_updates'] if phase=='representative' else 10**15
    telemetry_ids=ids if phase=='fixed' else train.selected(128,42)
    with preserve_rng():initial=window_metrics(net,train,telemetry_ids)
    state=dict(epoch=1,offset=0,updates=0,epoch_loss=0.,epoch_samples=0,history=[],initial=initial,
        best_diagnostic=None,best_qualified=None,best_fit=None,collapse_epochs=0,terminal_reason=None)
    if (destination/'last.pt').exists():
        saved=torch.load(destination/'last.pt',map_location='cpu',weights_only=False)
        if saved['signature']!=signature or saved['inventory_signature']!=inventory_signature:
            raise RuntimeError('Source/config/data/window hash changed; use a new experiment')
        net.load_state_dict(saved['model']);optimizer.load_state_dict(saved['optimizer'])
        if scheduler:scheduler.load_state_dict(saved['scheduler'])
        state=saved['state'];restore_rng(saved['rng'])
        print('RESUME',phase,spec['name'],seed,'update',state['updates'],flush=True)

    def checkpoint(name):
        c.torch_save(destination/name,dict(signature=signature,context=ctx['signature'],inventory_signature=inventory_signature,
            spec=spec,seed=seed,model=net.state_dict(),optimizer=optimizer.state_dict(),
            scheduler=None if scheduler is None else scheduler.state_dict(),state=state,rng=rng()))

    write(destination/'initial.json',state['initial']);started=time.monotonic()
    try:
        while state['epoch']<=max_epochs and state['updates']<max_updates and not state['terminal_reason']:
            epoch=state['epoch'];order=epoch_order(len(ids),seed,epoch)
            while state['offset']<len(ids) and state['updates']<max_updates:
                net.train();offset=state['offset'];index=ids[order[offset:offset+CONFIG['batch_size']]]
                optimizer.zero_grad(set_to_none=True);loss,parts,out,label=forward(net,train,index,spec)
                if not torch.isfinite(loss):raise FloatingPointError('Nonfinite loss')
                loss.backward()
                if any(p.grad is not None and not torch.isfinite(p.grad).all() for p in net.parameters()):
                    raise FloatingPointError('Nonfinite gradient')
                next_update=state['updates']+1
                stats=parameter_stats(net) if next_update%CONFIG['telemetry_interval']==0 else None
                optimizer.step()
                state['updates']=next_update;state['offset']=offset+len(index)
                state['epoch_loss']+=float(loss.detach())*len(index);state['epoch_samples']+=len(index)
                end_epoch=state['offset']==len(ids);end_budget=state['updates']==max_updates
                if stats is not None:
                    with preserve_rng():observed=window_metrics(net,train,telemetry_ids)
                    event=dict(time=now(),epoch=epoch,update=next_update,loss=parts,train=observed,
                               parameters_before_step=stats,encoder_max_after_step=encoder_stats(net)[0],lr=optimizer.param_groups[0]['lr'])
                    write(destination/'telemetry'/f'update_{next_update:07d}.json',event)
                    progress=dict(stage=phase,candidate=spec['name'],seed=seed,epoch=epoch,update=next_update,
                        max_updates=None if scheduled else max_updates,epoch_windows_done=state['offset'],epoch_windows=len(ids),
                        loss=state['epoch_loss']/state['epoch_samples'],train_rmse=observed['all_rmse'],
                        input_sensitivity=observed['input_sensitivity'],prediction_std_ratio=observed['prediction_std_ratio'],
                        lr=optimizer.param_groups[0]['lr'],elapsed_this_invocation_s=time.monotonic()-started,time=now())
                    write(destination/'progress.json',progress);write(HERE/'progress.json',progress)
                    print(json.dumps(progress),flush=True)
                evaluate=(next_update%CONFIG['validation_interval']==0 or end_budget or (scheduled and end_epoch))
                if evaluate:
                    with preserve_rng():
                        result=validation(net,val);train_metrics=window_metrics(net,train,telemetry_ids)
                    row=dict(epoch=epoch,update=next_update,train=train_metrics,validation=result,lr=optimizer.param_groups[0]['lr'])
                    state['history'].append(row)
                    score=result['xyz_velocity_rmse']
                    improved=state['best_diagnostic'] is None or score<state['best_diagnostic']['validation']['xyz_velocity_rmse']
                    fit_improved=state['best_fit'] is None or train_metrics['all_rmse']<state['best_fit']['train']['all_rmse']
                    qualified=result['qualification']['passed']
                    q_improved=qualified and (state['best_qualified'] is None or score<state['best_qualified']['validation']['xyz_velocity_rmse'])
                    if improved:state['best_diagnostic']=row
                    if fit_improved:state['best_fit']=row
                    if q_improved:state['best_qualified']=row
                    # Scheduler changes once per epoch, never at extra diagnostics.
                    if scheduled and end_epoch:
                        scheduler.step(score)
                        state['collapse_epochs']=state['collapse_epochs']+1 if collapse(result) else 0
                        if phase=='formal' and state['collapse_epochs']>=CONFIG['collapse_patience_epochs']:
                            state['terminal_reason']='consecutive_collapse'
                    if improved:checkpoint('best_diagnostic.pt')
                    if fit_improved:checkpoint('best_fit.pt')
                    if q_improved:checkpoint('best_qualified.pt')
                    write(destination/'history.json',state['history'])
                    print('VALIDATION',phase,spec['name'],'update',next_update,'RMSE',score,'passed',qualified,flush=True)
                if end_epoch:
                    state.update(epoch=epoch+1,offset=0,epoch_loss=0.,epoch_samples=0)
                if next_update%CONFIG['checkpoint_interval']==0 or (scheduled and end_epoch) or end_budget:
                    checkpoint('last.pt')
                if scheduled and end_epoch:checkpoint(f'epochs/epoch_{epoch:03d}.pt')
                if end_epoch or end_budget or state['terminal_reason']:break
        # Diagnostic-only upstream whole-record predictions do not affect selection.
        best=torch.load(destination/'best_diagnostic.pt',map_location='cpu',weights_only=False)
        net.load_state_dict(best['model']);net.eval()
        with preserve_rng():offline=offline_full_record(net)
        write(destination/'offline_diagnostic.json',offline)
        files=['last.pt','best_diagnostic.pt','best_fit.pt']
        if (destination/'best_qualified.pt').exists():files.append('best_qualified.pt')
        result=dict(signature=signature,spec=spec,phase=phase,seed=seed,updates=state['updates'],
            epochs_completed=state['epoch']-1,initial=state['initial'],best_diagnostic=state['best_diagnostic'],
            best_qualified=state['best_qualified'],best_fit=state['best_fit'],
            reason=state['terminal_reason'] or 'budget_complete',
            checkpoint_hashes={name:c.sha(destination/name) for name in files},completed=now())
        write(destination/'complete.json',result)
        return result
    except BaseException as exc:
        # Never overwrite last.pt with partly updated/nonfinite state on an exception.
        write(destination/'failure.json',dict(time=now(),error=repr(exc),last_committed_checkpoint=str(destination/'last.pt'),
                                             update=state['updates'],signature=signature))
        raise
    finally:
        del net,train,val,optimizer;gc.collect()
        if torch.cuda.is_available():torch.cuda.empty_cache()
