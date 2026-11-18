"""Bounded controlled experiments and evidence-based formal-training lock."""
import sys
import numpy as np
from common import HERE,REPORT,BASELINE,CANDIDATES,SUPPLEMENTAL,CONFIG,write,read,c,digest,now
from trainer import train_run,run_path
from recovery_core import choose_candidate,root_cause,qualification


def selected_validation(result):return result['best_diagnostic']['validation']


def collect_evidence(fixed,representative,pilot,repeat,controls):
    return dict(initial_fixed_rmse=fixed['initial']['all_rmse'],fixed_rmse=fixed['best_fit']['train']['all_rmse'],
                representative=selected_validation(representative),pilot=selected_validation(pilot),repeat=selected_validation(repeat),
                original_representative=selected_validation(controls['representative']),
                original_pilot=selected_validation(controls['pilot']),original_repeat=selected_validation(controls['repeat']))


def causal_evidence(value):
    checks=dict(fixed_fit=value['fixed_rmse']<=.5*value['initial_fixed_rmse'])
    for name in ('representative','pilot','repeat'):
        r=value[name];old=value['original_'+name];w=old['windows']
        checks[name+'_qualified']=qualification(r)['passed']
        checks[name+'_control_internal_effective']=w['all_rmse']<=.9*w['zero_all_rmse']
        checks[name+'_control_endpoint_poor']=w['endpoint_rmse']>=.99*w['zero_endpoint_rmse']
        checks[name+'_improved']=r['xyz_velocity_rmse']<old['xyz_velocity_rmse']-1e-4
    return dict(passed=all(checks.values()),checks=checks)


def confirm(ctx,spec,control,fixed,representative,control_rep,device):
    pilot=train_run(ctx,spec,'pilot',42,device)
    original_pilot=train_run(ctx,control,'pilot',42,device)
    repeat=train_run(ctx,spec,'representative',43,device)
    original_repeat=train_run(ctx,control,'representative',43,device)
    ev=collect_evidence(fixed,representative,pilot,repeat,dict(representative=control_rep,pilot=original_pilot,repeat=original_repeat))
    causal=spec['endpoint_weight']!=control['endpoint_weight']
    outcome=causal_evidence(ev) if causal else root_cause(ev)
    result=dict(candidate=spec,control=control,evidence=ev,outcome=outcome,signature=ctx['signature'],time=now(),
                interpretation='endpoint_supervision_mismatch' if causal else
                    'coupled_weight_decay' if spec['weight_decay']!=control['weight_decay'] else
                    'normalization_and_bias_decay' if spec['exempt_norm_bias']!=control['exempt_norm_bias'] else 'fp32_numeric_effect')
    write(HERE/'confirmations'/(spec['name']+'.json'),result)
    if outcome['passed']:
        # Store exact evidence file hashes; formal start revalidates them, not just a boolean.
        references=[run_path('fixed',spec,42)/'complete.json',run_path('representative',spec,42)/'complete.json',
            run_path('pilot',spec,42)/'complete.json',run_path('representative',spec,43)/'complete.json',
            run_path('representative',control,42)/'complete.json',run_path('pilot',control,42)/'complete.json',
            run_path('representative',control,43)/'complete.json',HERE/'confirmations'/(spec['name']+'.json')]
        locked=dict(ready=True,signature=ctx['signature'],spec=spec,seed=42,formal_epochs=CONFIG['formal_epochs'],
            reason=result['interpretation'],evidence={str(p.relative_to(HERE)):c.sha(p) for p in references},
            confirmation=str((HERE/'confirmations'/(spec['name']+'.json')).relative_to(HERE)),time=now(),
            automatic_training_started=False)
        write(HERE/'formal_config.lock.json',locked)
        write(HERE/'diagnosis_result.json',dict(ready=True,reason=result['interpretation'],candidate=spec,
            signature=ctx['signature'],time=now(),next_action='User starts run.py train; never started by diagnose'))
        print('DIAGNOSIS PASSED. Formal training has NOT started.',flush=True)
    return outcome['passed']


def execute(ctx,through='all',device='cuda'):
    fixed={};representative={};attempted=set()
    for spec in CANDIDATES:
        fixed[spec['name']]=train_run(ctx,spec,'fixed',42,device)
    if through=='fixed':return dict(ready=False,stage='fixed_complete')
    for spec in CANDIDATES:
        representative[spec['name']]=train_run(ctx,spec,'representative',42,device)
    if through=='representative':return dict(ready=False,stage='representative_complete')

    def choose(specs):
        rows=[dict(name=s['name'],spec=s,metric=selected_validation(representative[s['name']])['xyz_velocity_rmse'],
                   complexity=s['complexity']) for s in specs]
        return choose_candidate(rows)['spec']

    def attempt(spec,control=BASELINE):
        attempted.add(spec['name'])
        return confirm(ctx,spec,control,fixed[spec['name']],representative[spec['name']],representative[control['name']],device)

    first=choose(CANDIDATES[1:])
    if attempt(first):return read(HERE/'diagnosis_result.json')
    # Try the remaining decay arm only if it satisfies deployment and fit prerequisites.
    for spec in CANDIDATES[1:]:
        if spec['name'] not in attempted:
            r=representative[spec['name']];f=fixed[spec['name']]
            if qualification(selected_validation(r))['passed'] and f['best_fit']['train']['all_rmse']<=.5*f['initial']['all_rmse']:
                if attempt(spec):return read(HERE/'diagnosis_result.json')
    if through=='confirmation':
        result=dict(ready=False,stage='confirmation_inconclusive',signature=ctx['signature'],time=now())
        write(HERE/'diagnosis_result.json',result);return result

    # Ordered single-factor follow-ups. Each compares against the original arm.
    for spec in SUPPLEMENTAL:
        fixed[spec['name']]=train_run(ctx,spec,'fixed',42,device)
        representative[spec['name']]=train_run(ctx,spec,'representative',42,device)
        if qualification(selected_validation(representative[spec['name']]))['passed']:
            if attempt(spec):return read(HERE/'diagnosis_result.json')

    # Endpoint supervision is tested only after observing the specified mismatch.
    all_specs=CANDIDATES+SUPPLEMENTAL
    eligible=[]
    for spec in all_specs:
        v=selected_validation(representative[spec['name']]);w=v['windows']
        if w['all_rmse']<=.9*w['zero_all_rmse'] and w['endpoint_rmse']>=.99*w['zero_endpoint_rmse']:
            eligible.append(spec)
    if eligible:
        control=choose(eligible)
        spec={**control,'name':control['name']+'_endpoint','endpoint_weight':1.,'complexity':control['complexity']+10}
        fixed[spec['name']]=train_run(ctx,spec,'fixed',42,device)
        representative[spec['name']]=train_run(ctx,spec,'representative',42,device)
        if attempt(spec,control):return read(HERE/'diagnosis_result.json')
    result=dict(ready=False,signature=ctx['signature'],time=now(),reason='cause_not_confirmed',
        attempted=list(attempted),label_timing='Official get_label alignment verified by preflight; no unproven timestamp shift applied',
        next_action='Inspect evidence and sensor synchronization; do not start formal training or relax gates')
    write(HERE/'diagnosis_result.json',result)
    return result


def formal_lock(ctx):
    path=HERE/'formal_config.lock.json'
    if not path.exists():raise RuntimeError('No confirmed diagnosis / locked formal configuration; run diagnose first')
    lock=read(path)
    if not lock['ready'] or lock['signature']!=ctx['signature']:raise RuntimeError('Formal configuration is stale or unqualified')
    for name,sha in lock['evidence'].items():
        if c.sha(HERE/name)!=sha:raise RuntimeError('Diagnostic evidence changed: '+name)
    confirmation=read(HERE/lock['confirmation'])
    verified=causal_evidence(confirmation['evidence']) if confirmation['interpretation']=='endpoint_supervision_mismatch' else root_cause(confirmation['evidence'])
    if not verified['passed'] or lock['spec']!=confirmation['candidate']:raise RuntimeError('Root-cause gates do not pass independently')
    return lock
