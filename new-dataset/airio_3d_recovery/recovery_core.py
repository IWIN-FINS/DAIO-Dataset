"""Pure deterministic experiment design and fail-closed selection rules."""
import math
import numpy as np


def balanced_indices(groups, count, seed):
    groups=[np.asarray(g,dtype=np.int64) for g in groups if len(g)]
    if count>sum(map(len,groups)) or count<=0:raise ValueError('Insufficient distinct eligible windows')
    rng=np.random.default_rng(seed)
    shuffled=[rng.permutation(g) for g in groups]
    selected=[];used=[0]*len(groups)
    while len(selected)<count:
        for i,g in enumerate(shuffled):
            if used[i]<len(g):selected.append(int(g[used[i]]));used[i]+=1
            if len(selected)==count:break
    return np.asarray(selected,dtype=np.int64)


def epoch_order(count, seed, epoch):
    # Separate RNG from dropout, model initialization, evaluation and telemetry.
    return np.random.default_rng(np.random.SeedSequence([seed,epoch,9173])).permutation(count)


def choose_candidate(rows):
    rows=[r for r in rows if math.isfinite(r['metric'])]
    if not rows:raise ValueError('No finite candidate')
    best=min(r['metric'] for r in rows)
    return min((r for r in rows if r['metric']<=best+1e-4),key=lambda r:(r['complexity'],r['name']))


def qualification(value):
    checks=dict(
        finite=bool(value.get('finite',False)) and all(math.isfinite(value.get(k,float('nan'))) for k in
            ('xyz_velocity_rmse','zero_xyz_rmse','zero_ratio','input_sensitivity','prediction_std_ratio')),
        covariance=bool(value.get('covariance_valid',False)),
        better_than_zero_10pct=value.get('zero_ratio',float('inf'))<=.9,
        both_records_better=len(value.get('records',[]))==2 and all(
            r['deployed_xyz_rmse']<r['zero_xyz_rmse'] for r in value.get('records',[])),
        responsive=value.get('input_sensitivity',0)>1e-6,
        nonconstant=value.get('prediction_std_ratio',0)>=.01,
        encoder_healthy=bool(value.get('encoder_healthy',False)))
    return dict(passed=all(checks.values()),checks=checks)


def collapse(value):
    return (value.get('input_sensitivity',float('inf'))<=1e-6 and
            value.get('prediction_std_ratio',float('inf'))<.01 and value.get('zero_ratio',0)>=1.)


def degenerate(value):
    # For diagnosis, constant but slightly better-than-zero bias is still a collapse.
    return (not value.get('encoder_healthy',True) or
            (value.get('input_sensitivity',float('inf'))<=1e-6 and value.get('prediction_std_ratio',1)<.01))


def root_cause(evidence):
    checks=dict(fixed_fit=evidence['initial_fixed_rmse']>0 and evidence['fixed_rmse']<=.5*evidence['initial_fixed_rmse'])
    for name in ('representative','pilot','repeat'):
        candidate=evidence[name];control=evidence['original_'+name]
        checks[name+'_qualified']=qualification(candidate)['passed']
        checks[name+'_control_degenerated']=degenerate(control)
        checks[name+'_improves_control']=candidate['xyz_velocity_rmse']<control['xyz_velocity_rmse']-1e-4
    return dict(passed=all(checks.values()),checks=checks)
