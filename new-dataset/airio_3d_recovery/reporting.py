"""Diagnostic tables/curves and separate accepted comparison exports."""
import csv,json
import numpy as np
from common import HERE,REPORT,c,read,write,now


def diagnostic_report():
    REPORT.mkdir(parents=True,exist_ok=True);rows=[]
    for path in sorted((HERE/'runs').glob('*/*/seed_*/history.json')):
        cfg=read(path.parent/'config.json')
        for r in read(path):
            v=r['validation'];w=v['windows']
            rows.append(dict(phase=cfg['phase'],candidate=cfg['spec']['name'],seed=cfg['seed'],epoch=r['epoch'],update=r['update'],
                train_all_rmse=r['train']['all_rmse'],validation_all_rmse=w['all_rmse'],validation_endpoint_rmse=w['endpoint_rmse'],
                deployed_xyz_rmse=v['xyz_velocity_rmse'],teacher_xyz_rmse=v['teacher_xyz_rmse'],zero_xyz_rmse=v['zero_xyz_rmse'],
                official_window_mean_residual_rmse=w['official_window_mean_residual_rmse'],
                input_sensitivity=v['input_sensitivity'],prediction_std_ratio=v['prediction_std_ratio'],
                encoder_healthy=v['encoder_healthy'],qualified=v['qualification']['passed'],lr=r['lr']))
    if rows:
        with (REPORT/'diagnostic_metrics.csv').open('w',encoding='utf-8-sig',newline='') as f:
            w=csv.DictWriter(f,fieldnames=list(rows[0]));w.writeheader();w.writerows(rows)
        import matplotlib
        matplotlib.use('Agg')
        import matplotlib.pyplot as plt
        for phase in sorted(set(r['phase'] for r in rows)):
            fig,axes=plt.subplots(2,1,figsize=(10,7),sharex=True,layout='constrained')
            selected=[r for r in rows if r['phase']==phase]
            for name,seed in sorted(set((r['candidate'],r['seed']) for r in selected)):
                a=[r for r in selected if r['candidate']==name and r['seed']==seed]
                x=[r['update'] for r in a];axes[0].plot(x,[r['deployed_xyz_rmse'] for r in a],label=f'{name}, seed {seed}')
                axes[1].plot(x,[max(r['input_sensitivity'],1e-14) for r in a])
            axes[0].axhline(selected[0]['zero_xyz_rmse'],color='black',ls=':',label='Zero velocity')
            axes[0].set_ylabel('Causal XYZ RMSE [m/s]');axes[0].legend(fontsize=8,ncols=2)
            axes[1].set_ylabel('Input response [m/s]');axes[1].set_yscale('log');axes[1].set_xlabel('Optimizer updates')
            for ax in axes:ax.grid(alpha=.2)
            fig.savefig(REPORT/f'{phase}_diagnostic.png',dpi=200);plt.close(fig)
    conclusion=read(HERE/'diagnosis_result.json') if (HERE/'diagnosis_result.json').exists() else dict(ready=False,reason='Not run')
    write(REPORT/'diagnosis_status.json',conclusion)
    confirmations=[read(p) for p in sorted((HERE/'confirmations').glob('*.json'))]
    write(REPORT/'cause_evidence.json',confirmations)
    lines=['# AirIO recovery experiment','',f'Generated: {now()}',
        f'Formal training authorized by evidence: {conclusion.get("ready",False)}',
        f'Conclusion: {conclusion.get("reason",conclusion.get("stage","not run"))}',
        '', 'Only train/validation data determine diagnosis and configuration. Test data are evaluated after a qualified formal model.',
        'Official all-token, causal endpoint, teacher-pose and deployed-pose metrics are separate columns.',
        'Whole-record bidirectional inference is a noncausal diagnostic and never selects a formal model.',
        'Negative Gaussian log-covariance losses are allowed; velocity RMSE and input sensitivity determine usefulness.',
        '', '## Confirmation checks']
    for item in confirmations:
        lines.extend(['',item['candidate']['name']+': '+item['interpretation'],
                      'Passed: '+str(item['outcome']['passed']),
                      'Failed checks: '+', '.join(k for k,v in item['outcome']['checks'].items() if not v)])
    if not rows:lines.extend(['','No diagnostic training has been started. No curves or model results have been fabricated.'])
    (REPORT/'README.md').write_text('\n'.join(lines)+'\n',encoding='utf-8')


def comparison_figures():
    out=REPORT/'evaluation';provenance=read(out/'provenance.json')
    if not provenance['accepted']:raise RuntimeError('Evaluation not accepted')
    for name,sha in provenance['artifacts'].items():
        if c.sha(out/name)!=sha:raise RuntimeError('Evaluation artifact changed')
    bundle=c.ROOT/'ROV-ESKF-Odometry/reports/new_test1_4_odometry_all_methods'
    names=['ground_truth','DAIO','DEEPVL','AirIO','DAIO_withoutRPM']
    color=dict(zip(names,['#263746','#009E87','#AA4499','#0072B2','#D55E00']))
    labels=dict(zip(names,['Ground Truth (AprilTag)','DAIO + ESKF','DeepVL (Planar ROV)','AirIO (3D, causal adaptation)','DAIO without RPM + ESKF']))
    styles=dict(zip(names,['-','-','-.',':','--']))
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    from matplotlib.lines import Line2D
    from PIL import Image
    from figure_io import save_png
    plt.rcParams.update({'font.family':'sans-serif','font.sans-serif':['Arial','DejaVu Sans'],'font.size':22,
                         'axes.spines.top':False,'axes.spines.right':False,'axes.linewidth':1.3})
    destination=REPORT/'comparison_figures';destination.mkdir(parents=True,exist_ok=True)
    audited=[];errors={key:[] for key in ('velocity','position')};source_hashes={}
    def save(fig,name,ratio):
        fig.tight_layout(pad=.6);fig.canvas.draw();fig.tight_layout(pad=.6)
        path=destination/name;save_png(fig,path,1200);plt.close(fig)
        with Image.open(path) as im:
            size=im.size;dpi=im.info.get('dpi')
        if abs(size[0]/size[1]-ratio)>1e-8:raise RuntimeError('Figure aspect mismatch')
        if dpi is None or abs(dpi[0]-1200)>1:raise RuntimeError('Figure DPI mismatch')
        audited.append(dict(file=name,pixels=size,dpi=dpi,sha256=c.sha(path)))
    for number in range(1,5):
        source=bundle/f'test{number}.npz'
        # The export bundle uses stable names; resolve its actual data name if prefixed.
        if not source.exists():
            candidates=sorted(bundle.glob(f'test{number}_*.npz'))
            if len(candidates)!=1:raise RuntimeError('Ambiguous combined Test data: '+str(number))
            source=candidates[0]
        source_hashes[str(source)]=c.sha(source)
        with np.load(source) as z:old={k:z[k] for k in z.files}
        with np.load(out/f'test{number}_samples.npz') as z:new={k:z[k] for k in z.files}
        t=old.get('time_s',old.get('time'))
        np.testing.assert_allclose(t,new['time'],rtol=0,atol=1e-10)
        series={name:dict(position=old[name+'_position'],velocity=old[name+'_velocity']) for name in names if name!='AirIO'}
        series['AirIO']=dict(position=new['position'],velocity=new['velocity'])
        np.testing.assert_allclose(series['ground_truth']['position'],new['reference_position'][:,:2],rtol=0,atol=1e-10)
        # XY from reference is identical; historical velocity differentiation is checked explicitly.
        np.testing.assert_allclose(series['ground_truth']['velocity'],new['reference_velocity'][:,:2],rtol=0,atol=1e-8)
        display=np.unique(np.r_[np.arange(0,len(t),10),len(t)-1]);time=t-8
        fig,axes=plt.subplots(2,1,figsize=(8,4),sharex=True)
        for j,ax in enumerate(axes):
            for name in names:ax.plot(time[display],series[name]['velocity'][display,j],color=color[name],ls=styles[name],lw=1.8)
            ax.set_ylabel(r'${}^{W}v_'+('x' if j==0 else 'y')+r'$ [m/s]');ax.grid(axis='y',color='#dce4e9');ax.set_xlim(time[0],time[-1])
        axes[0].set_title(f'Test {number}',loc='left',fontweight='bold');axes[1].set_xlabel('Time(s)')
        save(fig,f'test{number}_velocity.png',2)
        fig,ax=plt.subplots(figsize=(8,4))
        for name in names:
            p=series[name]['position'];ax.plot(p[display,0],p[display,1],color=color[name],ls=styles[name],lw=1.8)
            ax.plot(*p[0],marker='o',mfc='white',mec=color[name],ms=7);ax.plot(*p[-1],marker='X',color=color[name],ms=8)
        ax.set_aspect('equal',adjustable='datalim');ax.set_xlabel(r'$X^W$ [m]');ax.set_ylabel(r'$Y^W$ [m]')
        ax.set_title(f'Test {number}',loc='left',fontweight='bold');ax.grid(color='#dce4e9')
        save(fig,f'test{number}_xy.png',2)
        for key in errors:errors[key].append({name:series[name][key]-series['ground_truth'][key] for name in names[1:]})
    for key,tests in errors.items():
        fig,axes=plt.subplots(1,3,figsize=(13.5,4.5));unit='m/s' if key=='velocity' else 'm'
        for component,ax in enumerate(axes):
            for n,values in enumerate(tests):
                for j,name in enumerate(names[1:]):
                    e=values[name];v=np.abs(e[:,component]) if component<2 else np.linalg.norm(e,axis=1)
                    b=ax.boxplot(v,positions=[n*5+j],widths=.65,patch_artist=True,
                        flierprops=dict(markersize=.6,alpha=.1),medianprops=dict(color='#263746'))
                    b['boxes'][0].set(facecolor=color[name],alpha=.7)
            ax.set_xticks(np.arange(4)*5+1.5,['T1','T2','T3','T4']);ax.set_ylim(bottom=0);ax.grid(axis='y',alpha=.2)
            ax.set_title(['|X error|','|Y error|','XY error norm'][component]);ax.set_ylabel(f'Error [{unit}]')
        save(fig,key+'_error_boxplot.png',3)
    fig,ax=plt.subplots(figsize=(12,3));ax.set_axis_off()
    handles=[Line2D([],[],color=color[n],ls=styles[n],lw=3,label=labels[n]) for n in names]
    handles.extend([Line2D([],[],marker='o',mfc='white',mec='#263746',ls='',label='Start'),
                    Line2D([],[],marker='X',color='#263746',ls='',label='End')])
    ax.legend(handles=handles,loc='center',ncol=2,frameon=False,fontsize=20)
    save(fig,'legend.png',4)
    if any(c.sha(path)!=sha for path,sha in source_hashes.items()):raise RuntimeError('Existing methods changed')
    write(destination/'verification.json',dict(figures=audited,unchanged_source_hashes=source_hashes,
        recovery_signature=provenance['signature'],existing_outputs_overwritten=False,time=now()))
