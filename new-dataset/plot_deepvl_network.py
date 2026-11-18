"""Plot direct DeepVL ensemble velocity, before any filter update or rotation."""
from pathlib import Path
import csv,json,sys,zipfile
import numpy as np
import torch
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from PIL import Image

DATA=Path(__file__).resolve().parent
sys.path.insert(0,str(DATA/'deepvl'))
import common as c
from figure_io import save_png

REPORT=DATA.parent/'ROV-ESKF-Odometry/reports/new_v1v2'
OUT=REPORT/'deepvl_network_velocity'
TRUTH='#263746';PREDICTION='#AA4499'


def save(fig,path):
    if fig.get_layout_engine() is not None:fig.set_constrained_layout_pads(w_pad=.09,h_pad=.05)
    fig.canvas.draw();fig.canvas.draw()
    b=fig.get_tightbbox(fig.canvas.get_renderer());w,h=fig.get_size_inches()
    if b.x0<-.01 or b.y0<-.01 or b.x1>w+.01 or b.y1>h+.01:
        raise RuntimeError(f'Figure extends outside canvas: {path.name}: {b.bounds}')
    save_png(fig,path,1200);plt.close(fig)


def main():
    OUT.mkdir(exist_ok=True);torch.set_num_threads(4)
    device='cuda' if torch.cuda.is_available() else 'cpu'
    checkpoint=c.HERE/'runs/best.pt';models,payload=c.load_ensemble(checkpoint,device)
    if payload['context']!=c.fingerprint():raise RuntimeError('Checkpoint source/data mismatch')
    replay_provenance=json.loads((REPORT/'deepvl_planar/provenance.json').read_text(encoding='utf-8'))
    if replay_provenance.get('network_weights_unchanged'):
        from evaluate_deepvl_fixed import runtime_signature
        signature=runtime_signature(checkpoint)
        if signature!=replay_provenance['method_signature']:raise RuntimeError('Repaired filter signature changed')
    else:signature=c.sha(checkpoint)+c.sha(c.HERE/'filter.py')+c.sha(c.HERE/'common.py')
    plt.rcParams.update({'font.family':'DejaVu Sans','font.size':22,'axes.labelsize':24,
        'xtick.labelsize':20,'ytick.labelsize':20,'axes.spines.top':False,'axes.spines.right':False,
        'axes.linewidth':1.2,'lines.linewidth':2.})
    figures=[];metrics=[];sources=[]
    for test,path in enumerate(c.splits()['demo'],1):
        # Keep online sensors separate from offline labels even for this plot.
        online=c.load_record(path,reference=False)
        ends,mean,variance=c.observations(online,models,payload['normalization'],device)
        cache=REPORT/'deepvl_planar'/f'test{test}_full_state.npz'
        with np.load(REPORT/'deepvl_planar'/f'test{test}_samples.npz') as samples:
            assert str(samples['checkpoint_sha256'])==c.sha(checkpoint)
            assert str(samples['source_sha256'])==c.sha(path)
            assert str(samples['method_signature'])==signature
        with np.load(cache) as old:
            np.testing.assert_allclose(old['observation_times'],online.time[ends],rtol=0,atol=1e-10)
            # These are stored pre-filter network observations, NOT fused state velocity.
            np.testing.assert_allclose(old['observation_mean'],mean,rtol=1e-5,atol=1e-7)
            np.testing.assert_allclose(old['observation_variance'],variance,rtol=2e-5,atol=1e-8)
        offline=c.load_record(path,reference=True);_,_,_,heading_truth=c.reference(offline)
        keep=ends>=1600;indices=ends[keep]
        t=online.time[indices]-8.;pred=mean[keep];truth=heading_truth[indices];var=variance[keep]
        e=pred-truth;row=dict(test=f'Test {test}',samples=len(t),start_record_s=float(online.time[indices[0]]),
            end_record_s=float(online.time[indices[-1]]),velocity_x_rmse_mps=float(np.sqrt(np.mean(e[:,0]**2))),
            velocity_y_rmse_mps=float(np.sqrt(np.mean(e[:,1]**2))),
            velocity_xy_vector_rmse_mps=float(np.sqrt(np.mean(np.sum(e**2,axis=1)))))
        metrics.append(row)
        np.savez_compressed(OUT/f'test{test}_network_predictions.npz',time_after_initialization=t,
             time_record=online.time[indices],prediction=pred,ground_truth=truth,variance=var,
             coordinate_frame='horizontal yaw-heading H',checkpoint_sha256=c.sha(checkpoint),source_sha256=c.sha(path))
        fig,axes=plt.subplots(2,1,figsize=(8,4),sharex=True,layout='constrained')
        for k,ax in enumerate(axes):
            ax.plot(t,truth[:,k],color=TRUTH,lw=1.7,zorder=2)
            ax.plot(t,pred[:,k],color=PREDICTION,lw=2.,ls='--',zorder=3)
            ax.set_ylabel(r'${}^{H}v_'+('x' if k==0 else 'y')+'$\n[m/s]')
            ax.set_xlim(0,float(online.time[-1]-8));ax.margins(y=.08)
            ax.grid(axis='y',color='#dce4e9');ax.set_axisbelow(True)
        axes[0].set_title(f'Test {test}',loc='left',fontsize=25,fontweight='bold')
        axes[1].set_xlabel('Time(s)')
        dest=OUT/f'0{test}_test{test}_network_velocity.png';save(fig,dest);figures.append(dest)
        sources.append(dict(test=test,data=str(path),data_sha256=c.sha(path),
            original_observations=str(cache),observations_sha256=c.sha(cache),recomputed_predictions_match=True))
        print(json.dumps(row),flush=True)
    mean=dict(test='Mean',samples='',start_record_s='',end_record_s='')
    for key in ('velocity_x_rmse_mps','velocity_y_rmse_mps','velocity_xy_vector_rmse_mps'):
        mean[key]=float(np.mean([m[key] for m in metrics]))
    metrics.append(mean)
    with (OUT/'network_velocity_metrics.csv').open('w',encoding='utf-8-sig',newline='') as f:
        w=csv.DictWriter(f,fieldnames=list(metrics[0]));w.writeheader();w.writerows(metrics)
    fig=plt.figure(figsize=(11,1.1))
    fig.legend(handles=[Line2D([],[],color=TRUTH,lw=3,label='Ground Truth (AprilTag)'),
        Line2D([],[],color=PREDICTION,lw=3,ls='--',label='DeepVL network prediction')],
        loc='center',ncol=2,frameon=False,fontsize=22,handlelength=2.5,columnspacing=1.5)
    dest=OUT/'05_legend.png';save(fig,dest);figures.append(dest)
    inventory=[];Image.MAX_IMAGE_PIXELS=None
    for p in figures:
        with Image.open(p) as im:
            assert abs(im.info['dpi'][0]-1200)<.1
            if p.name!='05_legend.png':assert im.size==(9600,4800)
            inventory.append(dict(file=p.name,pixels=im.size,dpi=im.info['dpi'],sha256=c.sha(p)))
    (OUT/'README.md').write_text('''# DeepVL direct network velocity, Test 1--4

These are the 8-member ensemble predictions from best update 1500, BEFORE filtering.
The outputs are horizontal yaw-heading H-frame vx,vy (not world-frame fused velocity).
Ground truth uses the identical SG101/poly3 AprilTag reference used in training/evaluation.
No estimated yaw or drifting filter state is used to transform these predictions.
The 20Hz predictions are re-inferred and checked against the original saved network observations.
Plots and network-only metrics use every native 20Hz prediction after the 8-second initialization.
Time(s)=record time-8; the first prediction is at +0.045s. No extra display smoothing.
Mean is the arithmetic mean of four trajectory RMSEs; these differ from 200Hz fused-odometry metrics.
PNG: 1200dpi, each test 9600x4800 (2:1). No in-panel legend; shared legend exported separately.
Reproduce: python -B new-dataset/plot_deepvl_network.py
''',encoding='utf-8')
    c.atomic_json(OUT/'provenance.json',dict(checkpoint=str(checkpoint),checkpoint_sha256=c.sha(checkpoint),
        checkpoint_update=payload['update'],frame='horizontal yaw-heading H',prediction_type='raw 8-network ensemble mean, before filtering',
        script_sha256=c.sha(Path(__file__)),scoring_hz=20,sources=sources,figures=inventory))
    with zipfile.ZipFile(OUT/'deepvl_network_velocity_figures.zip','w',zipfile.ZIP_DEFLATED) as z:
        for p in figures+[OUT/'README.md',OUT/'network_velocity_metrics.csv',OUT/'provenance.json']:z.write(p,p.name)
    print('Exported 4 network panels and separate legend:',OUT,flush=True)


if __name__=='__main__':main()
