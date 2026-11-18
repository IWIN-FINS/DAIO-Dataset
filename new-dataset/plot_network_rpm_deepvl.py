"""Native pre-filter RPM ablations and DeepVL; no training/evaluation mutation."""
from pathlib import Path
import csv,hashlib,json,sys,zipfile
import numpy as np
from scipy.signal import savgol_filter
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from PIL import Image,ImageOps,ImageDraw
from figure_io import save_png

DATA=Path(__file__).resolve().parent;ROOT=DATA.parent
REPORT=ROOT/'ROV-ESKF-Odometry/reports/new_v1v2'
OUT=REPORT/'network_rpm_deepvl_velocity'
MODELS=('horizontal_rpm','full_imu_rpm','deepvl')
COLORS={'truth':'#263746','horizontal_rpm':'#009E87','full_imu_rpm':'#0072B2','deepvl':'#AA4499'}
LABELS={'truth':'Ground Truth (AprilTag)','horizontal_rpm':'Horizontal IMU + RPM',
        'full_imu_rpm':'Full IMU + RPM','deepvl':'DeepVL (network)'}
STYLES={'truth':'-','horizontal_rpm':'-','full_imu_rpm':'--','deepvl':'-.'}


def sha(path):return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def reference(path):
    with path.open(encoding='utf-8-sig') as f:header=next(csv.reader(f))
    names=('sample_time_s','tag_x_m','tag_y_m','tag_yaw_rad')
    a=np.loadtxt(path,delimiter=',',skiprows=1,usecols=[header.index(k) for k in names],encoding='utf-8-sig')
    t=a[:,0];xy=savgol_filter(a[:,1:3],101,3,axis=0);yaw=savgol_filter(np.unwrap(a[:,3]),101,3)
    world=np.gradient(xy,t,axis=0,edge_order=1);c,s=np.cos(yaw),np.sin(yaw)
    return t,np.c_[c*world[:,0]+s*world[:,1],-s*world[:,0]+c*world[:,1]]


def sampled_line(ax,t,y,period,**kwargs):
    # Determine real gaps before thinning; never join a missing prediction span.
    cuts=np.r_[0,np.flatnonzero(np.diff(t)>period*1.5)+1,len(t)]
    for lo,hi in zip(cuts[:-1],cuts[1:]):
        x=t[lo:hi];v=y[lo:hi]
        if not len(x):continue
        stride=max(1,round(.1/period))
        display=np.unique(np.r_[np.arange(0,len(x),stride),len(x)-1,np.argmin(v),np.argmax(v)])
        ax.plot(x[display]-8.,v[display],**kwargs)


def save(fig,name):
    fig.canvas.draw();fig.canvas.draw()
    bounds=fig.get_tightbbox(fig.canvas.get_renderer());w,h=fig.get_size_inches()
    if bounds.x0<-.01 or bounds.y0<-.01 or bounds.x1>w+.01 or bounds.y1>h+.01:
        raise RuntimeError('Text exceeds canvas: '+name)
    save_png(fig,OUT/name,1200);plt.close(fig)
    print('Saved',name,flush=True)


def main():
    OUT.mkdir(parents=True,exist_ok=True)
    plt.rcParams.update({'font.family':'DejaVu Sans','font.size':22,'axes.labelsize':24,
        'xtick.labelsize':20,'ytick.labelsize':20,'axes.spines.top':False,'axes.spines.right':False,
        'axes.linewidth':1.2,'lines.linewidth':2.0})
    status=json.loads((DATA/'published_runs.json').read_text(encoding='utf-8'))
    original_metrics=list(csv.DictReader((REPORT/'network_ablation_metrics.csv').open(encoding='utf-8-sig')))
    deep_metrics=list(csv.DictReader((REPORT/'deepvl_network_velocity/network_velocity_metrics.csv').open(encoding='utf-8-sig')))
    deep_provenance=json.loads((REPORT/'deepvl_network_velocity/provenance.json').read_text(encoding='utf-8'))
    assert sha(ROOT/deep_provenance['checkpoint'])==deep_provenance['release_sha256']
    sources=[];metric_rows=[];figures=[]
    for number,record in enumerate(('025','028','029','030'),1):
        source=DATA/'processed/v2'/f'{record}_demo__segment_001.csv'
        time,truth=reference(source);values={}
        for model in MODELS:
            file=(REPORT/'deepvl_network_velocity'/f'test{number}_network_predictions.npz') if model=='deepvl' else (
                (ROOT / Path(status[model]['run_dir']))/'evaluation/demo'/f'{record}_demo__segment_001_predictions.npz')
            with np.load(file) as z:
                t=z['time_record'].copy() if model=='deepvl' else z['time'].copy()
                pred=z['prediction'][:,:2].copy();target=z['ground_truth'].copy() if model=='deepvl' else z['target'][:,:2].copy()
                if model=='deepvl':
                    assert str(z['checkpoint_sha256'])==deep_provenance['checkpoint_sha256']
                    assert str(z['source_sha256'])==sha(source)
            ref=np.c_[np.interp(t,time,truth[:,0]),np.interp(t,time,truth[:,1])]
            np.testing.assert_allclose(target,ref,atol=3e-6,rtol=1e-5)
            e=pred-target;rmse=np.sqrt(np.mean(e*e,axis=0));vector=np.sqrt(np.mean(np.sum(e*e,axis=1)))
            expected=next(r for r in (deep_metrics if model=='deepvl' else original_metrics)
                          if r['test']==f'Test {number}' and (model=='deepvl' or r['model']==model))
            keys=('velocity_x_rmse_mps','velocity_y_rmse_mps','velocity_xy_vector_rmse_mps') if model=='deepvl' else ('vx_rmse_mps','vy_rmse_mps','xy_vector_rmse_mps')
            np.testing.assert_allclose([*rmse,vector],[float(expected[k]) for k in keys],atol=2e-8,rtol=2e-6)
            metric_rows.append(dict(test=f'Test {number}',model=model,native_hz=20 if model=='deepvl' else 50,
                native_samples=len(t),native_start_s=float(t[0]),native_end_s=float(t[-1]),vx_rmse_mps=float(rmse[0]),
                vy_rmse_mps=float(rmse[1]),xy_vector_rmse_mps=float(vector)))
            visible=t>=8.;values[model]=(t[visible],pred[visible])
            sources.append(dict(test=number,model=model,predictions=str(file),sha256=sha(file),reference=str(source),reference_sha256=sha(source)))
        fig,axes=plt.subplots(2,1,sharex=True,figsize=(8,4),layout='constrained')
        fig.set_constrained_layout_pads(w_pad=.10,h_pad=.04)
        keep=time>=8.
        for k,ax in enumerate(axes):
            sampled_line(ax,time[keep],truth[keep,k],.005,color=COLORS['truth'],ls=STYLES['truth'],lw=1.8,zorder=4)
            for model in MODELS:
                t,p=values[model]
                sampled_line(ax,t,p[:,k],.05 if model=='deepvl' else .02,color=COLORS[model],ls=STYLES[model],lw=2.0,zorder=3)
            ax.set_ylabel(r'${}^{H}v_'+('x' if k==0 else 'y')+'$\n[m/s]')
            ax.set_xlim(0,time[-1]-8.);ax.margins(y=.10)
            ax.grid(axis='y',color='#dce4e9');ax.set_axisbelow(True)
        axes[0].set_title(f'Test {number}',loc='left',fontsize=25,fontweight='bold')
        axes[-1].set_xlabel('Time(s)')
        name=f'0{number}_test{number}_network_velocity.png';save(fig,name);figures.append(OUT/name)
    handles=[Line2D([],[],color=COLORS[k],ls=STYLES[k],lw=3,label=LABELS[k]) for k in ('truth',*MODELS)]
    fig=plt.figure(figsize=(12,1.6));fig.legend(handles=handles,loc='center',ncol=2,frameon=False,
        fontsize=22,handlelength=3.,columnspacing=2.,labelspacing=.65)
    name='05_legend.png';save(fig,name);figures.append(OUT/name)
    for model in MODELS:
        rows=[r for r in metric_rows if r['model']==model]
        metric_rows.append(dict(test='Mean',model=model,**{k:float(np.mean([r[k] for r in rows])) for k in ('vx_rmse_mps','vy_rmse_mps','xy_vector_rmse_mps')}))
    with (OUT/'native_network_metrics.csv').open('w',encoding='utf-8-sig',newline='') as f:
        writer=csv.DictWriter(f,fieldnames=list(metric_rows[0]));writer.writeheader();writer.writerows(metric_rows)
    (OUT/'README.md').write_text('''# RPM ablations and DeepVL: direct network velocities

Four independent Test 1--4 panels and one shared legend, PNG1200dpi.
Each test panel is9600x4800 (2:1), large fonts. No inset legends.
Outputs are horizontal heading-frame velocities, before ANY odometry fusion.
Time(s) = record time minus8; all panels show the common post-initialization interval.
Continuous reference is SG101/poly3 AprilTag XY/yaw, differentiated and rotated by reference yaw.
Both RPM models use depth; the baseline uses ax,ay,wz; the full model uses all six IMU channels.
DeepVL uses the original frozen eight-GRU ensemble, six-axis IMU, commands and voltage; no depth/RPM.
All inputs/outputs and original weights remain unchanged. DeepVL is an additional system/network baseline,
not a controlled one-input ablation. It has no yaw-rate output.
Missing RPM predictions are left as gaps, including the recorded saturation-affected interval.
Display thinning targets10Hz per curve, retaining each continuous segment's extrema/endpoints;
no added smoothing, interpolation of network predictions or missing-span filling is performed.
native_network_metrics.csv reproduces the existing table scores, using ALL ORIGINAL native samples:
RPM valid windows50Hz from7.675s, DeepVL20Hz after8s. Display cropping does not redefine these scores.
Mean is the arithmetic mean of four trajectory RMSEs. This table has different native scoring protocols.
Reproduce from project root: python -B new-dataset/plot_network_rpm_deepvl.py
''',encoding='utf-8')
    Image.MAX_IMAGE_PIXELS=None;inventory=[];thumbnails=[]
    for file in figures:
        with Image.open(file) as im:
            im.verify()
        with Image.open(file) as im:
            dpi=im.info.get('dpi');assert dpi and abs(dpi[0]-1200)<.1
            if file.name!='05_legend.png':
                assert im.size==(9600,4800);thumbnails.append(im.convert('RGB').resize((1200,600),Image.Resampling.LANCZOS))
            inventory.append(dict(file=file.name,pixels=im.size,dpi=dpi,sha256=sha(file)))
    preview=Image.new('RGB',(2400,1400),'white')
    for i,thumb in enumerate(thumbnails):preview.paste(thumb,((i%2)*1200,(i//2)*600))
    with Image.open(figures[-1]) as im:preview.paste(im.convert('RGB').resize((1500,200),Image.Resampling.LANCZOS),(450,1200))
    preview.save(OUT/'preview.png')
    (OUT/'provenance.json').write_text(json.dumps(dict(script_sha256=sha(__file__),sources=sources,figures=inventory,
        frame='horizontal yaw-heading H',time_origin_s=8.,network_only=True,display_hz=10,all_native_metrics_match=True),indent=2),encoding='utf-8')
    with zipfile.ZipFile(OUT/'network_velocity_test1_4.zip','w',zipfile.ZIP_DEFLATED) as z:
        for f in [*figures,OUT/'native_network_metrics.csv',OUT/'README.md',OUT/'provenance.json',Path(__file__)]:z.write(f,f.name)
    print('Verified and packaged',OUT,flush=True)


if __name__=='__main__':main()
