"""Three input ablations: identical test windows, standalone scientific PNG panels."""
from pathlib import Path
import sys,json,argparse,hashlib
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from figure_io import save_png

ROOT=Path(__file__).resolve().parents[1];DATA=ROOT/'new-dataset'
OUT=ROOT/'ROV-ESKF-Odometry/reports/new_v1v2/network_ablation'
MODELS=('horizontal_rpm','full_imu_rpm','horizontal_command')
COLOR={'truth':'#263746','horizontal_rpm':'#009E87','full_imu_rpm':'#0072B2','horizontal_command':'#D55E00'}
LABEL={'truth':'Ground Truth (AprilTag)','horizontal_rpm':'Horizontal IMU + RPM','full_imu_rpm':'Full IMU + RPM','horizontal_command':'Horizontal IMU + Command'}
STYLE={'horizontal_rpm':'-','full_imu_rpm':'--','horizontal_command':'-.'}

def main(dpi):
    OUT.mkdir(exist_ok=True,parents=True)
    plt.rcParams.update({'font.family':'DejaVu Sans','font.size':22,'axes.labelsize':24,'xtick.labelsize':20,'ytick.labelsize':20,
        'axes.spines.top':False,'axes.spines.right':False,'axes.linewidth':1.2,'lines.linewidth':2.0})
    status=json.loads((DATA/'published_runs.json').read_text(encoding='utf-8'));all_data=[];means={m:[] for m in MODELS};sources=[]
    def line_with_gaps(ax,time,value,indices,**style):
        x=np.asarray(time)[indices];y=np.asarray(value)[indices]
        # Gaps are determined at full 50 Hz before display sampling.
        missing=np.flatnonzero(np.diff(time)>.030)
        for k in missing[::-1]:
            at=int(np.searchsorted(indices,k+1))
            x=np.insert(x,at,np.nan);y=np.insert(y,at,np.nan)
        ax.plot(x,y,**style)
    def save(fig,name):
        fig.canvas.draw();bbox=fig.get_tightbbox(fig.canvas.get_renderer());w,h=fig.get_size_inches()
        if bbox.x0<-.01 or bbox.y0<-.01 or bbox.x1>w+.01 or bbox.y1>h+.01:raise RuntimeError('Clipped text: '+name)
        save_png(fig,OUT/name,dpi);plt.close(fig)
    def error_panel(ax,test,data,compact=False):
        time,truth,pred=data;time=time-time[0]
        errors={m:np.linalg.norm(pred[m][:,:2]-truth[:,:2],axis=1) for m in MODELS}
        indices=np.unique(np.concatenate((np.arange(0,len(time),10),np.array([len(time)-1]+[int(np.argmax(v)) for v in errors.values()]))))
        for m in MODELS:line_with_gaps(ax,time,errors[m],indices,color=COLOR[m],ls=STYLE[m])
        ax.set_xlim(0,time[-1]);ax.set_ylim(0,max(np.max(v) for v in errors.values())*1.001)
        ax.grid(axis='y',color='#dce4e9');ax.set_xlabel('Time(s)');ax.set_ylabel(r'$\|e_{v,xy}^{B}\|_2$ [m/s]')
        maximum=max(np.max(v) for v in errors.values());right=None
        for candidate in (.98,.73,.48):
            region=(time>=max(0,candidate-.25)*time[-1])&(time<=candidate*time[-1])
            if all(np.max(v[region],initial=0)<maximum*.78 for v in errors.values()):right=candidate;break
        if right is None:ax.set_title(f'Test {test}',loc='right',fontsize=22 if compact else 25,fontweight='bold',pad=3)
        else:ax.text(right,.95,f'Test {test}',ha='right',va='top',transform=ax.transAxes,fontsize=22 if compact else 25,fontweight='bold')
    for test,record in enumerate(('025','028','029','030'),1):
        arrays={}
        for m in MODELS:
            file=(ROOT / Path(status[m]['run_dir']))/'evaluation/demo'/f'{record}_demo__segment_001_predictions.npz'
            arrays[m]=np.load(file)
            sources.append({'model':m,'test':test,'file':str(file),'sha256':hashlib.sha256(file.read_bytes()).hexdigest()})
        reference=arrays[MODELS[0]]
        for m in MODELS:
            np.testing.assert_array_equal(reference['time'],arrays[m]['time'])
            np.testing.assert_array_equal(reference['target'],arrays[m]['target'])
            error=arrays[m]['prediction'][:,:2]-arrays[m]['target'][:,:2]
            means[m].append(float(np.sqrt(np.mean(np.sum(error**2,axis=1)))))
        data=(reference['time'],reference['target'],{m:arrays[m]['prediction'] for m in MODELS});all_data.append(data)
        fig,ax=plt.subplots(figsize=(8,4),layout='constrained');error_panel(ax,test,data);save(fig,f'test{test}_velocity_error.png')
        time=reference['time']-reference['time'][0];indices=np.unique(np.r_[np.arange(0,len(time),10),len(time)-1])
        fig,axes=plt.subplots(2,1,figsize=(8,4),sharex=True,layout='constrained')
        for component,ax in enumerate(axes):
            line_with_gaps(ax,time,reference['target'][:,component],indices,color=COLOR['truth'])
            for m in MODELS:line_with_gaps(ax,time,arrays[m]['prediction'][:,component],indices,color=COLOR[m],ls=STYLE[m])
            ax.set_ylabel(r'${}^{B}v_'+('x' if component==0 else 'y')+'$\n[m/s]');ax.grid(axis='y',color='#dce4e9');ax.set_xlim(0,time[-1]);ax.margins(y=.08)
        axes[0].set_title(f'Test {test}',loc='left',fontsize=25,fontweight='bold');axes[1].set_xlabel('Time(s)')
        save(fig,f'test{test}_body_velocity.png')
    fig,axes=plt.subplots(2,2,figsize=(12,6.5),layout='constrained')
    for test,(ax,data) in enumerate(zip(axes.flat,all_data),1):error_panel(ax,test,data,True)
    save(fig,'ablation_error_2x2.png')
    fig,ax=plt.subplots(figsize=(13.5,4.5),layout='constrained');x=np.arange(5)
    for i,m in enumerate(MODELS):ax.bar(x+(i-1)*.24,[*means[m],float(np.mean(means[m]))],width=.22,color=COLOR[m])
    ax.set_xticks(x,['Test 1','Test 2','Test 3','Test 4','Mean']);ax.set_ylabel('XY velocity RMSE [m/s]');ax.set_ylim(bottom=0);ax.set_axisbelow(True);ax.grid(axis='y',color='#dce4e9')
    save(fig,'ablation_velocity_rmse.png')
    for velocity in (False,True):
        keys=(('truth',)+MODELS) if velocity else MODELS
        handles=[Line2D([],[],color=COLOR[m],ls=STYLE.get(m,'-'),lw=3,label=LABEL[m]) for m in keys]
        fig=plt.figure(figsize=(12,2.4));fig.legend(handles=handles,loc='center',ncol=2,frameon=False,fontsize=22)
        save(fig,'legend_velocity.png' if velocity else 'legend_error.png')
    (OUT/'provenance.json').write_text(json.dumps({'dpi':dpi,'sources':sources,'identical_time_and_target_verified':True,
        'display':'5Hz plus global error maxima; no smoothing; missing predictions shown as gaps; all 50Hz samples used in metrics',
        'time_origin':'first valid network prediction (7.675 s in record)',
        'metric':'XY vector RMSE, arithmetic mean of four trajectories'},indent=2),encoding='utf-8')
    print('Saved 12 ablation PNGs:',OUT,flush=True)

if __name__=='__main__':
    parser=argparse.ArgumentParser();parser.add_argument('--dpi',type=int,default=1200);args=parser.parse_args();main(args.dpi)
