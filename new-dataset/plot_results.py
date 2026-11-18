"""Publication PNG panels, with all legends exported separately."""
from pathlib import Path
import sys,argparse
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from matplotlib.patches import Patch
from figure_io import save_png
ROOT=Path(__file__).resolve().parents[1];DATA=ROOT/'new-dataset'
sys.path[:0]=[str(DATA/'baselines')]
import planar_common as pc

OUT=ROOT/'ROV-ESKF-Odometry/reports/new_v1v2'
METHODS=('horizontal_rpm','horizontal_command','airio_planar','deepvl_planar')
COLORS={'truth':'#263746','horizontal_rpm':'#009E87','horizontal_command':'#D55E00','airio_planar':'#0072B2','deepvl_planar':'#AA4499'}
LABELS={'truth':'Ground Truth (AprilTag)','horizontal_rpm':'DAIO (RPM) + ESKF','horizontal_command':'DAIO (Command) + ESKF','airio_planar':'AirIO (Planar ROV)','deepvl_planar':'DeepVL (Planar ROV)'}
STYLES={'horizontal_rpm':'-','horizontal_command':'--','airio_planar':'-.','deepvl_planar':':'}
COLORS['airio_3d']='#0072B2'
LABELS['airio_3d']='AirIO (3D ROV)'
STYLES['airio_3d']='-.'

def save(fig,name):
    if fig.get_layout_engine() is not None:
        fig.set_constrained_layout_pads(w_pad=.12,h_pad=.06)
    # Equal-aspect datalim axes and scientific tick offsets need a second
    # layout pass to settle. Preserve the requested canvas/aspect and DPI.
    fig.canvas.draw()
    fig.canvas.draw()
    bounds=fig.get_tightbbox(fig.canvas.get_renderer())
    width,height=fig.get_size_inches()
    if bounds.x0 < -.01 or bounds.y0 < -.01 or bounds.x1>width+.01 or bounds.y1>height+.01:
        raise RuntimeError(f'Figure content outside canvas: {name}: {bounds.bounds}')
    save_png(fig,OUT/name,1200);plt.close(fig)

def main():
    plt.rcParams.update({'font.family':'DejaVu Sans','font.size':22,'axes.labelsize':24,'xtick.labelsize':20,'ytick.labelsize':20,
        'axes.spines.top':False,'axes.spines.right':False,'axes.linewidth':1.2,'lines.linewidth':2.,'savefig.dpi':1200})
    paths=pc.resolve_manifest(DATA/'processed/split.yaml',DATA)['demo'];err={'velocity':{},'position':{}}
    for test,path in enumerate(paths,1):
        seq=pc.load_sequence(path);xy,yaw,velocity,_=pc.reference(seq);time=seq.imu_ts[1600:]-8
        xy=xy[1600:];velocity=velocity[1600:]
        values={m:np.load(OUT/m/f'test{test}_samples.npz') for m in METHODS}
        for m in METHODS:
            np.testing.assert_allclose(values[m]['time'],seq.imu_ts[1600:],atol=1e-9,rtol=0)
        display=np.unique(np.r_[np.arange(0,len(time),10),len(time)-1])
        fig,axes=plt.subplots(2,1,figsize=(8,4),sharex=True,layout='constrained')
        for component,ax in enumerate(axes):
            ax.plot(time[display],velocity[display,component],color=COLORS['truth'],zorder=2)
            for m in METHODS:ax.plot(time[display],values[m]['velocity'][display,component],color=COLORS[m],ls=STYLES[m],zorder=3)
            ax.set_ylabel(r'${}^{W}v_'+('x' if component==0 else 'y')+'$\n[m/s]')
            ax.grid(axis='y',color='#dce4e9');ax.set_xlim(time[0],time[-1]);ax.margins(y=.08)
        axes[0].set_title(f'Test {test}',loc='left',fontsize=25,fontweight='bold');axes[-1].set_xlabel('Time(s)')
        save(fig,f'test{test}_world_velocity.png')
        fig,ax=plt.subplots(figsize=(8,4),layout='constrained')
        ax.plot(xy[display,0],xy[display,1],color=COLORS['truth'])
        for m in METHODS:ax.plot(values[m]['position'][display,0],values[m]['position'][display,1],color=COLORS[m],ls=STYLES[m])
        for x,c in [(xy,COLORS['truth'])]+[(values[m]['position'],COLORS[m]) for m in METHODS]:
            ax.plot(*x[0],marker='o',ms=9,mfc='white',mec=c,mew=2);ax.plot(*x[-1],marker='X',ms=10,color=c)
        ax.set_aspect('equal',adjustable='box');ax.set_xlabel(r'$X^W$ [m]');ax.set_ylabel(r'$Y^W$ [m]')
        ax.set_title(f'Test {test}',loc='left',fontsize=25,fontweight='bold');ax.grid(color='#dce4e9')
        save(fig,f'test{test}_xy_trajectory.png')
        err['velocity'][test]={m:np.linalg.norm(values[m]['velocity']-velocity,axis=1) for m in METHODS}
        err['position'][test]={m:np.linalg.norm(values[m]['position']-xy,axis=1) for m in METHODS}
    for kind in err:
        fig,ax=plt.subplots(figsize=(13.5,4.5),layout='constrained')
        for t in range(1,5):
            for j,m in enumerate(METHODS):
                x=t*4+(j-(len(METHODS)-1)/2)*.9
                result=ax.boxplot(err[kind][t][m],positions=[x],widths=.65,patch_artist=True,manage_ticks=False,
                    flierprops={'markersize':1.2,'alpha':.18,'markeredgecolor':COLORS[m]},medianprops={'color':'#263746','linewidth':2})
                result['boxes'][0].set(facecolor=COLORS[m],alpha=.70)
        ax.set_xticks([4,8,12,16],[f'Test {i}' for i in range(1,5)]);ax.set_xlim(2,18)
        ax.set_ylabel(r'$\|e_v^{XY}\|_2$ [m/s]' if kind=='velocity' else r'$\|e_p^{XY}\|_2$ [m]')
        ax.set_ylim(bottom=0);ax.grid(axis='y',color='#dce4e9');ax.set_axisbelow(True)
        save(fig,kind+'_error_boxplot.png')
    handles=[Line2D([],[],color=COLORS['truth'],lw=3,label=LABELS['truth'])]
    handles += [Line2D([],[],color=COLORS[m],ls=STYLES[m],lw=3,label=LABELS[m]) for m in METHODS]
    handles += [Line2D([],[],marker='o',mfc='white',mec='#263746',ls='',ms=9,label='Start'),Line2D([],[],marker='X',color='#263746',ls='',ms=9,label='End')]
    fig=plt.figure(figsize=(12,2.4));fig.legend(handles=handles,loc='center',ncol=2,frameon=False,fontsize=22);save(fig,'legend.png')
    print('Exported 11 publication PNGs at 1200 dpi',flush=True)
    if (OUT/'metrics.csv').exists():
        import write_eval_report
        if write_eval_report.OUT.resolve()==OUT.resolve():
            write_eval_report.main()
            import validate_evaluation
            validate_evaluation.main()

if __name__=='__main__':
    parser=argparse.ArgumentParser()
    parser.add_argument('--methods',nargs='+',choices=[*METHODS,'airio_3d'],default=['horizontal_rpm','horizontal_command'])
    args=parser.parse_args();METHODS=tuple(args.methods);main()
