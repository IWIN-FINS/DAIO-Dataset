"""Separate publication panels; no in-panel legends or metric display smoothing."""
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from PIL import Image
import common as c
from figure_io import save_png
COLOR={'truth':'#263746','daio':'#009E87','no_rpm':'#D55E00'}
STYLE={'truth':'-','daio':'-','no_rpm':'--'}
LABEL={'truth':'Ground Truth (AprilTag)','daio':'DAIO + ESKF','no_rpm':'DAIO without RPM + ESKF'}

def save(fig,path):
    fig.canvas.draw();fig.canvas.draw();b=fig.get_tightbbox(fig.canvas.get_renderer());w,h=fig.get_size_inches()
    if b.x0<-.01 or b.y0<-.01 or b.x1>w+.01 or b.y1>h+.01:raise RuntimeError('Clipped text '+str(path))
    save_png(fig,path,1200);plt.close(fig);print('PNG',path,flush=True)

def line(ax,time,value,**style):
    cuts=np.r_[0,np.flatnonzero(np.diff(time)>.03)+1,len(time)]
    for a,b in zip(cuts[:-1],cuts[1:]):
        i=np.unique(np.r_[np.arange(a,b,10),b-1]);ax.plot(time[i],value[i],**style)

def velocity(values,path,title,frame,origin):
    fig,axes=plt.subplots(2,1,figsize=(8,4),sharex=True,layout='constrained');fig.set_constrained_layout_pads(w_pad=.1,h_pad=.05)
    for k,ax in enumerate(axes):
        for name,(time,y) in values.items():line(ax,time-origin,y[:,k],color=COLOR[name],ls=STYLE[name],lw=1.7 if name=='truth' else 2.)
        ax.set_ylabel(r'${}^{'+frame+r'}v_'+('x' if k==0 else 'y')+'$\n[m/s]');ax.grid(axis='y',color='#dce4e9')
        ax.set_xlim(0,max(t[-1] for t,_ in values.values())-origin);ax.margins(y=.08)
    axes[0].set_title(title,loc='left',fontweight='bold',fontsize=25);axes[-1].set_xlabel('Time(s)');save(fig,path)

def main():
    plt.rcParams.update({'font.family':'DejaVu Sans','font.size':22,'axes.labelsize':24,'xtick.labelsize':20,'ytick.labelsize':20,
        'axes.spines.top':False,'axes.spines.right':False,'axes.linewidth':1.2})
    for domain in ('old','new'):
        folder=c.REPORT/'figures/network'/domain;folder.mkdir(parents=True,exist_ok=True)
        for number in range(1,5):
            values={}
            for name in ('daio','no_rpm'):
                with np.load(c.REPORT/'network'/domain/name/f'test{number}.npz') as z:
                    if not values:values['truth']=(z['time'],z['target'][:,:2]);first=z['time'].copy();gt=z['target'].copy()
                    else:np.testing.assert_array_equal(first,z['time']);np.testing.assert_array_equal(gt,z['target'])
                    values[name]=(z['time'],z['prediction'][:,:2])
            velocity(values,folder/f'test{number}_velocity.png',f'{domain.title()} Test {number}','H',float(first[0]))
    folder=c.REPORT/'figures/odometry';folder.mkdir(parents=True,exist_ok=True);errors={'velocity':{},'position':{}}
    for number in range(1,5):
        values={};positions={}
        for name in ('daio','no_rpm'):
            with np.load(c.REPORT/'odometry'/name/f'test{number}/samples.npz') as z:
                if not values:
                    values['truth']=(z['time'],z['reference_velocity']);positions['truth']=z['reference_position'];first=z['time'].copy();refv=z['reference_velocity'].copy();refp=z['reference_position'].copy()
                else:np.testing.assert_array_equal(first,z['time'])
                values[name]=(z['time'],z['velocity']);positions[name]=z['position']
                errors['velocity'][number,name]=np.linalg.norm(z['velocity']-refv,axis=1)
                errors['position'][number,name]=np.linalg.norm(z['position']-refp,axis=1)
        velocity(values,folder/f'test{number}_world_velocity.png',f'Test {number}','W',8.)
        fig,ax=plt.subplots(figsize=(8,4),layout='constrained');fig.set_constrained_layout_pads(w_pad=.1,h_pad=.05)
        for name,p in positions.items():
            i=np.unique(np.r_[np.arange(0,len(p),10),len(p)-1]);ax.plot(p[i,0],p[i,1],color=COLOR[name],ls=STYLE[name],lw=2.)
            ax.plot(*p[0],marker='o',ms=8,mfc='white',mec=COLOR[name]);ax.plot(*p[-1],marker='X',ms=9,color=COLOR[name])
        ax.set_aspect('equal',adjustable='box');ax.set_xlabel(r'$X^W$ [m]');ax.set_ylabel(r'$Y^W$ [m]');ax.grid(color='#dce4e9')
        ax.set_title(f'Test {number}',loc='left',fontsize=25,fontweight='bold');save(fig,folder/f'test{number}_xy.png')
    for kind in errors:
        fig,ax=plt.subplots(figsize=(13.5,4.5),layout='constrained')
        for number in range(1,5):
            for j,name in enumerate(('daio','no_rpm')):
                b=ax.boxplot(errors[kind][number,name],positions=[number*3+(j-.5)*.8],widths=.65,patch_artist=True,manage_ticks=False,
                    flierprops=dict(markersize=1.2,alpha=.18,markeredgecolor=COLOR[name]),medianprops=dict(color='#263746',linewidth=2))
                b['boxes'][0].set(facecolor=COLOR[name],alpha=.7)
        ax.set_xticks([3,6,9,12],[f'Test {n}' for n in range(1,5)]);ax.set_xlim(1.5,13.5);ax.set_ylim(bottom=0);ax.grid(axis='y',color='#dce4e9')
        ax.set_ylabel('XY velocity error [m/s]' if kind=='velocity' else 'XY position error [m]');save(fig,folder/f'{kind}_error_boxplot.png')
    for network in (False,True):
        handles=[Line2D([],[],color=COLOR[k],ls=STYLE[k],lw=3,label=(LABEL[k].replace(' + ESKF','') if network else LABEL[k])) for k in COLOR]
        if not network:handles += [Line2D([],[],marker='o',mfc='white',mec=COLOR['truth'],ls='',label='Start'),Line2D([],[],marker='X',color=COLOR['truth'],ls='',label='End')]
        fig=plt.figure(figsize=(13,2.2));fig.legend(handles=handles,loc='center',ncol=2,fontsize=22,frameon=False)
        save(fig,c.REPORT/'figures'/('network_legend.png' if network else 'odometry_legend.png'))
    Image.MAX_IMAGE_PIXELS=None;inventory=[]
    for path in (c.REPORT/'figures').rglob('*.png'):
        with Image.open(path) as im:
            assert abs(im.info['dpi'][0]-1200)<.1
            if 'boxplot' in path.name:assert im.size==(16200,5400)
            elif 'legend' not in path.name:assert im.size==(9600,4800)
            inventory.append(dict(path=str(path),pixels=im.size,sha256=c.sha(path)))
    assert len(inventory)==20
    c.atomic_json(c.REPORT/'figure_validation.json',dict(ok=True,figures=inventory))

if __name__=='__main__':main()
