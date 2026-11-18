"""Deliver current full-range velocity figures in Test 1--4 / boxes / legend order."""
from pathlib import Path
import hashlib,json,shutil,zipfile
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from PIL import Image
from plot_results import COLORS,LABELS,STYLES
from figure_io import save_png

ROOT=Path(__file__).resolve().parents[1]
REPORT=ROOT/'ROV-ESKF-Odometry/reports/new_v1v2'
OUT=REPORT/'velocity_figures'


def sha(path):return hashlib.sha256(path.read_bytes()).hexdigest()


def main():
    OUT.mkdir(exist_ok=True)
    validation=json.loads((REPORT/'artifact_validation.json').read_text(encoding='utf-8'))
    checked={x['file'].replace('\\','/'):x['sha256'] for x in validation['pngs']}
    files=[]
    sources=[(f'test{i}_world_velocity.png',f'0{i}_test{i}_world_velocity.png') for i in range(1,5)]
    sources += [('velocity_error_boxplot.png','05_velocity_error_boxplot.png'),
                ('position_error_boxplot.png','06_position_error_boxplot.png')]
    for name,target in sources:
        source=REPORT/name
        if sha(source)!=checked[name]:raise RuntimeError('Source PNG changed after independent validation: '+name)
        shutil.copy2(source,OUT/target);files.append(OUT/target)
    methods=json.loads((REPORT/'provenance.json').read_text(encoding='utf-8'))['evaluated_methods']
    labels=dict(LABELS)
    diagnostics=json.loads((REPORT/'deepvl_planar/filter_diagnostics.json').read_text(encoding='utf-8'))
    recovered=all(x['status']=='stable_after_train_sensor_calibration' for x in diagnostics['findings'])
    if not recovered:labels['deepvl_planar']='DeepVL (Planar ROV; diverged)'
    handles=[Line2D([],[],color=COLORS['truth'],lw=3,label=labels['truth'])]
    handles += [Line2D([],[],color=COLORS[m],ls=STYLES[m],lw=3,label=labels[m]) for m in methods]
    plt.rcParams.update({'font.family':'DejaVu Sans','font.size':22})
    fig=plt.figure(figsize=(14,2.6))
    fig.legend(handles=handles,loc='center',ncol=2,frameon=False,fontsize=22,
               handlelength=2.8,columnspacing=2.,labelspacing=.8)
    fig.canvas.draw();b=fig.get_tightbbox(fig.canvas.get_renderer());w,h=fig.get_size_inches()
    if b.x0<0 or b.y0<0 or b.x1>w or b.y1>h:raise RuntimeError('Legend extends outside canvas')
    legend=OUT/'07_legend.png';save_png(fig,legend,1200);plt.close(fig);files.append(legend)
    Image.MAX_IMAGE_PIXELS=None;inventory=[]
    for path in files:
        with Image.open(path) as im:
            dpi=im.info.get('dpi');assert dpi and abs(dpi[0]-1200)<.1
            if 'world_velocity' in path.name:assert im.width==2*im.height
            if 'boxplot' in path.name:assert im.width==3*im.height
            inventory.append(dict(file=path.name,size=im.size,dpi=dpi,sha256=sha(path)))
    (OUT/'README.md').write_text('''# Current odometry velocity figures

Order: Test 1--4 world-frame vx/vy; velocity error boxplot; position error boxplot; shared legend.
PNG at 1200 dpi. Velocity panels are 2:1, boxplots 3:1. Time(s) starts after the 8-second initialization.
Errors are XY vector norms from every scored 200Hz sample. Lines use display sampling only.
The six panels are byte-identical to the independently validated current report PNGs.
DeepVL uses the current validated replay; calibration/status are documented in the parent report.
Full scales and all error samples are retained.
The figures contain no legends. The shared legend omits XY start/end markers because no XY trajectories are delivered here.
Reproduce: python -B new-dataset/export_velocity_figures.py
''',encoding='utf-8')
    (OUT/'manifest.json').write_text(json.dumps(dict(methods=methods,metrics_sha256=sha(REPORT/'metrics.csv'),
          provenance_sha256=sha(REPORT/'provenance.json'),figures=inventory),indent=2),encoding='utf-8')
    with zipfile.ZipFile(OUT/'odometry_velocity_figures.zip','w',zipfile.ZIP_DEFLATED) as z:
        for p in files+[OUT/'README.md',OUT/'manifest.json']:z.write(p,p.name)
    print('Exported 7 validated 1200-dpi PNGs:',OUT,flush=True)


if __name__=='__main__':main()
