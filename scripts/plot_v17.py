"""Draw the complete four-metric V17 ablation summary from released results."""
from pathlib import Path
import argparse
import json
import sys

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib import font_manager


def plot(output):
    output=Path(output);output.mkdir(parents=True,exist_ok=True)
    for p in Path('/usr/share/fonts/truetype/msttcorefonts').glob('*.ttf'):
        font_manager.fontManager.addfont(str(p))
    fonts={f.name for f in font_manager.fontManager.ttflist}
    plt.rcParams.update({'font.family':'Arial' if 'Arial' in fonts else 'DejaVu Sans',
                         'font.size':10,'pdf.fonttype':42,'ps.fonttype':42,
                         'axes.spines.top':False,'axes.spines.right':False})
    groups=json.loads((ROOT/'configs/ablation_groups.json').read_text())
    labels=json.loads((ROOT/'configs/labels.json').read_text())
    arms=[a for _,aa in groups for a in aa]
    summary=pd.read_csv(ROOT/'results/v17/reference/tiger_summary.csv')
    means=summary.pivot(index='arm',columns='metric',values='mean').loc[arms]
    sd=summary.pivot(index='arm',columns='metric',values='sd').loc[arms]
    colors=['#F25C54','#45B8D8','#30B8A0','#F5A742','#9575CD','#EF84AE',
            '#5B9BD5','#60B98C','#C3A037','#899ACC','#DB8F55','#A7B43D','#E57A86']
    palette=[c for c,(_,aa) in zip(colors,groups) for _ in aa]
    fig,axes=plt.subplots(1,4,figsize=(15,13.8),sharey=True)
    y=np.arange(len(arms))
    for ax,m,title in zip(axes,['PCC','KCC','log10MAE','RSE'],
                           ['Family-macro Pearson r','Family-macro Kendall τb','Family-macro log10MAE','Family-macro RSE']):
        for i,a in enumerate(arms):
            ax.errorbar(means.loc[a,m],i,xerr=sd.loc[a,m],fmt='o',color=palette[i],
                        markersize=6.5,elinewidth=1.4,capsize=2.5)
        ax.axvline(means.loc['factor_P11',m],color='#F25C54',ls='--',lw=1,zorder=0)
        position=0
        for _,aa in groups:
            position+=len(aa);ax.axhline(position-.5,color='#E4E8EC',lw=.6,zorder=0)
        ax.set_title(title,pad=12);ax.grid(axis='x',color='#E9ECEF',lw=.6)
        ax.set_axisbelow(True);ax.tick_params(axis='y',length=0)
    axes[0].set_yticks(y,[labels[a] for a in arms]);axes[0].invert_yaxis()
    fig.subplots_adjust(left=.245,right=.985,top=.95,bottom=.07,wspace=.2)
    fig.text(.245,.026,'Mean ± sample SD across seeds 120–129. Dashed line: TIGER-pair (unsigned quota 20,000).',fontsize=10)
    for ext in ['pdf','png']:fig.savefig(output/f'ablation_four_metrics.{ext}',dpi=220)
    plt.close(fig)
    return output/'ablation_four_metrics.png'

if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--output',type=Path,default=ROOT/'outputs/figures')
    print(plot(p.parse_args().output))
