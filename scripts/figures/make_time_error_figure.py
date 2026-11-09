#!/usr/bin/env python3
# Run from the article directory; supply the original final_results.csv.
from pathlib import Path
import sys
import pandas as pd
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
out=Path(sys.argv[2]).resolve() if len(sys.argv)>2 else Path(__file__).resolve().parent
out.mkdir(parents=True, exist_ok=True)
DF=pd.read_csv(sys.argv[1] if len(sys.argv)>1 else out/'error_wallclock_source_data.csv')
assert DF.shape[0]==100 and DF[['method','lambda','seed']].duplicated().sum()==0
assert DF['state'].eq('success').all()
assert DF['n_candidates'].eq(5).all()
colors={'fixed_equal':'#555555','fixed_selected':'#2E6C9E','lra':'#22734D','relobralo':'#C47F16','ddqn':'#AB4053'}
labels={'fixed_equal':'Fixed (1,1)','fixed_selected':'Fixed (1,3)','lra':'LRA','relobralo':'ReLoBRaLo','ddqn':'DDQN'}
markers={'fixed_equal':'o','fixed_selected':'s','lra':'D','relobralo':'^','ddqn':'v'}
plt.rcParams.update({'font.family':'DejaVu Sans','font.size':8.5,'axes.labelsize':9,'axes.titlesize':10,'legend.fontsize':8,'xtick.labelsize':8,'ytick.labelsize':8,'pdf.fonttype':42})
fig,axes=plt.subplots(2,2,figsize=(7.15,5.4),sharex=True,sharey=True,constrained_layout=False)
for ax,lam in zip(axes.flat,[1.,2.,3.,3.4]):
    sub=DF[DF['lambda']==lam]
    for method in colors:
        t=sub[sub['method']==method]
        for _,r in t.iterrows():
            # Filled Intel, hollow AMD (do not interpret as performance-normalized)
            intel=('0-20' in str(r.host))
            ax.scatter(float(r.wall_time_seconds),float(r.relative_l2),
                       s=37,marker=markers[method],alpha=0.88,
                       facecolors=colors[method] if intel else 'white',
                       edgecolors=colors[method],linewidths=.85,zorder=3)
    ax.set_title(r'$\lambda = '+str(lam)+'$',pad=7)
    ax.set_yscale('log')
    ax.grid(True,axis='y',which='major',lw=.5,ls=':',color='#b2b2b2')
    ax.set_xlim(625,1080)
    ax.set_ylim(1e-5,.16)
    ax.set_xticks([650,750,850,950,1050])
    ax.tick_params(length=3)
for ax in axes[:,0]: ax.set_ylabel(r'Selected relative $L_2$ error')
for ax in axes[1,:]: ax.set_xlabel('Elapsed wall-clock time for five candidates (s)')
handles=[Line2D([0],[0],marker=markers[m],markersize=6,linestyle='',color=colors[m],markerfacecolor=colors[m],label=labels[m]) for m in colors]
fig.legend(handles=handles,loc='lower center',bbox_to_anchor=(.5,.048),ncol=5,frameon=False,columnspacing=1.25,handletextpad=.3)
# Node meaning kept in figure footer
fig.text(.5,.014,'Filled: Intel Xeon    |    Open: AMD Opteron',ha='center',fontsize=7.3,color='#444444')
fig.subplots_adjust(left=.11,right=.985,top=.94,bottom=.18,hspace=.20,wspace=.13)
fig.savefig(out/'Fig4.pdf',bbox_inches='tight',pad_inches=.04)
fig.savefig(out/'Fig4_preview.png',dpi=175,bbox_inches='tight',pad_inches=.04)
plt.close(fig)

