from __future__ import annotations
import argparse
from pathlib import Path
import pandas as pd
import matplotlib.pyplot as plt

ORDER=['1%','2%','4%','8%','16%']

def main():
    p=argparse.ArgumentParser(); p.add_argument('--results',required=True); p.add_argument('--out_dir',required=True); a=p.parse_args()
    d=pd.read_csv(a.results); out=Path(a.out_dir); out.mkdir(parents=True,exist_ok=True)
    d=d[d.selection_strategy.astype(str).ne('Initial set only')].copy()
    d['checkpoint']=pd.Categorical(d['checkpoint'],ORDER,ordered=True)
    for (task,initial,model),g in d.groupby(['task','initial_set','diagnostic_classifier'],sort=False):
        fig,ax=plt.subplots(figsize=(6.8,4.1))
        for strategy,s in g.groupby('selection_strategy',sort=False):
            s=s.sort_values('checkpoint')
            ax.plot(s['checkpoint'].astype(str),s['internal_validation_f1_mean'],marker='o',label=strategy)
        ax.set_xlabel('Added candidate percentage')
        ax.set_ylabel('Grouped-validation F1-Macro (%)')
        ax.set_title(f'{task} — {model} — {initial}')
        ax.grid(True,alpha=0.25); ax.legend(frameon=False,fontsize=8)
        fig.tight_layout()
        safe='__'.join(str(x).replace(':','').replace(' ','_').replace('/','_') for x in [task,model,initial])
        fig.savefig(out/f'{safe}.png',dpi=300,bbox_inches='tight')
        fig.savefig(out/f'{safe}.pdf',bbox_inches='tight')
        plt.close(fig)

if __name__=='__main__': main()
