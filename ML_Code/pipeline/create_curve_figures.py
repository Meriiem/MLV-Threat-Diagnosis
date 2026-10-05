from __future__ import annotations
import argparse
from pathlib import Path
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

CHECKPOINT_X={"Initial set only":0,"1%":1,"2%":2,"4%":4,"8%":8,"16%":16}
TASKS=["Task A: Component","Task B: Threat Type","Task C: Joint"]
INITS=["5-5-5","10-10-10","30-30-30"]
STRATEGIES=["Random","Entropy","Margin","Jaccard Diversity"]

def main():
    p=argparse.ArgumentParser();p.add_argument("--results",required=True);p.add_argument("--out_dir",required=True);a=p.parse_args()
    df=pd.read_csv(a.results);df["x"]=df["checkpoint"].map(CHECKPOINT_X)
    out=Path(a.out_dir);out.mkdir(parents=True,exist_ok=True)
    for i,task in enumerate(TASKS,1):
        fig,axes=plt.subplots(1,3,figsize=(15,4.8),sharey=True)
        for ax,init in zip(axes,INITS):
            sub=df[(df.task==task)&(df.initial_set==init)]
            for s in STRATEGIES:
                rows=sub[(sub.selection_strategy=="Initial set only")|(sub.selection_strategy==s)].sort_values("x")
                ax.plot(rows.x,rows.hidden_test_f1_macro,marker="o",linewidth=2,label=s)
            ax.set_title(init);ax.set_xticks([0,1,2,4,8,16]);ax.set_xticklabels(["Init","1%","2%","4%","8%","16%"]);ax.set_xlabel("Added candidate set (%)");ax.set_ylabel("Hidden Test F1-Macro (%)");ax.grid(alpha=.25)
        h,l=axes[0].get_legend_handles_labels();fig.legend(h,l,loc="lower center",ncol=4,frameon=False);fig.suptitle(f"{task} — LightGBM selection-strategy curves");fig.tight_layout(rect=(0,.08,1,.94));fig.savefig(out/f"{i:02d}_{task.split(':')[0].replace(' ','_')}.png",dpi=240,bbox_inches="tight");plt.close(fig)
if __name__=="__main__":main()
