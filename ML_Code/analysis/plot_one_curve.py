from __future__ import annotations
import argparse, csv
from pathlib import Path
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from curve_fit_common import saturation

def main():
    p=argparse.ArgumentParser(description="Plot one observed learning curve and its fitted saturation equation.")
    p.add_argument("--points_csv", required=True, help="CSV columns: x,y")
    p.add_argument("--fit_csv", required=True, help="Single-row CSV with y0,gain,rate_k,plateau,x95,r2")
    p.add_argument("--title", required=True); p.add_argument("--ylabel", required=True); p.add_argument("--output", required=True)
    a=p.parse_args()
    with open(a.points_csv, encoding="utf-8-sig", newline="") as f: points=list(csv.DictReader(f))
    with open(a.fit_csv, encoding="utf-8-sig", newline="") as f: fit=next(csv.DictReader(f))
    x=np.array([float(r["x"]) for r in points]); y=np.array([float(r["y"]) for r in points]); xx=np.linspace(0,16,250)
    yy=saturation(xx,float(fit["gain"]),float(fit["rate_k"]),float(fit["y0"]))
    fig,ax=plt.subplots(figsize=(8.2,5.2)); ax.scatter(x,y,s=48,label="Observed values"); ax.plot(xx,yy,linewidth=2.2,label="Saturating exponential fit")
    ax.set_xlabel("Added candidate set (%)"); ax.set_ylabel(a.ylabel); ax.set_title(a.title); ax.grid(alpha=.25); ax.legend(frameon=False)
    fig.tight_layout(); fig.savefig(a.output,dpi=240,bbox_inches="tight")
if __name__=="__main__": main()
