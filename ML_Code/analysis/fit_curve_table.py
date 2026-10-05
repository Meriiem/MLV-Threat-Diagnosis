from __future__ import annotations
import argparse, csv
from collections import defaultdict
from pathlib import Path
from curve_fit_common import read_csv, fit_saturation, XMAP

def write_csv(path, rows):
    with Path(path).open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0]))
        writer.writeheader(); writer.writerows(rows)

def main():
    p=argparse.ArgumentParser(description="Fit y = y0 + G(1-exp(-kx)) to learning curves.")
    p.add_argument("--input", required=True)
    p.add_argument("--metric", required=True)
    p.add_argument("--group", nargs="+", required=True, help="Columns defining one curve")
    p.add_argument("--output", required=True)
    a=p.parse_args()
    rows=read_csv(Path(a.input)); groups=defaultdict(list)
    for row in rows:
        groups[tuple(row[c] for c in a.group)].append((XMAP[row["checkpoint"]], float(row[a.metric])))
    out=[]
    for key, points in sorted(groups.items()):
        fit=fit_saturation(points)
        record={c:v for c,v in zip(a.group,key)}
        record.update(fit)
        record["equation"] = f'y = {fit["y0"]:.2f} + {fit["gain"]:.2f}(1 - exp(-{fit["rate_k"]:.4f}x))'
        out.append(record)
    write_csv(a.output,out)
if __name__=="__main__": main()
