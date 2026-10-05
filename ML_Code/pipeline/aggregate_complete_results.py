from __future__ import annotations
import argparse
from pathlib import Path
import pandas as pd

GROUP=['initial_set','task','diagnostic_classifier','selection_strategy','checkpoint','added_compositions','new_prob_runs']

def main():
    p=argparse.ArgumentParser(); p.add_argument('--raw_results',required=True); p.add_argument('--out',required=True); a=p.parse_args()
    d=pd.read_csv(a.raw_results)
    d=d[d['selection_strategy'].astype(str).ne('Initial set only')].copy()
    g=d.groupby(GROUP,dropna=False)['internal_validation_f1_mean'].agg(['mean','std','count']).reset_index()
    g=g.rename(columns={'mean':'internal_validation_f1_mean','std':'internal_validation_f1_std','count':'repetitions'})
    Path(a.out).parent.mkdir(parents=True,exist_ok=True)
    g.to_csv(a.out,index=False)
    print(f'Wrote {len(g)} aggregated active-learning combinations to {a.out}')

if __name__=='__main__': main()
