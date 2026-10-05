from __future__ import annotations
import argparse
import json
from pathlib import Path
import pandas as pd

KEYS=['diagnostic_classifier','selection_strategy','task','initial_set','checkpoint']

def main():
    p=argparse.ArgumentParser()
    p.add_argument('--results', required=True, help='Raw internal_comparison.csv or aggregated complete_results_by_checkpoint.csv')
    p.add_argument('--expected_grid', required=True)
    p.add_argument('--out_dir', required=True)
    a=p.parse_args()
    out=Path(a.out_dir); out.mkdir(parents=True, exist_ok=True)
    expected=pd.read_csv(a.expected_grid)[KEYS].drop_duplicates()
    results=pd.read_csv(a.results)
    # Ignore initial-set-only rows. Normalize expected active strategy names.
    actual=results.copy()
    if 'checkpoint' in actual:
        actual=actual[actual['checkpoint'].astype(str).ne('Initial set only')]
    actual=actual[KEYS].drop_duplicates() if all(k in actual for k in KEYS) else pd.DataFrame(columns=KEYS)
    merged=expected.merge(actual.assign(found=True), on=KEYS, how='left')
    missing=merged[merged['found'].isna()].drop(columns='found')
    missing.to_csv(out/'missing_after_run.csv', index=False)
    status={
        'expected_combinations': int(len(expected)),
        'found_combinations': int(len(expected)-len(missing)),
        'missing_combinations': int(len(missing)),
        'status': 'COMPLETE' if missing.empty else 'INCOMPLETE'
    }
    (out/'coverage_status.json').write_text(json.dumps(status,indent=2),encoding='utf-8')
    print(f"{status['status']}: {status['found_combinations']} / {status['expected_combinations']}")
    if not missing.empty:
        raise SystemExit(2)

if __name__=='__main__': main()
