from __future__ import annotations
import argparse
from pathlib import Path
import pandas as pd
from scipy import sparse


def main():
    p=argparse.ArgumentParser()
    p.add_argument('--features', required=True)
    p.add_argument('--metadata', required=True)
    p.add_argument('--candidate_combinations', required=True)
    p.add_argument('--out_dir', required=True)
    a=p.parse_args()
    out=Path(a.out_dir); out.mkdir(parents=True, exist_ok=True)
    x=sparse.load_npz(a.features).tocsr()
    m=pd.read_csv(a.metadata)
    c=pd.read_csv(a.candidate_combinations)
    aliases=c[c['existing_system_id'].fillna('').astype(str).str.strip().ne('')][['candidate_id','existing_system_id']]
    matrices=[x]; metadata=[m]; audit=[]
    existing=set(m['candidate_id'].astype(str))
    for row in aliases.itertuples(index=False):
        cid=str(row.candidate_id); sid=str(row.existing_system_id)
        if cid in existing:
            audit.append({'candidate_id':cid,'existing_system_id':sid,'status':'ALREADY_PRESENT','rows_added':0}); continue
        idx=m.index[m['candidate_id'].astype(str).eq(sid)].to_numpy()
        if len(idx)==0:
            audit.append({'candidate_id':cid,'existing_system_id':sid,'status':'SOURCE_TRACE_MISSING','rows_added':0}); continue
        alias=m.loc[idx].copy(); alias['candidate_id']=cid; alias['aliased_from_system_id']=sid
        matrices.append(x[idx]); metadata.append(alias)
        audit.append({'candidate_id':cid,'existing_system_id':sid,'status':'ALIASED_EXACT_COMPOSITION','rows_added':len(idx)})
        existing.add(cid)
    sparse.save_npz(out/'trace_features.npz', sparse.vstack(matrices).tocsr())
    pd.concat(metadata, ignore_index=True).to_csv(out/'trace_metadata.csv', index=False)
    pd.DataFrame(audit).to_csv(out/'alias_audit.csv', index=False)
    print(pd.DataFrame(audit)['status'].value_counts().to_dict())

if __name__=='__main__': main()
