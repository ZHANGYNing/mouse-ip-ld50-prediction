"""Export the original fixed training/validation/external tables from frozen IDs."""
from pathlib import Path
import argparse
import json
import numpy as np
import pandas as pd

ROOT=Path(__file__).resolve().parents[1]


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--output-dir',type=Path,default=ROOT/'data/processed')
    args=p.parse_args()
    read=lambda path:pd.read_csv(path,encoding='utf-8-sig',float_precision='round_trip')
    pool=read(ROOT/'data/oof/train_pool_oof_residual_report.csv').set_index('Canonical SMILES')
    master=read(ROOT/'data/model_compounds.csv')
    split=read(ROOT/'splits/source_holdout_and_internal.csv')
    core=[c for c in master if c not in ['compound_id','cohort','development_row_id','excluded_by_original_oof']]
    out=args.output_dir.resolve();out.mkdir(parents=True,exist_ok=True)
    exported={}
    for role,name,n in [('train_for_early_stopping','train_after_oof_no_leakage.csv',31059),
                        ('valid_for_early_stopping','valid_after_oof_no_leakage.csv',3452)]:
        ids=split.loc[split.original_internal_role==role].sort_values('partition_row_order')['Canonical SMILES']
        frame=pool.loc[ids].reset_index()
        if len(frame)!=n or frame.remove_by_oof_high_residual.any(): raise ValueError('Fixed internal split mismatch.')
        frame.to_csv(out/name,index=False,encoding='utf-8-sig');exported[name]=len(frame)
    external=master.loc[master.cohort=='pubchem_source_holdout',core]
    if len(external)!=1068: raise ValueError('External population mismatch.')
    name='external_pubchem_only_final_no_oof.csv'
    external.to_csv(out/name,index=False,encoding='utf-8-sig');exported[name]=len(external)
    print(json.dumps(exported,indent=2))


if __name__=='__main__':main()
