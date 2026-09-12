"""Replay the original rule curation and export exclusion records; no model fitting."""
from pathlib import Path
import argparse
import hashlib
import json
import sys

import numpy as np
import pandas as pd
import rdkit

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / 'src'))
import train_and_evaluate as original


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output-dir', type=Path, default=ROOT/'results/generated/rule_replay')
    args = parser.parse_args()
    raw_path = ROOT/'data/raw/mouse_intraperitoneal_ld50.xlsx'
    raw = pd.read_excel(raw_path)
    curated, removed_raw, removed_groups = original.rule_clean_and_aggregate(raw)
    frozen = pd.read_csv(ROOT/'data/model_compounds.csv', float_precision='round_trip')
    key = 'Canonical SMILES'
    if curated[key].duplicated().any() or set(curated[key]) != set(frozen[key]):
        raise ValueError('Replayed curation changes frozen compound identities; do not replace the frozen data.')
    current = curated.set_index(key).loc[frozen[key]]
    expected = frozen.set_index(key)
    diffs = {}
    for col in ['ld50_mgkg','molecular_weight','y','ld50_min_mgkg_raw','ld50_max_mgkg_raw','ld50_median_mgkg_raw']:
        a, b = current[col].to_numpy(float), expected[col].to_numpy(float)
        diffs[col] = float(np.max(np.abs(a-b)))
        if not np.allclose(a,b,rtol=1e-12,atol=1e-12):
            raise ValueError(f'Replayed {col} differs from the frozen study data: {diffs[col]}')
    for col in ['source_type','raw_row_ids','n_raw_records']:
        if not current[col].astype(str).equals(expected[col].astype(str)):
            raise ValueError(f'Replayed {col} differs from the frozen study data.')
    summary = json.loads((ROOT/'results/source_holdout/original_run_summary.json').read_text())['rule_cleaning']
    if len(removed_raw) != summary['n_removed_raw_by_rule'] or len(removed_groups) != summary['n_removed_aggregated_by_rule']:
        raise ValueError('Exclusion counts differ from the original run summary.')
    out = args.output_dir.resolve()
    out.mkdir(parents=True,exist_ok=True)
    for name,frame in [('rule_raw_records.csv',removed_raw),('rule_aggregated_records.csv',removed_groups)]:
        frame.to_csv(out/name,index=False,encoding='utf-8-sig')
    report = {
        'operation':'Replayed original rule_clean_and_aggregate without any model fitting',
        'rdkit_version':rdkit.__version__,
        'original_input_sha256':hashlib.sha256(raw_path.read_bytes()).hexdigest(),
        'n_raw':len(raw),'n_retained_compounds':len(curated),
        'n_excluded_raw_records':len(removed_raw),'n_excluded_aggregated_compounds':len(removed_groups),
        'all_retained_identities_sources_and_raw_record_links_match':True,
        'maximum_absolute_numeric_differences':diffs,
        'exclusion_provenance':'Regenerated from the supplied raw input and original curation function; not an archived historical exclusion CSV.'}
    (out/'rule_replay_verification.json').write_text(json.dumps(report,indent=2)+'\n')
    print(json.dumps(report,indent=2))


if __name__ == '__main__':
    main()
