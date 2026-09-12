"""Recompute achiral ECFP4 bit-vector AD and check the saved main-study AD table."""
from pathlib import Path
import argparse
import json

import numpy as np
import pandas as pd
from rdkit import Chem, DataStructs, rdBase
from rdkit.Chem import rdFingerprintGenerator

ROOT = Path(__file__).resolve().parents[1]


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output-dir',type=Path,default=ROOT/'results/generated/ad_replay')
    args = parser.parse_args()
    read = lambda p: pd.read_csv(p,encoding='utf-8-sig',float_precision='round_trip')
    pool = read(ROOT/'data/oof/train_pool_oof_residual_report.csv')
    reference = pool.loc[~pool.remove_by_oof_high_residual.astype(bool)]
    saved = read(ROOT/'predictions/source_holdout/external_with_ad.csv')
    if len(reference)!=34511 or len(saved)!=1068:
        raise ValueError('Unexpected AD reference or external population.')
    generator = rdFingerprintGenerator.GetMorganGenerator(radius=2,fpSize=2048,includeChirality=False)
    def fingerprint(s):
        mol = Chem.MolFromSmiles(s)
        if mol is None: raise ValueError('Unparsable frozen SMILES: '+s)
        return generator.GetFingerprint(mol)
    fps = [fingerprint(s) for s in reference['Canonical SMILES']]
    thresholds = [.50,.55,.60,.65,.70,.75,.80,.85,.90]
    rows = []
    for s in saved['Canonical SMILES']:
        sims = np.asarray(DataStructs.BulkTanimotoSimilarity(fingerprint(s),fps),dtype=float)
        row = {'max_similarity':float(sims.max()),'mean_top5_similarity':float(np.sort(sims)[-5:].mean())}
        for threshold in thresholds:
            n = int(np.sum(sims >= threshold))
            row[f'n_train_sim_ge_{threshold:.2f}'] = n
            for count in [1,3,5,10,20]:
                row[f'in_AD_S{threshold:.2f}_N{count}'] = bool(n>=count)
        rows.append(row)
    actual = pd.DataFrame(rows)
    checks = {}
    for col in actual:
        if col not in saved: raise ValueError('Expected saved AD field missing: '+col)
        a, b = actual[col].to_numpy(), saved[col].to_numpy()
        if col in ['max_similarity','mean_top5_similarity']:
            difference = float(np.max(np.abs(a.astype(float)-b.astype(float))))
            checks[col] = difference
            if difference > 1e-8: raise ValueError(f'AD replay differs in {col}: {difference}')
        elif not np.array_equal(a,b):
            raise ValueError(f'AD replay differs in {col} for {int(np.sum(a!=b))} records.')
    out=args.output_dir.resolve();out.mkdir(parents=True,exist_ok=True)
    table = pd.concat([saved[['Canonical SMILES']].reset_index(drop=True),actual],axis=1)
    table.to_csv(out/'ad_replayed.csv',index=False,encoding='utf-8-sig')
    report={'rdkit_version':rdBase.rdkitVersion,'n_reference':len(reference),'n_external':len(saved),
            'fingerprint':'achiral 2048-bit Morgan radius 2; binary fingerprint for AD',
            'n_fields_checked':len(actual.columns),'all_saved_AD_fields_match':True,
            'n_max_similarity_exactly_one':int((actual.max_similarity==1.0).sum()),
            'max_absolute_numeric_differences':checks,
            'provenance':'New replay utility verified against saved AD output; not asserted to be the original AD source code.'}
    (out/'ad_replay_verification.json').write_text(json.dumps(report,indent=2)+'\n')
    print(json.dumps(report,indent=2))


if __name__ == '__main__':
    main()
