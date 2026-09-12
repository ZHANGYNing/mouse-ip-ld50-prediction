"""Recompute main-study numerical results from saved predictions, without training."""
from pathlib import Path
import argparse
import hashlib
import json

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
SMILES = 'Canonical SMILES'


def read(path):
    return pd.read_csv(path,encoding='utf-8-sig',float_precision='round_trip',low_memory=False)


def metrics(y,p):
    y,p=np.asarray(y,dtype=float),np.asarray(p,dtype=float)
    if y.ndim != 1 or p.shape != y.shape or not np.isfinite(y).all() or not np.isfinite(p).all():
        raise ValueError('Invalid target/prediction arrays; no records were removed.')
    err=p-y;ae=np.abs(err);n=len(y)
    if n == 0:
        return {'n':0,'R2':None,'RMSE':None,'MAE':None,'median_fold_error':None,
                'within_2_fold_percent':None,'within_5_fold_percent':None,'within_10_fold_percent':None}
    sse=float(np.dot(err,err));sst=float(np.sum((y-y.mean())**2))
    return {'n':n,'R2':1-sse/sst if n>1 and sst>0 else None,'RMSE':float(np.sqrt(sse/n)),
            'MAE':float(ae.mean()),'median_fold_error':float(np.median(10**ae)),
            'within_2_fold_percent':float(np.mean(ae<=np.log10(2))*100),
            'within_5_fold_percent':float(np.mean(ae<=np.log10(5))*100),
            'within_10_fold_percent':float(np.mean(ae<=1)*100),'SSE':sse,'SST':sst}


def close_metrics(actual,expected,tol=2e-6):
    for key in ['R2','RMSE','MAE']:
        if abs(actual[key]-expected[key]) > tol:
            raise ValueError(f'{key} differs from the recorded result: {actual[key]} vs {expected[key]}')


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output-dir',type=Path,default=ROOT/'results/recomputed')
    args=parser.parse_args()
    pool=read(ROOT/'data/oof/train_pool_oof_residual_report.csv')
    excluded=read(ROOT/'data/exclusions/removed_oof_448.csv')
    master=read(ROOT/'data/model_compounds.csv')
    split=read(ROOT/'splits/source_holdout_and_internal.csv')
    original_oof=read(ROOT/'splits/original_oof.csv')
    historical=json.loads((ROOT/'results/source_holdout/original_run_summary.json').read_text())
    if len(master)!=36027 or len(pool)!=34959 or len(excluded)!=448:
        raise ValueError('Unexpected frozen cohort size.')
    mask=pool.oof_fold_error.to_numpy()>=20.0
    if not np.array_equal(mask,pool.remove_by_oof_high_residual.to_numpy(bool)):
        raise ValueError('OOF flags do not follow the recorded 20-fold threshold.')
    if set(pool.loc[mask,SMILES])!=set(excluded[SMILES]):
        raise ValueError('448-record exclusion identities do not match the frozen OOF report.')
    if set(split.compound_id)!=set(master.compound_id) or split.compound_id.duplicated().any():
        raise ValueError('Split identities do not cover the master data exactly once.')
    roles=split.original_internal_role.value_counts().to_dict()
    expected_roles={'train_for_early_stopping':31059,'valid_for_early_stopping':3452,'external_test':1068,'excluded_oof':448}
    if roles!=expected_roles: raise ValueError(f'Unexpected split membership: {roles}')
    joined=split.merge(master[['compound_id','source_type']],on='compound_id',validate='one_to_one')
    for role,hkey in [('train_for_early_stopping','train_source_counts'),('valid_for_early_stopping','valid_source_counts')]:
        actual=joined.loc[joined.original_internal_role==role,'source_type'].value_counts().to_dict()
        if actual!=historical['final_modeling_data'][hkey]:
            raise ValueError(f'Recovered {role} source composition differs from the original summary.')
    label_check=-np.log10(master.ld50_mgkg.to_numpy()/(1000*master.molecular_weight.to_numpy()))
    if not np.allclose(label_check,master.y.to_numpy(),rtol=0,atol=1e-12):
        raise ValueError('Target transformation mismatch.')
    oof=metrics(pool.y,pool.oof_pred_y)
    close_metrics(oof,historical['oof_filter_train_pool_only']['oof_metrics_on_train_pool_before_filter'])
    oof_rows=[]
    for fold in range(1,6):
        idx=original_oof.loc[original_oof.screening_fold==fold,'development_row_id'].to_numpy(int)
        row=metrics(pool.iloc[idx].y,pool.iloc[idx].oof_pred_y)
        h=historical['oof_filter_train_pool_only']['fold_records'][fold-1]
        close_metrics(row,{'R2':h['valid_R2'],'RMSE':h['valid_RMSE'],'MAE':h['valid_MAE']})
        oof_rows.append({'fold':fold,**row})
    external_rows=[]
    for variant in ['single','bagging']:
        frame=read(ROOT/f'predictions/source_holdout/external_{variant}.csv')
        if len(frame)!=1068 or frame[SMILES].duplicated().any() or set(frame[SMILES])&set(pool[SMILES]):
            raise ValueError('Invalid external source-holdout population.')
        row=metrics(frame.y_true_neglog_molkg,frame.y_pred_neglog_molkg)
        historical_key='external_pubchem_only_single_model_no_leakage' if variant=='single' else 'external_pubchem_only_bagging_no_leakage'
        close_metrics(row,historical[historical_key]['metrics'])
        external_rows.append({'model':variant,**row})
    ad=read(ROOT/'predictions/source_holdout/external_with_ad.csv')
    bag=read(ROOT/'predictions/source_holdout/external_bagging.csv').set_index(SMILES).loc[ad[SMILES]]
    if len(ad)!=1068 or ad[SMILES].duplicated().any(): raise ValueError('Invalid AD population.')
    for col in ['y_true_neglog_molkg','y_pred_neglog_molkg']:
        if not np.allclose(bag[col],ad[col],atol=1e-12,rtol=0):
            raise ValueError('AD table differs from frozen ensemble predictions.')
    ad_rows=[]
    for col in ad.columns:
        if col.startswith('in_AD_'):
            for inside in [True,False]:
                part=ad.loc[ad[col].to_numpy(bool)==inside]
                ad_rows.append({'AD_rule':col,'region':'inside' if inside else 'outside',
                                **metrics(part.y_true_neglog_molkg,part.y_pred_neglog_molkg)})
    out=args.output_dir.resolve();out.mkdir(parents=True,exist_ok=True)
    pd.DataFrame([{'population':'pre_screening_development',**oof}]).to_csv(out/'oof_screening_metrics.csv',index=False)
    pd.DataFrame(oof_rows).to_csv(out/'oof_screening_fold_metrics.csv',index=False)
    pd.DataFrame(external_rows).to_csv(out/'external_metrics.csv',index=False)
    pd.DataFrame(ad_rows).to_csv(out/'AD_metrics.csv',index=False)
    report={'status':'PASS','scope':'Main source-holdout workflow; no model fitting or new prediction generation',
            'n_raw_rows':historical['rule_cleaning']['n_raw_rows'],'n_rule_curated_compounds':len(master),
            'n_development_before_oof':len(pool),'n_excluded_oof':int(mask.sum()),'split_membership':roles,
            'all_five_recovered_OOF_fold_metrics_match_original_summary':True,
            'single_and_ensemble_external_metrics_match_original_summary':True,
            'AD_predictions_match_original_ensemble_predictions':True,
            'numpy':np.__version__,'pandas':pd.__version__,
            'limits':['Original internal holdout per-compound predictions are not included.',
                      'No original trained tree files or main-model preprocessing objects were supplied.',
                      'Exact original main-training dependency versions were not recorded in the supplied original summary.']}
    (out/'verification.json').write_text(json.dumps(report,indent=2)+'\n')
    print(pd.DataFrame(external_rows)[['model','n','R2','RMSE','MAE','within_2_fold_percent']].to_string(index=False))
    print('Verification: PASS; output:',out)


if __name__=='__main__':
    main()
