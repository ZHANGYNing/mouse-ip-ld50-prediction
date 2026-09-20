# -*- coding: utf-8 -*-
"""Nested training-only residual-screening ablation for JHM revision.

Default: random outer 5-fold CV, inner 5-fold OOF, no filter / 10 / 20 / 30,
fixed screening seed 42 and final model seed 42. Add --seeds to average models.
Optional --mode scaffold uses achiral Murcko scaffold groups in outer/inner CV.

This is a NEW nested validation implementation whose feature/preprocessing
and model settings have been checked against the uploaded original 4(3).py.
The evaluation design intentionally differs from the original non-nested design.
See README and original_source_audit.txt for verified facts and remaining limits.
No external PubChem-only records or old OOF values are used for model fitting.
"""
from __future__ import annotations
import argparse
import gc
import json
import logging
import math
import os
from pathlib import Path
import platform
import sys
import time
import warnings
from typing import Any

import numpy as np
import pandas as pd
import sklearn
from sklearn.impute import SimpleImputer
from sklearn.feature_selection import VarianceThreshold
from sklearn.model_selection import KFold, GroupShuffleSplit, train_test_split
import xgboost as xgb
import rdkit
from rdkit import Chem, DataStructs
from rdkit.Chem import Descriptors, rdFingerprintGenerator
from rdkit.Chem.Scaffolds import MurckoScaffold

from revision_common import (finite_column, object_hash, read_csv,
                             regression_metrics, save_csv, save_json,
                             save_npy, sha256_file, locate_data_file)

ROOT = Path(__file__).resolve().parent
LOG = logging.getLogger('JHM_revision')
SMILES = 'Canonical SMILES'
BASE_COLUMNS = [SMILES, 'y', 'ld50_mgkg', 'molecular_weight', 'source_type']
OLD_OOF_COLUMNS = ['oof_pred_y', 'oof_abs_error_y', 'oof_fold_error',
                   'remove_by_oof_high_residual']
SCHEMA_VERSION = 'nested_oof_revision_2.0_source_checked'


def load_json(p: Path) -> Any:
    return json.loads(p.read_text(encoding='utf-8-sig'))


def npz_atomic(path: Path, **kwargs) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(path.name + '.tmp')
    with tmp.open('wb') as f:
        np.savez_compressed(f, **kwargs)
    os.replace(tmp, path)


def resolve_file(name: str, data_dir: Path, config_dir: Path) -> Path:
    p = Path(name)
    if p.is_absolute():
        return p
    # User data directory has priority over the code/config directory.
    candidate = data_dir / p
    return candidate if candidate.is_file() else config_dir / p


def load_development(path: Path, expected_n: int) -> pd.DataFrame:
    df = read_csv(path)
    missing = [c for c in BASE_COLUMNS if c not in df]
    if missing:
        raise ValueError(f'Missing required columns {missing}; do not substitute old predictions for y.')
    if len(df) != expected_n:
        raise ValueError(f'Expected {expected_n} PRE-OOF development rows, got {len(df)}. '
                         'Do not use the 34511-row filtered set.')
    if df[SMILES].isna().any() or df[SMILES].duplicated().any():
        raise ValueError('SMILES missing/duplicated. Fix the input provenance; no silent deduplication.')
    counts = df['source_type'].value_counts(dropna=False).to_dict()
    if not df['source_type'].isin(['TOXRIC_only', 'mixed']).all():
        raise ValueError(f'Only TOXRIC_only + mixed are allowed; source counts: {counts}')
    if expected_n == 34959 and counts != {'TOXRIC_only':31944, 'mixed':3015}:
        raise ValueError(f'Unexpected source composition: {counts}')
    for c in ['y', 'ld50_mgkg', 'molecular_weight']:
        df[c] = finite_column(df, c)
    if (df['ld50_mgkg'] <= 0).any() or (df['molecular_weight'] <= 0).any():
        raise ValueError('Dose and molecular weight must be positive.')
    y_calc = -np.log10(df['ld50_mgkg'].to_numpy() /
                       (1000 * df['molecular_weight'].to_numpy()))
    if not np.allclose(df['y'], y_calc, rtol=0, atol=1e-6):
        raise ValueError('y does not agree with -log10(ld50_mgkg/(1000*MW)); check target scale.')
    df = df.copy()
    df.insert(0, 'sample_id', [object_hash({'smiles':s})[:20] for s in df[SMILES]])
    df.insert(1, 'row_id', np.arange(len(df), dtype=int))
    if df['sample_id'].duplicated().any():
        raise ValueError('Unexpected sample identifier collision.')
    # Retain the complete original table for provenance only. Never use old OOF columns as features.
    return df


class TrainOnlyPreprocessor:
    """Same estimator calls as original source, fitted ONLY on this training partition."""
    def __init__(self):
        self.keep_cols = None
        self.imputer = None
        self.selector = None
        self.indices = None
        self.medians = None

    def fit_transform(self, raw: np.ndarray) -> np.ndarray:
        a = np.array(raw, dtype=np.float32, order='C', copy=True)
        a[~np.isfinite(a)] = np.nan
        self.keep_cols = ~np.all(np.isnan(a), axis=0)
        if not self.keep_cols.any():
            raise ValueError('All features missing in this training partition.')
        a = a[:, self.keep_cols]
        self.imputer = SimpleImputer(strategy='median')
        imputed = self.imputer.fit_transform(a).astype(np.float32)
        self.selector = VarianceThreshold(threshold=0.0)
        self.selector.fit(imputed)
        support = self.selector.get_support()
        self.indices = np.flatnonzero(self.keep_cols)[support]
        self.medians = np.asarray(self.imputer.statistics_[support], dtype=np.float32)
        return np.ascontiguousarray(self.selector.transform(imputed), dtype=np.float32)

    def transform(self, raw: np.ndarray) -> np.ndarray:
        if self.indices is None:
            raise RuntimeError('Preprocessor not fitted.')
        a = np.array(raw, dtype=np.float32, order='C', copy=True)
        a[~np.isfinite(a)] = np.nan
        a = self.imputer.transform(a[:, self.keep_cols]).astype(np.float32)
        return np.ascontiguousarray(self.selector.transform(a), dtype=np.float32)

    def save(self, path: Path) -> None:
        # These arrays suffice to reproduce transform: select indices, fill with medians.
        npz_atomic(path, indices=self.indices, medians=self.medians,
                   all_nan_keep_cols=self.keep_cols)


def molecule_feature_row(smiles: str, cfg: dict, names: list[str],
                         fpgen=None) -> tuple[np.ndarray, dict]:
    """Match original count/log1p and descriptor routines without any training.

    Parsing separately for the two blocks mirrors the old build_raw_features.
    Invalid structures raise instead of silently turning into an all-zero sample.
    Descriptor errors become NaN, as in the original script.
    """
    mol_fp = Chem.MolFromSmiles(str(smiles))
    mol_desc = Chem.MolFromSmiles(str(smiles))
    if mol_fp is None or mol_desc is None:
        raise ValueError(f'Invalid standardized SMILES: {smiles}')
    if fpgen is None:
        fpgen = rdFingerprintGenerator.GetMorganGenerator(
            radius=int(cfg['fingerprint_radius']), fpSize=int(cfg['fingerprint_size']),
            includeChirality=bool(cfg['include_chirality']))
    counts = np.zeros(int(cfg['fingerprint_size']), dtype=np.float32)
    fp = fpgen.GetCountFingerprint(mol_fp)
    DataStructs.ConvertToNumpyArray(fp, counts)
    if cfg['count_log1p']:
        counts = np.log1p(counts)
    lookup = dict(Descriptors._descList)
    values = []
    failures = {}
    for name in names:
        try:
            v = float(lookup[name](mol_desc))
        except Exception:
            v = np.nan
        if not np.isfinite(v):
            v = np.nan
            failures[name] = 1
        values.append(v)
    with np.errstate(over='ignore', invalid='ignore'):
        desc = np.asarray(values, dtype=np.float32)
    result = np.hstack([counts.astype(np.float32), desc]).astype(np.float32)
    result[~np.isfinite(result)] = np.nan
    return result, failures


def build_raw_features(df: pd.DataFrame, cfg: dict, names: list[str], out: Path,
                       mode: str) -> tuple[np.ndarray, np.ndarray]:
    """Per-molecule, label-free calculation; no dataset fitting at this stage."""
    feat_dir = out / 'raw_features'; feat_dir.mkdir(parents=True, exist_ok=True)
    shape = (len(df), cfg['fingerprint_size'] + len(names))
    feature_path = feat_dir / 'raw_features.npy'
    feature_names = ([f'ECFP4_count_{k}' for k in range(cfg['fingerprint_size'])]
                     + ['RDKit2D_'+n for n in names])
    save_json(feat_dir / 'feature_names.json', feature_names)
    state_path = feat_dir / 'state.json'
    group_path = feat_dir / 'scaffold_groups.npy'
    desc_lookup = dict(Descriptors._descList)
    unknown = sorted(set(names) - set(desc_lookup))
    if unknown:
        raise ValueError(f'This RDKit version lacks descriptors: {unknown}. Do not silently replace them.')
    fpgen = rdFingerprintGenerator.GetMorganGenerator(
        radius=int(cfg['fingerprint_radius']), fpSize=int(cfg['fingerprint_size']),
        includeChirality=bool(cfg['include_chirality']))
    if state_path.is_file():
        state = load_json(state_path)
        if list(shape) != state['shape']:
            raise ValueError('Cached feature shape mismatch. Use a clean run directory.')
        x = np.load(feature_path, mmap_mode='r+' if not state['complete'] else 'r')
        groups = np.load(group_path, allow_pickle=False).astype(str)
        if state['complete']:
            LOG.info('Reusing completed raw feature cache: %s', shape)
            return x, groups
        start = int(state['done_rows'])
        failures = state.get('descriptor_failure_counts', {})
    else:
        # Incomplete files lacking a committed state are safe to reconstruct.
        x = np.lib.format.open_memmap(feature_path, mode='w+', dtype=np.float32, shape=shape)
        groups = np.full(len(df), '', dtype='<U64')
        start = 0; failures = {}
        save_npy(group_path, groups)
        save_json(state_path, {'shape':list(shape),'done_rows':0,'complete':False,
                              'descriptor_failure_counts':failures})
    LOG.info('Computing label-free features from row %d/%d, %d columns', start, len(df), shape[1])
    for i in range(start, len(df)):
        smiles = str(df.iloc[i][SMILES])
        mol = Chem.MolFromSmiles(smiles)
        if mol is None:
            raise ValueError(f'RDKit could not parse row {i}, sample_id={df.iloc[i]["sample_id"]}; '
                             'no sample will be dropped automatically.')
        row, row_failures = molecule_feature_row(smiles, cfg, names, fpgen)
        for name, count in row_failures.items():
            failures[name] = int(failures.get(name, 0)) + count
        x[i] = row
        if mode == 'scaffold':
            scaffold = MurckoScaffold.MurckoScaffoldSmiles(mol=mol, includeChirality=False)
            # All acyclic molecules intentionally share a group, instead of becoming singletons.
            groups[i] = object_hash({'achiral_murcko':scaffold or '__ACYCLIC__'})
        else:
            groups[i] = str(df.iloc[i]['sample_id'])
        if (i + 1) % 250 == 0 or i + 1 == len(df):
            x.flush(); save_npy(group_path, groups)
            save_json(state_path, {'shape':list(shape),'done_rows':i+1,
                                  'complete':i+1==len(df), 'descriptor_failure_counts':failures})
            LOG.info('Features: %d/%d', i+1, len(df))
    feature_names = ([f'ECFP4_count_{k}' for k in range(cfg['fingerprint_size'])]
                     + ['RDKit2D_'+n for n in names])
    save_json(feat_dir / 'feature_names.json', feature_names)
    LOG.info('Raw features cached. Fitted imputation/selection will be redone within EVERY training partition.')
    return x, groups


def group_folds(indices: np.ndarray, groups: np.ndarray, k: int,
                seed: int) -> list[tuple[np.ndarray,np.ndarray]]:
    """Greedy size-balanced whole-scaffold allocation, independent of labels."""
    sub = groups[indices]
    uniq, inverse, counts = np.unique(sub, return_inverse=True, return_counts=True)
    if len(uniq) < k:
        raise ValueError(f'Only {len(uniq)} distinct scaffold groups, cannot make {k} folds.')
    rng = np.random.default_rng(seed)
    # Random tie ordering, then stable descending-size sort.
    order = rng.permutation(len(uniq))
    order = order[np.argsort(-counts[order], kind='stable')]
    total = np.zeros(k, dtype=np.int64)
    assigned = np.full(len(uniq), -1, dtype=np.int64)
    for u in order:
        candidates = np.flatnonzero(total == total.min())
        fold = int(rng.choice(candidates))
        assigned[u] = fold; total[fold] += counts[u]
    row_fold = assigned[inverse]
    answer = []
    for f in range(k):
        te = indices[row_fold == f]; tr = indices[row_fold != f]
        if len(te) == 0 or set(groups[te]) & set(groups[tr]):
            raise AssertionError('Scaffold group isolation failed.')
        answer.append((tr, te))
    LOG.info('Scaffold fold sizes: %s; largest group=%d (%.2f%%)',
             total.tolist(), counts.max(), 100*counts.max()/len(indices))
    return answer


def cv_splits(indices: np.ndarray, groups: np.ndarray, mode: str,
              k: int, seed: int) -> list[tuple[np.ndarray,np.ndarray]]:
    if mode == 'scaffold':
        return group_folds(indices, groups, k, seed)
    split = KFold(n_splits=k, shuffle=True, random_state=seed)
    return [(indices[a], indices[b]) for a,b in split.split(indices)]


def pilot_split(indices: np.ndarray, groups: np.ndarray, mode: str,
                fraction: float, seed: int) -> tuple[np.ndarray,np.ndarray]:
    if mode == 'scaffold':
        if len(np.unique(groups[indices])) < 2:
            raise ValueError('Insufficient scaffold groups for training-only early stopping.')
        gss = GroupShuffleSplit(n_splits=1, test_size=fraction, random_state=seed)
        a,b = next(gss.split(indices, groups=groups[indices]))
        a,b = indices[a],indices[b]
        if set(groups[a]) & set(groups[b]):
            raise AssertionError('Pilot scaffold overlap.')
    else:
        a,b = train_test_split(indices, test_size=fraction, random_state=seed, shuffle=True)
    if len(a)<2 or len(b)<2 or set(a)&set(b):
        raise ValueError('Invalid pilot split.')
    return np.asarray(a),np.asarray(b)


class DeviceGuard(xgb.callback.TrainingCallback):
    def __init__(self, requested: str):
        self.requested = requested
    def after_iteration(self, model, epoch, evals_log):
        if epoch == 0:
            device = load_booster_device(model)
            if self.requested.startswith('cuda') and not device.startswith('cuda'):
                raise RuntimeError(f'Requested {self.requested} but XGBoost is using {device}. '
                                   'No silent CPU fallback. Repair GPU environment or set device=cpu.')
        return False


def load_booster_device(model) -> str:
    cfg = json.loads(model.save_config())
    return str(cfg.get('learner',{}).get('generic_param',{}).get('device','unknown'))


def params_for(cfg: dict, seed: int) -> dict:
    return {**cfg['xgboost_params'], 'seed':int(seed), 'device':cfg['device'],
            'nthread':int(cfg['nthread']), 'verbosity':1, 'validate_parameters':True}


def training_matrix(a: np.ndarray, y: np.ndarray, cfg: dict):
    """Use DMatrix as in the original source; hist/CUDA remain unchanged."""
    return xgb.DMatrix(a, label=np.asarray(y, dtype=np.float32),
                       nthread=int(cfg['nthread']))


def check_done(task: Path, signature: str) -> dict | None:
    p = task / 'complete.json'
    if not p.is_file():
        return None
    state = load_json(p)
    if state.get('signature') != signature:
        raise RuntimeError(f'Completed-task signature mismatch: {task}')
    for name,digest in state.get('checksums',{}).items():
        if not (task/name).is_file() or sha256_file(task/name)!=digest:
            raise RuntimeError(f'Cached result failed integrity check: {task/name}')
    return state


def select_rounds(x: np.ndarray, y: np.ndarray, train_idx: np.ndarray,
                  valid_idx: np.ndarray, cfg: dict, seed: int, task: Path,
                  run_signature: str) -> int:
    """Early stop only inside the INNER TRAIN portion, never on its OOF target fold."""
    signature = object_hash({'run':run_signature,'task':'pilot','seed':int(seed),
                             'train':train_idx.tolist(),'valid':valid_idx.tolist()})
    done = check_done(task, signature)
    if done is not None:
        LOG.info('Resume pilot: %s, rounds=%d', task.name, done['best_rounds'])
        return int(done['best_rounds'])
    task.mkdir(parents=True, exist_ok=True)
    save_npy(task/'train_row_ids.npy', train_idx); save_npy(task/'valid_row_ids.npy', valid_idx)
    pre = TrainOnlyPreprocessor()
    xt = pre.fit_transform(x[train_idx]); xv = pre.transform(x[valid_idx])
    pre.save(task/'preprocessor.npz')
    dt = training_matrix(xt, y[train_idx], cfg)
    dv = training_matrix(xv, y[valid_idx], cfg)
    history = {}
    t0 = time.monotonic()
    LOG.info('Pilot fit: train=%d, early-stop=%d, features=%d',len(train_idx),len(valid_idx),xt.shape[1])
    booster = xgb.train(params_for(cfg,seed), dt, num_boost_round=int(cfg['max_boost_rounds']),
                        evals=[(dv,'training_only_earlystop')],
                        early_stopping_rounds=int(cfg['early_stopping_rounds']),
                        evals_result=history, verbose_eval=int(cfg['verbose_eval']),
                        callbacks=[DeviceGuard(cfg['device'])])
    rounds = int(booster.best_iteration) + 1
    save_json(task/'learning_curve.json', history)
    save_json(task/'parameters.json', params_for(cfg,seed))
    (task/'booster_config.json').write_text(booster.save_config(),encoding='utf-8')
    if cfg['save_models']:
        booster.save_model(task/'model.ubj')
    save_json(task/'complete.json',{'signature':signature,'best_rounds':rounds,
                                   'n_features':int(xt.shape[1]),'seconds':time.monotonic()-t0,
                                   'actual_device':load_booster_device(booster),
                                   'checksums':{'preprocessor.npz':sha256_file(task/'preprocessor.npz')}})
    del booster,dt,dv,xt,xv,pre; gc.collect()
    return rounds


def fit_predict(x: np.ndarray, y_train: np.ndarray, train_idx: np.ndarray,
                pred_idx: np.ndarray, cfg: dict, seed: int, rounds: int,
                task: Path, run_signature: str) -> np.ndarray:
    """The predictor receives training labels only. Prediction indices carry no labels."""
    if len(train_idx)!=len(y_train) or set(train_idx)&set(pred_idx):
        raise AssertionError('Training/held-out separation failed.')
    signature = object_hash({'run':run_signature,'task':'fixed_round_fit','seed':int(seed),
                             'rounds':int(rounds),'train':train_idx.tolist(),'pred':pred_idx.tolist()})
    done = check_done(task, signature)
    if done is not None:
        p = np.load(task/'predictions.npy', allow_pickle=False)
        if p.shape!=(len(pred_idx),) or not np.isfinite(p).all():
            raise ValueError('Invalid cached predictions.')
        LOG.info('Resume predictions: %s', task)
        return p
    task.mkdir(parents=True, exist_ok=True)
    save_npy(task/'train_row_ids.npy',train_idx); save_npy(task/'prediction_row_ids.npy',pred_idx)
    pre = TrainOnlyPreprocessor()
    xt = pre.fit_transform(x[train_idx]); xp = pre.transform(x[pred_idx])
    pre.save(task/'preprocessor.npz')
    dt = training_matrix(xt, y_train, cfg)
    # Deliberately no labels for the held-out prediction matrix.
    dp = xgb.DMatrix(xp, nthread=int(cfg['nthread']))
    t0 = time.monotonic()
    LOG.info('Fixed-round fit: train=%d, predict=%d, seed=%d, rounds=%d, features=%d',
             len(train_idx),len(pred_idx),seed,rounds,xt.shape[1])
    booster = xgb.train(params_for(cfg,seed), dt, num_boost_round=int(rounds),
                        evals=[(dt,'train')], verbose_eval=int(cfg['verbose_eval']),
                        callbacks=[DeviceGuard(cfg['device'])])
    p = np.asarray(booster.predict(dp), dtype=np.float64)
    if p.shape!=(len(pred_idx),) or not np.isfinite(p).all():
        raise ValueError('Invalid predictions. No output metrics will be fabricated.')
    save_npy(task/'predictions.npy',p)
    save_json(task/'parameters.json', {**params_for(cfg,seed),'num_boost_round':int(rounds)})
    (task/'booster_config.json').write_text(booster.save_config(),encoding='utf-8')
    if cfg['save_models']:
        booster.save_model(task/'model.ubj')
    save_json(task/'complete.json',{'signature':signature,'n_train':int(len(train_idx)),
                                   'n_predict':int(len(pred_idx)),'seed':int(seed),
                                   'rounds':int(rounds),'n_features':int(xt.shape[1]),
                                   'seconds':time.monotonic()-t0,'actual_device':load_booster_device(booster),
                                   'checksums':{'predictions.npy':sha256_file(task/'predictions.npy'),
                                                'preprocessor.npz':sha256_file(task/'preprocessor.npz')}})
    del booster,dt,dp,xt,xp,pre; gc.collect()
    return p


def write_summaries(result_dir: Path, fold_rows: list[dict], single_rows: list[dict],
                    prediction_tables: list[pd.DataFrame], k: int,
                    expected_n: int) -> None:
    fold_df = pd.DataFrame(fold_rows)
    save_csv(result_dir/'outer_fold_metrics.csv',fold_df)
    save_csv(result_dir/'individual_seed_metrics.csv',pd.DataFrame(single_rows))
    all_pred = pd.concat(prediction_tables,ignore_index=True)
    save_csv(result_dir/'outer_test_predictions.csv',all_pred)
    summary=[]; changes=[]
    for arm, part in all_pred.groupby('arm',sort=False):
        if part['row_id'].duplicated().any():
            raise AssertionError(f'Repeated outer test predictions within arm {arm}')
        nfold = part['outer_fold'].nunique()
        m = regression_metrics(part['y_true'].to_numpy(),part['y_pred'].to_numpy())
        rows = fold_df.loc[fold_df.arm==arm]
        s = {'arm':arm,'completed_outer_folds':int(nfold),
             'complete':bool(nfold==k and len(part)==expected_n),**m}
        for key in ['R2','RMSE','MAE','within_2_fold_percent']:
            s[f'fold_mean_{key}']=float(rows[key].mean())
            s[f'fold_sd_{key}']=float(rows[key].std(ddof=1)) if len(rows)>1 else None
        s['mean_removed_training_percent']=float(rows['removed_percent'].mean())
        summary.append(s)
    save_csv(result_dir/'pooled_and_fold_summary.csv',pd.DataFrame(summary))
    baseline = fold_df.loc[fold_df.arm=='no_filter'].set_index('outer_fold')
    for _,row in fold_df.loc[fold_df.arm!='no_filter'].iterrows():
        f=int(row['outer_fold'])
        if f not in baseline.index:
            continue
        base=baseline.loc[f]
        r={'outer_fold':f,'arm':row['arm'], 'comparison':'arm minus no_filter, same held-out compounds'}
        for key in ['R2','RMSE','MAE','within_2_fold_percent']:
            r['delta_'+key]=float(row[key]-base[key])
        changes.append(r)
    save_csv(result_dir/'paired_fold_changes_vs_no_filter.csv',pd.DataFrame(changes))


def main() -> None:
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--config',type=Path,default=ROOT/'revision_config.json')
    parser.add_argument('--data-dir',type=Path,default=ROOT)
    parser.add_argument('--mode',choices=['random','scaffold'],default='random')
    parser.add_argument('--seeds',type=int,nargs='+',help='Final model seeds; default is config model_seeds.')
    parser.add_argument('--device',help='cpu, cuda, or cuda:0. Default from config.')
    parser.add_argument('--preflight-only',action='store_true',help='Input/config checks only. No feature calculation or model fit.')
    parser.add_argument('--prepare-only',action='store_true',help='Prepare raw features and fixed outer folds, then stop before any fit.')
    args=parser.parse_args()
    cfg=load_json(args.config)
    thresholds=[float(t) for t in cfg['thresholds']]
    if (not thresholds or len(set(thresholds)) != len(thresholds) or
            any(not math.isfinite(t) or t <= 1 for t in thresholds)):
        raise ValueError('Thresholds must be distinct finite numbers >1.')
    for key in ['outer_folds', 'inner_folds']:
        if int(cfg[key]) < 2:
            raise ValueError(f'{key} must be >=2.')
    if int(cfg['max_boost_rounds']) < 1 or int(cfg['early_stopping_rounds']) < 1:
        raise ValueError('Training-round limits must be positive.')
    if not 0 < float(cfg['pilot_validation_fraction']) < 1:
        raise ValueError('pilot_validation_fraction must lie in (0,1).')
    if cfg.get('screening_seed_policy') != 'base_plus_inner_fold_index':
        raise ValueError('This version expects screening_seed_policy=base_plus_inner_fold_index.')
    if cfg['final_rounds_policy'] != 'median_inner_pilot_best_rounds_common_to_all_arms':
        raise ValueError('Unsupported final_rounds_policy; do not silently change experiment definition.')
    data_dir=args.data_dir.resolve(); config_dir=args.config.resolve().parent
    if args.device:
        cfg['device']=args.device
    seeds=[int(s) for s in (args.seeds if args.seeds else cfg['model_seeds'])]
    if len(set(seeds))!=len(seeds) or not seeds:
        raise ValueError('Seed list must be nonempty and unique.')
    seeds=sorted(seeds)
    if any(s<0 or s>2**31-1 for s in seeds):
        raise ValueError('Seeds outside supported integer range.')
    if not isinstance(cfg['include_chirality'],bool):
        raise ValueError('include_chirality must be explicitly true or false.')
    if int(xgb.__version__.split('.')[0])<2:
        raise RuntimeError('This implementation requires XGBoost >=2.0. Do not use old gpu_hist parameters.')
    if cfg['device'].startswith('cuda') and hasattr(xgb,'build_info'):
        info=xgb.build_info()
        if str(info.get('USE_CUDA',True)).lower() in ['false','0']:
            raise RuntimeError('Installed XGBoost was built without CUDA. Use the original GPU environment.')
    if cfg.get('matrix_type') != 'DMatrix':
        raise ValueError('This source-checked version uses matrix_type=DMatrix.')
    input_path=locate_data_file(cfg['input_csv'], data_dir, required=True)
    df=load_development(input_path,int(cfg['expected_development_n']))
    if cfg.get('descriptor_policy') != 'runtime_descList':
        raise ValueError('Expected descriptor_policy=runtime_descList to match original source.')
    names=[name for name, _ in Descriptors._descList]
    if not names or len(names) != len(set(names)):
        raise ValueError('Local RDKit descriptor registry is empty or has duplicate names.')
    # The attached list is a reference, not a substitute for the original registry rule.
    reference_path=config_dir/cfg['descriptor_names_file']
    reference_names=load_json(reference_path) if reference_path.is_file() else None
    descriptor_catalog=[{'index':i, 'name':name,
                         'function_version':str(getattr(fn, 'version', 'not_recorded'))}
                        for i,(name,fn) in enumerate(Descriptors._descList)]
    actual_dim=int(cfg['fingerprint_size'])+len(names)
    expected_dim=int(cfg['expected_raw_feature_dim'])
    if actual_dim != expected_dim:
        raise ValueError(
            f'Original source uses the full local RDKit Descriptors._descList. '
            f'This environment has {len(names)} descriptors / {actual_dim} raw columns; '
            f'historical summary reports {expected_dim}. '
            'Do not truncate the list or blindly upgrade packages. Restore the original '
            'RDKit environment, or document a deliberate new representation before training.')
    hist_path=locate_data_file(cfg['historical_summary_json'],data_dir,required=False)
    if hist_path is not None:
        hist=load_json(hist_path)
        mismatches={k:(v,hist.get('xgboost_params',{}).get(k))
                    for k,v in cfg['xgboost_params'].items()
                    if k in hist.get('xgboost_params',{}) and v!=hist['xgboost_params'][k]}
        if mismatches:
            raise ValueError(f'Configured parameters differ from historical summary: {mismatches}')
        hd=hist.get('feature_preprocess_final_train_full',{}).get('raw_dim')
        if hd is not None and hd != actual_dim:
            raise ValueError(f'Historical raw feature dimension {hd} != local {actual_dim}')
    versions={'python':sys.version,'platform':platform.platform(),'numpy':np.__version__,
              'pandas':pd.__version__,'sklearn':sklearn.__version__,
              'xgboost':xgb.__version__,'rdkit':rdkit.__version__}
    identity_cfg={k:v for k,v in cfg.items() if k not in
                  ['model_seeds','output_directory','verbose_eval','historical_summary_json']}
    identity={'schema':SCHEMA_VERSION,'mode':args.mode,'config':identity_cfg,
              'input_sha256':sha256_file(input_path),'descriptor_names':names,
              'descriptor_catalog':descriptor_catalog,
              'versions':versions,'code_sha256':sha256_file(Path(__file__)),
              'common_code_sha256':sha256_file(ROOT/'revision_common.py')}
    sig=object_hash(identity)
    out=data_dir/cfg['output_directory']/args.mode/('run_'+sig[:12])
    out.mkdir(parents=True,exist_ok=True)
    logging.basicConfig(level=logging.INFO,format='%(asctime)s | %(message)s',
                        handlers=[logging.StreamHandler(sys.stdout),
                                  logging.FileHandler(out/'run.log',encoding='utf-8')])
    LOG.info('Mode: %s | input rows: %d | seed list: %s',args.mode,len(df),seeds)
    LOG.info('Old OOF predictions/flags are ignored, including all historically excluded 448 compounds.')
    LOG.info('Features: %d counts + %d descriptors | include_chirality=%s',
             cfg['fingerprint_size'],len(names),cfg['include_chirality'])
    LOG.info(cfg['representation_note'])
    LOG.info('Resolved input: %s', input_path)
    LOG.info('Historical summary: %s', hist_path if hist_path is not None else 'not found; source-checked defaults used')
    LOG.info('Local descriptor order equals supplied REFERENCE list: %s', reference_names == names)
    LOG.info('Early stopping: patience=%s, maximum rounds=%s; DMatrix input',
             cfg['early_stopping_rounds'], cfg['max_boost_rounds'])
    LOG.info('Output: %s',out)
    LOG.info('Screening is nested; hyperparameters are frozen, not re-optimized by outer CV.')
    LOG.info('Fixed thresholds are compared; no best threshold is selected by this script.')
    save_json(out/'run_manifest.json',{'signature':sig,**identity,
               'protocol':'Outer folds untouched. Each inner OOF model chooses rounds with a separate pilot split '
                          'inside its inner-training rows, then refits that inner-training partition. '
                          'Final round count is the median of those pilot round counts, shared by all threshold arms. '
                          'Outer held-out labels are used only after prediction for evaluation.',
               'source_level_feature_definition_checked':True,
               'historical_package_versions_verified':False,
               'historical_feature_matrix_numerical_identity_verified':False,
               'local_descriptor_order_matches_reference':reference_names == names,
               'resolved_input_path':str(input_path),
               'resolved_historical_summary_path':str(hist_path) if hist_path is not None else None,
               'expected_historical_raw_feature_dim':2265,
               'actual_configured_raw_feature_dim':int(cfg['fingerprint_size']+len(names)),
               'old_oof_columns_ignored':[c for c in OLD_OOF_COLUMNS if c in df]})
    save_csv(out/'input_records_provenance_only.csv',df)
    save_json(out/'descriptor_names_used.json',names)
    save_json(out/'descriptor_catalog_used.json',descriptor_catalog)
    if args.preflight_only:
        LOG.info('Preflight complete. No features or models were computed.');return
    x,groups=build_raw_features(df,cfg,names,out,args.mode)
    indices=np.arange(len(df),dtype=np.int64)
    outer=cv_splits(indices,groups,args.mode,int(cfg['outer_folds']),int(cfg['split_seed']))
    fold_assign=np.zeros(len(df),dtype=int)
    for f,(tr,te) in enumerate(outer,1):
        fold_assign[te]=f
        if set(tr)&set(te) or len(tr)+len(te)!=len(df):
            raise AssertionError('Invalid outer partition.')
    assignment=df[['row_id','sample_id',SMILES,'source_type']].copy()
    assignment['outer_fold']=fold_assign;assignment['group_id']=groups
    save_csv(out/'fixed_outer_splits.csv',assignment)
    if args.prepare_only:
        LOG.info('Features and splits prepared. No model was fitted.');return
    result_dir=out/('results_seeds_'+'_'.join(map(str,seeds)))
    result_dir.mkdir(parents=True,exist_ok=True)
    save_json(result_dir/'ensemble_definition.json',{'seeds':seeds,'n_models_per_arm':len(seeds),
            'aggregation':'arithmetic mean of y predictions; then back-transform dose',
            'single_seed_result':len(seeds)==1,'threshold_selection':'none; all arms reported'})
    y=df['y'].to_numpy(dtype=np.float64)
    fold_rows=[];single_rows=[];prediction_tables=[]
    thresholds=[float(t) for t in cfg['thresholds']]
    if len(set(thresholds))!=len(thresholds) or any(t<=1 for t in thresholds):
        raise ValueError('Thresholds must be unique and greater than 1.')
    for f,(otr,ote) in enumerate(outer,1):
        LOG.info('===== OUTER FOLD %d/%d: train=%d, untouched test=%d =====',
                 f,len(outer),len(otr),len(ote))
        fold_dir=out/f'outer_fold_{f:02d}'
        fold_dir.mkdir(exist_ok=True)
        inner=cv_splits(otr,groups,args.mode,int(cfg['inner_folds']),int(cfg['split_seed'])+1000*f)
        inner_pred=np.full(len(df),np.nan,dtype=np.float64)
        inner_fold=np.zeros(len(df),dtype=int)
        best_rounds=[]
        for j,(itr,ival) in enumerate(inner,1):
            if set(itr)&set(ote) or set(ival)&set(ote) or set(itr)&set(ival):
                raise AssertionError('Outer test contamination in screening.')
            task=fold_dir/f'inner_fold_{j:02d}'
            pilot_tr,pilot_va=pilot_split(itr,groups,args.mode,
                     float(cfg['pilot_validation_fraction']),int(cfg['split_seed'])+10000*f+j)
            if set(pilot_tr)&set(ival) or set(pilot_va)&set(ival):
                raise AssertionError('Inner OOF labels would leak into early stopping.')
            screen_seed=int(cfg['screening_seed'])+j
            rounds=select_rounds(x,y,pilot_tr,pilot_va,cfg,screen_seed,task/'pilot',sig)
            best_rounds.append(rounds)
            p=fit_predict(x,y[itr],itr,ival,cfg,screen_seed,rounds,
                          task/'oof_refit',sig)
            inner_pred[ival]=p;inner_fold[ival]=j
        if not np.isfinite(inner_pred[otr]).all() or np.isfinite(inner_pred[ote]).any():
            raise AssertionError('Incomplete inner OOF, or outer test accidentally screened.')
        ae=np.abs(y[otr]-inner_pred[otr])
        if ae.max()>300:
            raise ValueError('Implausible inner OOF log error; check units.')
        screening=df.iloc[otr][['row_id','sample_id',SMILES,'source_type','y']].copy()
        screening['inner_fold']=inner_fold[otr]
        screening['new_inner_oof_prediction']=inner_pred[otr]
        screening['new_abs_log_error']=ae
        screening['new_fold_error']=np.power(10.,ae)
        for t in thresholds:
            screening[f'remove_ge_{t:g}fold']=ae>=np.log10(t)
        save_csv(fold_dir/'training_only_screening.csv',screening)
        final_rounds=int(np.rint(np.median(best_rounds)))
        final_rounds=max(1,min(int(cfg['max_boost_rounds']),final_rounds))
        save_json(fold_dir/'common_round_count.json',{'inner_pilot_best_rounds':best_rounds,
                  'final_num_boost_round':final_rounds,'policy':cfg['final_rounds_policy']})
        arms=[('no_filter',None)]+[(f'filter_{t:g}fold',t) for t in thresholds]
        for arm,t in arms:
            retain=np.ones(len(otr),dtype=bool) if t is None else ae<np.log10(t)
            used=otr[retain]
            if len(used)<2:
                raise ValueError(f'{arm} retained too few samples; no fallback threshold is allowed.')
            removed=int(len(otr)-len(used));per_seed=[]
            for seed in seeds:
                p=fit_predict(x,y[used],used,ote,cfg,seed,final_rounds,
                              fold_dir/arm/f'seed_{seed}',sig)
                per_seed.append(p)
                single_rows.append({'mode':args.mode,'outer_fold':f,'arm':arm,'seed':seed,
                                    'n_train_after':len(used),'removed':removed,
                                    **regression_metrics(y[ote],p)})
            avg=np.mean(np.vstack(per_seed),axis=0)
            m=regression_metrics(y[ote],avg)
            fold_rows.append({'mode':args.mode,'outer_fold':f,'arm':arm,
                              'n_train_before':len(otr),'n_train_after':len(used),
                              'removed':removed,'removed_percent':100*removed/len(otr),
                              'n_model_seeds':len(seeds),'num_boost_round':final_rounds,**m})
            pp=df.iloc[ote][['row_id','sample_id',SMILES,'source_type','ld50_mgkg','molecular_weight']].copy()
            pp['outer_fold']=f;pp['group_id']=groups[ote];pp['arm']=arm
            pp['y_true']=y[ote];pp['y_pred']=avg
            pp['residual']=avg-y[ote];pp['fold_error']=10**np.abs(avg-y[ote])
            pp['predicted_ld50_mgkg']=1000*pp['molecular_weight'].to_numpy()*10**(-avg)
            for seed,p in zip(seeds,per_seed):
                pp[f'prediction_seed_{seed}']=p
            prediction_tables.append(pp)
            write_summaries(result_dir,fold_rows,single_rows,prediction_tables,len(outer),len(df))
            LOG.info('Fold %d | %s | removed=%d | R2=%.6f RMSE=%.6f MAE=%.6f',
                     f,arm,removed,m['R2'],m['RMSE'],m['MAE'])
    final=read_csv(result_dir/'pooled_and_fold_summary.csv')
    LOG.info('===== ALL OUTER FOLDS COMPLETE =====\n%s',final.to_string(index=False))
    LOG.info('Results: %s',result_dir)
    LOG.info('Do not choose a winning threshold on these test results and call that selection independently validated.')


if __name__=='__main__':
    try:
        main()
    except KeyboardInterrupt:
        print('\nInterrupted. Completed fits and feature blocks are saved. Rerun the same command to resume.',file=sys.stderr)
        sys.exit(130)
    except Exception:
        logging.exception('Stopped. No rows/thresholds/parameters were silently changed.')
        raise
