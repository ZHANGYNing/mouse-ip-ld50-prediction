# -*- coding: utf-8 -*-
"""
JHM revision: scaffold-based nested cross-validation for mouse i.p. LD50.

Purpose
-------
Answer Reviewer 2's request for scaffold-based evaluation with residual
screening nested inside cross-validation.

Protocol
--------
1) Start from the rule-cleaned 34,959-compound development pool only
   (TOXRIC_only + mixed). PubChem-only compounds are NOT used here.
2) Build deterministic 5-fold OUTER splits by Bemis-Murcko scaffold.
3) For each outer fold, keep the outer test fold completely untouched.
4) On the outer-training compounds only, generate 5-fold RANDOM OOF
   predictions using the original screening logic.
5) Compare four arms on the SAME outer test compounds:
      no_filter, filter_10fold, filter_20fold, filter_30fold
   A filtered arm removes only outer-training compounds whose nested OOF
   fold-error >= the corresponding threshold.
6) Fit preprocessing only on the retained outer-training compounds, then
   transform the untouched outer test fold.
7) Train one seed-42 XGBoost model per arm/fold. The number of boosting
   rounds is the median of the five inner-OOF best iterations for that
   outer fold, so the four arms share the same round count within an outer
   fold and the outer test fold is never used for early stopping.
8) Save fold metrics, pooled metrics, mean +/- SD, removal records,
   scaffold assignments, and compound-level predictions.

Important terminology
---------------------
- "10fold/20fold/30fold" below means 10x/20x/30x residual threshold,
  NOT 10-fold/20-fold/30-fold cross-validation.
- Inner OOF screening deliberately remains random 5-fold to preserve the
  original OOF screening method. The requested structural generalization
  test is implemented in the OUTER scaffold split.
- For molecules with an empty Bemis-Murcko scaffold (acyclic molecules),
  a compound-specific key is used instead of putting every acyclic molecule
  into one giant scaffold group. This rule is recorded in the manifest.

Default Windows location follows the user's existing project layout.
You can run this file directly in PyCharm.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import random
import sys
import time
from pathlib import Path
from typing import Dict, Iterable, List, Tuple

import joblib
import numpy as np
import pandas as pd
from rdkit import Chem, DataStructs, RDLogger, rdBase
from rdkit.Chem import Descriptors, rdFingerprintGenerator
from rdkit.Chem.Scaffolds import MurckoScaffold
from sklearn.feature_selection import VarianceThreshold
from sklearn.impute import SimpleImputer
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import KFold
import sklearn
import xgboost as xgb


# ============================================================================
# 1. User-editable defaults
# ============================================================================

DEFAULT_BASE_DIR = Path(r"C:\Users\Admin\Desktop\新建文件夹 (2)\LD50mouse")
DEFAULT_OUTPUT_NAME = "JHM_scaffold_nested_cv"

OUTER_FOLDS = 5
INNER_FOLDS = 5
THRESHOLDS = [10.0, 20.0, 30.0]
SEED = 42

ECFP_RADIUS = 2
ECFP_BITS = 2048
EXPECTED_RAW_DIM = 2265  # 2048 ECFP + 217 RDKit2D in the original environment

NUM_BOOST_ROUND_MAX = 7000
EARLY_STOPPING_ROUNDS = 250
SAVE_MODELS = False

XGB_PARAMS = {
    "objective": "reg:squarederror",
    "eval_metric": "rmse",
    "tree_method": "hist",
    "max_depth": 9,
    "eta": 0.02,
    "subsample": 0.75,
    "colsample_bytree": 0.85,
    "min_child_weight": 2,
    "gamma": 0.01,
    "lambda": 0.5,
    "alpha": 0,
    "max_bin": 256,
    "seed": SEED,
    "device": "cuda",
}

RDLogger.DisableLog("rdApp.warning")


# ============================================================================
# 2. Small utilities
# ============================================================================

def set_seed(seed: int = SEED) -> None:
    random.seed(seed)
    np.random.seed(seed)


def to_jsonable(obj):
    if isinstance(obj, dict):
        return {str(k): to_jsonable(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return [to_jsonable(v) for v in obj]
    if isinstance(obj, np.ndarray):
        return obj.tolist()
    if isinstance(obj, (np.floating,)):
        return float(obj)
    if isinstance(obj, (np.integer,)):
        return int(obj)
    if isinstance(obj, Path):
        return str(obj)
    return obj


def sha256_file(path: Path, block_size: int = 1024 * 1024) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        while True:
            block = f.read(block_size)
            if not block:
                break
            h.update(block)
    return h.hexdigest()


def calc_metrics(y_true: np.ndarray, y_pred: np.ndarray) -> Dict[str, float]:
    y_true = np.asarray(y_true, dtype=float)
    y_pred = np.asarray(y_pred, dtype=float)
    return {
        "R2": float(r2_score(y_true, y_pred)),
        "RMSE": float(math.sqrt(mean_squared_error(y_true, y_pred))),
        "MAE": float(mean_absolute_error(y_true, y_pred)),
    }


def calc_fold_error_metrics(y_true: np.ndarray, y_pred: np.ndarray) -> Dict[str, float]:
    # Because y = -log10(LD50 in mol/kg), the fold discrepancy is exactly
    # 10**abs(y_true-y_pred). Molecular weight cancels in the ratio.
    abs_err = np.abs(np.asarray(y_true, dtype=float) - np.asarray(y_pred, dtype=float))
    fold_error = np.power(10.0, abs_err)
    return {
        "median_fold_error": float(np.median(fold_error)),
        "mean_fold_error": float(np.mean(fold_error)),
        "within_2_fold_percent": float(np.mean(fold_error <= 2.0) * 100.0),
        "within_5_fold_percent": float(np.mean(fold_error <= 5.0) * 100.0),
        "within_10_fold_percent": float(np.mean(fold_error <= 10.0) * 100.0),
    }


def safe_read_csv(path: Path) -> pd.DataFrame:
    return pd.read_csv(path, low_memory=False)


# ============================================================================
# 3. Find and validate the 34,959-compound development pool
# ============================================================================

def candidate_input_paths(base_dir: Path) -> List[Path]:
    preferred = [
        base_dir / "mouse_intraperitoneal_LD50_strict_cleaned" /
        "no_leakage_pubchem_external_validation" /
        "train_pool_toxric_mixed_before_oof.csv",
        base_dir / "train_pool_toxric_mixed_before_oof.csv",
    ]
    seen = set()
    out: List[Path] = []
    for p in preferred:
        if p.is_file() and p.resolve() not in seen:
            out.append(p)
            seen.add(p.resolve())
    for p in base_dir.rglob("train_pool_toxric_mixed_before_oof.csv"):
        try:
            rp = p.resolve()
        except OSError:
            continue
        if p.is_file() and rp not in seen:
            out.append(p)
            seen.add(rp)
    return out


def validate_development_df(df: pd.DataFrame, path: Path) -> None:
    required = {"Canonical SMILES", "y"}
    missing = required - set(df.columns)
    if missing:
        raise ValueError(f"{path} 缺少必要列: {sorted(missing)}")
    if len(df) != 34959:
        raise ValueError(f"开发池应为 34,959 个化合物，但 {path} 有 {len(df):,} 行。")
    if df["Canonical SMILES"].isna().any():
        raise ValueError("开发池存在空 Canonical SMILES。")
    if df["Canonical SMILES"].duplicated().any():
        ndup = int(df["Canonical SMILES"].duplicated().sum())
        raise ValueError(f"开发池 Canonical SMILES 有 {ndup} 个重复行。")
    y = pd.to_numeric(df["y"], errors="coerce")
    if y.isna().any() or not np.isfinite(y.to_numpy(float)).all():
        raise ValueError("开发池 y 列存在无效值。")
    if "source_type" in df.columns:
        bad = sorted(set(df["source_type"].dropna().astype(str)) - {"TOXRIC_only", "mixed"})
        if bad:
            raise ValueError(f"开发池出现不应存在的 source_type: {bad}")


def resolve_input(base_dir: Path, explicit_input: str | None) -> Path:
    if explicit_input:
        p = Path(explicit_input)
        if not p.is_file():
            raise FileNotFoundError(f"指定输入文件不存在: {p}")
        df = safe_read_csv(p)
        validate_development_df(df, p)
        return p

    candidates = candidate_input_paths(base_dir)
    valid: List[Tuple[Path, str]] = []
    errors = []
    for p in candidates:
        try:
            df = safe_read_csv(p)
            validate_development_df(df, p)
            valid.append((p, sha256_file(p)))
        except Exception as exc:
            errors.append((p, str(exc)))

    if not valid:
        msg = ["没有找到可用的 34,959-compound development pool."]
        if errors:
            msg.append("找到过同名候选，但未通过检查:")
            msg.extend([f"  - {p}: {e}" for p, e in errors])
        raise FileNotFoundError("\n".join(msg))

    hashes = {h for _, h in valid}
    if len(hashes) > 1:
        msg = ["找到多份内容不同、但都看起来有效的开发池文件。为避免跑错，请用 --input 显式指定一份:"]
        msg.extend([f"  - {p}  sha256={h}" for p, h in valid])
        raise RuntimeError("\n".join(msg))

    return valid[0][0]


# ============================================================================
# 4. Bemis-Murcko scaffold construction and deterministic outer folds
# ============================================================================

def scaffold_key(smiles: str) -> Tuple[str, bool]:
    mol = Chem.MolFromSmiles(str(smiles))
    if mol is None:
        raise ValueError(f"RDKit cannot parse development SMILES: {smiles}")
    scaffold = MurckoScaffold.GetScaffoldForMol(mol)
    smi = Chem.MolToSmiles(scaffold, canonical=True, isomericSmiles=False)
    if smi:
        return f"BM::{smi}", False

    # Acyclic molecules have an empty Murcko scaffold. Treat them as
    # compound-specific groups to avoid putting all acyclic molecules into
    # one huge fold. Canonical non-isomeric SMILES is used for determinism.
    canonical = Chem.MolToSmiles(mol, canonical=True, isomericSmiles=False)
    return f"ACYCLIC::{canonical}", True


def assign_groups_to_folds(group_keys: Iterable[str], n_splits: int = 5) -> Dict[str, int]:
    series = pd.Series(list(group_keys), dtype=str)
    counts = series.value_counts().to_dict()
    groups = sorted(counts.items(), key=lambda kv: (-kv[1], kv[0]))

    fold_sizes = [0 for _ in range(n_splits)]
    group_to_fold: Dict[str, int] = {}
    for key, size in groups:
        fold = min(range(n_splits), key=lambda f: (fold_sizes[f], f))
        group_to_fold[key] = fold + 1  # 1..5 for human-readable output
        fold_sizes[fold] += int(size)
    return group_to_fold


def build_scaffold_assignments(df: pd.DataFrame, output_dir: Path) -> pd.DataFrame:
    cache = output_dir / "scaffold_assignments.csv"
    if cache.is_file():
        cached = pd.read_csv(cache, low_memory=False)
        if len(cached) == len(df) and set(cached["Canonical SMILES"]) == set(df["Canonical SMILES"]):
            print(f"复用 scaffold assignment: {cache}")
            # Reorder to exactly match df.
            return df[["Canonical SMILES"]].merge(cached, on="Canonical SMILES", how="left", validate="one_to_one")

    print("计算 Bemis-Murcko scaffolds ...")
    keys: List[str] = []
    acyclic_flags: List[bool] = []
    for i, smi in enumerate(df["Canonical SMILES"].astype(str), start=1):
        key, is_acyclic = scaffold_key(smi)
        keys.append(key)
        acyclic_flags.append(is_acyclic)
        if i % 5000 == 0 or i == len(df):
            print(f"  scaffold: {i:,}/{len(df):,}")

    group_to_fold = assign_groups_to_folds(keys, OUTER_FOLDS)
    out = pd.DataFrame({
        "Canonical SMILES": df["Canonical SMILES"].astype(str).values,
        "scaffold_key": keys,
        "is_acyclic_empty_murcko": acyclic_flags,
        "outer_fold": [group_to_fold[k] for k in keys],
    })
    out.to_csv(cache, index=False, encoding="utf-8-sig")
    return out


def validate_outer_scaffold_splits(assignments: pd.DataFrame) -> Dict[str, object]:
    summary: Dict[str, object] = {
        "n_compounds": int(len(assignments)),
        "n_scaffold_groups_total": int(assignments["scaffold_key"].nunique()),
        "n_acyclic_compounds": int(assignments["is_acyclic_empty_murcko"].sum()),
        "n_nonempty_murcko_scaffolds": int(
            assignments.loc[~assignments["is_acyclic_empty_murcko"], "scaffold_key"].nunique()
        ),
        "folds": [],
    }
    for fold in range(1, OUTER_FOLDS + 1):
        te = assignments[assignments["outer_fold"] == fold]
        tr = assignments[assignments["outer_fold"] != fold]
        tr_nonempty = set(tr.loc[~tr["is_acyclic_empty_murcko"], "scaffold_key"])
        te_nonempty = set(te.loc[~te["is_acyclic_empty_murcko"], "scaffold_key"])
        overlap = tr_nonempty & te_nonempty
        if overlap:
            raise RuntimeError(f"Outer fold {fold}: non-empty Murcko scaffold leakage: {len(overlap)} groups")
        summary["folds"].append({
            "outer_fold": fold,
            "n_train": int(len(tr)),
            "n_test": int(len(te)),
            "n_test_scaffold_groups": int(te["scaffold_key"].nunique()),
            "n_test_acyclic": int(te["is_acyclic_empty_murcko"].sum()),
            "nonempty_scaffold_overlap_count": 0,
        })
    return summary


# ============================================================================
# 5. Original molecular representation
# ============================================================================

MORGAN_GENERATOR = rdFingerprintGenerator.GetMorganGenerator(
    radius=ECFP_RADIUS,
    fpSize=ECFP_BITS,
    includeChirality=False,
)
RDKit_2D_DESC_LIST = Descriptors._descList
RDKit_2D_NAMES = [name for name, _ in RDKit_2D_DESC_LIST]


def smiles_to_raw_feature(smiles: str) -> np.ndarray:
    mol = Chem.MolFromSmiles(str(smiles))
    if mol is None:
        raise ValueError(f"RDKit cannot parse SMILES during feature generation: {smiles}")

    ecfp = np.zeros((ECFP_BITS,), dtype=np.float32)
    fp = MORGAN_GENERATOR.GetCountFingerprint(mol)
    DataStructs.ConvertToNumpyArray(fp, ecfp)
    ecfp = np.log1p(ecfp).astype(np.float32)

    # Some RDKit descriptors can be finite in float64 but overflow when
    # cast to float32 (for example, extremely large Ipc-like values).
    # The original final-training script converted non-finite raw values
    # to NaN before imputation. Reproduce that behavior explicitly here.
    desc = np.empty((len(RDKit_2D_DESC_LIST),), dtype=np.float32)
    f32_max = np.finfo(np.float32).max
    for j, (_, func) in enumerate(RDKit_2D_DESC_LIST):
        try:
            v = float(func(mol))
            if np.isfinite(v) and abs(v) <= f32_max:
                desc[j] = np.float32(v)
            else:
                desc[j] = np.nan
        except Exception:
            desc[j] = np.nan

    raw = np.concatenate([ecfp, desc]).astype(np.float32)
    raw[~np.isfinite(raw)] = np.nan
    return raw


def build_or_load_raw_features(df: pd.DataFrame, output_dir: Path, input_sha256: str) -> np.ndarray:
    feature_file = output_dir / "raw_features_ecfp4count_rdkit2d.npy"
    meta_file = output_dir / "raw_features_meta.json"

    if feature_file.is_file() and meta_file.is_file():
        try:
            meta = json.loads(meta_file.read_text(encoding="utf-8-sig"))
            if (
                meta.get("input_sha256") == input_sha256
                and meta.get("n_compounds") == len(df)
                and meta.get("raw_dim") == EXPECTED_RAW_DIM
            ):
                X = np.load(feature_file, mmap_mode="r")
                if X.shape == (len(df), EXPECTED_RAW_DIM):
                    print(f"复用 raw feature cache: {feature_file}")
                    return X
        except Exception:
            pass

    raw_dim = ECFP_BITS + len(RDKit_2D_DESC_LIST)
    print(f"当前 RDKit descriptors: {len(RDKit_2D_DESC_LIST)}; raw_dim={raw_dim}")
    if raw_dim != EXPECTED_RAW_DIM:
        raise RuntimeError(
            f"当前环境 raw feature 维度为 {raw_dim}，但原模型应为 {EXPECTED_RAW_DIM}.\n"
            "为避免与原模型特征定义不一致，程序停止。请确认 RDKit 版本与原实验环境。"
        )

    X = np.empty((len(df), raw_dim), dtype=np.float32)
    for i, smi in enumerate(df["Canonical SMILES"].astype(str), start=1):
        X[i - 1] = smiles_to_raw_feature(smi)
        if i % 1000 == 0 or i == len(df):
            print(f"  features: {i:,}/{len(df):,}")

    np.save(feature_file, X)
    meta = {
        "input_sha256": input_sha256,
        "n_compounds": int(len(df)),
        "raw_dim": int(raw_dim),
        "ecfp_radius": ECFP_RADIUS,
        "ecfp_bits": ECFP_BITS,
        "include_chirality": False,
        "fingerprint_type": "Morgan count fingerprint, log1p",
        "rdkit2d_count": len(RDKit_2D_DESC_LIST),
        "rdkit2d_names": RDKit_2D_NAMES,
        "rdkit_version": rdBase.rdkitVersion,
    }
    meta_file.write_text(json.dumps(to_jsonable(meta), ensure_ascii=False, indent=2), encoding="utf-8")
    return np.load(feature_file, mmap_mode="r")


# ============================================================================
# 6. Leakage-safe preprocessing
# ============================================================================

def _sanitize_raw_matrix(X: np.ndarray) -> Tuple[np.ndarray, int]:
    """Match the original pipeline: replace +/-inf or float32 overflow with NaN."""
    X = np.asarray(X, dtype=np.float32).copy()
    bad = ~np.isfinite(X)
    n_bad = int(np.count_nonzero(bad))
    if n_bad:
        X[bad] = np.nan
    return X, n_bad


def fit_preprocess(X_train_raw: np.ndarray, X_other_raw: List[np.ndarray]):
    # IMPORTANT: the original final-training code sanitized raw features with
    # np.where(np.isfinite(X_raw), X_raw, np.nan) before median imputation.
    # Do the same here so very large RDKit descriptor values cannot reach
    # sklearn's SimpleImputer as +/-inf.
    X_train_raw, n_bad_train = _sanitize_raw_matrix(X_train_raw)
    keep_cols = ~np.all(np.isnan(X_train_raw), axis=0)
    X_train = X_train_raw[:, keep_cols]

    imputer = SimpleImputer(strategy="median")
    X_train_imp = imputer.fit_transform(X_train).astype(np.float32)

    selector = VarianceThreshold(threshold=0.0)
    X_train_final = selector.fit_transform(X_train_imp).astype(np.float32)

    outputs = [X_train_final]
    n_bad_other = []
    for X in X_other_raw:
        X, n_bad = _sanitize_raw_matrix(X)
        n_bad_other.append(n_bad)
        X = X[:, keep_cols]
        X = imputer.transform(X).astype(np.float32)
        X = selector.transform(X).astype(np.float32)
        outputs.append(X)

    info = {
        "raw_dim": int(X_train_raw.shape[1]),
        "after_all_nan_filter_dim": int(np.sum(keep_cols)),
        "after_variance_filter_dim": int(X_train_final.shape[1]),
        "nonfinite_raw_values_replaced_with_nan_train": n_bad_train,
        "nonfinite_raw_values_replaced_with_nan_other": n_bad_other,
    }
    return outputs, info, keep_cols, imputer, selector


# ============================================================================
# 7. XGBoost training helpers
# ============================================================================

def train_early_stopping(
    X_train: np.ndarray,
    y_train: np.ndarray,
    X_valid: np.ndarray,
    y_valid: np.ndarray,
    seed: int,
):
    params = dict(XGB_PARAMS)
    params["seed"] = int(seed)
    dtrain = xgb.DMatrix(X_train, label=y_train)
    dvalid = xgb.DMatrix(X_valid, label=y_valid)

    actual_device = "cuda"
    try:
        booster = xgb.train(
            params,
            dtrain,
            num_boost_round=NUM_BOOST_ROUND_MAX,
            evals=[(dtrain, "train"), (dvalid, "valid")],
            early_stopping_rounds=EARLY_STOPPING_ROUNDS,
            verbose_eval=False,
        )
    except Exception as exc:
        print(f"GPU training failed; fallback CPU: {str(exc)[:250]}")
        actual_device = "cpu"
        params.pop("device", None)
        params["tree_method"] = "hist"
        booster = xgb.train(
            params,
            dtrain,
            num_boost_round=NUM_BOOST_ROUND_MAX,
            evals=[(dtrain, "train"), (dvalid, "valid")],
            early_stopping_rounds=EARLY_STOPPING_ROUNDS,
            verbose_eval=False,
        )

    best_iter = int(booster.best_iteration) + 1
    pred = booster.predict(dvalid, iteration_range=(0, best_iter))
    return pred.astype(np.float64), best_iter, actual_device


def train_fixed_round(
    X_train: np.ndarray,
    y_train: np.ndarray,
    X_test: np.ndarray,
    rounds: int,
    seed: int,
):
    params = dict(XGB_PARAMS)
    params["seed"] = int(seed)
    dtrain = xgb.DMatrix(X_train, label=y_train)
    dtest = xgb.DMatrix(X_test)

    actual_device = "cuda"
    try:
        booster = xgb.train(
            params,
            dtrain,
            num_boost_round=int(rounds),
            evals=[(dtrain, "train")],
            verbose_eval=False,
        )
    except Exception as exc:
        print(f"GPU training failed; fallback CPU: {str(exc)[:250]}")
        actual_device = "cpu"
        params.pop("device", None)
        params["tree_method"] = "hist"
        booster = xgb.train(
            params,
            dtrain,
            num_boost_round=int(rounds),
            evals=[(dtrain, "train")],
            verbose_eval=False,
        )

    pred = booster.predict(dtest).astype(np.float64)
    return booster, pred, actual_device


# ============================================================================
# 8. Nested OOF residual screening inside ONE outer-training fold
# ============================================================================

def generate_inner_oof(
    outer_fold: int,
    outer_train_global_idx: np.ndarray,
    X_raw: np.ndarray,
    y_all: np.ndarray,
    df: pd.DataFrame,
    fold_dir: Path,
):
    pred_file = fold_dir / "inner_oof_predictions.csv"
    meta_file = fold_dir / "inner_oof_summary.json"
    fold_metrics_file = fold_dir / "inner_oof_fold_metrics.csv"

    if pred_file.is_file() and meta_file.is_file() and fold_metrics_file.is_file():
        p = pd.read_csv(pred_file, low_memory=False)
        meta = json.loads(meta_file.read_text(encoding="utf-8-sig"))
        if len(p) == len(outer_train_global_idx) and int(meta.get("n_outer_train", -1)) == len(outer_train_global_idx):
            print(f"Outer fold {outer_fold}: reuse inner OOF screening")
            # Ensure order by local_index.
            p = p.sort_values("local_index")
            return (
                p["oof_pred_y"].to_numpy(float),
                p["oof_fold_error"].to_numpy(float),
                [int(v) for v in meta["best_iterations"]],
                pd.read_csv(fold_metrics_file),
            )

    print(f"Outer fold {outer_fold}: nested inner {INNER_FOLDS}-fold OOF screening ...")
    n = len(outer_train_global_idx)
    y_outer = y_all[outer_train_global_idx]
    oof_pred = np.full(n, np.nan, dtype=np.float64)
    inner_fold_id = np.zeros(n, dtype=int)
    best_iterations: List[int] = []
    records = []

    # Preserve the original OOF screening style: random KFold inside outer train.
    kf = KFold(n_splits=INNER_FOLDS, shuffle=True, random_state=SEED)
    for inner_fold, (tr_local, va_local) in enumerate(kf.split(np.arange(n)), start=1):
        tr_global = outer_train_global_idx[tr_local]
        va_global = outer_train_global_idx[va_local]

        X_tr_raw = np.asarray(X_raw[tr_global], dtype=np.float32)
        X_va_raw = np.asarray(X_raw[va_global], dtype=np.float32)
        (Xs, prep_info, _, _, _) = fit_preprocess(X_tr_raw, [X_va_raw])
        X_tr, X_va = Xs

        pred, best_iter, actual_device = train_early_stopping(
            X_tr,
            y_all[tr_global],
            X_va,
            y_all[va_global],
            seed=SEED + inner_fold,
        )
        oof_pred[va_local] = pred
        inner_fold_id[va_local] = inner_fold
        best_iterations.append(best_iter)

        m = calc_metrics(y_all[va_global], pred)
        f = calc_fold_error_metrics(y_all[va_global], pred)
        rec = {
            "outer_fold": outer_fold,
            "inner_fold": inner_fold,
            "n_train": int(len(tr_global)),
            "n_valid": int(len(va_global)),
            "best_iteration": int(best_iter),
            "actual_device": actual_device,
            "feature_dim": int(prep_info["after_variance_filter_dim"]),
            **m,
            **f,
        }
        records.append(rec)
        print(
            f"  inner {inner_fold}/{INNER_FOLDS}: n_valid={len(va_global):,}, "
            f"R2={m['R2']:.4f}, RMSE={m['RMSE']:.4f}, best_iter={best_iter}"
        )

    if np.isnan(oof_pred).any():
        raise RuntimeError(f"Outer fold {outer_fold}: inner OOF predictions incomplete")

    fold_error = np.power(10.0, np.abs(y_outer - oof_pred))
    pred_df = pd.DataFrame({
        "local_index": np.arange(n),
        "global_index": outer_train_global_idx,
        "Canonical SMILES": df.iloc[outer_train_global_idx]["Canonical SMILES"].astype(str).values,
        "y_true": y_outer,
        "oof_pred_y": oof_pred,
        "oof_abs_error_y": np.abs(y_outer - oof_pred),
        "oof_fold_error": fold_error,
        "inner_fold": inner_fold_id,
    })
    pred_df.to_csv(pred_file, index=False, encoding="utf-8-sig")
    pd.DataFrame(records).to_csv(fold_metrics_file, index=False, encoding="utf-8-sig")

    pooled = calc_metrics(y_outer, oof_pred)
    pooled.update(calc_fold_error_metrics(y_outer, oof_pred))
    meta = {
        "outer_fold": outer_fold,
        "n_outer_train": int(n),
        "inner_folds": INNER_FOLDS,
        "inner_split": "random KFold(shuffle=True, random_state=42), nested inside outer training only",
        "best_iterations": best_iterations,
        "median_best_iteration": int(round(float(np.median(best_iterations)))),
        "pooled_inner_oof_metrics": pooled,
        "threshold_removal_counts": {
            str(int(t)): int(np.sum(fold_error >= t)) for t in THRESHOLDS
        },
    }
    meta_file.write_text(json.dumps(to_jsonable(meta), ensure_ascii=False, indent=2), encoding="utf-8")
    return oof_pred, fold_error, best_iterations, pd.DataFrame(records)


# ============================================================================
# 9. Run one outer fold
# ============================================================================

def run_outer_fold(
    outer_fold: int,
    df: pd.DataFrame,
    assignments: pd.DataFrame,
    X_raw: np.ndarray,
    y_all: np.ndarray,
    output_dir: Path,
):
    fold_dir = output_dir / f"outer_fold_{outer_fold}"
    fold_dir.mkdir(parents=True, exist_ok=True)

    test_mask = assignments["outer_fold"].to_numpy(int) == outer_fold
    te_idx = np.where(test_mask)[0]
    tr_idx = np.where(~test_mask)[0]

    # Save exact outer split.
    split_df = pd.DataFrame({
        "global_index": np.concatenate([tr_idx, te_idx]),
        "split": ["train"] * len(tr_idx) + ["test"] * len(te_idx),
    })
    split_df = split_df.merge(
        pd.DataFrame({
            "global_index": np.arange(len(df)),
            "Canonical SMILES": df["Canonical SMILES"].astype(str).values,
            "y": y_all,
            "scaffold_key": assignments["scaffold_key"].values,
            "is_acyclic_empty_murcko": assignments["is_acyclic_empty_murcko"].values,
        }),
        on="global_index",
        how="left",
        validate="one_to_one",
    )
    split_df.to_csv(fold_dir / "outer_split.csv", index=False, encoding="utf-8-sig")

    # Nested screening uses ONLY outer training data.
    _, inner_fold_error, best_iterations, _ = generate_inner_oof(
        outer_fold, tr_idx, X_raw, y_all, df, fold_dir
    )
    common_rounds = int(max(1, min(NUM_BOOST_ROUND_MAX, round(float(np.median(best_iterations))))))
    print(f"Outer fold {outer_fold}: common final rounds={common_rounds}")

    arms = [("no_filter", None)] + [(f"filter_{int(t)}fold", t) for t in THRESHOLDS]
    fold_records = []

    for arm, threshold in arms:
        result_file = fold_dir / f"result_{arm}.json"
        pred_file = fold_dir / f"predictions_{arm}.csv"

        if result_file.is_file() and pred_file.is_file():
            try:
                rec = json.loads(result_file.read_text(encoding="utf-8-sig"))
                p = pd.read_csv(pred_file, low_memory=False)
                if len(p) == len(te_idx) and int(rec.get("n_test", -1)) == len(te_idx):
                    print(f"Outer fold {outer_fold} / {arm}: reuse completed result")
                    fold_records.append(rec)
                    continue
            except Exception:
                pass

        if threshold is None:
            keep_local = np.ones(len(tr_idx), dtype=bool)
        else:
            keep_local = inner_fold_error < float(threshold)

        retained_tr_idx = tr_idx[keep_local]
        removed_tr_idx = tr_idx[~keep_local]

        removed = pd.DataFrame({
            "global_index": removed_tr_idx,
            "Canonical SMILES": df.iloc[removed_tr_idx]["Canonical SMILES"].astype(str).values,
            "y": y_all[removed_tr_idx],
            "scaffold_key": assignments.iloc[removed_tr_idx]["scaffold_key"].values,
            "nested_oof_fold_error": inner_fold_error[~keep_local] if threshold is not None else np.array([], dtype=float),
            "threshold_fold": threshold,
        })
        removed.to_csv(fold_dir / f"removed_{arm}.csv", index=False, encoding="utf-8-sig")

        X_tr_raw = np.asarray(X_raw[retained_tr_idx], dtype=np.float32)
        X_te_raw = np.asarray(X_raw[te_idx], dtype=np.float32)
        (Xs, prep_info, keep_cols, imputer, selector) = fit_preprocess(X_tr_raw, [X_te_raw])
        X_tr, X_te = Xs

        # Save small preprocessing objects for exact reproducibility.
        prep_dir = fold_dir / f"preprocess_{arm}"
        prep_dir.mkdir(exist_ok=True)
        np.save(prep_dir / "keep_cols.npy", keep_cols)
        joblib.dump(imputer, prep_dir / "median_imputer.joblib")
        joblib.dump(selector, prep_dir / "variance_selector.joblib")
        (prep_dir / "preprocess_info.json").write_text(
            json.dumps(to_jsonable(prep_info), ensure_ascii=False, indent=2), encoding="utf-8"
        )

        booster, pred, actual_device = train_fixed_round(
            X_tr,
            y_all[retained_tr_idx],
            X_te,
            rounds=common_rounds,
            seed=SEED,
        )
        if SAVE_MODELS:
            booster.save_model(str(fold_dir / f"model_{arm}.ubj"))

        metrics = calc_metrics(y_all[te_idx], pred)
        fold_metrics = calc_fold_error_metrics(y_all[te_idx], pred)
        fold_error = np.power(10.0, np.abs(y_all[te_idx] - pred))

        pred_df = pd.DataFrame({
            "global_index": te_idx,
            "Canonical SMILES": df.iloc[te_idx]["Canonical SMILES"].astype(str).values,
            "outer_fold": outer_fold,
            "scaffold_key": assignments.iloc[te_idx]["scaffold_key"].values,
            "is_acyclic_empty_murcko": assignments.iloc[te_idx]["is_acyclic_empty_murcko"].values,
            "y_true": y_all[te_idx],
            "y_pred": pred,
            "fold_error": fold_error,
            "arm": arm,
            "threshold_fold": threshold,
        })
        pred_df.to_csv(pred_file, index=False, encoding="utf-8-sig")

        rec = {
            "outer_fold": outer_fold,
            "arm": arm,
            "threshold_fold": threshold,
            "n_outer_train_before_screening": int(len(tr_idx)),
            "n_train_retained": int(len(retained_tr_idx)),
            "n_train_removed": int(len(removed_tr_idx)),
            "removed_percent": float(len(removed_tr_idx) / len(tr_idx) * 100.0),
            "n_test": int(len(te_idx)),
            "common_rounds": int(common_rounds),
            "final_seed": SEED,
            "actual_device": actual_device,
            "feature_dim": int(prep_info["after_variance_filter_dim"]),
            **metrics,
            **fold_metrics,
        }
        result_file.write_text(json.dumps(to_jsonable(rec), ensure_ascii=False, indent=2), encoding="utf-8")
        fold_records.append(rec)
        print(
            f"Outer {outer_fold} / {arm}: retained={len(retained_tr_idx):,}, "
            f"removed={len(removed_tr_idx):,}, test={len(te_idx):,}, "
            f"R2={metrics['R2']:.4f}, RMSE={metrics['RMSE']:.4f}, MAE={metrics['MAE']:.4f}"
        )

    return fold_records


# ============================================================================
# 10. Aggregate all 5 outer folds
# ============================================================================

def aggregate_results(output_dir: Path) -> None:
    arms = ["no_filter"] + [f"filter_{int(t)}fold" for t in THRESHOLDS]
    fold_records = []
    pred_by_arm = {arm: [] for arm in arms}

    for fold in range(1, OUTER_FOLDS + 1):
        fold_dir = output_dir / f"outer_fold_{fold}"
        for arm in arms:
            rfile = fold_dir / f"result_{arm}.json"
            pfile = fold_dir / f"predictions_{arm}.csv"
            if not rfile.is_file() or not pfile.is_file():
                raise RuntimeError(f"Missing completed output: outer fold {fold}, {arm}")
            rec = json.loads(rfile.read_text(encoding="utf-8-sig"))
            fold_records.append(rec)
            pred_by_arm[arm].append(pd.read_csv(pfile, low_memory=False))

    fold_df = pd.DataFrame(fold_records)
    fold_df.to_csv(output_dir / "scaffold_nested_cv_fold_results.csv", index=False, encoding="utf-8-sig")

    mean_sd_records = []
    pooled_records = []
    comparison_records = []

    metric_cols = [
        "R2", "RMSE", "MAE", "median_fold_error", "mean_fold_error",
        "within_2_fold_percent", "within_5_fold_percent", "within_10_fold_percent",
    ]

    for arm in arms:
        f = fold_df[fold_df["arm"] == arm].sort_values("outer_fold")
        if len(f) != OUTER_FOLDS:
            raise RuntimeError(f"{arm}: expected {OUTER_FOLDS} fold records, got {len(f)}")

        mean_sd = {
            "arm": arm,
            "threshold_fold": f["threshold_fold"].dropna().iloc[0] if f["threshold_fold"].notna().any() else np.nan,
            "mean_removed_training_percent": float(f["removed_percent"].mean()),
        }
        for c in metric_cols:
            mean_sd[f"mean_{c}"] = float(f[c].mean())
            mean_sd[f"sd_{c}"] = float(f[c].std(ddof=1))
        mean_sd_records.append(mean_sd)

        p = pd.concat(pred_by_arm[arm], ignore_index=True)
        if len(p) != 34959:
            raise RuntimeError(f"{arm}: pooled outer predictions should contain 34,959 rows, got {len(p)}")
        if p["global_index"].duplicated().any() or set(p["global_index"]) != set(range(34959)):
            raise RuntimeError(f"{arm}: pooled outer predictions do not cover each development compound exactly once")
        p = p.sort_values("global_index").reset_index(drop=True)
        p.to_csv(output_dir / f"scaffold_nested_predictions_{arm}.csv", index=False, encoding="utf-8-sig")

        m = calc_metrics(p["y_true"].to_numpy(float), p["y_pred"].to_numpy(float))
        fe = calc_fold_error_metrics(p["y_true"].to_numpy(float), p["y_pred"].to_numpy(float))
        pooled = {
            "arm": arm,
            "threshold_fold": mean_sd["threshold_fold"],
            "n": int(len(p)),
            "mean_removed_training_percent": mean_sd["mean_removed_training_percent"],
            **m,
            **fe,
        }
        pooled_records.append(pooled)

        comparison_records.append({
            "arm": arm,
            "threshold_fold": mean_sd["threshold_fold"],
            "n_outer_predictions": int(len(p)),
            "mean_removed_training_percent": mean_sd["mean_removed_training_percent"],
            "pooled_R2": m["R2"],
            "fold_mean_R2": mean_sd["mean_R2"],
            "fold_sd_R2": mean_sd["sd_R2"],
            "pooled_RMSE": m["RMSE"],
            "fold_mean_RMSE": mean_sd["mean_RMSE"],
            "fold_sd_RMSE": mean_sd["sd_RMSE"],
            "pooled_MAE": m["MAE"],
            "fold_mean_MAE": mean_sd["mean_MAE"],
            "fold_sd_MAE": mean_sd["sd_MAE"],
            "pooled_median_fold_error": fe["median_fold_error"],
            "pooled_within_2_fold_percent": fe["within_2_fold_percent"],
            "pooled_within_5_fold_percent": fe["within_5_fold_percent"],
            "pooled_within_10_fold_percent": fe["within_10_fold_percent"],
        })

    pd.DataFrame(mean_sd_records).to_csv(
        output_dir / "scaffold_nested_cv_mean_sd.csv", index=False, encoding="utf-8-sig"
    )
    pd.DataFrame(pooled_records).to_csv(
        output_dir / "scaffold_nested_cv_pooled_results.csv", index=False, encoding="utf-8-sig"
    )
    comparison_df = pd.DataFrame(comparison_records)
    comparison_df.to_csv(
        output_dir / "scaffold_nested_cv_comparison.csv", index=False, encoding="utf-8-sig"
    )

    print("\n================ FINAL SCAFFOLD NESTED CV ================")
    print(comparison_df.to_string(index=False))
    print("==========================================================")


# ============================================================================
# 11. Main
# ============================================================================

def parse_args():
    parser = argparse.ArgumentParser(description="JHM scaffold-based nested CV")
    parser.add_argument("--base-dir", default=str(DEFAULT_BASE_DIR))
    parser.add_argument("--input", default=None, help="Optional explicit train_pool_toxric_mixed_before_oof.csv")
    parser.add_argument("--output-dir", default=None)
    parser.add_argument("--save-models", action="store_true", help="Save outer-fold XGBoost model files")
    return parser.parse_args()


def main() -> int:
    global SAVE_MODELS
    args = parse_args()
    set_seed(SEED)
    SAVE_MODELS = bool(args.save_models)

    base_dir = Path(args.base_dir)
    if not base_dir.exists():
        raise FileNotFoundError(f"BASE_DIR does not exist: {base_dir}")

    input_path = resolve_input(base_dir, args.input)
    output_dir = Path(args.output_dir) if args.output_dir else base_dir / DEFAULT_OUTPUT_NAME
    output_dir.mkdir(parents=True, exist_ok=True)

    print("\n================ CONFIG ================")
    print("Input:", input_path)
    print("Output:", output_dir)
    print("Outer split: Bemis-Murcko scaffold, 5 folds")
    print("Inner screening: random 5-fold OOF within outer training only")
    print("Arms: no_filter, 10x, 20x, 30x")
    print("Final model seed: 42")
    print("XGBoost max rounds / early stopping:", NUM_BOOST_ROUND_MAX, "/", EARLY_STOPPING_ROUNDS)
    print("Save models:", SAVE_MODELS)
    print("========================================\n")

    input_sha = sha256_file(input_path)
    df = safe_read_csv(input_path).reset_index(drop=True)
    validate_development_df(df, input_path)
    df["y"] = pd.to_numeric(df["y"], errors="raise").astype(float)
    y_all = df["y"].to_numpy(float)

    assignments_core = build_scaffold_assignments(df, output_dir)
    # build_scaffold_assignments may return only assignment columns after merge; keep canonical order.
    assignments = assignments_core[[
        "Canonical SMILES", "scaffold_key", "is_acyclic_empty_murcko", "outer_fold"
    ]].copy()
    assignments.to_csv(output_dir / "scaffold_assignments.csv", index=False, encoding="utf-8-sig")

    scaffold_summary = validate_outer_scaffold_splits(assignments)
    (output_dir / "scaffold_split_summary.json").write_text(
        json.dumps(to_jsonable(scaffold_summary), ensure_ascii=False, indent=2), encoding="utf-8"
    )
    print("Scaffold split summary:")
    print(json.dumps(to_jsonable(scaffold_summary), ensure_ascii=False, indent=2))

    X_raw = build_or_load_raw_features(df, output_dir, input_sha)

    manifest = {
        "experiment": "JHM scaffold-based nested CV with training-only OOF residual screening",
        "timestamp_local": time.strftime("%Y-%m-%d %H:%M:%S"),
        "input_file": str(input_path),
        "input_sha256": input_sha,
        "n_development_compounds": int(len(df)),
        "outer_split": {
            "method": "Bemis-Murcko scaffold grouping",
            "n_folds": OUTER_FOLDS,
            "assignment": "deterministic greedy balancing by scaffold-group size",
            "acyclic_rule": "empty Murcko scaffold -> compound-specific canonical non-isomeric SMILES key",
            "test_fold_used_for_screening": False,
        },
        "nested_screening": {
            "inner_folds": INNER_FOLDS,
            "inner_split": "random KFold(shuffle=True, random_state=42) inside outer training only",
            "fold_error": "10**abs(y_true - y_inner_oof_pred)",
            "thresholds": THRESHOLDS,
            "arms": ["no_filter"] + [f"filter_{int(t)}fold" for t in THRESHOLDS],
        },
        "final_training": {
            "seed": SEED,
            "round_selection": "median of the 5 inner-OOF early-stopping best iterations within each outer fold; same rounds for all arms in that outer fold",
            "outer_test_used_for_early_stopping": False,
        },
        "feature": {
            "ecfp": "ECFP4 Morgan count fingerprint, radius=2, fpSize=2048, includeChirality=False, log1p",
            "rdkit2d": "Descriptors._descList",
            "expected_raw_dim": EXPECTED_RAW_DIM,
            "preprocessing": "drop all-NaN columns on training only; median imputation on training only; variance threshold 0 on training only",
        },
        "xgboost_params": XGB_PARAMS,
        "max_boost_rounds": NUM_BOOST_ROUND_MAX,
        "early_stopping_rounds": EARLY_STOPPING_ROUNDS,
        "versions": {
            "python": sys.version,
            "numpy": np.__version__,
            "pandas": pd.__version__,
            "sklearn": sklearn.__version__,
            "rdkit": rdBase.rdkitVersion,
            "xgboost": xgb.__version__,
        },
    }
    (output_dir / "run_manifest.json").write_text(
        json.dumps(to_jsonable(manifest), ensure_ascii=False, indent=2), encoding="utf-8"
    )

    all_fold_records = []
    for outer_fold in range(1, OUTER_FOLDS + 1):
        print(f"\n\n################ OUTER FOLD {outer_fold}/{OUTER_FOLDS} ################")
        recs = run_outer_fold(
            outer_fold=outer_fold,
            df=df,
            assignments=assignments,
            X_raw=X_raw,
            y_all=y_all,
            output_dir=output_dir,
        )
        all_fold_records.extend(recs)

    aggregate_results(output_dir)
    (output_dir / "COMPLETED.txt").write_text(
        "All 5 scaffold outer folds and all four screening arms completed.\n",
        encoding="utf-8",
    )

    print("\n完成。重点文件:")
    print(output_dir / "scaffold_nested_cv_comparison.csv")
    print(output_dir / "scaffold_nested_cv_fold_results.csv")
    print(output_dir / "scaffold_nested_cv_mean_sd.csv")
    print(output_dir / "scaffold_nested_cv_pooled_results.csv")
    print(output_dir / "scaffold_split_summary.json")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
