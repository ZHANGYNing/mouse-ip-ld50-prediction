# -*- coding: utf-8 -*-
"""Original numerical functions extracted from the uploaded final 4.py.
No execution of the old main() or its filesystem setup. Algorithmic bodies
are copied exactly. This module is specific to the development-pool CV job.
"""
import os
import json
import numpy as np
import pandas as pd
import joblib
from tqdm import tqdm
from rdkit import Chem, DataStructs
from rdkit.Chem import Descriptors, rdFingerprintGenerator
from sklearn.impute import SimpleImputer
from sklearn.feature_selection import VarianceThreshold
import xgboost as xgb

OUTPUT_DIR = None
ECFP_BITS = 2048
ECFP_RADIUS = 2
NUM_BOOST_ROUND = 7000
EARLY_STOPPING_ROUNDS = 250
morgan_generator = rdFingerprintGenerator.GetMorganGenerator(
    radius=ECFP_RADIUS, fpSize=ECFP_BITS
)
RDKit_2D_DESC_LIST = Descriptors._descList

# Uploaded original 4.py: lines 488-502
def smiles_to_ecfp4_count(smiles):
    arr = np.zeros((ECFP_BITS,), dtype=np.float32)

    mol = Chem.MolFromSmiles(str(smiles))

    if mol is None:
        return arr

    fp = morgan_generator.GetCountFingerprint(mol)
    DataStructs.ConvertToNumpyArray(fp, arr)

    # count fingerprint 做 log1p
    arr = np.log1p(arr)

    return arr.astype(np.float32)

# Uploaded original 4.py: lines 505-525
def smiles_to_rdkit2d(smiles):
    mol = Chem.MolFromSmiles(str(smiles))

    if mol is None:
        return np.full(len(RDKit_2D_DESC_LIST), np.nan, dtype=np.float32)

    values = []

    for name, func in RDKit_2D_DESC_LIST:
        try:
            v = func(mol)
            v = float(v)
        except Exception:
            v = np.nan

        if not np.isfinite(v):
            v = np.nan

        values.append(v)

    return np.array(values, dtype=np.float32)

# Uploaded original 4.py: lines 528-542
def build_raw_features(df, split_name):
    ecfp_list = []
    desc_list = []

    for smi in tqdm(df["Canonical SMILES"].tolist(), desc=f"Features {split_name}"):
        ecfp_list.append(smiles_to_ecfp4_count(smi))
        desc_list.append(smiles_to_rdkit2d(smi))

    X_ecfp = np.vstack(ecfp_list).astype(np.float32)
    X_desc = np.vstack(desc_list).astype(np.float32)

    X_raw = np.hstack([X_ecfp, X_desc]).astype(np.float32)
    X_raw = np.where(np.isfinite(X_raw), X_raw, np.nan)

    return X_raw

# Uploaded original 4.py: lines 545-585
def fit_preprocess_and_transform(X_fit_raw, X_list_raw, save_prefix=None):
    """
    只在 X_fit_raw 上 fit imputer 和 variance selector。
    然后 transform X_list_raw。
    """

    X_fit_raw = np.where(np.isfinite(X_fit_raw), X_fit_raw, np.nan)

    keep_cols = ~np.all(np.isnan(X_fit_raw), axis=0)

    X_fit = X_fit_raw[:, keep_cols]

    imputer = SimpleImputer(strategy="median")
    X_fit_imp = imputer.fit_transform(X_fit).astype(np.float32)

    selector = VarianceThreshold(threshold=0.0)
    selector.fit(X_fit_imp)

    transformed = []

    for X_raw in X_list_raw:
        X_raw = np.where(np.isfinite(X_raw), X_raw, np.nan)
        X = X_raw[:, keep_cols]
        X = imputer.transform(X).astype(np.float32)
        X = selector.transform(X).astype(np.float32)
        transformed.append(X)

    info = {
        "raw_dim": int(X_fit_raw.shape[1]),
        "after_all_nan_filter_dim": int(np.sum(keep_cols)),
        "after_variance_filter_dim": int(transformed[0].shape[1]),
    }

    if save_prefix is not None:
        joblib.dump(imputer, os.path.join(OUTPUT_DIR, f"{save_prefix}_imputer.joblib"))
        joblib.dump(selector, os.path.join(OUTPUT_DIR, f"{save_prefix}_variance_selector.joblib"))

        with open(os.path.join(OUTPUT_DIR, f"{save_prefix}_feature_preprocess_info.json"), "w", encoding="utf-8") as f:
            json.dump(info, f, ensure_ascii=False, indent=4)

    return transformed, info

# Uploaded original 4.py: lines 592-624
def train_with_early_stopping(params, X_train, y_train, X_valid, y_valid):
    dtrain = xgb.DMatrix(X_train, label=y_train)
    dvalid = xgb.DMatrix(X_valid, label=y_valid)

    try:
        booster = xgb.train(
            params=params,
            dtrain=dtrain,
            num_boost_round=NUM_BOOST_ROUND,
            evals=[(dtrain, "train"), (dvalid, "valid")],
            early_stopping_rounds=EARLY_STOPPING_ROUNDS,
            verbose_eval=200,
        )

    except Exception as e:
        print("GPU 训练失败，切换 CPU。错误信息：", str(e)[:300])

        params_cpu = params.copy()
        params_cpu.pop("device", None)
        params_cpu["tree_method"] = "hist"

        booster = xgb.train(
            params=params_cpu,
            dtrain=dtrain,
            num_boost_round=NUM_BOOST_ROUND,
            evals=[(dtrain, "train"), (dvalid, "valid")],
            early_stopping_rounds=EARLY_STOPPING_ROUNDS,
            verbose_eval=200,
        )

    best_iter = booster.best_iteration + 1

    return booster, best_iter

