import os
import re
import json
import math
import random
import warnings

import numpy as np
import pandas as pd
import joblib

from tqdm import tqdm

from rdkit import Chem, DataStructs, RDLogger
from rdkit.Chem import Descriptors, rdFingerprintGenerator
from rdkit.Chem.MolStandardize import rdMolStandardize

from sklearn.model_selection import KFold, train_test_split
from sklearn.impute import SimpleImputer
from sklearn.feature_selection import VarianceThreshold
from sklearn.metrics import r2_score, mean_squared_error, mean_absolute_error

import xgboost as xgb


warnings.filterwarnings("ignore")
RDLogger.DisableLog("rdApp.warning")


# =========================================================
# 1. 路径设置
# =========================================================

from pathlib import Path
import argparse
import importlib.metadata
import platform

REPOSITORY_ROOT = Path(__file__).resolve().parents[1]
_parser = argparse.ArgumentParser(description="Original source-holdout training workflow")
_parser.add_argument("--input", type=Path, default=REPOSITORY_ROOT / "data/raw/mouse_intraperitoneal_ld50.xlsx")
_parser.add_argument("--output-dir", type=Path, default=REPOSITORY_ROOT / "results/generated/main")
_parser.add_argument("--device", choices=["cuda", "cpu"], default="cuda")
_args = _parser.parse_args() if __name__ == "__main__" else _parser.parse_args([])
BASE_DIR = str(REPOSITORY_ROOT)
INPUT_FILE = str(_args.input.resolve())
FALLBACK_INPUT_FILE = INPUT_FILE
OUTPUT_DIR = str(_args.output_dir.resolve())
os.makedirs(OUTPUT_DIR, exist_ok=True)

# =========================================================
# 2. 清洗与建模参数
# =========================================================

SEED = 42

# 同一标准化 SMILES 内 LD50 最大值 / 最小值超过该倍数，删除整组
DUP_CONFLICT_FOLD = 5.0

# 全局 LD50 范围，单位 mg/kg
LD50_MIN_MGKG = 0.1
LD50_MAX_MGKG = 10000.0

# PubChem_only 外部集只做规则清洗，可额外删除特别极端值
PUBCHEM_ONLY_MIN_MGKG = 0.5
PUBCHEM_ONLY_MAX_MGKG = 5000.0

# 只在 TOXRIC_only + mixed 训练池中做 OOF 高残差删除
HIGH_RESIDUAL_FOLD_TRAIN_POOL = 20.0

N_SPLITS_OOF = 5
VALID_SIZE = 0.10

USE_GPU = _args.device == "cuda"

NUM_BOOST_ROUND = 7000
EARLY_STOPPING_ROUNDS = 250

BAGGING_SEEDS = [
    1, 2, 3, 4, 5,
    42, 123, 2024, 2025, 3407
]

# 允许保留的常见有机元素
ALLOWED_ELEMENTS = {
    "H", "C", "N", "O", "F", "Cl", "Br", "I",
    "P", "S", "B", "Si", "Se"
}

MIN_HEAVY_ATOMS = 4


# =========================================================
# 3. XGBoost 参数
# 使用你之前 ECFP4 count + RDKit 2D 的最优风格
# =========================================================

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
}

if USE_GPU:
    XGB_PARAMS["device"] = "cuda"


# =========================================================
# 4. 工具函数
# =========================================================

def set_seed(seed=42):
    random.seed(seed)
    np.random.seed(seed)


set_seed(SEED)


def to_jsonable(obj):
    if isinstance(obj, dict):
        return {str(k): to_jsonable(v) for k, v in obj.items()}
    if isinstance(obj, list):
        return [to_jsonable(v) for v in obj]
    if isinstance(obj, tuple):
        return [to_jsonable(v) for v in obj]
    if isinstance(obj, np.ndarray):
        return obj.tolist()
    if isinstance(obj, (np.float32, np.float64)):
        return float(obj)
    if isinstance(obj, (np.int32, np.int64)):
        return int(obj)
    return obj


def normalize_colname(x):
    return str(x).strip().lower().replace(" ", "").replace("_", "")


def find_column(df, candidates):
    norm_map = {normalize_colname(c): c for c in df.columns}

    for cand in candidates:
        key = normalize_colname(cand)
        if key in norm_map:
            return norm_map[key]

    for col in df.columns:
        col_norm = normalize_colname(col)
        for cand in candidates:
            cand_norm = normalize_colname(cand)
            if cand_norm in col_norm or col_norm in cand_norm:
                return col

    return None


def parse_ld50_value(x):
    """
    返回：
    value, is_limit, note

    is_limit=True 表示含 >、<、≥、≤ 等限值符号。
    """

    if pd.isna(x):
        return np.nan, False, "missing"

    s = str(x).strip()

    s = (
        s.replace("&gt;", ">")
         .replace("&lt;", "<")
         .replace("≥", ">=")
         .replace("≤", "<=")
    )

    s_lower = s.lower()

    limit_patterns = [
        ">", "<", ">=", "<=",
        "greater than", "less than",
        "more than", "not less than",
        "not greater than"
    ]

    is_limit = any(p in s_lower for p in limit_patterns)

    nums = re.findall(r"[-+]?\d*\.?\d+(?:[eE][-+]?\d+)?", s)

    if len(nums) == 0:
        return np.nan, is_limit, "no_numeric"

    values = [float(v) for v in nums]

    # 如果出现范围，例如 10-20，取平均
    value = float(np.mean(values))

    return value, is_limit, "ok"


def normalize_source(x):
    """
    label=1 -> TOXRIC
    label=2 -> PubChem
    """

    if pd.isna(x):
        return "Unknown"

    s = str(x).strip().lower()

    if s in ["1", "1.0"]:
        return "TOXRIC"

    if s in ["2", "2.0"]:
        return "PubChem"

    if "toxric" in s:
        return "TOXRIC"

    if "pubchem" in s:
        return "PubChem"

    return "Unknown"


def calc_metrics(y_true, y_pred):
    return {
        "R2": float(r2_score(y_true, y_pred)) if len(y_true) >= 2 else np.nan,
        "RMSE": float(math.sqrt(mean_squared_error(y_true, y_pred))),
        "MAE": float(mean_absolute_error(y_true, y_pred)),
    }


def calc_fold_error(y_true_neglog_molkg, y_pred_neglog_molkg, molecular_weight):
    true_molkg = 10 ** (-y_true_neglog_molkg)
    pred_molkg = 10 ** (-y_pred_neglog_molkg)

    true_mgkg = true_molkg * molecular_weight * 1000.0
    pred_mgkg = pred_molkg * molecular_weight * 1000.0

    fold_error = np.maximum(
        pred_mgkg / true_mgkg,
        true_mgkg / pred_mgkg
    )

    return {
        "median_fold_error": float(np.median(fold_error)),
        "mean_fold_error": float(np.mean(fold_error)),
        "within_2_fold_percent": float(np.mean(fold_error <= 2) * 100),
        "within_5_fold_percent": float(np.mean(fold_error <= 5) * 100),
        "within_10_fold_percent": float(np.mean(fold_error <= 10) * 100),
    }


def save_predictions(df, y_pred, save_path):
    pred_df = df.copy()

    pred_df["y_true_neglog_molkg"] = pred_df["y"].values
    pred_df["y_pred_neglog_molkg"] = y_pred

    pred_df["ld50_true_molkg"] = 10 ** (-pred_df["y_true_neglog_molkg"])
    pred_df["ld50_pred_molkg"] = 10 ** (-pred_df["y_pred_neglog_molkg"])

    pred_df["ld50_true_mgkg"] = (
        pred_df["ld50_true_molkg"] * pred_df["molecular_weight"] * 1000.0
    )

    pred_df["ld50_pred_mgkg"] = (
        pred_df["ld50_pred_molkg"] * pred_df["molecular_weight"] * 1000.0
    )

    pred_df["fold_error"] = np.maximum(
        pred_df["ld50_pred_mgkg"] / pred_df["ld50_true_mgkg"],
        pred_df["ld50_true_mgkg"] / pred_df["ld50_pred_mgkg"]
    )

    pred_df.to_csv(save_path, index=False, encoding="utf-8-sig")

    return pred_df


# =========================================================
# 5. SMILES 标准化：盐形式只保留主体分子
# =========================================================

fragment_chooser = rdMolStandardize.LargestFragmentChooser(preferOrganic=True)
uncharger = rdMolStandardize.Uncharger()


def has_carbon(mol):
    return any(atom.GetSymbol() == "C" for atom in mol.GetAtoms())


def get_element_set(mol):
    return {atom.GetSymbol() for atom in mol.GetAtoms()}


def standardize_smiles(smiles):
    if pd.isna(smiles):
        return {
            "valid": False,
            "parent_smiles": None,
            "reason": "missing_smiles",
            "n_fragments": np.nan,
            "element_set": None,
            "heavy_atoms": np.nan,
            "molecular_weight": np.nan,
        }

    s = str(smiles).strip()

    if s == "":
        return {
            "valid": False,
            "parent_smiles": None,
            "reason": "empty_smiles",
            "n_fragments": np.nan,
            "element_set": None,
            "heavy_atoms": np.nan,
            "molecular_weight": np.nan,
        }

    mol = Chem.MolFromSmiles(s)

    if mol is None:
        return {
            "valid": False,
            "parent_smiles": None,
            "reason": "rdkit_parse_failed",
            "n_fragments": np.nan,
            "element_set": None,
            "heavy_atoms": np.nan,
            "molecular_weight": np.nan,
        }

    try:
        Chem.SanitizeMol(mol)
    except Exception:
        return {
            "valid": False,
            "parent_smiles": None,
            "reason": "sanitize_failed",
            "n_fragments": np.nan,
            "element_set": None,
            "heavy_atoms": np.nan,
            "molecular_weight": np.nan,
        }

    try:
        frags = Chem.GetMolFrags(mol, asMols=True, sanitizeFrags=True)
    except Exception:
        return {
            "valid": False,
            "parent_smiles": None,
            "reason": "fragment_failed",
            "n_fragments": np.nan,
            "element_set": None,
            "heavy_atoms": np.nan,
            "molecular_weight": np.nan,
        }

    n_fragments = len(frags)

    organic_frags = [f for f in frags if has_carbon(f)]
    organic_sizes = sorted(
        [f.GetNumHeavyAtoms() for f in organic_frags],
        reverse=True
    )

    # 多个大小接近的有机片段，认为是复杂混合物
    if len(organic_sizes) >= 2:
        if organic_sizes[1] / organic_sizes[0] >= 0.5:
            return {
                "valid": False,
                "parent_smiles": None,
                "reason": "complex_multiple_organic_fragments",
                "n_fragments": n_fragments,
                "element_set": None,
                "heavy_atoms": np.nan,
                "molecular_weight": np.nan,
            }

    try:
        parent = fragment_chooser.choose(mol)
        parent = uncharger.uncharge(parent)
        parent_smiles = Chem.MolToSmiles(parent, canonical=True)
        parent = Chem.MolFromSmiles(parent_smiles)
    except Exception:
        return {
            "valid": False,
            "parent_smiles": None,
            "reason": "largest_fragment_or_uncharge_failed",
            "n_fragments": n_fragments,
            "element_set": None,
            "heavy_atoms": np.nan,
            "molecular_weight": np.nan,
        }

    if parent is None:
        return {
            "valid": False,
            "parent_smiles": None,
            "reason": "parent_parse_failed",
            "n_fragments": n_fragments,
            "element_set": None,
            "heavy_atoms": np.nan,
            "molecular_weight": np.nan,
        }

    if not has_carbon(parent):
        return {
            "valid": False,
            "parent_smiles": parent_smiles,
            "reason": "no_carbon_in_parent",
            "n_fragments": n_fragments,
            "element_set": None,
            "heavy_atoms": parent.GetNumHeavyAtoms(),
            "molecular_weight": np.nan,
        }

    element_set = get_element_set(parent)
    bad_elements = sorted(list(element_set - ALLOWED_ELEMENTS))

    if len(bad_elements) > 0:
        return {
            "valid": False,
            "parent_smiles": parent_smiles,
            "reason": "uncommon_or_metal_elements:" + ",".join(bad_elements),
            "n_fragments": n_fragments,
            "element_set": ",".join(sorted(element_set)),
            "heavy_atoms": parent.GetNumHeavyAtoms(),
            "molecular_weight": np.nan,
        }

    heavy_atoms = parent.GetNumHeavyAtoms()

    if heavy_atoms < MIN_HEAVY_ATOMS:
        return {
            "valid": False,
            "parent_smiles": parent_smiles,
            "reason": "too_few_heavy_atoms",
            "n_fragments": n_fragments,
            "element_set": ",".join(sorted(element_set)),
            "heavy_atoms": heavy_atoms,
            "molecular_weight": np.nan,
        }

    mw = Descriptors.MolWt(parent)

    return {
        "valid": True,
        "parent_smiles": parent_smiles,
        "reason": "ok",
        "n_fragments": n_fragments,
        "element_set": ",".join(sorted(element_set)),
        "heavy_atoms": heavy_atoms,
        "molecular_weight": mw,
    }


# =========================================================
# 6. 特征：ECFP4 count + RDKit 2D
# =========================================================

ECFP_BITS = 2048
ECFP_RADIUS = 2

morgan_generator = rdFingerprintGenerator.GetMorganGenerator(
    radius=ECFP_RADIUS,
    fpSize=ECFP_BITS
)

RDKit_2D_DESC_LIST = Descriptors._descList


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


# =========================================================
# 7. XGBoost 训练函数
# =========================================================

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


def train_fixed_round(params, X_train, y_train, num_boost_round):
    dtrain = xgb.DMatrix(X_train, label=y_train)

    try:
        booster = xgb.train(
            params=params,
            dtrain=dtrain,
            num_boost_round=num_boost_round,
            evals=[(dtrain, "train")],
            verbose_eval=500,
        )

    except Exception as e:
        print("GPU 训练失败，切换 CPU。错误信息：", str(e)[:300])

        params_cpu = params.copy()
        params_cpu.pop("device", None)
        params_cpu["tree_method"] = "hist"

        booster = xgb.train(
            params=params_cpu,
            dtrain=dtrain,
            num_boost_round=num_boost_round,
            evals=[(dtrain, "train")],
            verbose_eval=500,
        )

    return booster


# =========================================================
# 8. 规则清洗，从原始数据到聚合化合物层面
# =========================================================

def rule_clean_and_aggregate(raw_df):
    raw_df = raw_df.copy()
    raw_df["_row_id"] = np.arange(len(raw_df))

    smiles_col = find_column(
        raw_df,
        [
            "Canonical SMILES",
            "SMILES",
            "smiles",
            "canonical_smiles",
            "Canonical_SMILES"
        ]
    )

    value_col = find_column(
        raw_df,
        [
            "Toxicity Value",
            "LD50_mgkg",
            "ld50_mgkg",
            "mgkg",
            "value",
            "Dose",
            "dose"
        ]
    )

    label_col = find_column(
        raw_df,
        [
            "label",
            "source_label",
            "Source",
            "source",
            "Database",
            "database"
        ]
    )

    if smiles_col is None:
        raise ValueError(f"找不到 SMILES 列。当前列名：{list(raw_df.columns)}")

    if value_col is None:
        raise ValueError(f"找不到 LD50 数值列。当前列名：{list(raw_df.columns)}")

    if label_col is None:
        raise ValueError(
            f"找不到来源 label/source 列。当前列名：{list(raw_df.columns)}\n"
            f"需要 label=1 表示 TOXRIC，label=2 表示 PubChem。"
        )

    print("识别到 SMILES 列：", smiles_col)
    print("识别到 LD50 列：", value_col)
    print("识别到来源列：", label_col)

    work_df = raw_df.copy()

    work_df["raw_smiles"] = work_df[smiles_col]
    work_df["raw_ld50_value"] = work_df[value_col]
    work_df["source_norm"] = work_df[label_col].apply(normalize_source)

    removed_parts = []

    def remove_rows(df, mask, reason):
        removed = df.loc[mask].copy()
        removed["remove_reason"] = reason
        removed_parts.append(removed)
        return df.loc[~mask].copy()

    # 1. LD50 解析和限值删除
    parsed = work_df["raw_ld50_value"].apply(parse_ld50_value)
    work_df["ld50_mgkg"] = parsed.apply(lambda x: x[0])
    work_df["is_limit_value"] = parsed.apply(lambda x: x[1])
    work_df["ld50_parse_note"] = parsed.apply(lambda x: x[2])

    work_df = remove_rows(
        work_df,
        work_df["is_limit_value"],
        "limit_value_with_greater_less_symbols"
    )

    work_df = remove_rows(
        work_df,
        work_df["ld50_mgkg"].isna(),
        "ld50_numeric_parse_failed"
    )

    work_df = remove_rows(
        work_df,
        work_df["ld50_mgkg"] <= 0,
        "ld50_non_positive"
    )

    # 2. 结构标准化
    print("\n正在标准化 SMILES、处理盐形式和结构规则清洗...")

    std_records = []

    for smi in tqdm(work_df["raw_smiles"].tolist(), desc="Standardizing SMILES"):
        std_records.append(standardize_smiles(smi))

    std_df = pd.DataFrame(std_records, index=work_df.index)

    for col in std_df.columns:
        work_df[col] = std_df[col]

    work_df = remove_rows(
        work_df,
        ~work_df["valid"],
        "invalid_structure_or_removed_by_structure_rule"
    )

    # 3. 全局 LD50 极端值
    work_df = remove_rows(
        work_df,
        work_df["ld50_mgkg"] < LD50_MIN_MGKG,
        f"ld50_below_{LD50_MIN_MGKG}_mgkg"
    )

    work_df = remove_rows(
        work_df,
        work_df["ld50_mgkg"] > LD50_MAX_MGKG,
        f"ld50_above_{LD50_MAX_MGKG}_mgkg"
    )

    # 4. 同一标准化 SMILES 冲突删除
    print("\n正在检查重复化合物冲突...")

    group_stats = (
        work_df
        .groupby("parent_smiles")["ld50_mgkg"]
        .agg(["count", "min", "max", "median"])
        .reset_index()
    )

    group_stats["conflict_fold"] = group_stats["max"] / group_stats["min"]

    conflict_smiles = set(
        group_stats.loc[
            (group_stats["count"] >= 2) &
            (group_stats["conflict_fold"] > DUP_CONFLICT_FOLD),
            "parent_smiles"
        ]
    )

    work_df = remove_rows(
        work_df,
        work_df["parent_smiles"].isin(conflict_smiles),
        f"duplicate_conflict_fold_above_{DUP_CONFLICT_FOLD}"
    )

    # 5. 聚合到化合物层面
    print("\n正在聚合到标准化化合物层面...")

    agg_records = []

    for smi, g in tqdm(work_df.groupby("parent_smiles"), desc="Aggregating"):
        mol = Chem.MolFromSmiles(smi)

        if mol is None:
            continue

        mw = Descriptors.MolWt(mol)

        sources = sorted(set(g["source_norm"].tolist()))

        has_toxric = "TOXRIC" in sources
        has_pubchem = "PubChem" in sources

        if has_toxric and has_pubchem:
            source_type = "mixed"
        elif has_toxric:
            source_type = "TOXRIC_only"
        elif has_pubchem:
            source_type = "PubChem_only"
        else:
            source_type = "Unknown"

        ld50_values = g["ld50_mgkg"].values.astype(float)
        ld50_median = float(np.median(ld50_values))

        ld50_molkg = (ld50_median / 1000.0) / mw
        y = -math.log10(ld50_molkg)

        agg_records.append({
            "Canonical SMILES": smi,
            "Toxicity Value": ld50_median,
            "ld50_mgkg": ld50_median,
            "molecular_weight": mw,
            "y": y,
            "source_type": source_type,
            "source_set": ";".join(sources),
            "has_toxric": has_toxric,
            "has_pubchem": has_pubchem,
            "n_raw_records": int(len(g)),
            "ld50_min_mgkg_raw": float(np.min(ld50_values)),
            "ld50_max_mgkg_raw": float(np.max(ld50_values)),
            "ld50_mean_mgkg_raw": float(np.mean(ld50_values)),
            "ld50_median_mgkg_raw": ld50_median,
            "raw_row_ids": ";".join(map(str, g["_row_id"].tolist())),
            "n_fragments_original_max": int(g["n_fragments"].max()),
            "heavy_atoms": int(g["heavy_atoms"].iloc[0]),
            "element_set": g["element_set"].iloc[0],
        })

    agg_df = pd.DataFrame(agg_records)

    removed_agg_parts = []

    def remove_agg(df, mask, reason):
        removed = df.loc[mask].copy()
        removed["remove_reason"] = reason
        removed_agg_parts.append(removed)
        return df.loc[~mask].copy()

    # 6. PubChem-only 外部集只做规则清洗，这里属于规则清洗
    agg_df = remove_agg(
        agg_df,
        (agg_df["source_type"] == "PubChem_only") &
        (agg_df["ld50_mgkg"] < PUBCHEM_ONLY_MIN_MGKG),
        f"pubchem_only_ld50_below_{PUBCHEM_ONLY_MIN_MGKG}_mgkg"
    )

    agg_df = remove_agg(
        agg_df,
        (agg_df["source_type"] == "PubChem_only") &
        (agg_df["ld50_mgkg"] > PUBCHEM_ONLY_MAX_MGKG),
        f"pubchem_only_ld50_above_{PUBCHEM_ONLY_MAX_MGKG}_mgkg"
    )

    removed_raw_df = (
        pd.concat(removed_parts, axis=0, ignore_index=True)
        if len(removed_parts) > 0
        else pd.DataFrame()
    )

    removed_agg_df = (
        pd.concat(removed_agg_parts, axis=0, ignore_index=True)
        if len(removed_agg_parts) > 0
        else pd.DataFrame()
    )

    return agg_df.reset_index(drop=True), removed_raw_df, removed_agg_df


# =========================================================
# 9. 只在训练池中做 OOF 高残差过滤
# =========================================================

def run_oof_filter_on_train_pool(train_pool_df):
    print("\n\n===================================================")
    print("只在 TOXRIC_only + mixed 训练池中做 OOF 高残差过滤")
    print("===================================================")

    train_pool_df = train_pool_df.reset_index(drop=True).copy()

    y = train_pool_df["y"].values.astype(np.float32)

    X_raw = build_raw_features(train_pool_df, "train_pool_for_oof")

    oof_pred = np.zeros(len(train_pool_df), dtype=np.float32)

    kf = KFold(
        n_splits=N_SPLITS_OOF,
        shuffle=True,
        random_state=SEED
    )

    fold_records = []

    for fold, (tr_idx, va_idx) in enumerate(kf.split(X_raw), start=1):
        print(f"\n========== OOF Fold {fold}/{N_SPLITS_OOF} ==========")

        X_tr_raw = X_raw[tr_idx]
        X_va_raw = X_raw[va_idx]

        (X_tr, X_va), prep_info = fit_preprocess_and_transform(
            X_tr_raw,
            [X_tr_raw, X_va_raw],
            save_prefix=None
        )

        y_tr = y[tr_idx]
        y_va = y[va_idx]

        params = XGB_PARAMS.copy()
        params["seed"] = SEED + fold

        booster, best_iter = train_with_early_stopping(
            params,
            X_tr,
            y_tr,
            X_va,
            y_va
        )

        pred_va = booster.predict(
            xgb.DMatrix(X_va),
            iteration_range=(0, best_iter)
        )

        oof_pred[va_idx] = pred_va

        fold_metrics = calc_metrics(y_va, pred_va)

        fold_record = {
            "fold": fold,
            "best_iteration": int(best_iter),
            "valid_R2": fold_metrics["R2"],
            "valid_RMSE": fold_metrics["RMSE"],
            "valid_MAE": fold_metrics["MAE"],
            "feature_dim": prep_info["after_variance_filter_dim"],
        }

        fold_records.append(fold_record)

        booster.save_model(
            os.path.join(OUTPUT_DIR, f"oof_filter_fold_{fold}.model")
        )

        print("OOF fold result:", fold_record)

    report_df = train_pool_df.copy()
    report_df["oof_pred_y"] = oof_pred
    report_df["oof_abs_error_y"] = np.abs(report_df["y"].values - oof_pred)
    report_df["oof_fold_error"] = 10 ** report_df["oof_abs_error_y"]

    report_df["remove_by_oof_high_residual"] = (
        report_df["oof_fold_error"] >= HIGH_RESIDUAL_FOLD_TRAIN_POOL
    )

    oof_metrics = calc_metrics(y, oof_pred)

    removed_by_oof = report_df.loc[
        report_df["remove_by_oof_high_residual"]
    ].copy()

    kept_df = report_df.loc[
        ~report_df["remove_by_oof_high_residual"]
    ].copy()

    report_df.to_csv(
        os.path.join(OUTPUT_DIR, "train_pool_oof_residual_report.csv"),
        index=False,
        encoding="utf-8-sig"
    )

    removed_by_oof.to_csv(
        os.path.join(OUTPUT_DIR, "removed_by_oof_train_pool_only.csv"),
        index=False,
        encoding="utf-8-sig"
    )

    kept_df.to_csv(
        os.path.join(OUTPUT_DIR, "train_pool_after_oof_filter.csv"),
        index=False,
        encoding="utf-8-sig"
    )

    fold_df = pd.DataFrame(fold_records)

    fold_df.to_csv(
        os.path.join(OUTPUT_DIR, "oof_filter_fold_results.csv"),
        index=False,
        encoding="utf-8-sig"
    )

    oof_summary = {
        "oof_metrics_on_train_pool_before_filter": oof_metrics,
        "n_train_pool_before_oof": int(len(train_pool_df)),
        "n_removed_by_oof": int(len(removed_by_oof)),
        "n_train_pool_after_oof": int(len(kept_df)),
        "high_residual_threshold_fold": HIGH_RESIDUAL_FOLD_TRAIN_POOL,
        "fold_records": fold_records,
    }

    with open(os.path.join(OUTPUT_DIR, "oof_filter_summary.json"), "w", encoding="utf-8") as f:
        json.dump(to_jsonable(oof_summary), f, ensure_ascii=False, indent=4)

    print("\nOOF 高残差过滤完成：")
    print(oof_summary)

    return kept_df.reset_index(drop=True), removed_by_oof.reset_index(drop=True), oof_summary


# =========================================================
# 10. 主程序
# =========================================================

def main():
    input_file = INPUT_FILE

    if not os.path.exists(input_file):
        if os.path.exists(FALLBACK_INPUT_FILE):
            input_file = FALLBACK_INPUT_FILE
        else:
            raise FileNotFoundError(
                f"找不到输入文件：\n{INPUT_FILE}\n也找不到：\n{FALLBACK_INPUT_FILE}"
            )

    print("正在读取原始数据：")
    print(input_file)

    if input_file.lower().endswith(".csv"):
        raw_df = pd.read_csv(input_file)
    else:
        raw_df = pd.read_excel(input_file)

    print("原始行数：", len(raw_df))

    # =====================================================
    # 10.1 从原始数据做规则清洗和聚合
    # =====================================================

    agg_df, removed_raw_df, removed_agg_rule_df = rule_clean_and_aggregate(raw_df)

    agg_df.to_csv(
        os.path.join(OUTPUT_DIR, "rule_cleaned_aggregated_all.csv"),
        index=False,
        encoding="utf-8-sig"
    )

    removed_raw_df.to_csv(
        os.path.join(OUTPUT_DIR, "removed_raw_records_by_rule.csv"),
        index=False,
        encoding="utf-8-sig"
    )

    removed_agg_rule_df.to_csv(
        os.path.join(OUTPUT_DIR, "removed_aggregated_records_by_rule.csv"),
        index=False,
        encoding="utf-8-sig"
    )

    with pd.ExcelWriter(
        os.path.join(OUTPUT_DIR, "rule_cleaning_outputs.xlsx"),
        engine="openpyxl"
    ) as writer:
        agg_df.to_excel(writer, sheet_name="rule_cleaned_aggregated", index=False)

        if len(removed_agg_rule_df) > 0:
            removed_agg_rule_df.to_excel(writer, sheet_name="removed_agg_rule", index=False)

        if len(removed_raw_df) > 0:
            removed_raw_df.head(100000).to_excel(
                writer,
                sheet_name="removed_raw_head100000",
                index=False
            )

    print("\n规则清洗和聚合后 source_type 分布：")
    print(agg_df["source_type"].value_counts())

    # =====================================================
    # 10.2 先划出 PubChem_only external set
    # 关键：此后 PubChem_only 不参与 OOF 高残差过滤
    # =====================================================

    train_pool_df = agg_df[
        agg_df["source_type"].isin(["TOXRIC_only", "mixed"])
    ].copy()

    external_pubchem_df = agg_df[
        agg_df["source_type"] == "PubChem_only"
    ].copy()

    unknown_df = agg_df[
        agg_df["source_type"] == "Unknown"
    ].copy()

    train_pool_df = train_pool_df.reset_index(drop=True)
    external_pubchem_df = external_pubchem_df.reset_index(drop=True)

    train_pool_df.to_csv(
        os.path.join(OUTPUT_DIR, "train_pool_toxric_mixed_before_oof.csv"),
        index=False,
        encoding="utf-8-sig"
    )

    external_pubchem_df.to_csv(
        os.path.join(OUTPUT_DIR, "external_pubchem_only_rule_cleaned_no_oof.csv"),
        index=False,
        encoding="utf-8-sig"
    )

    if len(unknown_df) > 0:
        unknown_df.to_csv(
            os.path.join(OUTPUT_DIR, "unknown_source_records.csv"),
            index=False,
            encoding="utf-8-sig"
        )

    print("\n来源划分：")
    print("train_pool TOXRIC_only + mixed:", len(train_pool_df))
    print(train_pool_df["source_type"].value_counts())
    print("external PubChem_only:", len(external_pubchem_df))

    if len(external_pubchem_df) < 50:
        raise ValueError(
            f"PubChem_only external 样本太少：{len(external_pubchem_df)}"
        )

    # =====================================================
    # 10.3 只在 train_pool 上做 OOF 高残差过滤
    # =====================================================

    train_pool_after_oof_df, removed_by_oof_df, oof_summary = run_oof_filter_on_train_pool(
        train_pool_df
    )

    # =====================================================
    # 10.4 内部 train / valid 划分
    # =====================================================

    train_df, valid_df = train_test_split(
        train_pool_after_oof_df,
        test_size=VALID_SIZE,
        random_state=SEED,
        shuffle=True
    )

    train_df = train_df.reset_index(drop=True)
    valid_df = valid_df.reset_index(drop=True)
    external_df = external_pubchem_df.reset_index(drop=True)

    train_df.to_csv(
        os.path.join(OUTPUT_DIR, "train_after_oof_no_leakage.csv"),
        index=False,
        encoding="utf-8-sig"
    )

    valid_df.to_csv(
        os.path.join(OUTPUT_DIR, "valid_after_oof_no_leakage.csv"),
        index=False,
        encoding="utf-8-sig"
    )

    external_df.to_csv(
        os.path.join(OUTPUT_DIR, "external_pubchem_only_final_no_oof.csv"),
        index=False,
        encoding="utf-8-sig"
    )

    print("\n最终建模划分：")
    print("train:", len(train_df), train_df["source_type"].value_counts().to_dict())
    print("valid:", len(valid_df), valid_df["source_type"].value_counts().to_dict())
    print("external PubChem_only:", len(external_df))

    # =====================================================
    # 10.5 内部 valid 早停
    # =====================================================

    print("\n\n===================================================")
    print("内部 train/valid 建模，确定 best_iteration")
    print("===================================================")

    X_train_raw = build_raw_features(train_df, "train_internal")
    X_valid_raw = build_raw_features(valid_df, "valid_internal")
    X_external_raw = build_raw_features(external_df, "external_pubchem_only")

    (X_train, X_valid, X_external), prep_info_internal = fit_preprocess_and_transform(
        X_train_raw,
        [X_train_raw, X_valid_raw, X_external_raw],
        save_prefix="internal_valid"
    )

    y_train = train_df["y"].values.astype(np.float32)
    y_valid = valid_df["y"].values.astype(np.float32)
    y_external = external_df["y"].values.astype(np.float32)

    params = XGB_PARAMS.copy()
    params["seed"] = SEED

    booster_valid, best_iter = train_with_early_stopping(
        params,
        X_train,
        y_train,
        X_valid,
        y_valid
    )

    pred_valid = booster_valid.predict(
        xgb.DMatrix(X_valid),
        iteration_range=(0, best_iter)
    )

    valid_metrics = calc_metrics(y_valid, pred_valid)

    valid_fold = calc_fold_error(
        y_valid,
        pred_valid,
        valid_df["molecular_weight"].values
    )

    booster_valid.save_model(
        os.path.join(OUTPUT_DIR, "xgb_internal_valid_earlystop.model")
    )

    save_predictions(
        valid_df,
        pred_valid,
        os.path.join(OUTPUT_DIR, "valid_predictions_internal_earlystop.csv")
    )

    print("\n内部 valid 结果：")
    print(valid_metrics)
    print(valid_fold)
    print("best_iter:", best_iter)

    # =====================================================
    # 10.6 train + valid 全量重训，不使用 external 参与预处理
    # =====================================================

    print("\n\n===================================================")
    print("train + valid 全量重训并预测 PubChem_only external")
    print("===================================================")

    train_full_df = pd.concat([train_df, valid_df], axis=0).reset_index(drop=True)

    X_train_full_raw = np.vstack([X_train_raw, X_valid_raw]).astype(np.float32)

    (X_train_full, X_external_final), prep_info_final = fit_preprocess_and_transform(
        X_train_full_raw,
        [X_train_full_raw, X_external_raw],
        save_prefix="final_train_full"
    )

    y_train_full = train_full_df["y"].values.astype(np.float32)

    # 单模型
    final_params = XGB_PARAMS.copy()
    final_params["seed"] = SEED

    final_booster = train_fixed_round(
        final_params,
        X_train_full,
        y_train_full,
        best_iter
    )

    final_booster.save_model(
        os.path.join(OUTPUT_DIR, "final_single_xgb_no_leakage.model")
    )

    pred_external_single = final_booster.predict(
        xgb.DMatrix(X_external_final)
    )

    external_single_metrics = calc_metrics(y_external, pred_external_single)

    external_single_fold = calc_fold_error(
        y_external,
        pred_external_single,
        external_df["molecular_weight"].values
    )

    save_predictions(
        external_df,
        pred_external_single,
        os.path.join(OUTPUT_DIR, "external_pubchem_only_predictions_single_no_leakage.csv")
    )

    print("\n========== PubChem-only external 单模型结果 ==========")
    print(external_single_metrics)
    print(external_single_fold)

    # =====================================================
    # 10.7 多 seed bagging
    # =====================================================

    print("\n\n===================================================")
    print("多 seed bagging")
    print("===================================================")

    pred_list = []
    seed_records = []

    d_external = xgb.DMatrix(X_external_final)

    for i, seed in enumerate(BAGGING_SEEDS, start=1):
        print(f"\n========== Bagging seed {seed} ({i}/{len(BAGGING_SEEDS)}) ==========")

        p = XGB_PARAMS.copy()
        p["seed"] = seed

        model = train_fixed_round(
            p,
            X_train_full,
            y_train_full,
            best_iter
        )

        model.save_model(
            os.path.join(OUTPUT_DIR, f"xgb_bagging_seed_{seed}_no_leakage.model")
        )

        pred = model.predict(d_external)

        pred_list.append(pred)

        m = calc_metrics(y_external, pred)

        f = calc_fold_error(
            y_external,
            pred,
            external_df["molecular_weight"].values
        )

        record = {
            "seed": seed,
            **m,
            **f,
        }

        seed_records.append(record)

        print(record)

    pred_matrix = np.column_stack(pred_list)
    pred_bagging = pred_matrix.mean(axis=1)

    external_bagging_metrics = calc_metrics(y_external, pred_bagging)

    external_bagging_fold = calc_fold_error(
        y_external,
        pred_bagging,
        external_df["molecular_weight"].values
    )

    save_predictions(
        external_df,
        pred_bagging,
        os.path.join(OUTPUT_DIR, "external_pubchem_only_predictions_bagging_no_leakage.csv")
    )

    pd.DataFrame(
        pred_matrix,
        columns=[f"seed_{s}" for s in BAGGING_SEEDS]
    ).assign(
        mean_pred=pred_bagging,
        y_true=y_external
    ).to_csv(
        os.path.join(OUTPUT_DIR, "external_pubchem_only_predictions_each_seed_no_leakage.csv"),
        index=False,
        encoding="utf-8-sig"
    )

    seed_df = pd.DataFrame(seed_records).sort_values("R2", ascending=False)

    seed_df.to_csv(
        os.path.join(OUTPUT_DIR, "external_pubchem_only_single_seed_results_no_leakage.csv"),
        index=False,
        encoding="utf-8-sig"
    )

    print("\n========== PubChem-only external bagging 结果 ==========")
    print(external_bagging_metrics)
    print(external_bagging_fold)

    # =====================================================
    # 10.8 保存总结
    # =====================================================

    summary = {
        "experiment": "strict no-leakage source external validation",
        "input_file": input_file,
        "important_design": {
            "step_1": "Start from pre-cleaning raw mixed data.",
            "step_2": "Rule-clean and aggregate compounds.",
            "step_3": "Hold out PubChem_only as external set before any OOF high-residual filtering.",
            "step_4": "Apply OOF high-residual filtering only to TOXRIC_only + mixed training pool.",
            "step_5": "PubChem_only external set receives rule-based cleaning only and no OOF residual filtering.",
            "step_6": "Fit preprocessing only on training data; external set is never used for fitting imputer or variance selector.",
        },
        "rule_cleaning": {
            "n_raw_rows": int(len(raw_df)),
            "n_rule_cleaned_aggregated_all": int(len(agg_df)),
            "source_type_counts_after_rule_cleaning": agg_df["source_type"].value_counts().to_dict(),
            "n_removed_raw_by_rule": int(len(removed_raw_df)),
            "n_removed_aggregated_by_rule": int(len(removed_agg_rule_df)),
        },
        "source_split_before_oof": {
            "n_train_pool_toxric_mixed_before_oof": int(len(train_pool_df)),
            "n_external_pubchem_only_rule_cleaned_no_oof": int(len(external_pubchem_df)),
            "train_pool_source_counts": train_pool_df["source_type"].value_counts().to_dict(),
        },
        "oof_filter_train_pool_only": oof_summary,
        "final_modeling_data": {
            "n_train": int(len(train_df)),
            "n_valid": int(len(valid_df)),
            "n_train_full": int(len(train_full_df)),
            "n_external_pubchem_only": int(len(external_df)),
            "train_source_counts": train_df["source_type"].value_counts().to_dict(),
            "valid_source_counts": valid_df["source_type"].value_counts().to_dict(),
        },
        "feature": "ECFP4 count fingerprint + RDKit 2D descriptors",
        "feature_preprocess_internal_valid": prep_info_internal,
        "feature_preprocess_final_train_full": prep_info_final,
        "xgboost_params": XGB_PARAMS,
        "best_iteration_from_internal_valid": int(best_iter),
        "internal_valid": {
            "metrics": valid_metrics,
            "fold_error": valid_fold,
        },
        "external_pubchem_only_single_model_no_leakage": {
            "metrics": external_single_metrics,
            "fold_error": external_single_fold,
        },
        "external_pubchem_only_bagging_no_leakage": {
            "metrics": external_bagging_metrics,
            "fold_error": external_bagging_fold,
        },
        "single_seed_records": seed_records,
        "output_files": {
            "rule_cleaned_aggregated_all": os.path.join(OUTPUT_DIR, "rule_cleaned_aggregated_all.csv"),
            "external_pubchem_only_rule_cleaned_no_oof": os.path.join(OUTPUT_DIR, "external_pubchem_only_rule_cleaned_no_oof.csv"),
            "train_pool_oof_residual_report": os.path.join(OUTPUT_DIR, "train_pool_oof_residual_report.csv"),
            "removed_by_oof_train_pool_only": os.path.join(OUTPUT_DIR, "removed_by_oof_train_pool_only.csv"),
            "external_predictions_single": os.path.join(OUTPUT_DIR, "external_pubchem_only_predictions_single_no_leakage.csv"),
            "external_predictions_bagging": os.path.join(OUTPUT_DIR, "external_pubchem_only_predictions_bagging_no_leakage.csv"),
            "seed_results": os.path.join(OUTPUT_DIR, "external_pubchem_only_single_seed_results_no_leakage.csv"),
        }
    }

    summary_path = os.path.join(
        OUTPUT_DIR,
        "no_leakage_pubchem_external_validation_summary.json"
    )

    with open(summary_path, "w", encoding="utf-8") as f:
        json.dump(to_jsonable(summary), f, ensure_ascii=False, indent=4)

    print("\n\n===================================================")
    print("完全无泄漏 PubChem-only 来源外验证完成")
    print("===================================================")

    print("\n结果目录：")
    print(OUTPUT_DIR)

    print("\n总结文件：")
    print(summary_path)

    print("\n重点结果：")
    print("Internal valid:", valid_metrics)
    print("External PubChem-only single:", external_single_metrics)
    print("External PubChem-only bagging:", external_bagging_metrics)



def save_reproduction_metadata():
    names = [name for name, _ in RDKit_2D_DESC_LIST]
    if len(names) != 217:
        raise RuntimeError(f"This study expects 217 RDKit descriptors; found {len(names)}. See environment/.")
    versions = {"python": platform.python_version(), "platform": platform.platform()}
    for package in ["numpy","pandas","scikit-learn","rdkit","xgboost","joblib","tqdm","openpyxl"]:
        try:
            versions[package] = importlib.metadata.version(package)
        except importlib.metadata.PackageNotFoundError:
            versions[package] = "not recorded"
    with open(os.path.join(OUTPUT_DIR, "run_environment.json"), "w", encoding="utf-8") as f:
        json.dump(versions, f, indent=2)
    with open(os.path.join(OUTPUT_DIR, "feature_names.json"), "w", encoding="utf-8") as f:
        json.dump([f"ECFP4_count_bit_{i}" for i in range(ECFP_BITS)] + [f"RDKit2D_{n}" for n in names], f, indent=2)

if __name__ == "__main__":
    save_reproduction_metadata()
    main()
