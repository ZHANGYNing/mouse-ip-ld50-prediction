import os
import math
import json
import warnings

import numpy as np
import pandas as pd

from tqdm import tqdm

import matplotlib.pyplot as plt

from rdkit import Chem, DataStructs, RDLogger
from rdkit.Chem import Descriptors, rdFingerprintGenerator

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

REPOSITORY_ROOT = Path(__file__).resolve().parents[1]
_parser = argparse.ArgumentParser(description="TreeSHAP for the original single XGBoost model")
_parser.add_argument("--model-dir", type=Path, default=REPOSITORY_ROOT / "results/generated/main")
_parser.add_argument("--output-dir", type=Path, default=REPOSITORY_ROOT / "results/generated/shap")
_args = _parser.parse_args() if __name__ == "__main__" else _parser.parse_args([])
BASE_DIR = str(REPOSITORY_ROOT)
CLEAN_DIR = str(_args.model_dir.resolve())
NO_LEAK_DIR = CLEAN_DIR
OUTPUT_DIR = str(_args.output_dir.resolve())
os.makedirs(OUTPUT_DIR, exist_ok=True)
TRAIN_FILE = os.path.join(NO_LEAK_DIR, "train_after_oof_no_leakage.csv")
VALID_FILE = os.path.join(NO_LEAK_DIR, "valid_after_oof_no_leakage.csv")
EXTERNAL_FILE = os.path.join(NO_LEAK_DIR, "external_pubchem_only_final_no_oof.csv")
MODEL_FILE = os.path.join(NO_LEAK_DIR, "final_single_xgb_no_leakage.model")

# =========================================================
# 2. SHAP 分析参数
# =========================================================

EXPLAIN_SET = "external_pubchem_only"
# 可选：
# EXPLAIN_SET = "external_pubchem_only"
# EXPLAIN_SET = "internal_valid"

MAX_EXPLAIN_SAMPLES = 2000
RANDOM_SEED = 42

TOP_N_ALL_FEATURES = 40
TOP_N_RDKIT_FEATURES = 30
TOP_N_DEPENDENCE = 8
TOP_N_LOCAL_FEATURES = 15

ECFP_RADIUS = 2
ECFP_BITS = 2048


# =========================================================
# 3. 工具函数
# =========================================================

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
        "fold_error_array": fold_error,
    }


def safe_spearman(x, y):
    try:
        return float(pd.Series(x).corr(pd.Series(y), method="spearman"))
    except Exception:
        return np.nan


def ensure_files():
    required = [
        TRAIN_FILE,
        VALID_FILE,
        EXTERNAL_FILE,
        MODEL_FILE,
    ]

    for f in required:
        if not os.path.exists(f):
            raise FileNotFoundError(f"找不到文件：{f}")


# =========================================================
# 4. 特征计算：ECFP4 count + RDKit 2D
# =========================================================

morgan_generator = rdFingerprintGenerator.GetMorganGenerator(
    radius=ECFP_RADIUS,
    fpSize=ECFP_BITS
)

RDKit_2D_DESC_LIST = Descriptors._descList
RDKit_2D_NAMES = [x[0] for x in RDKit_2D_DESC_LIST]


def smiles_to_ecfp4_count(smiles):
    arr = np.zeros((ECFP_BITS,), dtype=np.float32)

    mol = Chem.MolFromSmiles(str(smiles))

    if mol is None:
        return arr

    fp = morgan_generator.GetCountFingerprint(mol)
    DataStructs.ConvertToNumpyArray(fp, arr)

    # count fingerprint 做 log1p，和之前模型保持一致
    arr = np.log1p(arr)

    return arr.astype(np.float32)


def smiles_to_rdkit2d(smiles):
    mol = Chem.MolFromSmiles(str(smiles))

    if mol is None:
        return np.full(len(RDKit_2D_DESC_LIST), np.nan, dtype=np.float32)

    values = []

    for name, func in RDKit_2D_DESC_LIST:
        try:
            v = float(func(mol))
        except Exception:
            v = np.nan

        if not np.isfinite(v):
            v = np.nan

        values.append(v)

    return np.array(values, dtype=np.float32)


def build_raw_features(df, split_name):
    ecfp_list = []
    desc_list = []

    smiles_list = df["Canonical SMILES"].tolist()

    for smi in tqdm(smiles_list, desc=f"Calculating features: {split_name}"):
        ecfp_list.append(smiles_to_ecfp4_count(smi))
        desc_list.append(smiles_to_rdkit2d(smi))

    X_ecfp = np.vstack(ecfp_list).astype(np.float32)
    X_desc = np.vstack(desc_list).astype(np.float32)

    X_raw = np.hstack([X_ecfp, X_desc]).astype(np.float32)
    X_raw = np.where(np.isfinite(X_raw), X_raw, np.nan)

    raw_feature_names = (
        [f"ECFP4_count_bit_{i}" for i in range(ECFP_BITS)]
        +
        [f"RDKit2D_{name}" for name in RDKit_2D_NAMES]
    )

    return X_raw, raw_feature_names


def fit_preprocess_train_transform_explain(X_train_raw, X_explain_raw, raw_feature_names):
    """
    注意：
    这里只在 train_full 上 fit imputer 和 variance selector，
    external set 不参与任何 fit。
    """

    keep_cols = ~np.all(np.isnan(X_train_raw), axis=0)

    X_train_keep = X_train_raw[:, keep_cols]
    X_explain_keep = X_explain_raw[:, keep_cols]

    feature_names_keep = np.array(raw_feature_names)[keep_cols].tolist()

    imputer = SimpleImputer(strategy="median")
    X_train_imp = imputer.fit_transform(X_train_keep).astype(np.float32)
    X_explain_imp = imputer.transform(X_explain_keep).astype(np.float32)

    selector = VarianceThreshold(threshold=0.0)
    X_train_sel = selector.fit_transform(X_train_imp).astype(np.float32)
    X_explain_sel = selector.transform(X_explain_imp).astype(np.float32)

    support = selector.get_support()
    feature_names_selected = np.array(feature_names_keep)[support].tolist()

    preprocess_info = {
        "raw_dim": int(X_train_raw.shape[1]),
        "after_all_nan_filter_dim": int(np.sum(keep_cols)),
        "after_variance_filter_dim": int(X_train_sel.shape[1]),
    }

    return X_train_sel, X_explain_sel, feature_names_selected, preprocess_info


# =========================================================
# 5. 作图函数
# =========================================================

def plot_barh(df, value_col, label_col, title, xlabel, save_path, top_n=30):
    plot_df = df.head(top_n).copy()
    plot_df = plot_df.iloc[::-1]

    plt.figure(figsize=(9, max(5, 0.28 * len(plot_df))))
    plt.barh(plot_df[label_col], plot_df[value_col])
    plt.xlabel(xlabel)
    plt.title(title)
    plt.tight_layout()
    plt.savefig(save_path, dpi=300)
    plt.close()


def plot_group_importance(group_df, save_path):
    plot_df = group_df.copy().sort_values("mean_abs_shap", ascending=True)

    plt.figure(figsize=(7, 4))
    plt.barh(plot_df["feature_group"], plot_df["mean_abs_shap"])
    plt.xlabel("Mean absolute SHAP value")
    plt.title("SHAP contribution by feature group")
    plt.tight_layout()
    plt.savefig(save_path, dpi=300)
    plt.close()


def plot_dependence(feature_values, shap_values_one_feature, feature_name, save_path):
    plt.figure(figsize=(6, 5))
    plt.scatter(feature_values, shap_values_one_feature, s=12, alpha=0.55)
    plt.axhline(0, linewidth=1)
    plt.xlabel(feature_name)
    plt.ylabel("SHAP value")
    plt.title(f"Dependence plot: {feature_name}")
    plt.tight_layout()
    plt.savefig(save_path, dpi=300)
    plt.close()


def plot_local_bar(local_df, title, save_path):
    plot_df = local_df.copy()
    plot_df["abs_shap"] = plot_df["shap_value"].abs()
    plot_df = plot_df.sort_values("abs_shap", ascending=False).head(TOP_N_LOCAL_FEATURES)
    plot_df = plot_df.sort_values("shap_value", ascending=True)

    plt.figure(figsize=(9, max(5, 0.35 * len(plot_df))))
    plt.barh(plot_df["feature"], plot_df["shap_value"])
    plt.axvline(0, linewidth=1)
    plt.xlabel("SHAP value")
    plt.title(title)
    plt.tight_layout()
    plt.savefig(save_path, dpi=300)
    plt.close()


# =========================================================
# 6. 主程序
# =========================================================

def main():
    ensure_files()

    print("读取数据...")

    train_df = pd.read_csv(TRAIN_FILE)
    valid_df = pd.read_csv(VALID_FILE)
    external_df = pd.read_csv(EXTERNAL_FILE)

    train_full_df = pd.concat([train_df, valid_df], axis=0).reset_index(drop=True)

    if EXPLAIN_SET == "external_pubchem_only":
        explain_df = external_df.copy().reset_index(drop=True)
    elif EXPLAIN_SET == "internal_valid":
        explain_df = valid_df.copy().reset_index(drop=True)
    else:
        raise ValueError("EXPLAIN_SET 只能是 external_pubchem_only 或 internal_valid")

    print("train_full:", len(train_full_df))
    print("explain set:", EXPLAIN_SET, len(explain_df))

    # =====================================================
    # 6.1 计算特征
    # =====================================================

    print("\n计算 train_full 特征...")
    X_train_raw, raw_feature_names = build_raw_features(train_full_df, "train_full")

    print("\n计算 explain set 特征...")
    X_explain_raw, _ = build_raw_features(explain_df, EXPLAIN_SET)

    X_train, X_explain, feature_names, preprocess_info = fit_preprocess_train_transform_explain(
        X_train_raw,
        X_explain_raw,
        raw_feature_names
    )

    print("\n特征预处理信息：")
    print(preprocess_info)
    print("X_train:", X_train.shape)
    print("X_explain:", X_explain.shape)

    # =====================================================
    # 6.2 如果 explain set 太大，抽样计算 SHAP
    # =====================================================

    rng = np.random.default_rng(RANDOM_SEED)

    if len(explain_df) > MAX_EXPLAIN_SAMPLES:
        explain_indices = rng.choice(
            np.arange(len(explain_df)),
            size=MAX_EXPLAIN_SAMPLES,
            replace=False
        )
        explain_indices = np.sort(explain_indices)
    else:
        explain_indices = np.arange(len(explain_df))

    explain_sample_df = explain_df.iloc[explain_indices].copy().reset_index(drop=True)
    X_shap = X_explain[explain_indices]

    print("用于 SHAP 的样本数：", len(explain_sample_df))

    # =====================================================
    # 6.3 加载 XGBoost 模型
    # =====================================================

    booster = xgb.Booster()
    booster.load_model(MODEL_FILE)

    model_n_features = booster.num_features()

    if model_n_features != X_train.shape[1]:
        raise ValueError(
            f"模型特征数和当前特征数不一致：\n"
            f"model_n_features = {model_n_features}\n"
            f"current_feature_dim = {X_train.shape[1]}\n"
            f"请确认用的是 no_leakage 脚本生成的模型和同一套特征。"
        )

    print("\n模型特征数匹配：", model_n_features)

    # =====================================================
    # 6.4 预测和性能
    # =====================================================

    d_explain_all = xgb.DMatrix(X_explain)
    pred_all = booster.predict(d_explain_all)

    y_true_all = explain_df["y"].values.astype(float)
    mw_all = explain_df["molecular_weight"].values.astype(float)

    metrics_all = calc_metrics(y_true_all, pred_all)
    fold_all = calc_fold_error(y_true_all, pred_all, mw_all)

    fold_error_array = fold_all.pop("fold_error_array")

    prediction_df = explain_df.copy()
    prediction_df["y_pred"] = pred_all
    prediction_df["abs_error_y"] = np.abs(prediction_df["y"].values - pred_all)
    prediction_df["fold_error"] = fold_error_array

    prediction_df.to_csv(
        os.path.join(OUTPUT_DIR, f"{EXPLAIN_SET}_predictions_for_shap.csv"),
        index=False,
        encoding="utf-8-sig"
    )

    print("\n解释集预测结果：")
    print(metrics_all)
    print(fold_all)

    # =====================================================
    # 6.5 TreeSHAP
    # =====================================================

    print("\n正在计算 TreeSHAP...")

    d_shap = xgb.DMatrix(X_shap)

    # XGBoost 内置 TreeSHAP，最后一列是 bias / expected value
    contribs = booster.predict(d_shap, pred_contribs=True)

    shap_values = contribs[:, :-1]
    expected_values = contribs[:, -1]

    pred_sample = booster.predict(d_shap)

    shap_reconstruct = shap_values.sum(axis=1) + expected_values

    max_diff = float(np.max(np.abs(shap_reconstruct - pred_sample)))

    print("SHAP 重构预测最大误差：", max_diff)

    np.save(os.path.join(OUTPUT_DIR, f"{EXPLAIN_SET}_shap_values.npy"), shap_values)
    np.save(os.path.join(OUTPUT_DIR, f"{EXPLAIN_SET}_X_shap.npy"), X_shap)

    pd.DataFrame({
        "sample_index_in_original_explain_set": explain_indices,
        "expected_value": expected_values,
        "prediction_from_model": pred_sample,
        "prediction_from_shap_sum": shap_reconstruct,
        "shap_reconstruct_abs_diff": np.abs(shap_reconstruct - pred_sample)
    }).to_csv(
        os.path.join(OUTPUT_DIR, f"{EXPLAIN_SET}_shap_reconstruction_check.csv"),
        index=False,
        encoding="utf-8-sig"
    )

    # =====================================================
    # 6.6 全部特征 SHAP 重要性
    # =====================================================

    mean_abs_shap = np.mean(np.abs(shap_values), axis=0)
    mean_shap = np.mean(shap_values, axis=0)

    feature_type = [
        "ECFP4_count" if name.startswith("ECFP4_count_bit_") else "RDKit2D"
        for name in feature_names
    ]

    importance_df = pd.DataFrame({
        "feature": feature_names,
        "feature_type": feature_type,
        "mean_abs_shap": mean_abs_shap,
        "mean_shap": mean_shap,
    })

    # 计算方向性：特征值和 SHAP value 的 Spearman 相关
    shap_corr_list = []

    for j in range(len(feature_names)):
        corr = safe_spearman(X_shap[:, j], shap_values[:, j])
        shap_corr_list.append(corr)

    importance_df["spearman_feature_value_vs_shap"] = shap_corr_list

    importance_df["interpretation_for_y"] = np.where(
        importance_df["spearman_feature_value_vs_shap"] > 0,
        "Higher feature value tends to increase predicted y, indicating higher toxicity / lower LD50",
        np.where(
            importance_df["spearman_feature_value_vs_shap"] < 0,
            "Higher feature value tends to decrease predicted y, indicating lower toxicity / higher LD50",
            "No clear monotonic direction"
        )
    )

    importance_df = importance_df.sort_values("mean_abs_shap", ascending=False).reset_index(drop=True)

    importance_df.to_csv(
        os.path.join(OUTPUT_DIR, "shap_feature_importance_all_features.csv"),
        index=False,
        encoding="utf-8-sig"
    )

    rdkit_importance_df = (
        importance_df[importance_df["feature_type"] == "RDKit2D"]
        .copy()
        .reset_index(drop=True)
    )

    rdkit_importance_df.to_csv(
        os.path.join(OUTPUT_DIR, "shap_feature_importance_rdkit2d_only.csv"),
        index=False,
        encoding="utf-8-sig"
    )

    print("\nTop 20 all features:")
    print(importance_df.head(20)[["feature", "feature_type", "mean_abs_shap", "spearman_feature_value_vs_shap"]])

    print("\nTop 20 RDKit 2D descriptors:")
    print(rdkit_importance_df.head(20)[["feature", "mean_abs_shap", "spearman_feature_value_vs_shap"]])

    # =====================================================
    # 6.7 分组重要性：ECFP vs RDKit 2D
    # =====================================================

    group_df = (
        importance_df
        .groupby("feature_type")["mean_abs_shap"]
        .sum()
        .reset_index()
        .rename(columns={"feature_type": "feature_group"})
    )

    group_df["relative_percent"] = group_df["mean_abs_shap"] / group_df["mean_abs_shap"].sum() * 100

    group_df.to_csv(
        os.path.join(OUTPUT_DIR, "shap_group_importance_ecfp_vs_rdkit2d.csv"),
        index=False,
        encoding="utf-8-sig"
    )

    print("\n分组 SHAP 贡献：")
    print(group_df)

    # =====================================================
    # 6.8 画全局图
    # =====================================================

    plot_barh(
        importance_df,
        value_col="mean_abs_shap",
        label_col="feature",
        title=f"Top {TOP_N_ALL_FEATURES} global SHAP importance - all features",
        xlabel="Mean absolute SHAP value",
        save_path=os.path.join(OUTPUT_DIR, f"shap_top{TOP_N_ALL_FEATURES}_all_features.png"),
        top_n=TOP_N_ALL_FEATURES
    )

    plot_barh(
        rdkit_importance_df,
        value_col="mean_abs_shap",
        label_col="feature",
        title=f"Top {TOP_N_RDKIT_FEATURES} global SHAP importance - RDKit 2D descriptors",
        xlabel="Mean absolute SHAP value",
        save_path=os.path.join(OUTPUT_DIR, f"shap_top{TOP_N_RDKIT_FEATURES}_rdkit2d_descriptors.png"),
        top_n=TOP_N_RDKIT_FEATURES
    )

    plot_group_importance(
        group_df,
        save_path=os.path.join(OUTPUT_DIR, "shap_group_importance_ecfp_vs_rdkit2d.png")
    )

    # =====================================================
    # 6.9 RDKit 描述符 dependence plot
    # =====================================================

    dependence_dir = os.path.join(OUTPUT_DIR, "dependence_plots_rdkit2d")
    os.makedirs(dependence_dir, exist_ok=True)

    top_rdkit_features = rdkit_importance_df.head(TOP_N_DEPENDENCE)["feature"].tolist()

    dependence_records = []

    feature_to_index = {f: i for i, f in enumerate(feature_names)}

    for feat in top_rdkit_features:
        j = feature_to_index[feat]

        save_path = os.path.join(
            dependence_dir,
            f"dependence_{feat.replace('/', '_').replace(' ', '_')}.png"
        )

        plot_dependence(
            feature_values=X_shap[:, j],
            shap_values_one_feature=shap_values[:, j],
            feature_name=feat,
            save_path=save_path
        )

        dependence_records.append({
            "feature": feat,
            "feature_index": int(j),
            "mean_abs_shap": float(importance_df.loc[importance_df["feature"] == feat, "mean_abs_shap"].iloc[0]),
            "spearman_feature_value_vs_shap": float(importance_df.loc[importance_df["feature"] == feat, "spearman_feature_value_vs_shap"].iloc[0]),
            "plot_file": save_path
        })

    pd.DataFrame(dependence_records).to_csv(
        os.path.join(OUTPUT_DIR, "dependence_plots_rdkit2d_index.csv"),
        index=False,
        encoding="utf-8-sig"
    )

    # =====================================================
    # 6.10 局部解释：几个代表样本
    # =====================================================

    local_dir = os.path.join(OUTPUT_DIR, "local_explanations")
    os.makedirs(local_dir, exist_ok=True)

    # 注意：这里从 explain_indices 对应的 SHAP 子样本中选代表样本
    sample_pred_df = prediction_df.iloc[explain_indices].copy().reset_index(drop=True)
    sample_pred_df["sample_pos_in_shap_matrix"] = np.arange(len(sample_pred_df))

    selected_samples = []

    # 预测毒性最高：y_pred 最大，LD50 越低
    selected_samples.append((
        "highest_predicted_toxicity",
        int(sample_pred_df["y_pred"].idxmax())
    ))

    # 预测毒性最低：y_pred 最小，LD50 越高
    selected_samples.append((
        "lowest_predicted_toxicity",
        int(sample_pred_df["y_pred"].idxmin())
    ))

    # 误差最大
    selected_samples.append((
        "largest_fold_error",
        int(sample_pred_df["fold_error"].idxmax())
    ))

    # 误差接近中位数
    median_fe = sample_pred_df["fold_error"].median()
    selected_samples.append((
        "median_fold_error_case",
        int((sample_pred_df["fold_error"] - median_fe).abs().idxmin())
    ))

    local_records = []

    for label, pos in selected_samples:
        row_info = sample_pred_df.iloc[pos].copy()
        shap_row = shap_values[pos]
        x_row = X_shap[pos]

        local_df = pd.DataFrame({
            "feature": feature_names,
            "feature_value": x_row,
            "shap_value": shap_row,
            "abs_shap": np.abs(shap_row),
            "feature_type": [
                "ECFP4_count" if name.startswith("ECFP4_count_bit_") else "RDKit2D"
                for name in feature_names
            ]
        }).sort_values("abs_shap", ascending=False).reset_index(drop=True)

        local_csv = os.path.join(local_dir, f"local_{label}_top_contributions.csv")

        local_df.head(60).to_csv(
            local_csv,
            index=False,
            encoding="utf-8-sig"
        )

        local_png = os.path.join(local_dir, f"local_{label}_top{TOP_N_LOCAL_FEATURES}.png")

        plot_local_bar(
            local_df,
            title=(
                f"{label}\n"
                f"y_true={row_info['y']:.3f}, y_pred={row_info['y_pred']:.3f}, "
                f"fold_error={row_info['fold_error']:.2f}"
            ),
            save_path=local_png
        )

        local_records.append({
            "case_label": label,
            "sample_pos_in_shap_matrix": int(pos),
            "original_index_in_explain_set": int(explain_indices[pos]),
            "Canonical SMILES": row_info.get("Canonical SMILES", ""),
            "y_true": float(row_info["y"]),
            "y_pred": float(row_info["y_pred"]),
            "abs_error_y": float(row_info["abs_error_y"]),
            "fold_error": float(row_info["fold_error"]),
            "local_csv": local_csv,
            "local_plot": local_png,
        })

    pd.DataFrame(local_records).to_csv(
        os.path.join(OUTPUT_DIR, "local_explanation_cases_index.csv"),
        index=False,
        encoding="utf-8-sig"
    )

    # =====================================================
    # 6.11 保存总结
    # =====================================================

    summary = {
        "experiment": "SHAP analysis for no-leakage XGBoost model",
        "model_file": MODEL_FILE,
        "explain_set": EXPLAIN_SET,
        "n_train_full": int(len(train_full_df)),
        "n_explain_all": int(len(explain_df)),
        "n_explain_used_for_shap": int(len(explain_sample_df)),
        "target": "-log10(LD50 mol/kg)",
        "interpretation": {
            "positive_shap": "increases predicted y, indicating lower LD50 and higher acute toxicity",
            "negative_shap": "decreases predicted y, indicating higher LD50 and lower acute toxicity"
        },
        "metrics_on_explain_set": metrics_all,
        "fold_error_on_explain_set": fold_all,
        "preprocess_info": preprocess_info,
        "model_num_features": int(model_n_features),
        "shap_reconstruction_max_abs_diff": max_diff,
        "top_10_all_features": importance_df.head(10).to_dict(orient="records"),
        "top_10_rdkit2d_features": rdkit_importance_df.head(10).to_dict(orient="records"),
        "group_importance": group_df.to_dict(orient="records"),
        "output_files": {
            "all_feature_importance": os.path.join(OUTPUT_DIR, "shap_feature_importance_all_features.csv"),
            "rdkit2d_feature_importance": os.path.join(OUTPUT_DIR, "shap_feature_importance_rdkit2d_only.csv"),
            "group_importance": os.path.join(OUTPUT_DIR, "shap_group_importance_ecfp_vs_rdkit2d.csv"),
            "top_all_plot": os.path.join(OUTPUT_DIR, f"shap_top{TOP_N_ALL_FEATURES}_all_features.png"),
            "top_rdkit_plot": os.path.join(OUTPUT_DIR, f"shap_top{TOP_N_RDKIT_FEATURES}_rdkit2d_descriptors.png"),
            "group_plot": os.path.join(OUTPUT_DIR, "shap_group_importance_ecfp_vs_rdkit2d.png"),
            "dependence_plot_dir": dependence_dir,
            "local_explanation_dir": local_dir,
        }
    }

    with open(os.path.join(OUTPUT_DIR, "shap_analysis_summary.json"), "w", encoding="utf-8") as f:
        json.dump(summary, f, ensure_ascii=False, indent=4)

    print("\n========== SHAP 分析完成 ==========")
    print("输出目录：")
    print(OUTPUT_DIR)

    print("\n最重要的文件：")
    print(os.path.join(OUTPUT_DIR, "shap_feature_importance_rdkit2d_only.csv"))
    print(os.path.join(OUTPUT_DIR, f"shap_top{TOP_N_RDKIT_FEATURES}_rdkit2d_descriptors.png"))
    print(os.path.join(OUTPUT_DIR, "shap_group_importance_ecfp_vs_rdkit2d.csv"))
    print(os.path.join(OUTPUT_DIR, "shap_analysis_summary.json"))


if __name__ == "__main__":
    main()
