"""Recalculate available revision results from archived records, without fitting models.

Run from any directory. Outputs go to results/recomputed/revision by default.
The threshold split export is reconstructed from the supplied algorithm and row
order; it is not represented as a recovered historical split file.
"""
from pathlib import Path
import argparse
import json
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
SMILES = "Canonical SMILES"
ARMS = ["no_filter", "filter_10fold", "filter_20fold", "filter_30fold"]


def read(path):
    return pd.read_csv(path, encoding="utf-8-sig", float_precision="round_trip")


def require(condition, message):
    if not condition:
        raise ValueError(message)


def close(actual, expected, context, tol=1e-8):
    require(np.allclose(actual, expected, rtol=0, atol=tol), context)


def metrics(y, prediction):
    y, prediction = np.asarray(y, float), np.asarray(prediction, float)
    require(y.ndim == 1 and y.shape == prediction.shape and len(y) > 1,
            "Invalid metric input shape")
    require(np.isfinite(y).all() and np.isfinite(prediction).all(), "Nonfinite predictions/targets")
    absolute = np.abs(prediction-y)
    sse, sst = float(np.sum((prediction-y)**2)), float(np.sum((y-y.mean())**2))
    require(sst > 0 and absolute.max() < 300, "Invalid target variation or units")
    return dict(n=len(y), R2=1-sse/sst, RMSE=float(np.sqrt(sse/len(y))),
                MAE=float(absolute.mean()), median_fold_error=float(np.median(10**absolute)),
                within_2_fold_percent=float(100*np.mean(absolute <= np.log10(2))),
                within_5_fold_percent=float(100*np.mean(absolute <= np.log10(5))),
                within_10_fold_percent=float(100*np.mean(absolute <= 1)), SSE=sse, SST=sst)


def save(frame, path):
    path.parent.mkdir(parents=True, exist_ok=True)
    frame.to_csv(path, index=False, encoding="utf-8-sig", float_format="%.17g")


def subgroup_results(output):
    ad = read(ROOT/"predictions/source_holdout/external_with_ad.csv")
    original = read(ROOT/"predictions/source_holdout/external_bagging.csv")
    compounds = read(ROOT/"data/model_compounds.csv").set_index(SMILES)
    for frame in (ad, original):
        require(len(frame) == 1068 and frame[SMILES].is_unique and frame[SMILES].notna().all(),
                "Expected 1,068 uniquely identified external compounds")
        require(frame.source_type.eq("PubChem_only").all(), "Unexpected source membership")
    require(set(ad[SMILES]) == set(original[SMILES]), "External identities differ")
    original = original.set_index(SMILES).loc[ad[SMILES]]
    for column in ("y_true_neglog_molkg", "y_pred_neglog_molkg"):
        close(ad[column], original[column], "Frozen prediction/target mismatch", 1e-12)
    s = ad.max_similarity.to_numpy(float)
    require(np.isfinite(s).all() and ((s >= 0) & (s <= 1)).all(), "Invalid similarity values")
    masks = {
        "all_pubchem_only": np.ones(1068, dtype=bool),
        "fingerprint_identical": s == 1.0,
        "fingerprint_nonidentical": s < 1.0,
        "in_domain": s >= 0.70,
        "similarity_ge_0p70_lt_1": (s >= 0.70) & (s < 1.0),
        "out_of_domain": s < 0.70,
    }
    expected_n = [1068, 641, 427, 733, 92, 335]
    expected_r2 = [.7440, .9794, .4290, .9136, .5268, .3884]
    rows = []
    for (name, mask), n, r2 in zip(masks.items(), expected_n, expected_r2):
        require(mask.sum() == n, f"Unexpected subgroup size: {name}")
        row = metrics(ad.loc[mask, "y_true_neglog_molkg"], ad.loc[mask, "y_pred_neglog_molkg"])
        close(row["R2"], r2, f"Reported R2 mismatch: {name}", 5.01e-5)
        rows.append(dict(subset=name, **row))
    groups = ad[[SMILES, "source_type", "y_true_neglog_molkg", "y_pred_neglog_molkg", "max_similarity"]].copy()
    groups.insert(0, "compound_id", compounds.loc[ad[SMILES], "compound_id"].to_numpy())
    groups["similarity_group"] = np.where(s == 1, "fingerprint_identical",
                                         np.where(s >= .7, "similarity_ge_0p70_lt_1", "out_of_domain"))
    groups["fingerprint_nonidentical"] = s < 1
    groups["in_domain_at_0p70"] = s >= .7
    save(groups, output/"pubchem_similarity_groups.csv")
    table = pd.DataFrame(rows)
    save(table, output/"pubchem_subgroup_metrics.csv")
    return dict(n=1068, mutually_exclusive_groups=[641, 92, 335], fixed_predictions_match=True)


def random_results(development, output):
    splits = read(ROOT/"splits/revision/random_nested/outer_folds.csv").set_index("row_id")
    require(len(splits) == 34959 and splits.index.is_unique, "Invalid random outer splits")
    require(set(splits.index) == set(range(len(development))), "Random split row coverage")
    summary = read(ROOT/"results/revision/random_nested/random_nested_cv_comparison.csv").set_index("arm")
    folds = read(ROOT/"results/revision/random_nested/outer_fold_metrics.csv")
    rows = []
    for arm in ARMS:
        pred = read(ROOT/f"predictions/revision/random_nested/{arm}.csv")
        require(len(pred) == 34959 and pred.row_id.is_unique and pred.arm.eq(arm).all(), f"Invalid {arm} predictions")
        require(set(pred.row_id) == set(splits.index), "Prediction row coverage differs")
        matched = splits.loc[pred.row_id]
        require(np.array_equal(matched.outer_fold, pred.outer_fold), "Prediction fold mismatch")
        require(np.array_equal(matched[SMILES], pred[SMILES]), "Prediction structure mismatch")
        close(development.iloc[pred.row_id].y, pred.y_true, "Prediction target mismatch", 1e-12)
        row = metrics(pred.y_true, pred.y_pred)
        for k in row:
            close(row[k], summary.loc[arm, k], f"Random summary mismatch: {arm}/{k}")
        for fold, block in pred.groupby("outer_fold"):
            reference = folds[(folds.arm == arm) & (folds.outer_fold == fold)]
            require(len(reference) == 1, "Missing/duplicated fold summary")
            for k, v in metrics(block.y_true, block.y_pred).items():
                close(v, reference.iloc[0][k], f"Random fold metric mismatch: {arm}/{fold}/{k}")
        rows.append(dict(arm=arm, **row))
    for fold in range(1, 6):
        outer_test = set(splits.index[splits.outer_fold == fold])
        outer_train = set(splits.index)-outer_test
        screening = read(ROOT/f"data/revision/random_nested/outer_fold_{fold:02d}_screening.csv")
        require(screening.row_id.is_unique and set(screening.row_id) == outer_train,
                "Screening rows differ from outer training partition")
        with np.load(ROOT/f"splits/revision/random_nested/outer_fold_{fold:02d}_inner_indices.npz", allow_pickle=False) as archive:
            for inner in range(1, 6):
                prefix = f"inner_{inner:02d}_"
                tr = set(archive[prefix+"oof_train"])
                va = set(archive[prefix+"oof_prediction"])
                pilot_tr, pilot_va = set(archive[prefix+"pilot_train"]), set(archive[prefix+"pilot_valid"])
                require(tr.isdisjoint(va) and tr|va == outer_train, "Inner split partition mismatch")
                require(pilot_tr.isdisjoint(pilot_va) and pilot_tr|pilot_va == tr, "Pilot split partition mismatch")
                require(set(screening.loc[screening.inner_fold == inner, "row_id"]) == va,
                        "Inner screening fold mismatch")
            for arm in ARMS:
                actual = set(archive[arm+"_train"])
                threshold = None if arm == "no_filter" else float(arm.removeprefix("filter_").removesuffix("fold"))
                keep = np.ones(len(screening), bool) if threshold is None else screening.new_abs_log_error < np.log10(threshold)
                require(actual == set(screening.loc[keep, "row_id"]), "Final training selection mismatch")
                require(set(archive[arm+"_prediction"]) == outer_test, "Final prediction partition mismatch")
    save(pd.DataFrame(rows), output/"random_nested_recomputed.csv")
    return dict(arms=4, predictions_per_arm=34959, outer_folds=5, inner_splits_checked=25,
                archived_prediction_metrics_match=True)


def threshold_results(development, output):
    summary = read(ROOT/"results/revision/threshold/development_cv_comparison.csv").set_index("arm")
    folds = read(ROOT/"results/revision/threshold/development_cv_fold_results.csv")
    assignments, checks = [], []
    for threshold in [None, 5, 10, 20, 30]:
        arm = "no_filter" if threshold is None else f"filter_{threshold}fold"
        keep = np.ones(len(development), bool) if threshold is None else development.oof_fold_error < threshold
        row_ids = np.flatnonzero(keep)
        y = development.y.to_numpy(float)[row_ids]
        reference, stats = summary.loc[arm], folds[folds.arm == arm].sort_values("fold")
        require(len(stats) == 5 and int(reference.n_evaluation_cohort) == len(y), "Threshold cohort size mismatch")
        require(int(stats.n.sum()) == len(y), "Threshold fold size mismatch")
        sse = float(stats.SSE.sum())
        pooled = dict(R2=1-sse/np.sum((y-y.mean())**2), RMSE=np.sqrt(sse/len(y)),
                      MAE=np.average(stats.MAE, weights=stats.n),
                      within_2_fold_percent=np.average(stats.within_2_fold_percent, weights=stats.n))
        for k, value in pooled.items():
            close(value, reference["pooled_"+k], f"Threshold summary arithmetic: {arm}/{k}")
            close(stats[k].mean(), reference["fold_mean_"+k], f"Threshold mean: {arm}/{k}")
            close(stats[k].std(ddof=1), reference["fold_sd_"+k], f"Threshold SD: {arm}/{k}")
        # Equivalent to KFold(5, shuffle=True, random_state=42) on this arm's ordered rows.
        shuffled = np.arange(len(row_ids))
        np.random.RandomState(42).shuffle(shuffled)
        assigned = np.zeros(len(row_ids), dtype=int)
        for fold, positions in enumerate(np.array_split(shuffled, 5), 1):
            assigned[positions] = fold
            require(len(positions) == int(stats.iloc[fold-1].n), "Reconstructed fold size mismatch")
        assignments.append(pd.DataFrame(dict(arm=arm, development_row_id=row_ids, evaluation_fold=assigned)))
        checks.append(dict(arm=arm, n=len(row_ids), **pooled))
    save(pd.concat(assignments, ignore_index=True), output/"threshold_cv_assignments_reconstructed.csv")
    save(pd.DataFrame(checks), output/"threshold_summary_arithmetic.csv")
    return dict(arms=5, fold_summary_rows=25, summary_arithmetic_checked=True,
                per_compound_evaluation_predictions_available=False,
                split_export="Reconstructed from code, seed 42 and frozen ordered screening report")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", type=Path, default=ROOT/"results/recomputed/revision")
    args = parser.parse_args()
    output = args.output_dir.resolve()
    output.mkdir(parents=True, exist_ok=True)
    development = read(ROOT/"data/oof/train_pool_oof_residual_report.csv")
    require(len(development) == 34959 and development[SMILES].is_unique, "Invalid development report")
    result = dict(subgroups=subgroup_results(output), random_nested=random_results(development, output),
                  threshold=threshold_results(development, output),
                  scaffold=dict(supplied="Code and reported summary table", per_compound_recomputation=False))
    (output/"verification.json").write_text(json.dumps(result, indent=2), encoding="utf-8")
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
