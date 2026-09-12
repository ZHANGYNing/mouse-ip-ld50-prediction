# Main-workflow reproducibility coverage

| Requested component | Included files / status |
|---|---|
| Compound-level data | Original raw workbook; all 36,027 curated compounds; source labels, structures, targets, raw-row links; fixed processed partitions |
| Exclusion records | Original 448-record OOF list, complete OOF predictions/flags; replayed raw/group rule exclusions with reasons |
| Fixed splits | Source holdout, internal early-stopping partition and five original screening folds, with explicit compound IDs and order |
| Individual predictions | All 34,959 screening predictions and all 1,068 external single-model/ensemble predictions; saved AD fields |
| Complete main-model parameters | Curation thresholds, feature definition, XGBoost settings, 10 ensemble seeds, early stopping and recorded fitted round count |
| Code | Original main modeling and SHAP functions with portable paths; verified curation, AD and metric replay utilities |

## Exact limits of the supplied archive

1. `valid_predictions_internal_earlystop.csv` is not included. Its 3,452-compound membership and original aggregate metrics are supplied, and the main training script generates this prediction file. Do not claim that the archived internal holdout metric has been independently recomputed from individual predictions.
2. Original trained XGBoost model files and fitted main-model preprocessing objects are not included. Training code generates them. They are needed for inference/TreeSHAP without retraining, not for recalculation of the saved external metrics.
3. The ten per-seed external prediction arrays are not included. The ensemble's per-compound predictions and each seed's aggregate metrics in the original summary are included. The original averaging operation cannot be rechecked against all ten raw arrays until they are archived.
4. Exact original main-training package versions are not recovered. The supplied requirements/reference environment is identified separately from historical model training. Numerical results of new fits must not silently replace the frozen historical results.
5. Supplementary nested validation, threshold comparisons and other historical comparison runs are outside this main-workflow package. Their code/data can be supplied separately with those analyses.

The manuscript's full 448-record database verification belongs in SI. This repository retains the original exclusion identities, predictions and computational rule required to reproduce selection.

## Files to add when the original run directory is available

| Original filename | Purpose |
|---|---|
| `valid_predictions_internal_earlystop.csv` | Exact internal-holdout result verification |
| `external_pubchem_only_predictions_each_seed_no_leakage.csv` | Verify the ten-model average directly |
| `final_single_xgb_no_leakage.model` | Reuse the model explained by the original SHAP analysis |
| `xgb_bagging_seed_*_no_leakage.model` | Reuse all ten original ensemble members |
| `final_train_full_imputer.joblib`, `final_train_full_variance_selector.joblib`, matching preprocessing metadata | Reuse fitted main-model transformations; retain exact feature order |
| Original training environment export / recorded dependency versions | Preserve the historical training runtime |

These filenames refer to the original source-holdout workflow. To capture relevant versions from the original training environment, run `python tools/capture_training_environment.py` there. The resulting file describes that environment at capture time; its historical identity should be established by the author.
