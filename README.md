# Mouse intraperitoneal LD50 prediction

Code, compound-level data, fixed assignments, exclusion records and saved predictions for the main source-holdout workflow in **Data quality-aware machine learning for reliable prediction of mouse intraperitoneal acute toxicity**.

Repository: <https://github.com/ZHANGYNing/mouse-ip-ld50-prediction>

The workflow integrates TOXRIC and PubChem records, applies rule-based curation, reserves PubChem-only compounds before residual screening, screens the development pool with five-fold out-of-fold (OOF) predictions, and trains single-seed and ten-seed XGBoost models. Features are log1p-transformed ECFP4 count fingerprints and RDKit 2D descriptors. Applicability-domain (AD) analysis and TreeSHAP support interpretation of the saved predictions.

This package covers the original main workflow. Random/scaffold nested validation and threshold-sensitivity experiments are separate supplementary analyses and are not included here. Detailed database verification and structural annotations for the 448 high-residual compounds belong in the manuscript's Supplementary Information; the computational exclusion list is included here.

## Data and target

| Stage | Number |
|---|---:|
| Original input records | 40,579 |
| Rule-curated unique compounds | 36,027 |
| Development pool before OOF screening | 34,959 |
| Excluded at the original 20-fold residual threshold | 448 |
| Retained development compounds | 34,511 |
| Original internal training / early-stopping validation | 31,059 / 3,452 |
| PubChem-only source-holdout compounds | 1,068 |

Raw `label=1` denotes TOXRIC; `label=2` denotes PubChem. Modeling targets are medians of linked source values after curation. The response is:

```text
y = -log10(LD50_mgkg / (1000 * molecular_weight))
LD50_mgkg = 1000 * molecular_weight * 10**(-y)
fold_error = 10**abs(y_true - y_pred)
```

OOF exclusion is `fold_error >= 20`, and five is the number of CV folds. PubChem-only compounds are not screened by OOF residuals. Source separation does not imply scaffold independence. A high model residual does not establish an erroneous experimental value.

## Contents

| Path | Purpose |
|---|---|
| `data/raw/` | Original merged input, unchanged |
| `data/model_compounds.csv` | Curated compound-level data, stable IDs, source membership and targets |
| `data/processed/` | Exported original training, validation and external tables in fixed order |
| `data/oof/` | Frozen 34,959-compound OOF prediction/residual report |
| `data/exclusions/` | The 448 residual exclusions and replayed rule-based exclusions with reasons |
| `splits/` | Compound-level source/internal assignments and original OOF fold assignments |
| `src/train_and_evaluate.py` | Original curation, screening, training and prediction functions with portable paths |
| `src/shap_analysis.py` | Original TreeSHAP analysis of the single XGBoost model |
| `tools/` | Saved-result verification, fixed-data export, rule replay and AD replay |
| `configs/` | Complete main-model and curation parameter snapshot |
| `predictions/source_holdout/` | Original per-compound single/ensemble predictions and saved AD fields |
| `results/` | Original run summary, AD/SHAP source tables and independently recomputed metrics |
| `provenance/` | Input/code hashes, transformation records and verification scope |
| `docs/` | Protocol, data dictionary and the mapping from reproducibility requirements to files |

## 1. Recompute saved results without training

Use Python 3.12 for the supplied reference installation. From the repository root:

```bash
python -m pip install -r requirements-recompute.txt
python tools/verify_file_integrity.py
python tools/recompute_results.py
```

This checks the cohort sizes, 448 exclusion identities, target transformation, saved split membership, original OOF fold metrics and external predictions. It recalculates AD-stratified metrics from the saved AD fields. Outputs are written to `results/recomputed/`.

Expected ensemble source-holdout results from the frozen predictions are R² = **0.744008**, RMSE = **0.470260**, MAE = **0.244540**, and **78.1835%** within twofold error. R², RMSE and MAE are evaluated on the transformed target scale. Minor differences from historical JSON values reflect stored precision and metric arithmetic.

## 2. Replay curation, fixed data exports and AD

```bash
python -m pip install -r requirements.txt
python tools/replay_rule_curation.py
python tools/export_fixed_data.py
python tools/replay_applicability_domain.py
```

The rule replay calls the unchanged original curation function and checks its output against all 36,027 frozen compounds before exporting exclusions. It reproduced the original 1,198 excluded raw records and 49 excluded aggregated compounds. These counts describe different processing levels.

The fixed source and internal assignments were recovered deterministically from the frozen input order and original seed-42 split calls. Original OOF fold assignments were additionally checked against all five saved fold metrics. `data/processed/` contains exports of those assignments, not newly selected partitions.

The AD replay uses **binary**, achiral 2,048-bit Morgan fingerprints with radius 2 against the 34,511 retained development compounds. This differs from the count/log1p features used for model training. Its 56 AD fields match the saved table, including maximum similarity, mean top-five similarity, neighbor counts and domain flags. This is a new replay utility, validated against the saved results.

## 3. Retrain the original main workflow

```bash
python src/train_and_evaluate.py --device cuda
```

For a CPU run:

```bash
python src/train_and_evaluate.py --device cpu
```

Optional paths:

```bash
python src/train_and_evaluate.py --input data/raw/mouse_intraperitoneal_ld50.xlsx --output-dir results/generated/main --device cuda
```

This performs model fitting and can take substantial time. It recomputes screening using the original algorithm; it does not reuse the frozen OOF predictions as features. Generated results go to `results/generated/main/`, keeping the distributed historical predictions intact. Compare regenerated compound identities and exclusions with the frozen tables before interpreting any new scores.

The original functions, hyperparameters, seeds, early stopping and training logic are preserved. Changes are limited to portable path/device options and recording environment/feature names. The parameter snapshot is in `configs/main_parameters.json`; the operational constants remain visible in the source. The full raw feature count is 2,265 (2,048 count features plus 217 descriptors); the historical fitted feature count was 2,262.

## 4. Run SHAP

After the main training command has generated its model and data files:

```bash
python src/shap_analysis.py --model-dir results/generated/main
```

This explains `final_single_xgb_no_leakage.model`. The archived SHAP analysis explains the **single model**, whereas the main reported ensemble prediction averages ten models. Do not label the supplied SHAP tables as explanations of the ensemble. The CSV importance tables and original SHAP summary are already supplied for inspection without retraining.

## Reproduction coverage

Saved OOF and external predictions can be verified without an ML runtime. The original curation was replayed successfully, all retained identities/source links were checked, and all saved AD fields were reproduced. Main-model training functions were compared with the uploaded originals and are unchanged.

Full XGBoost training and TreeSHAP inference were not rerun while assembling this repository. The requirements specify a reference installation, not a recovered historical main-training environment. Curation/AD were checked with RDKit 2025.09.5; XGBoost import/command checks used the CPU build. Exact historical main-training dependency versions, original trained tree files/preprocessing objects and original internal-holdout per-compound predictions have not yet been archived in this package. See `docs/availability.md` for precise coverage. Saved predictions are authoritative for the reported numerical results; a retraining run should record its actual environment and may not be bit-identical across versions or devices.

## Sources and citation

Input records were supplied as the study's merged [TOXRIC](https://toxric.bioinforai.tech/) and [PubChem](https://pubchem.ncbi.nlm.nih.gov/) dataset. Source labels and raw-row links are retained. See the manuscript and Supplementary Information for database collection and the detailed 448-record source verification.

When citing this code, use this repository URL and the commit or release corresponding to the revised manuscript. Add the final manuscript citation/DOI when available. Data-source attribution is distinct from any license subsequently selected for the repository code.
