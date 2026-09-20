# Analyses added during revision

The main workflow remains in `src/`. The supplementary comparisons and frozen-prediction subgroup analysis have separate code and result directories. Reported summary files are retained; new training outputs go to `results/generated/revision/`.

| Manuscript/SI item | Code | Results and supporting records |
|---|---|---|
| Table S7: threshold comparison | `revision/threshold/05_development_pool_cv.py`, `devpool_cv_core.py`; `configs/threshold_cv.json` | `results/revision/threshold/`; `splits/revision/threshold_cv_assignments_reconstructed.csv` |
| Table S8: random nested comparison | `revision/random_nested/02_nested_oof_ablation.py`, `revision_common.py`; `configs/random_nested.json` | `results/revision/random_nested/`; `predictions/revision/random_nested/`; `splits/revision/random_nested/`; `data/revision/random_nested/` |
| Table S9: scaffold nested comparison | `revision/scaffold_nested/06_scaffold_nested_cv_fixed.py` (parameters in source) | `results/revision/scaffold_nested/scaffold_nested_cv_comparison.csv` |
| Table S10 and source-holdout subgroup discussion | `tools/recompute_revision_results.py`; original similarities may be replayed with `tools/replay_applicability_domain.py` | `predictions/revision/pubchem_similarity_groups.csv`; `results/revision/pubchem_subgroup_metrics.csv`; original source-holdout predictions/AD tables |

## Inspect saved results without training

From the repository root, with `requirements-recompute.txt` installed:

```bash
python tools/verify_file_integrity.py
python tools/recompute_results.py
python tools/recompute_revision_results.py
```

The last command writes to `results/recomputed/revision/`. Its checks cover:

- Alignment of the 1,068 saved ensemble predictions and AD records, followed by recalculation of all, 641, 427, 733, 92 and 335-compound subgroup metrics.
- Recalculation of the four random nested arms from 34,959 outer predictions per arm, with checks against original fold and pooled metrics.
- Random outer/inner/pilot/final fitting indices and their correspondence to the original training-only screening records.
- Consistency of Table S7 pooled R²/RMSE/MAE/twofold coverage and fold mean/SD with the 25 supplied fold summaries. This check uses fold summaries and retained-cohort targets; the original threshold evaluation predictions are not in this archive.
- Export of threshold fold assignments from the supplied selection rule, original row order and split seed. These assignments are labelled **reconstructed**.

The original scaffold individual predictions have not been supplied with this archive, so Table S9 is included as the reported summary without an individual-level recalculation claim.

## Rerun supplementary model fitting

These commands fit new models and can take substantial time. They are not required for inspecting the supplied results. Install the reference training requirements and run commands from the repository root.

Threshold comparison:

```bash
python revision/threshold/05_development_pool_cv.py --config configs/threshold_cv.json
```

Add `--preflight-only` to check its input and configuration without fitting.

Random nested comparison:

```bash
python revision/random_nested/02_nested_oof_ablation.py --config configs/random_nested.json --data-dir . --mode random
```

Add `--preflight-only` for input/configuration checks. `--device cpu` selects CPU execution. The archived result uses one final model seed (42) per arm/fold; this supplementary comparison is distinct from the ten-seed main ensemble.

Scaffold nested comparison:

```bash
python revision/scaffold_nested/06_scaffold_nested_cv_fixed.py --base-dir . --input data/oof/train_pool_oof_residual_report.csv --output-dir results/generated/revision/scaffold_nested
```

Use this separate scaffold implementation for Table S9's protocol. The random script also exposes a scaffold mode, but that mode uses a different inner grouping/round-selection design and is not the supplied Table S9 implementation. In the Table S9 code, Bemis–Murcko groups define outer folds; acyclic compounds use canonical non-isomeric whole-structure keys, and inner screening uses random five-fold splits within outer training.

## Interpretation and record scope

The threshold analysis describes performance in separately retained development cohorts. Random and scaffold nested comparisons retain the common outer test population across threshold arms and confine residual screening to outer training. The subgroup analysis evaluates fixed predictions according to their saved structural similarities; fingerprint nonidentity alone does not establish experimental independence or scaffold novelty.

The original random code and shared-utility hashes, and the input-report hash, match the archived random run manifest. Its recorded versions are Python 3.13.5, NumPy 2.3.3, pandas 2.3.3, scikit-learn 1.7.2, XGBoost 3.2.0 and RDKit 2025.09.5. These are records for that run, not a claim about the original main model's historical environment. Repository `requirements.txt` remains a separately identified reference installation.

The supplied supplementary training scripts and numerical routines are unchanged. Portable input/output paths and explanatory metadata are set in the new config files. The new recalculation script performs no fitting. See `docs/availability.md` for the precise historical output coverage and `provenance/revision_package.json` for source-file hashes and export transformations.

The 448-compound source-verification workbook is provided with the manuscript's Supplementary Data. The repository includes its computational exclusion records and OOF report. The full upstream provenance audit of the 641-compound subgroup is not included in this update.
