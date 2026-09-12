# Data dictionary

The original workbook has three columns: `Canonical SMILES`, `Toxicity Value` (mg/kg) and `label` (1 = TOXRIC; 2 = PubChem). Worksheet data-row positions are zero based in all `raw_row_ids` links; the header is not counted.

| Field | Meaning |
|---|---|
| `compound_id` | First 20 hexadecimal characters of SHA256 of compact sorted JSON `{"smiles": <Canonical SMILES>}`; an identifier, never a model feature |
| `Canonical SMILES` in modeling tables | Structure after the original fragment/charge handling and canonicalization |
| `cohort` | `development` or `pubchem_source_holdout` |
| `development_row_id` | Zero-based row in the frozen 34,959-compound development report; -1 for external compounds |
| `excluded_by_original_oof` | Original 20-fold residual exclusion flag; external compounds were not screened |
| `source_type` | `TOXRIC_only`, `mixed`, or `PubChem_only` |
| `source_set`, `has_toxric`, `has_pubchem` | Original source membership |
| `Toxicity Value`, `ld50_mgkg` | Median dose after curation, mg/kg |
| `molecular_weight` | RDKit molecular weight, numerically g/mol |
| `y` | −log10(LD50 in mol/kg) |
| `n_raw_records`, `raw_row_ids` | Number and semicolon-separated zero-based positions of linked raw rows |
| `ld50_min_mgkg_raw`, `ld50_max_mgkg_raw`, `ld50_mean_mgkg_raw`, `ld50_median_mgkg_raw` | Linked-source summary before final aggregation |
| `n_fragments_original_max`, `heavy_atoms`, `element_set` | Original structural curation metadata |
| `oof_pred_y`, `oof_abs_error_y`, `oof_fold_error` | Saved development-pool screening prediction, absolute transformed error and fold error |
| `remove_by_oof_high_residual` | `oof_fold_error >= 20` |
| `y_true_neglog_molkg`, `y_pred_neglog_molkg` | External true target and saved model prediction |
| `ld50_true_*`, `ld50_pred_*`, `fold_error` | Original dose-scale exports and multiplicative discrepancy |
| `max_similarity`, `mean_top5_similarity` | Maximum and mean five largest achiral binary-ECFP4 similarities to the retained development reference |
| `n_train_sim_ge_0.70` (example) | Number of reference compounds with similarity ≥0.70 |
| `in_AD_S0.70_N3` (example) | At least 3 reference compounds have similarity ≥0.70 |
| `remove_reason` in rule exclusions | The original curation rule excluding the raw row or aggregated group |
| `reason` in raw rule exclusions | More specific structural-processing reason where available |

`splits/source_holdout_and_internal.csv` records role, order within the original train/validation partitions and order in the final train+validation fit. `splits/original_oof.csv` records which of the five original screening folds predicted each development compound. These assignments were recovered deterministically from the original code and frozen row order, with numerical cross-checks documented in `provenance/`.

The rule-based exclusion tables are replayed outputs. The 448-record OOF exclusion CSV and the complete OOF residual report are unchanged supplied files. Detailed source verification of those 448 compounds is in SI and is not a computational exclusion rule.
