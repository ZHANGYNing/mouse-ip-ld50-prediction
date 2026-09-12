# Environment records

The root requirements files define the reference installation used for repository assembly. They do not establish the exact historical main-model training environment.

`provenance/packaging_environment_and_checks.json` records the environment used to verify saved metrics, replay rule curation and AD, and check code imports. Full training was not performed during packaging; the XGBoost CPU build was used for import checks.

To record the relevant package versions while working in the original training environment:

```bash
python tools/capture_training_environment.py
```

This creates `environment/captured_training_environment.json`. Record whether that environment has remained unchanged since the reported training run. A current version export alone does not prove historical version identity.
