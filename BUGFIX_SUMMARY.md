# Critical Bug Fixes for RG-ICL LAG Glaucoma Experiment

To verify the fixes work:

```bash
# 1. Extract features (DINOv3 on LAG dataset)
venv/bin/python scripts/extract_features.py --config configs/experiments/classification.yaml --datasets lag

# 2. Run classification (all 3 methods)
venv/bin/python scripts/run_classification.py --config configs/experiments/classification.yaml --datasets lag

# 3. Verify against reference tables
venv/bin/python scripts/verify_results.py --pred-root outputs/classification --tolerance 0.02
```

## Files Modified

1. `configs/default.yaml` - model + device
2. `configs/experiments/classification.yaml` - model + device + datasets
3. `configs/experiments/encoder_ablation.yaml` - model
4. `configs/experiments/vqa.yaml` - model
5. `configs/experiments/robustness.yaml` - model
6. `configs/experiments/k_sweep.yaml` - model
7. `src/config.py` - model default
8. `src/inference/mllm_client.py` - model default
9. `src/prompting/templates.py` - JSON prompt format + P(positive) semantics
10. `src/inference/output_parser.py` - JSON parsing + label fallback fix
11. `scripts/run_classification.py` - confidence direction fix
12. `prepare_lag_data.py` - manifest.json generation

## Testing Notes

The parser now handles three response formats in priority order:
1. JSON (gpt-4o preferred): `{"label": "glaucoma", "confidence": 0.91, "evidence": "..."}`
2. Structured legacy: `Label: glaucoma\nConfidence: 0.87`
3. Natural language fallback: word-boundary substring matching

The confidence semantics are now unambiguous: **always P(positive class)**, regardless of which class is predicted.
