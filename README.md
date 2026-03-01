# RG-ICL

Representation-Guided In-Context Learning for medical image classification and visual question answering. Training-free framework using frozen vision encoders + GPT-4V.

## Install

```bash
pip install -r requirements.txt
export OPENAI_API_KEY="your-key"
```

## Usage

### 1. Extract Features

```bash
python scripts/extract_features.py --config configs/default.yaml
```

### 2. Zero-Shot (no references)

```bash
python scripts/run_classification.py --methods zero_shot
python scripts/run_vqa.py --methods zero_shot
```

### 3. In-Context Learning (random 6-shot references)

```bash
python scripts/run_classification.py --methods naive_icl
python scripts/run_vqa.py --methods naive_icl
```

### 4. RG-ICL Global (DINOv2 CLS-token retrieval)

```bash
python scripts/run_classification.py --methods rg_icl_global
python scripts/run_vqa.py --methods rg_icl_global
```

### 5. RG-ICL Global + Spatial (CLS-token + patch-level retrieval)

```bash
python scripts/run_classification.py --methods rg_icl_global_spatial
python scripts/run_vqa.py --methods rg_icl_global_spatial
```

### 6. Run All Methods at Once

```bash
python scripts/run_classification.py --config configs/experiments/classification.yaml
python scripts/run_vqa.py --config configs/experiments/vqa.yaml
```

### 7. Full Pipeline (end-to-end)

```bash
python scripts/run_all.py
```

## Analysis

```bash
python scripts/run_k_sweep.py --config configs/experiments/k_sweep.yaml
python scripts/run_robustness.py --config configs/experiments/robustness.yaml
python scripts/run_encoder_ablation.py --config configs/experiments/encoder_ablation.yaml
python scripts/run_judge.py --config configs/default.yaml
python scripts/run_stats.py --config configs/default.yaml
```

## Datasets

Place data in `data/` — see `data/README_DATA.md` for structure.

**Classification:** LAG, DDR, CheXpert, BreakHis
**VQA:** Medical-CXR-VQA, VQA-RAD, PathVQA, PMC-VQA
