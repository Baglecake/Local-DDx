# Benchmark: Local-DDx v10 on Open-XDDx

Automated evaluation of the Local-DDx v10 pipeline against the [Open-XDDx dataset](https://doi.org/10.1038/s44401-025-00015-6) (570 cases, 9 specialties).

## Components

| File | Purpose |
|------|---------|
| `batch_runner.py` | Batch orchestrator with checkpoint/resume and case range selection |
| `vllm_backend.py` | GPU inference backend (vLLM, for Colab A100) |
| `ddx_evaluator.py` | Deterministic clinical equivalence evaluator |
| `benchmark_config.yaml` | Llama-3-8B-Instruct config |
| `benchmark_config_qwen32b.yaml` | Qwen2.5-32B-Instruct config |
| `colab_benchmark.ipynb` | Google Colab notebook for running benchmarks |

## Dataset

`v10_full_pipeline/data/Open-XDDx.xlsx` — 570 clinical vignettes with ground truth differential diagnoses across 9 medical specialties (cardiology, neurology, pulmonology, gastroenterology, nephrology, endocrinology, hematology, rheumatology, infectious disease).

Source: Zhou et al. (2025), *Explainable differential diagnosis with dual-inference large language models*, npj Health Systems.

## Results

| System | Model | Cases | Recall | Precision | Safety |
|--------|-------|-------|--------|-----------|--------|
| **Local-DDx v10** | Qwen2.5-32B-GPTQ | 399 | **57.6%** | 49.2% | 67.5% |
| Zhou Dual-Inf | GPT-4 | 570 | 53.3% | — | — |

Local-DDx v10 outperforms Zhou et al.'s GPT-4 Dual-Inference system by **+4.3 percentage points** on clinical recall, using a 32B open-weight model running on a single A100 GPU.

## Evaluator Design

The evaluator uses **deterministic clinical equivalence matching** (no LLM-as-judge) with a 4-tier matching pipeline:

1. **Direct match** (score: 1.0) — exact string after normalization
2. **Synonym match** (0.95) — 260+ bidirectional medical synonym groups
3. **Hierarchy match** (0.90) — 75+ clinical supertype/subtype trees
4. **Word overlap** (0.80) — Jaccard similarity ≥ 0.8 on content words

Normalization pipeline (11 steps): lowercase, unicode, apostrophe variants, parenthetical stripping, dash/slash→space, stop-word removal, whitespace collapse, and more.

### Metrics

- **Recall**: TP / (TP + FN) — percentage of ground truth diagnoses identified
- **Precision**: TP / (TP + FP) — accuracy of pipeline diagnoses
- **CRQ** (Clinical Reasoning Quality): (TP + 0.5×AE) / (TP + FN + AE)
- **Safety**: (TP + AE) / (TP + FN + AE) — where AE = appropriately excluded (considered but not finalized)

## How to Run

Benchmarks are designed for **Google Colab with A100 GPU** — not for local M4 execution. See `colab_benchmark.ipynb` for the complete workflow.

```bash
# On Colab after setup:
!cd /content/Local-DDx && python benchmark/batch_runner.py \
    --config benchmark/benchmark_config_qwen32b.yaml \
    --cases 0-99 --resume
```

Result JSONs are gitignored (large, regenerable via batch_runner).
