# Experiment Scripts

This directory now keeps only the public scripts used for the paper's main Qwen-based experiments.

## Kept Surface

- `run_qwen_suite.sh`: shared runner used by all model-specific wrappers
- `qwen2_5_7b_fixed.sh`
- `qwen2_5_7b_adaptive.sh`
- `qwen3_0_6b_fixed.sh`
- `qwen3_0_6b_adaptive.sh`
- `qwen3_1_7b_fixed.sh`
- `qwen3_1_7b_adaptive.sh`
- `qwen3_4b_fixed.sh`
- `qwen3_4b_adaptive.sh`
- `qwen3_8b_fixed.sh`
- `qwen3_8b_adaptive.sh`
- `calculate_bm25_baseline_all.sh`

The naming is intentional:

- `fixed` means a fixed calibration strength
- `adaptive` means entropy/perplexity-aware adaptive calibration

## Usage

Run any wrapper directly:

```bash
bash code/scripts/qwen3_1_7b_fixed.sh
bash code/scripts/qwen3_1_7b_adaptive.sh
```

Override runtime settings with environment variables:

```bash
NUM_QUERIES=100 \
BIAS_RATES="0.5 1.0 1.5" \
TREC_DL_DATASETS="dl21 dl22 dl23" \
BEIR_DATASETS="trec-covid" \
bash code/scripts/qwen3_4b_fixed.sh
```

All wrappers resolve the repository root automatically and default to:

- `RERANK_BIAS_DATA_ROOT=./data` if unset
- `RERANK_BIAS_HF_HOME=$RERANK_BIAS_DATA_ROOT/huggingface` if unset
