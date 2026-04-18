# Reproduction Guideline

This document gives a compact, paper-oriented reproduction path for the main experiments in this repository.

## Step 1: Create the environment

```bash
conda create -n rerank-bias python=3.10 -y
conda activate rerank-bias
pip install -r requirements.txt
```

## Step 2: Choose a shared data root

```bash
export RERANK_BIAS_DATA_ROOT=/path/to/your/data
export RERANK_BIAS_HF_HOME=$RERANK_BIAS_DATA_ROOT/huggingface
```

The repository assumes that all datasets live under this root.

## Step 3: Prepare datasets

### BEIR

```bash
python code/dataloader/download_datasets.py --datasets trec-covid
```

### TREC DL

Manually place the data at:

- `$RERANK_BIAS_DATA_ROOT/trec-dl-2019`
- `$RERANK_BIAS_DATA_ROOT/trec-dl-2020`
- `$RERANK_BIAS_DATA_ROOT/trec/trec21`
- `$RERANK_BIAS_DATA_ROOT/trec/trec22`
- `$RERANK_BIAS_DATA_ROOT/trec/trec23`

For `dl21`, `dl22`, and `dl23`, also place MSMARCO v2 passages at:

- `$RERANK_BIAS_DATA_ROOT/msmarco_v2_passage`

or set:

```bash
export RERANK_MSMARCO_V2_PASSAGE_DIR=/path/to/msmarco_v2_passage
```

## Step 4: Verify dataset loading

```bash
python code/main/main.py \
  --dataset_type beir \
  --dataset_name trec-covid \
  --num_queries 10 \
  --dry_run
```

If this succeeds, path resolution and loader imports are working.

## Step 5: Run experiments

### Fixed-rate example

```bash
bash code/scripts/qwen3_1_7b_fixed.sh
```

### Adaptive example

```bash
bash code/scripts/qwen3_1_7b_adaptive.sh
```

### Override datasets or bias rates

```bash
NUM_QUERIES=100 \
BIAS_RATES="1.0 1.5 2.0" \
TREC_DL_DATASETS="dl21 dl22 dl23" \
BEIR_DATASETS="trec-covid" \
bash code/scripts/qwen3_4b_fixed.sh
```

## Step 6: Compute BM25 baselines

```bash
bash code/scripts/calculate_bm25_baseline_all.sh
```

## Outputs

By default, experiment outputs are written under `results/qwen/`, and BM25 summaries are written under `results/bm25_baseline/`.

## Notes

- The shell scripts now work from any current working directory.
- No proxy variables are required by default.
- The repository now exposes only the paper's main Qwen-based TREC-COVID / TREC-DL surface.
- If you want to override model, device, or query count, pass them as environment variables when invoking the shell scripts.
