# Main Entry Point

`code/main/main.py` is the unified entry point for the cleaned repository.

## Supported Tasks

- `--dataset_type beir`
- `--dataset_type trec_dl`

## Supported Methods

- `--experiment_type fixed`
- `--experiment_type adaptive`

`fixed` applies a fixed calibration strength.

`adaptive` adjusts calibration strength from model uncertainty during decoding.

## Minimal Examples

```bash
python3 code/main/main.py \
  --dataset_type beir \
  --dataset_name trec-covid \
  --experiment_type fixed \
  --model_name Qwen/Qwen3-1.7B \
  --num_queries 50 \
  --passages_per_query 20 \
  --bias_rate 1.0
```

```bash
python3 code/main/main.py \
  --dataset_type trec_dl \
  --trec_dl_dataset dl23 \
  --experiment_type adaptive \
  --model_name Qwen/Qwen3-4B \
  --num_queries 50 \
  --passages_per_query 20 \
  --bias_rate 2.0
```
