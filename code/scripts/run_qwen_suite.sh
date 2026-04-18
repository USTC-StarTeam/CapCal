#!/usr/bin/env bash
set -euo pipefail

source "$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)/common.sh"

MODEL_NAME="${MODEL_NAME:?MODEL_NAME must be set}"
MODEL_SLUG="${MODEL_SLUG:?MODEL_SLUG must be set}"
EXPERIMENT_TYPE="${EXPERIMENT_TYPE:?EXPERIMENT_TYPE must be set}"

NUM_QUERIES="${NUM_QUERIES:-500}"
PASSAGES_PER_QUERY="${PASSAGES_PER_QUERY:-20}"
BIAS_RATES="${BIAS_RATES:-1.0}"
SEED="${SEED:-42}"
DEVICE="${DEVICE:-auto}"
MAX_LENGTH="${MAX_LENGTH:-8192}"
USE_BM25="${USE_BM25:-true}"
BM25_TOP_K="${BM25_TOP_K:-100}"
BEIR_DATASETS="${BEIR_DATASETS:-trec-covid}"
TREC_DL_DATASETS="${TREC_DL_DATASETS:-dl19 dl20 dl21 dl22 dl23}"

run_dataset() {
  local dataset_type="$1"
  local dataset_name="$2"
  local bias_rate="$3"
  local output_dir="${PROJECT_ROOT}/results/qwen/${MODEL_SLUG}/${EXPERIMENT_TYPE}/${dataset_type}/${dataset_name}/bias_${bias_rate}"

  local cmd=(
    "${PYTHON_BIN}" "${PROJECT_ROOT}/code/main/main.py"
    --dataset_type "${dataset_type}"
    --experiment_type "${EXPERIMENT_TYPE}"
    --output_dir "${output_dir}"
    --num_queries "${NUM_QUERIES}"
    --passages_per_query "${PASSAGES_PER_QUERY}"
    --bias_rate "${bias_rate}"
    --seed "${SEED}"
    --model_name "${MODEL_NAME}"
    --device "${DEVICE}"
    --max_length "${MAX_LENGTH}"
    --cache_use
    --verbose
  )

  if [[ "${dataset_type}" == "beir" ]]; then
    cmd+=(--dataset_name "${dataset_name}")
  else
    cmd+=(--trec_dl_dataset "${dataset_name}")
  fi

  if [[ "${USE_BM25}" == "true" ]]; then
    cmd+=(--use_bm25_retrieval --bm25_top_k "${BM25_TOP_K}")
  fi

  "${cmd[@]}"
}

echo "Model: ${MODEL_NAME}"
echo "Method: ${EXPERIMENT_TYPE}"
echo "TREC-COVID datasets: ${BEIR_DATASETS}"
echo "TREC-DL datasets: ${TREC_DL_DATASETS}"
echo "Bias rates: ${BIAS_RATES}"

for dataset in ${BEIR_DATASETS}; do
  for bias_rate in ${BIAS_RATES}; do
    echo "Running beir/${dataset} with bias_rate=${bias_rate}"
    run_dataset "beir" "${dataset}" "${bias_rate}"
  done
done

for dataset in ${TREC_DL_DATASETS}; do
  for bias_rate in ${BIAS_RATES}; do
    echo "Running trec_dl/${dataset} with bias_rate=${bias_rate}"
    run_dataset "trec_dl" "${dataset}" "${bias_rate}"
  done
done
