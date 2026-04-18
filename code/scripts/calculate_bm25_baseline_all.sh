#!/usr/bin/env bash
set -euo pipefail

source "$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)/common.sh"

BASE_OUTPUT_DIR="${BASE_OUTPUT_DIR:-results/bm25_baseline}"

BM25_TOP_K="${BM25_TOP_K:-100}"

NUM_QUERIES="${NUM_QUERIES:-}"

BEIR_DATASETS=("trec-covid")

TREC_DL_DATASETS=("dl19" "dl20" "dl21" "dl22" "dl23")

TOTAL_DATASETS=0
SUCCESS_COUNT=0
FAILED_DATASETS=()

run_baseline() {
    local dataset_type="$1"
    local dataset_name="$2"
    local output_dir="$3"
    local cmd=(
        "${PYTHON_BIN}" "${PROJECT_ROOT}/code/main/calculate_bm25_baseline.py"
        --dataset_type "$dataset_type"
        --bm25_top_k "$BM25_TOP_K"
        --output_dir "$output_dir"
    )
    if [[ -n "$dataset_name" ]]; then
        cmd+=(--dataset_name "$dataset_name")
    fi
    if [[ "$dataset_type" == "trec_dl" ]]; then
        cmd+=(--trec_dl_dataset "$dataset_name")
    fi
    if [ -n "${NUM_QUERIES:-}" ]; then
        cmd+=(--num_queries "$NUM_QUERIES")
    fi
    "${cmd[@]}"
}

# 计算 TREC-DL 数据集
for dataset in "${TREC_DL_DATASETS[@]}"; do
    TOTAL_DATASETS=$((TOTAL_DATASETS + 1))
    OUTPUT_DIR="${BASE_OUTPUT_DIR}/trec_dl/${dataset}"
    if run_baseline "trec_dl" "$dataset" "$OUTPUT_DIR"; then
        SUCCESS_COUNT=$((SUCCESS_COUNT + 1))
    else
        FAILED_DATASETS+=("trec_dl/$dataset")
        echo "Failed on trec_dl/${dataset}; continuing."
    fi
done

for dataset in "${BEIR_DATASETS[@]}"; do
    TOTAL_DATASETS=$((TOTAL_DATASETS + 1))
    OUTPUT_DIR="${BASE_OUTPUT_DIR}/beir/${dataset}"
    if run_baseline "beir" "$dataset" "$OUTPUT_DIR"; then
        SUCCESS_COUNT=$((SUCCESS_COUNT + 1))
    else
        FAILED_DATASETS+=("beir/$dataset")
        echo "Failed on beir/${dataset}; continuing."
    fi
done

echo "总数据集数: $TOTAL_DATASETS"
echo "成功计算: $SUCCESS_COUNT"
echo "失败数量: ${#FAILED_DATASETS[@]}"

if [ ${#FAILED_DATASETS[@]} -gt 0 ]; then
    echo ""
    echo "失败的数据集:"
    for failed in "${FAILED_DATASETS[@]}"; do
        echo "  - $failed"
    done
fi
