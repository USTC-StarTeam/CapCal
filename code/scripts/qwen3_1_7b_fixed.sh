#!/usr/bin/env bash
set -euo pipefail

MODEL_NAME="${MODEL_NAME:-Qwen/Qwen3-1.7B}"
MODEL_SLUG="qwen3_1_7b"
EXPERIMENT_TYPE="fixed"
BIAS_RATES="${BIAS_RATES:-0.1 0.3 0.5 0.7 0.9 1.0 1.5 2.0}"

source "$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)/run_qwen_suite.sh"
