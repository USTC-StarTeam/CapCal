#!/usr/bin/env bash
set -euo pipefail

MODEL_NAME="${MODEL_NAME:-Qwen/Qwen2.5-7B-Instruct}"
MODEL_SLUG="qwen2_5_7b"
EXPERIMENT_TYPE="adaptive"
BIAS_RATES="${BIAS_RATES:-2.0}"

source "$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)/run_qwen_suite.sh"
