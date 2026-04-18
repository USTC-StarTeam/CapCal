#!/usr/bin/env bash
set -euo pipefail

MODEL_NAME="${MODEL_NAME:-Qwen/Qwen3-8B}"
MODEL_SLUG="qwen3_8b"
EXPERIMENT_TYPE="fixed"
BIAS_RATES="${BIAS_RATES:-0.5 0.7 0.9 1.0}"

source "$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)/run_qwen_suite.sh"
