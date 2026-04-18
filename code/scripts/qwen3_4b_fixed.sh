#!/usr/bin/env bash
set -euo pipefail

MODEL_NAME="${MODEL_NAME:-Qwen/Qwen3-4B}"
MODEL_SLUG="qwen3_4b"
EXPERIMENT_TYPE="fixed"
BIAS_RATES="${BIAS_RATES:-0.0}"

source "$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)/run_qwen_suite.sh"
