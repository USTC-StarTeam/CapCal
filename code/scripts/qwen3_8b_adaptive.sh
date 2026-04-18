#!/usr/bin/env bash
set -euo pipefail

MODEL_NAME="${MODEL_NAME:-Qwen/Qwen3-8B}"
MODEL_SLUG="qwen3_8b"
EXPERIMENT_TYPE="adaptive"
BIAS_RATES="${BIAS_RATES:-2.0}"

source "$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)/run_qwen_suite.sh"
