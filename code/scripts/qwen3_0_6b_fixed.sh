#!/usr/bin/env bash
set -euo pipefail

MODEL_NAME="${MODEL_NAME:-Qwen/Qwen3-0.6B}"
MODEL_SLUG="qwen3_0_6b"
EXPERIMENT_TYPE="fixed"
BIAS_RATES="${BIAS_RATES:-1.5 2.0}"

source "$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)/run_qwen_suite.sh"
