#!/usr/bin/env bash

# Shared helpers for reproducible experiment scripts.

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd -- "${SCRIPT_DIR}/../.." && pwd)"
if [[ -z "${PYTHON_BIN:-}" ]]; then
  if command -v python >/dev/null 2>&1; then
    PYTHON_BIN="python"
  else
    PYTHON_BIN="python3"
  fi
fi

if [[ -z "${RERANK_BIAS_DATA_ROOT:-}" ]]; then
  export RERANK_BIAS_DATA_ROOT="${PROJECT_ROOT}/data"
fi

if [[ -z "${RERANK_BIAS_HF_HOME:-}" ]]; then
  export RERANK_BIAS_HF_HOME="${RERANK_BIAS_DATA_ROOT}/huggingface"
fi

run_python() {
  "${PYTHON_BIN}" "$@"
}
