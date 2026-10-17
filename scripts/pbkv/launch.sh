#!/usr/bin/env bash
set -euo pipefail

: "${MODEL_PATH:?Set MODEL_PATH to the local LLM directory}"
: "${PREDICTOR_DIR:?Set PREDICTOR_DIR to an existing model.pt or its directory}"
if [[ "${PBKV_MODE:-full}" != full ]]; then
  printf 'PBKV supports only full mode; unset PBKV_MODE or set it to full.\n' >&2
  exit 2
fi
PBKV_REPO_ROOT="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/../.." && pwd)"
export PYTHONPATH="${PBKV_REPO_ROOT}/python${PYTHONPATH:+:$PYTHONPATH}"
PBKV_ARGS=(--model-path "$MODEL_PATH" --tp 2 --page-size 16
           --enable-hierarchical-cache --hicache-ratio 1
           --my-pred-chain-evict-enable
           --my-pred-chain-evict-model-type graphsage
           --my-pred-chain-evict-model "$PREDICTOR_DIR"
           --my-pred-chain-evict-device "${PREDICTOR_DEVICE:-cuda:2}"
           --my-pred-evict-K 3 --my-pred-evict-gamma 0.7
           --my-prefetch-enable)
exec "${PYTHON_BIN:-python}" -m sglang.launch_server "${PBKV_ARGS[@]}" "$@"
