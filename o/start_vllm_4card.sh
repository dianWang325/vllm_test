#!/usr/bin/env bash
set -euo pipefail

MODEL_PATH="${1:-/mnt/weight/DeepSeek-V4-Flash-w8a8-mtp}"
HOST="${HOST:-0.0.0.0}"
PORT="${PORT:-10800}"

export ASCEND_RT_VISIBLE_DEVICES=0,1,2,3,4,5,6,7
export VLLM_WORKER_MULTIPROC_METHOD=spawn

exec vllm serve "${MODEL_PATH}" \
  --host "${HOST}" \
  --port "${PORT}" \
  --tensor-parallel-size 8 \
  --trust-remote-code