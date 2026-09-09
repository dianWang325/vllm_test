#!/usr/bin/env bash
set -euo pipefail

MODEL_PATH="${1:-/mnt/weight/DeepSeek-V4-Flash-w8a8-mtp}"
HOST="${HOST:-0.0.0.0}"
PORT="${PORT:-10800}"

export ASCEND_RT_VISIBLE_DEVICES=0,1,2,3,4,5,6,7,8,9,10,11,12,13,14,15
export VLLM_WORKER_MULTIPROC_METHOD=spawn

exec vllm serve "${MODEL_PATH}" \
  --host "${HOST}" \
  --port "${PORT}" \
  --data-parallel-size 2 \
  --tensor-parallel-size 8 \
  --trust-remote-code