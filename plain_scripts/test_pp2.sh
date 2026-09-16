#!/usr/bin/env bash
set -euo pipefail

# TP4 x PP2 requires eight visible Ascend devices.
export ASCEND_RT_VISIBLE_DEVICES=0,1,2,3,4,5,6,7
export HCCL_BUFFSIZE=1024
export HCCL_OP_EXPANSION_MODE=AIV
export PYTORCH_NPU_ALLOC_CONF=expandable_segments:True
export VLLM_USE_V2_MODEL_RUNNER=1
export VLLM_LOGGING_LEVEL=INFO

vllm serve /mnt/weight/Qwen3-30B-A3B-W8A8 \
  --host 127.0.0.1 \
  --port 18080 \
  --served-model-name qwen3-30b-a3b-w8a8 \
  --trust-remote-code \
  --tensor-parallel-size 4 \
  --pipeline-parallel-size 2 \
  --max-num-seqs 100 \
  --max-model-len 40960 \
  --max-num-batched-tokens 16384 \
  --block-size 128 \
  --enable-expert-parallel \
  --quantization ascend \
  --distributed-executor-backend mp \
  --no-enable-prefix-caching \
  --compilation-config '{"cudagraph_mode":"FULL_DECODE_ONLY"}' \
  --additional-config '{"weight_nz_mode":2}' \
  --gpu-memory-utilization 0.95
