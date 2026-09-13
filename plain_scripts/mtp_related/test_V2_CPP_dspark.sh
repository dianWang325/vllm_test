#!/usr/bin/env bash
set -euo pipefail

export ASCEND_RT_VISIBLE_DEVICES=0,1,2,3,4,5,6,7,8,9,10,11,12,13,14,15

export VLLM_LOGGING_LEVEL=INFO
export HCCL_BUFFSIZE=1024
export VLLM_EXECUTE_MODEL_TIMEOUT_SECONDS=3000
export VLLM_USE_V2_MODEL_RUNNER=1
export OMP_PROC_BIND=false
export OMP_NUM_THREADS=10
export PYTORCH_NPU_ALLOC_CONF=expandable_segments:True
export TASK_QUEUE_ENABLE=1
export HCCL_OP_EXPANSION_MODE=AIV

vllm serve /mnt/weight/DeepSeek-V4-Flash-w8a8-mtp \
  --host 127.0.0.1 \
  --port 18080 \
  --trust-remote-code \
  --tensor-parallel-size 8 \
  --pipeline-parallel-size 2 \
  --max-num-batched-tokens 24576 \
  --block-size 64 \
  --enable-chunked-prefill \
  --enable-request-id-headers \
  --enforce-eager \
  --no-enable-prefix-caching \
  --no-async-scheduling \
  --enable-expert-parallel \
  --served-model-name deepseek-v4-flash \
  --quantization ascend \
  --max-model-len 87040 \
  --additional-config '{"scheduler_config":{"profiling_chunk_config":{"enabled":true,"smooth_factor":0.8,"need_timing":true}}}' \
  --speculative_config='{"method": "dspark","num_speculative_tokens": 5,"enforce_eager": true}' \
  --tokenizer-mode deepseek_v4 \
  --tool-call-parser deepseek_v4 \
  --reasoning-parser deepseek_v4 \
  --enable-auto-tool-choice \
  --no-disable-hybrid-kv-cache-manager \
  --model-loader-extra-config '{"enable_multithread_load":true,"num_threads":128}'