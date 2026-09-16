#!/usr/bin/env bash
set -euo pipefail

export ASCEND_RT_VISIBLE_DEVICES=0,1,2,3
export VLLM_HOST_IP=80.5.17.105
export HCCL_IF_IP=80.5.17.105
export GLOO_SOCKET_IFNAME=enp48s3u1u1
export TP_SOCKET_IFNAME=enp48s3u1u1
export HCCL_SOCKET_IFNAME=enp48s3u1u1
export HCCL_BUFFSIZE=1024
export HCCL_CONNECT_TIMEOUT=120
export HCCL_EXEC_TIMEOUT=204
export VLLM_RPC_TIMEOUT=3600000
export VLLM_EXECUTE_MODEL_TIMEOUT_SECONDS=3000
export PYTORCH_NPU_ALLOC_CONF=expandable_segments:True
export HCCL_OP_EXPANSION_MODE=AIV
export VLLM_USE_V2_MODEL_RUNNER=1
export VLLM_LOGGING_LEVEL=INFO

vllm serve /mnt/weight/Qwen3-30B-A3B-W8A8 \
  --host 127.0.0.1 \
  --port 18080 \
  --served-model-name qwen3-30b-a3b-w8a8 \
  --trust-remote-code \
  --tensor-parallel-size 4 \
  --pipeline-parallel-size 2 \
  --distributed-executor-backend mp \
  --nnodes 2 \
  --node-rank 1 \
  --master-addr 80.5.9.113 \
  --master-port 29500 \
  --headless \
  --max-num-seqs 100 \
  --max-model-len 40960 \
  --max-num-batched-tokens 16384 \
  --block-size 128 \
  --enable-expert-parallel \
  --quantization ascend \
  --no-enable-prefix-caching \
  --compilation-config '{"cudagraph_mode":"FULL_DECODE_ONLY"}' \
  --additional-config '{"weight_nz_mode":2}' \
  --gpu-memory-utilization 0.95
