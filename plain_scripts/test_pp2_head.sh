#!/usr/bin/env bash
set -euo pipefail

export ASCEND_RT_VISIBLE_DEVICES=0,1,2,3,4,5,6,7,8,9,10,11,12,13,14,15
export VLLM_HOST_IP=80.5.17.111
export HCCL_IF_IP=80.5.17.111
export GLOO_SOCKET_IFNAME=enp48s3u1u1
export TP_SOCKET_IFNAME=enp48s3u1u1
export HCCL_SOCKET_IFNAME=enp48s3u1u1
export HCCL_BUFFSIZE=1024
export HCCL_CONNECT_TIMEOUT=120
export HCCL_EXEC_TIMEOUT=204
# Two logical devices share each physical NPU, so let HCCL allocate
# per-process host/NPU ports instead of reusing the default NPU port 16666.
export HCCL_HOST_SOCKET_PORT_RANGE=auto
export HCCL_NPU_SOCKET_PORT_RANGE=auto
export TE_PARALLEL_COMPILER=1
export OMP_NUM_THREADS=10
export VLLM_RPC_TIMEOUT=3600000
export VLLM_EXECUTE_MODEL_TIMEOUT_SECONDS=3000
export PYTORCH_NPU_ALLOC_CONF=expandable_segments:True
export HCCL_OP_EXPANSION_MODE=AIV
export VLLM_USE_V2_MODEL_RUNNER=1
export VLLM_LOGGING_LEVEL=INFO
# Keep the PP boundary on a full Indexer layer for GLM-5.2.
export VLLM_PP_LAYER_PARTITION=38,40

vllm serve /mnt/weight/GLM-5.2-W4A8C8-0713-MTP \
  --host 127.0.0.1 \
  --port 18080 \
  --served-model-name glm-5.2 \
  --trust-remote-code \
  --tensor-parallel-size 16 \
  --pipeline-parallel-size 2 \
  --distributed-executor-backend mp \
  --nnodes 2 \
  --node-rank 0 \
  --master-addr 80.5.17.111 \
  --master-port 29500 \
  --max-num-seqs 100 \
  --max-model-len 40960 \
  --max-num-batched-tokens 16384 \
  --block-size 128 \
  --enforce-eager \
  --quantization ascend \
  --no-enable-prefix-caching \
  --additional-config '{"weight_nz_mode":2}' \
  --gpu-memory-utilization 0.85
