#!/usr/bin/env bash
# TRIAL 2026-09-16: 80.5.9.113 PP0 + 80.5.9.127 PP1.
# TP16/PP2/EP, max_model_len=87040; verify before marking successful.
set -euo pipefail

role="${1:?expected head or worker}"
case "$role" in
  head)
    local_ip=80.5.9.113
    node_rank=0
    extra_args=()
    ;;
  worker)
    local_ip=80.5.9.127
    node_rank=1
    extra_args=(--headless)
    ;;
  *) echo "invalid role: $role" >&2; exit 2 ;;
esac

log_dir=/home/w00985415/pp2_glm_ep_113_127_85k_20260916
mkdir -p "$log_dir/${role}_cann"

export ASCEND_RT_VISIBLE_DEVICES=0,1,2,3,4,5,6,7,8,9,10,11,12,13,14,15
export VLLM_HOST_IP="$local_ip"
export HCCL_IF_IP="$local_ip"
export GLOO_SOCKET_IFNAME=enp194s0f0
export TP_SOCKET_IFNAME=enp194s0f0
export HCCL_SOCKET_IFNAME=enp194s0f0
export HCCL_BUFFSIZE=1024
export HCCL_CONNECT_TIMEOUT=120
export HCCL_EXEC_TIMEOUT=204
export HCCL_HOST_SOCKET_PORT_RANGE=auto
export HCCL_NPU_SOCKET_PORT_RANGE=auto
export HCCL_OP_EXPANSION_MODE=AIV
export ASCEND_PROCESS_LOG_PATH="$log_dir/${role}_cann"
export ASCEND_MODULE_LOG_LEVEL=HCCL=0
export ASCEND_SLOG_PRINT_TO_STDOUT=0
export TE_PARALLEL_COMPILER=1
export OMP_NUM_THREADS=10
export VLLM_RPC_TIMEOUT=3600000
export VLLM_EXECUTE_MODEL_TIMEOUT_SECONDS=3000
export PYTORCH_NPU_ALLOC_CONF=expandable_segments:True
export VLLM_USE_V2_MODEL_RUNNER=1
export VLLM_LOGGING_LEVEL=DEBUG
export VLLM_PP_LAYER_PARTITION=38,40
export PYTHONUNBUFFERED=1

printf 'role=%s host=%s peer=%s max_model_len=87040 started=%s\n' "$role" "$local_ip" "$(if [ "$role" = head ]; then echo 80.5.9.127; else echo 80.5.9.113; fi)" "$(date -Is)" > "$log_dir/${role}.meta"
exec vllm serve /mnt/weight/GLM-5.2-W4A8C8-0713-MTP \
  --host 127.0.0.1 \
  --port 18080 \
  --served-model-name glm-5.2 \
  --trust-remote-code \
  --tensor-parallel-size 16 \
  --pipeline-parallel-size 2 \
  --distributed-executor-backend mp \
  --nnodes 2 \
  --node-rank "$node_rank" \
  --master-addr 80.5.9.113 \
  --master-port 29500 \
  "${extra_args[@]}" \
  --max-num-seqs 100 \
  --max-model-len 87040 \
  --max-num-batched-tokens 16384 \
  --block-size 128 \
  --enforce-eager \
  --enable-expert-parallel \
  --quantization ascend \
  --no-enable-prefix-caching \
  --additional-config '{"weight_nz_mode":2}' \
  --gpu-memory-utilization 0.85 > "$log_dir/${role}.log" 2>&1
