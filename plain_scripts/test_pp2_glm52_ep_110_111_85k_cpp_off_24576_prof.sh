#!/usr/bin/env bash
# GLM-5.2 TP16/PP2/EP with CPP off and MS Service Profiler on both nodes.
# Defaults: PP0 80.5.17.110, PP1 80.5.17.111. Override both hosts and RUN_ID together when moving nodes.
set -euo pipefail

head_ip=${PP0_HOST:-80.5.17.110}
worker_ip=${PP1_HOST:-80.5.17.111}
run_id=${RUN_ID:-110_111}
role="${1:?expected head or worker}"
case "$role" in
  head)
    local_ip="$head_ip"
    node_rank=0
    prof_role=pp0
    extra_args=()
    ;;
  worker)
    local_ip="$worker_ip"
    node_rank=1
    prof_role=pp1
    extra_args=(--headless)
    ;;
  *) echo "invalid role: $role" >&2; exit 2 ;;
esac

log_dir="/home/w00985415/pp2_glm_ep_${run_id}_85k_cpp_off_24576_prof_20260916"
mkdir -p "$log_dir/${role}_cann"

prof_root=/home/w00985415/vllm_test_glm/plain_scripts/profiling
prof_source="$prof_root/$prof_role/ms_service_profiler_config.json"
prof_runtime="$log_dir/${prof_role}_ms_service_profiler_config.json"
test -r "$prof_source"
test -r "$prof_root/service_profiling_symbols.yaml"
grep -Eq '"enable"[[:space:]]*:[[:space:]]*0' "$prof_source"
cp "$prof_source" "$prof_runtime"
export SERVICE_PROF_CONFIG_PATH="$prof_runtime"
export PROFILING_SYMBOLS_PATH="$prof_root/service_profiling_symbols.yaml"
test -r "$SERVICE_PROF_CONFIG_PATH"
test -r "$PROFILING_SYMBOLS_PATH"
unset VLLM_TORCH_PROFILER_DIR

export ASCEND_RT_VISIBLE_DEVICES=0,1,2,3,4,5,6,7,8,9,10,11,12,13,14,15
export VLLM_HOST_IP="$local_ip"
export HCCL_IF_IP="$local_ip"
export GLOO_SOCKET_IFNAME=enp48s3u1u1
export TP_SOCKET_IFNAME=enp48s3u1u1
export HCCL_SOCKET_IFNAME=enp48s3u1u1
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

printf 'role=%s host=%s peer=%s cpp_enabled=false max_num_batched_tokens=24576 profiler_config=%s profiler_symbols=%s started=%s\n' "$role" "$local_ip" "$(if [ "$role" = head ]; then echo "$worker_ip"; else echo "$head_ip"; fi)" "$SERVICE_PROF_CONFIG_PATH" "$PROFILING_SYMBOLS_PATH" "$(date -Is)" > "$log_dir/${role}.meta"
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
  --master-addr "$head_ip" \
  --master-port 29500 \
  "${extra_args[@]}" \
  --max-num-seqs 100 \
  --max-model-len 87040 \
  --max-num-batched-tokens 24576 \
  --block-size 128 \
  --enforce-eager \
  --enable-expert-parallel \
  --quantization ascend \
  --no-enable-prefix-caching \
  --additional-config '{"weight_nz_mode":2,"scheduler_config":{"profiling_chunk_config":{"enabled":false}}}' \
  --gpu-memory-utilization 0.85 > "$log_dir/${role}.log" 2>&1
