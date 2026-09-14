#!/usr/bin/env bash
# Called by launch_online_dp_two_p.py for one local vLLM process.
set -euo pipefail

if [[ $# -ne 14 ]]; then
  echo "Usage: $0 ROLE DEVICES HTTP_PORT DP_SIZE DP_RANK DP_ADDRESS DP_RPC_PORT TP_SIZE PP_SIZE LOCAL_IP NIC_NAME NODE_RANK MASTER_ADDR MASTER_PORT" >&2
  exit 2
fi

role=$1
devices=$2
http_port=$3
dp_size=$4
dp_rank=$5
dp_address=$6
dp_rpc_port=$7
tp_size=$8
pp_size=$9
local_ip=${10}
nic_name=${11}
node_rank=${12}
master_addr=${13}
master_port=${14}

export ASCEND_RT_VISIBLE_DEVICES="$devices"
export VLLM_HOST_IP="$local_ip"
export HCCL_IF_IP="$local_ip"
export GLOO_SOCKET_IFNAME="$nic_name"
export TP_SOCKET_IFNAME="$nic_name"
export HCCL_SOCKET_IFNAME="$nic_name"
export VLLM_RPC_TIMEOUT=3600000
export VLLM_EXECUTE_MODEL_TIMEOUT_SECONDS=30000
export HCCL_EXEC_TIMEOUT=204
export OMP_PROC_BIND=false
export OMP_NUM_THREADS=10
export PYTORCH_NPU_ALLOC_CONF=expandable_segments:True
export TASK_QUEUE_ENABLE=1
export HCCL_OP_EXPANSION_MODE=AIV
export VLLM_USE_V2_MODEL_RUNNER=1
export VLLM_LOGGING_LEVEL=INFO

jemalloc=/usr/lib/aarch64-linux-gnu/libjemalloc.so.2
if [[ -f $jemalloc ]]; then
  export LD_PRELOAD="$jemalloc${LD_PRELOAD:+:$LD_PRELOAD}"
fi

model=/mnt/weight/DeepSeek-V4-Flash-w8a8-mtp
model_loader_config='{"enable_multithread_load":true,"num_threads":128}'
# Both roles must advertise the same topology to the Mooncake connector.
topology='{"prefill":{"dp_size":2,"tp_size":8,"pp_size":2},"decode":{"dp_size":2,"tp_size":8,"pp_size":1}}'

case "$role" in
  prefill)
    export HCCL_CONNECT_TIMEOUT=120
    export HCCL_BUFFSIZE=2560
    http_host=${PD_HTTP_HOST:-127.0.0.1}
    kv_transfer_config='{"kv_connector":"MooncakeHybridConnector","kv_role":"kv_producer","kv_port":"30000","engine_id":"0","kv_connector_extra_config":'"$topology"'}'
    if [[ "$node_rank" == 1 ]]; then
      headless_args=(--headless)
    else
      headless_args=()
    fi
    case "${PD_P_SCHEDULER_MODE:-enhanced}" in
      baseline)
        scheduler_args=(--additional-config '{"enable_dsa_cp":true}')
        ;;
      enhanced)
        scheduler_args=(--additional-config '{"enable_dsa_cp":true,"scheduler_config":{"profiling_chunk_config":{"enabled":true,"smooth_factor":0.8,"need_timing":false},"short_request_first_config":{"enabled":true,"threshold":65546,"long_max_wait_ms":2000}}}')
        ;;
      *)
        echo "PD_P_SCHEDULER_MODE must be baseline or enhanced" >&2
        exit 2
        ;;
    esac
    exec vllm serve "$model" \
      --host "$http_host" \
      --port "$http_port" \
      --trust-remote-code \
      --data-parallel-size "$dp_size" \
      --data-parallel-size-local 2 \
      --data-parallel-start-rank 0 \
      --data-parallel-address "$dp_address" \
      --data-parallel-rpc-port "$dp_rpc_port" \
      --tensor-parallel-size "$tp_size" \
      --pipeline-parallel-size "$pp_size" \
      --distributed-executor-backend mp \
      --nnodes 2 \
      --node-rank "$node_rank" \
      --master-addr "$master_addr" \
      --master-port "$master_port" \
      "${headless_args[@]}" \
      --max-num-batched-tokens 24576 \
      --block-size 32 \
      --enable-chunked-prefill \
      --enable-request-id-headers \
      --enforce-eager \
      --no-enable-prefix-caching \
      --no-async-scheduling \
      --enable-expert-parallel \
      "${scheduler_args[@]}" \
      --served-model-name deepseek-v4-flash \
      --quantization ascend \
      --gpu-memory-utilization 0.85 \
      --max-model-len 87040 \
      --tokenizer-mode deepseek_v4 \
      --tool-call-parser deepseek_v4 \
      --reasoning-parser deepseek_v4 \
      --enable-auto-tool-choice \
      --no-disable-hybrid-kv-cache-manager \
      --model-loader-extra-config "$model_loader_config" \
      --kv-transfer-config "$kv_transfer_config"
    ;;
  decode)
    export HCCL_CONNECT_TIMEOUT=1200
    export HCCL_BUFFSIZE=1024
    http_host=${PD_HTTP_HOST:-0.0.0.0}
    kv_transfer_config='{"kv_connector":"MooncakeHybridConnector","kv_role":"kv_consumer","kv_port":"30100","engine_id":"1","kv_connector_extra_config":'"$topology"'}'
    exec vllm serve "$model" \
      --host "$http_host" \
      --port "$http_port" \
      --trust-remote-code \
      --data-parallel-size "$dp_size" \
      --data-parallel-rank "$dp_rank" \
      --data-parallel-address "$dp_address" \
      --data-parallel-rpc-port "$dp_rpc_port" \
      --tensor-parallel-size "$tp_size" \
      --pipeline-parallel-size "$pp_size" \
      --max-num-batched-tokens 120 \
      --max-num-seqs 60 \
      --block-size 32 \
      --enable-request-id-headers \
      --no-enable-prefix-caching \
      --no-async-scheduling \
      --enable-expert-parallel \
      --additional-config '{"ascend_compilation_config":{"enable_npugraph_ex":true,"enable_static_kernel":false},"enable_cpu_binding":true,"multistream_overlap_shared_expert":true,"recompute_scheduler_enable":true}' \
      --compilation-config '{"cudagraph_mode":"FULL_DECODE_ONLY"}' \
      --served-model-name deepseek-v4-flash \
      --quantization ascend \
      --gpu-memory-utilization 0.9 \
      --max-model-len 87040 \
      --tokenizer-mode deepseek_v4 \
      --tool-call-parser deepseek_v4 \
      --reasoning-parser deepseek_v4 \
      --enable-auto-tool-choice \
      --no-disable-hybrid-kv-cache-manager \
      --model-loader-extra-config "$model_loader_config" \
      --kv-transfer-config "$kv_transfer_config"
    ;;
  *)
    echo "Unknown role: $role (expected prefill or decode)" >&2
    exit 2
    ;;
esac
