#!/usr/bin/env bash
set -euo pipefail

repo=/home/w00985415/vllm_test
suite=prefill_baseline_cpp_v2_runner_1_performance
log_dir="$repo/launch_logs"
stamp=$(date +%Y%m%d-%H%M%S)
mkdir -p "$log_dir"
cd "$repo"

launch() {
    local tag=$1
    local devices=$2
    local port=$3
    local log="$log_dir/${stamp}_${tag}.log"
    local pid_file="$log_dir/${stamp}_${tag}.pid"

    nohup env \
        ASCEND_RT_VISIBLE_DEVICES="$devices" \
        VTEST_SERVER_PORT="$port" \
        VTEST_RUN_TAG="$tag" \
        ./vtest run suite "$suite" \
        >"$log" 2>&1 </dev/null &
    local pid=$!
    printf '%s\n' "$pid" >"$pid_file"
    printf '%s\tpid=%s\tport=%s\tdevices=%s\tlog=%s\n' \
        "$tag" "$pid" "$port" "$devices" "$log"
}

launch cards0-7 0,1,2,3,4,5,6,7 18080
launch cards8-15 8,9,10,11,12,13,14,15 18090
