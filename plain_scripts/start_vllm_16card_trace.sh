#!/usr/bin/env bash
set -Eeuo pipefail

SOURCE_SCRIPT="${SOURCE_SCRIPT:-/home/w00985415/start_vllm_16card.sh}"
TRACE_ROOT="${TRACE_ROOT:-/home/w00985415/vllm_trace_runs}"
RUN_STAMP="$(date +%Y%m%d_%H%M%S)"
RUN_DIR="${TRACE_ROOT}/${RUN_STAMP}_pid$$"
SERVER_LOG="${RUN_DIR}/server.log"
STATUS_FILE="${RUN_DIR}/status.txt"
SERVER_PID=""

mkdir -p "${RUN_DIR}"
printf '%s\n' "${RUN_DIR}" > "${TRACE_ROOT}/LATEST"

finish() {
    local exit_code=$?
    set +e
    {
        printf 'finished_at=%s\n' "$(date --iso-8601=seconds)"
        printf 'exit_code=%s\n' "${exit_code}"
        printf 'server_pid=%s\n' "${SERVER_PID}"
    } >> "${STATUS_FILE}"
    npu-smi info > "${RUN_DIR}/npu_after.txt" 2>&1
    ps -ef > "${RUN_DIR}/processes_after.txt" 2>&1
}

forward_signal() {
    if [[ -n "${SERVER_PID}" ]] && kill -0 "${SERVER_PID}" 2>/dev/null; then
        kill -TERM "${SERVER_PID}" 2>/dev/null || true
        wait "${SERVER_PID}" || true
    fi
    exit 143
}

trap finish EXIT
trap forward_signal INT TERM

if [[ ! -f "${SOURCE_SCRIPT}" ]]; then
    printf 'source script not found: %s\n' "${SOURCE_SCRIPT}" >&2
    exit 2
fi

cp -p "${SOURCE_SCRIPT}" "${RUN_DIR}/source_start_script.sh"
sha256sum "${SOURCE_SCRIPT}" > "${RUN_DIR}/source_start_script.sha256"

{
    printf 'started_at=%s\n' "$(date --iso-8601=seconds)"
    printf 'hostname=%s\n' "$(hostname)"
    printf 'working_directory=%s\n' "$(pwd)"
    printf 'source_script=%s\n' "${SOURCE_SCRIPT}"
    printf 'run_directory=%s\n' "${RUN_DIR}"
    printf 'uid=%s\n' "$(id -u)"
    printf 'gid=%s\n' "$(id -g)"
    printf 'command=' 
    printf '%q ' bash "${SOURCE_SCRIPT}" "$@"
    printf '\n'
} > "${RUN_DIR}/metadata.txt"

{
    for name in \
        ASCEND_HOME_PATH \
        ASCEND_RT_VISIBLE_DEVICES \
        ASCEND_TOOLKIT_HOME \
        HCCL_CONNECT_TIMEOUT \
        HCCL_EXEC_TIMEOUT \
        HOST \
        LD_LIBRARY_PATH \
        PATH \
        PORT \
        PYTHONPATH \
        SOC_VERSION \
        VLLM_EXECUTE_MODEL_TIMEOUT_SECONDS \
        VLLM_LOGGING_LEVEL \
        VLLM_USE_V1 \
        VLLM_WORKER_MULTIPROC_METHOD; do
        if [[ -v "${name}" ]]; then
            printf '%s=%s\n' "${name}" "${!name}"
        fi
    done
} > "${RUN_DIR}/environment.txt"

{
    printf 'vllm_commit=' 
    git -C /home/w00985415/proj_0825/deps/vllm rev-parse HEAD 2>/dev/null || true
    printf 'vllm_ascend_commit=' 
    git -C /home/w00985415/proj_0825/deps/vllm-ascend rev-parse HEAD 2>/dev/null || true
    printf 'vllm_ascend_verified_vllm=' 
    cat /home/w00985415/proj_0825/deps/vllm-ascend/.github/vllm-main-verified.commit 2>/dev/null || true
    python - <<'PY'
import importlib.metadata
for package in ("vllm", "vllm_ascend", "torch", "torch-npu", "triton-ascend"):
    try:
        version = importlib.metadata.version(package)
    except importlib.metadata.PackageNotFoundError:
        version = "not-installed"
    print(f"{package}_version={version}")
PY
} > "${RUN_DIR}/versions.txt" 2>&1

git -C /home/w00985415/proj_0825/deps/vllm status --short --branch \
    > "${RUN_DIR}/vllm_git_status.txt" 2>&1 || true
git -C /home/w00985415/proj_0825/deps/vllm-ascend status --short --branch \
    > "${RUN_DIR}/vllm_ascend_git_status.txt" 2>&1 || true
git -C /home/w00985415/proj_0825/deps/vllm-ascend submodule status --recursive \
    > "${RUN_DIR}/vllm_ascend_submodules.txt" 2>&1 || true
npu-smi info > "${RUN_DIR}/npu_before.txt" 2>&1 || true
ps -ef > "${RUN_DIR}/processes_before.txt" 2>&1 || true
ulimit -a > "${RUN_DIR}/ulimit.txt" 2>&1 || true

export PYTHONFAULTHANDLER="${PYTHONFAULTHANDLER:-1}"
export PYTHONUNBUFFERED="${PYTHONUNBUFFERED:-1}"
export TORCH_SHOW_CPP_STACKTRACES="${TORCH_SHOW_CPP_STACKTRACES:-1}"
export VLLM_LOGGING_LEVEL="${VLLM_LOGGING_LEVEL:-DEBUG}"

{
    printf 'launching_at=%s\n' "$(date --iso-8601=seconds)"
    printf 'run_directory=%s\n' "${RUN_DIR}"
    printf 'server_log=%s\n' "${SERVER_LOG}"
} >> "${STATUS_FILE}"

bash "${SOURCE_SCRIPT}" "$@" >> "${SERVER_LOG}" 2>&1 &
SERVER_PID=$!
printf 'server_pid=%s\n' "${SERVER_PID}" >> "${STATUS_FILE}"
wait "${SERVER_PID}"
