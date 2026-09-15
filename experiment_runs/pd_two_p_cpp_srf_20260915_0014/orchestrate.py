"""Run the approved eight-run A3 PD experiment with two Prefill hosts."""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
import time
from pathlib import Path


HERE = Path(__file__).resolve().parent
MANIFEST = json.loads((HERE / "manifest.json").read_text(encoding="utf-8"))
P0_HOST = MANIFEST["p0_host"]
P1_HOST = MANIFEST["p1_host"]
D_HOST = MANIFEST["d_host"]
REMOTE = MANIFEST["remote_root"]
PROJECT = "/home/w00985415/vllm_test"
PYTHON = "/usr/local/python3.11.10/bin/python3"
AISBENCH = "/usr/local/python3.11.10/bin/ais_bench"
MANAGER = f"{PROJECT}/scripts/remote_process.py"
PROXY = (
    "/home/w00985415/proj_0825/deps/vllm-ascend/examples/"
    "disaggregated_prefill_v1/load_balance_proxy_server_example.py"
)
SSH = ["ssh", "-T", "-o", "BatchMode=yes", "-o", "ConnectTimeout=8"]


def ssh(host: str, command: list[str], timeout: float = 30) -> subprocess.CompletedProcess[str]:
    return subprocess.run(
        [*SSH, f"root@{host}", *command], capture_output=True,
        text=True, timeout=timeout,
    )


def snapshot(host: str, target: Path) -> None:
    result = ssh(host, ["npu-smi", "info"], 30)
    target.write_text(result.stdout + result.stderr, encoding="utf-8")
    if result.returncode:
        raise RuntimeError(f"npu-smi failed on {host}: {result.returncode}")


def assert_idle(host: str) -> None:
    deadline = time.monotonic() + 180
    while True:
        result = ssh(host, ["npu-smi", "info"], 30)
        if result.returncode == 0 and result.stdout.count("No running processes found in NPU") == 8:
            return
        if time.monotonic() >= deadline:
            raise RuntimeError(f"{host} no longer has all 16 chips idle")
        time.sleep(10)


def health(host: str, port: int, path: str = "/health") -> tuple[bool, str]:
    result = ssh(host, [
        "docker", "exec", "wd_test0825", "curl", "--noproxy", "127.0.0.1",
        "-fsS", "--connect-timeout", "3", "--max-time", "6",
        f"http://127.0.0.1:{port}{path}",
    ], 20)
    return result.returncode == 0, result.stdout.strip()


class Service:
    def __init__(self, name: str, host: str, command: list[str], tag: str):
        self.name = name
        self.host = host
        self.pid_file = f"/tmp/{MANIFEST['run_id']}-{tag}-{name}.pid"
        self.log_path = HERE / tag / f"{name}.log"
        self.log = self.log_path.open("wb")
        remote_command = [
            "docker", "exec", "-i", "-w", PROJECT, "wd_test0825",
            PYTHON, MANAGER, "run", "--pid-file", self.pid_file,
            "--stop-on-stdin-close", "--", *command,
        ]
        self.process = subprocess.Popen(
            [*SSH, f"root@{host}", *remote_command],
            stdin=subprocess.PIPE, stdout=self.log, stderr=subprocess.STDOUT,
        )

    def alive(self) -> bool:
        return self.process.poll() is None

    def stop(self) -> None:
        if self.process.stdin and not self.process.stdin.closed:
            self.process.stdin.close()
        try:
            self.process.wait(timeout=120)
        except subprocess.TimeoutExpired:
            result = ssh(self.host, [
                "docker", "exec", "wd_test0825", PYTHON, MANAGER,
                "stop", "--pid-file", self.pid_file, "--timeout-seconds", "120",
            ], 140)
            if result.returncode:
                raise RuntimeError(f"Could not stop {self.name}: {result.stderr}")
            self.process.wait(timeout=30)
        self.log.close()


def wait_for_health(services: list[Service], host: str, ports: list[int], path: str, timeout: int) -> None:
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        for service in services:
            if not service.alive():
                raise RuntimeError(f"{service.name} exited; see {service.log_path}")
        checks = [health(host, port, path) for port in ports]
        if all(ok for ok, _ in checks):
            if path == "/healthcheck":
                result = json.loads(checks[0][1])
                if result.get("prefill_instances") != 1 or result.get("decode_instances") != 2:
                    raise RuntimeError(f"Proxy registered unexpected backend counts: {result}")
            return
        time.sleep(10)
    raise RuntimeError(f"Timed out waiting for {host}:{ports} {path}")


def run_bench(tag: str, config: str, phase: str, expected: int, services: list[Service]) -> None:
    log_path = HERE / tag / f"{phase}.log"
    work_dir = f"{REMOTE}/results/{tag}/{phase}/{time.time_ns()}"
    command = [
        "docker", "exec", "-w", PROJECT, "wd_test0825", AISBENCH,
        f"{REMOTE}/configs/{config}", "--mode", "perf", "--work-dir", work_dir,
        "--debug",
    ]
    with log_path.open("wb") as log:
        process = subprocess.Popen([*SSH, f"root@{P0_HOST}", *command], stdout=log, stderr=subprocess.STDOUT)
        deadline = time.monotonic() + (3600 if phase == "warmup" else 7200)
        try:
            while process.poll() is None:
                exited = [service for service in services if not service.alive()]
                if exited:
                    process.terminate()
                    details = ", ".join(
                        f"{service.name} (ssh exit {service.process.returncode})"
                        for service in exited
                    )
                    raise RuntimeError(f"{details} exited during {phase}; see service logs")
                if time.monotonic() >= deadline:
                    process.terminate()
                    raise RuntimeError(f"AISBench {phase} exceeded its timeout")
                time.sleep(5)
        finally:
            if process.poll() is None:
                process.terminate()
                process.wait(timeout=30)
    if process.returncode:
        raise RuntimeError(f"AISBench {phase} exited {process.returncode}: {log_path}")
    paths = []
    for name in ("customdataset.json", "vtest_data.json"):
        found = ssh(P0_HOST, ["find", work_dir, "-type", "f", "-name", name, "-size", "+0c"], 30)
        if found.returncode == 0:
            paths.extend(found.stdout.splitlines())
    if len(paths) != 1:
        raise RuntimeError(f"AISBench {phase} performance JSON missing under {work_dir}")
    data = ssh(P0_HOST, ["cat", paths[0]], 30)
    metrics = json.loads(data.stdout)
    success = metrics.get("Success Requests", {}).get("total")
    failed = metrics.get("Failed Requests", {}).get("total")
    if success != expected or failed != 0:
        raise RuntimeError(f"AISBench {phase} request counts invalid: success={success}, failed={failed}; {log_path}")
    (HERE / tag / f"{phase}_metrics.json").write_text(json.dumps(metrics, indent=2), encoding="utf-8")
    csv_path = paths[0][:-5] + ".csv"
    csv_check = ssh(P0_HOST, ["test", "-s", csv_path], 30)
    if csv_check.returncode:
        raise RuntimeError(f"AISBench {phase} performance CSV missing: {csv_path}")
    subprocess.run([
        "scp", "-o", "BatchMode=yes", f"root@{P0_HOST}:{csv_path}",
        str(HERE / tag / f"{phase}_metrics.csv"),
    ], check=True, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)


def run_one(index: int, mode: str, dataset: str) -> None:
    tag = f"{index:02d}_{mode}_{dataset}"
    folder = HERE / tag
    folder.mkdir(parents=True, exist_ok=True)
    status_path = folder / "status.json"
    if status_path.exists() and json.loads(status_path.read_text(encoding="utf-8")).get("status") == "complete":
        print(f"{tag}: already complete", flush=True)
        return
    assert_idle(P0_HOST)
    assert_idle(P1_HOST)
    assert_idle(D_HOST)
    snapshot(P0_HOST, folder / "p0-before-npu.txt")
    snapshot(P1_HOST, folder / "p1-before-npu.txt")
    snapshot(D_HOST, folder / "d-before-npu.txt")
    services: list[Service] = []
    status = {
        "tag": tag, "mode": mode, "dataset": dataset,
        "p0_host": P0_HOST, "p1_host": P1_HOST, "d_host": D_HOST,
        "commit": MANIFEST["commit"],
        "status": "starting", "started_at": time.time(),
    }
    status_path.write_text(json.dumps(status, indent=2), encoding="utf-8")
    try:
        p1_command = [
            "env", f"PD_P_SCHEDULER_MODE={mode}", PYTHON,
            f"{REMOTE}/scripts/launch_online_dp_two_p.py", "--role", "prefill",
            "--node-rank", "1", "--local-ip", P1_HOST,
            "--master-addr", P0_HOST, "--nic-name", MANIFEST["nic"],
        ]
        services.append(Service("prefill_p1", P1_HOST, p1_command, tag))
        p0_command = [
            "env", f"PD_P_SCHEDULER_MODE={mode}", PYTHON,
            f"{REMOTE}/scripts/launch_online_dp_two_p.py", "--role", "prefill",
            "--node-rank", "0", "--local-ip", P0_HOST,
            "--master-addr", P0_HOST, "--nic-name", MANIFEST["nic"],
        ]
        services.append(Service("prefill_p0", P0_HOST, p0_command, tag))
        wait_for_health(services, P0_HOST, [18080], "/health", 3600)
        d_command = [
            PYTHON, f"{PROJECT}/plain_scripts/launch_online_dp_two_p.py", "--role", "decode",
            "--local-ip", D_HOST, "--nic-name", MANIFEST["nic"],
        ]
        services.append(Service("decode", D_HOST, d_command, tag))
        wait_for_health(services, D_HOST, [18082, 18083], "/health", 3600)
        proxy_command = [
            PYTHON, PROXY, "--host", "127.0.0.1", "--port", "18090",
            "--prefiller-hosts", "127.0.0.1", "--prefiller-ports", "18080",
            "--decoder-hosts", D_HOST, D_HOST,
            "--decoder-ports", "18082", "18083", "--workers", "1",
        ]
        services.append(Service("proxy", P0_HOST, proxy_command, tag))
        wait_for_health(services, P0_HOST, [18090], "/healthcheck", 300)
        status["status"] = "healthy"
        status_path.write_text(json.dumps(status, indent=2), encoding="utf-8")
        if mode == "enhanced":
            run_bench(tag, "prefill_warmup.py", "warmup", 5, services)
            if "Mooncake transfer failed" in (folder / "decode.log").read_text(encoding="utf-8", errors="replace"):
                raise RuntimeError(f"D reported Mooncake transfer failures during warmup; see {folder / 'decode.log'}")
            status["status"] = "warmed"
            status_path.write_text(json.dumps(status, indent=2), encoding="utf-8")
        run_bench(tag, f"prefill_{dataset}.py", "formal", 24, services)
        status["status"] = "formal_passed"
    except BaseException as exc:
        status["status"] = "failed"
        status["error"] = str(exc)
        print(f"{tag}: FAILED: {exc}", file=sys.stderr, flush=True)
        raise
    finally:
        errors = []
        for service in reversed(services):
            try:
                service.stop()
            except BaseException as exc:
                errors.append(f"{service.name}: {exc}")
        status["stop_errors"] = errors
        if status["status"] == "formal_passed" and not errors:
            status["status"] = "complete"
            status["completed_at"] = time.time()
            print(f"{tag}: complete", flush=True)
        status_path.write_text(json.dumps(status, indent=2), encoding="utf-8")
        if errors:
            raise RuntimeError("Service teardown incomplete: " + "; ".join(errors))


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--max-runs", type=int, default=8)
    args = parser.parse_args()
    for index, (mode, dataset) in enumerate(MANIFEST["schedule"][:args.max_runs], 1):
        run_one(index, mode, dataset)
    if args.max_runs == 8:
        subprocess.run([sys.executable, str(HERE / "summarize.py")], check=True)
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except (OSError, RuntimeError, subprocess.TimeoutExpired) as exc:
        print(f"Experiment stopped: {exc}", file=sys.stderr, flush=True)
        raise SystemExit(1) from exc
