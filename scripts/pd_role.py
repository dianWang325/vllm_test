"""Run one PD role locally from the same resolved vtest suite configuration."""

from __future__ import annotations

import argparse
import subprocess
from pathlib import Path

from scripts.config import deep_merge, resolve_case, resolve_suite
from scripts.remote_process import run as run_process
from scripts.remote_process import stop as stop_process
from scripts.server import build_command, build_environment


DEFAULT_SUITE = "deepseek_v4_flash_pd_performance"


def _server_config(case_name: str | None, suite_name: str) -> dict:
    if case_name is not None:
        return resolve_case(case_name)["effective"]["server"]
    return resolve_suite(suite_name)["cases"][0]["effective"]["server"]


def _role_config(server: dict, role: str) -> dict:
    pd = server.get("pd")
    if not isinstance(pd, dict) or not isinstance(pd.get(role), dict):
        raise RuntimeError(f"server has no PD role: {role}")
    base = {key: value for key, value in server.items() if key != "pd"}
    return deep_merge(base, pd[role])


def _environment_command(config: dict) -> list[str]:
    build_environment(config)
    command = ["env"]
    for name in config.get("unset_environment", []):
        command.extend(["-u", name])
    for name, value in config.get("environment", {}).items():
        command.append(f"{name}={value}")
    command.extend(build_command(config))
    return command


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("action", choices=("run", "stop", "command"))
    parser.add_argument("--role", choices=("prefill", "decode"), required=True)
    parser.add_argument("--suite", default=DEFAULT_SUITE)
    parser.add_argument("--case")
    parser.add_argument("--pid-file", type=Path)
    parser.add_argument("--timeout-seconds", type=float, default=120)
    args = parser.parse_args()
    pid_file = args.pid_file or Path(f"/tmp/vtest-{args.role}.pid")
    if args.action == "stop":
        return stop_process(pid_file, args.timeout_seconds)
    config = _role_config(_server_config(args.case, args.suite), args.role)
    command = _environment_command(config)
    if args.action == "command":
        print(subprocess.list2cmdline(command))
        return 0
    return run_process(pid_file, command)


if __name__ == "__main__":
    raise SystemExit(main())
