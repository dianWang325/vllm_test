"""Run one PD role locally from the same resolved vtest suite configuration."""

from __future__ import annotations

import argparse
import shlex
from pathlib import Path

from scripts.config import pd_role_nodes, resolve_case, resolve_suite
from scripts.remote_process import run as run_process
from scripts.remote_process import stop as stop_process
from scripts.server import build_environment_command


DEFAULT_SUITE = "deepseek_v4_flash_pd_performance"


def _server_config(case_name: str | None, suite_name: str | None) -> dict:
    if case_name is not None and suite_name is None:
        return resolve_case(case_name)["effective"]["server"]
    cases = resolve_suite(suite_name or DEFAULT_SUITE)["cases"]
    if case_name is None:
        return cases[0]["effective"]["server"]
    for case in cases:
        if case["name"] == case_name:
            return case["effective"]["server"]
    raise RuntimeError(f"case {case_name} is not in suite {suite_name}")


def _role_config(server: dict, role: str, node: str | None = None) -> dict:
    nodes = pd_role_nodes(server, role)
    if node is None:
        if len(nodes) != 1:
            raise RuntimeError(f"pd.{role} has multiple nodes; specify --node")
        return next(iter(nodes.values()))
    return nodes[node]


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("action", choices=("run", "stop", "command"))
    parser.add_argument("--role", choices=("prefill", "decode"), required=True)
    parser.add_argument("--suite")
    parser.add_argument("--case")
    parser.add_argument("--node")
    parser.add_argument("--pid-file", type=Path)
    parser.add_argument("--timeout-seconds", type=float, default=120)
    args = parser.parse_args()
    node_suffix = f"-{args.node}" if args.node else ""
    pid_file = args.pid_file or Path(f"/tmp/vtest-{args.role}{node_suffix}.pid")
    if args.action == "stop":
        return stop_process(pid_file, args.timeout_seconds)
    config = _role_config(_server_config(args.case, args.suite), args.role, args.node)
    command = build_environment_command(config)
    if args.action == "command":
        print(shlex.join(command))
        return 0
    return run_process(pid_file, command)


if __name__ == "__main__":
    raise SystemExit(main())
