#!/usr/bin/env python3
"""Launch the A3 DeepSeek-V4-Flash Prefill or Decode ranks on one host.

Examples (run on the corresponding host):
  python launch_online_dp.py --role prefill --local-ip P_IP --nic-name NIC
  python launch_online_dp.py --role decode --local-ip D_IP --nic-name NIC

The Prefill host uses DP1/TP8/PP2 and ports 18080; the Decode host uses
DP2/TP8/PP1 and ports 18082-18083. Each role consumes devices 0-15.
"""

from __future__ import annotations

import argparse
import os
import signal
import subprocess
import sys
import time
from pathlib import Path


SCRIPT = Path(__file__).with_name("run_dp_template.sh")
LAYOUTS = {
    "prefill": {"dp": 1, "tp": 8, "pp": 2, "port": 18080},
    "decode": {"dp": 2, "tp": 8, "pp": 1, "port": 18082},
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--role", choices=LAYOUTS, required=True)
    parser.add_argument("--local-ip", required=True, help="This host's reachable communication IP")
    parser.add_argument("--nic-name", required=True, help="Interface carrying the local IP")
    parser.add_argument(
        "--dp-address",
        help="DP master IP; defaults to --local-ip (both ranks of D are on one host)",
    )
    parser.add_argument("--dp-rpc-port", type=int, default=12321)
    parser.add_argument("--vllm-start-port", type=int, help="First HTTP port for this role")
    parser.add_argument("--device-start", type=int, default=0, help="First local NPU ID")
    parser.add_argument("--dry-run", action="store_true", help="Print rank commands without starting vLLM")
    args = parser.parse_args()
    layout = LAYOUTS[args.role]
    args.dp_address = args.dp_address or args.local_ip
    if args.vllm_start_port is None:
        args.vllm_start_port = layout["port"]
    if args.device_start < 0:
        parser.error("--device-start must be nonnegative")
    for port in (args.dp_rpc_port, args.vllm_start_port, args.vllm_start_port + layout["dp"] - 1):
        if not 1 <= port <= 65535:
            parser.error("ports must be between 1 and 65535")
    if args.dp_rpc_port in range(args.vllm_start_port, args.vllm_start_port + layout["dp"]):
        parser.error("DP RPC port must differ from the HTTP ports")
    return args


def main() -> int:
    args = parse_args()
    if not SCRIPT.is_file():
        print(f"Missing template: {SCRIPT}", file=sys.stderr)
        return 2

    layout = LAYOUTS[args.role]
    ranks: list[list[str]] = []
    devices_per_rank = layout["tp"] * layout["pp"]
    for rank in range(layout["dp"]):
        start = args.device_start + rank * devices_per_rank
        visible_devices = ",".join(map(str, range(start, start + devices_per_rank)))
        ranks.append([
            "bash", str(SCRIPT), args.role, visible_devices,
            str(args.vllm_start_port + rank), str(layout["dp"]), str(rank),
            args.dp_address, str(args.dp_rpc_port), str(layout["tp"]),
            str(layout["pp"]), args.local_ip, args.nic_name,
        ])

    for command in ranks:
        print("Launching:", " ".join(command), flush=True)
    if args.dry_run:
        return 0

    processes: list[subprocess.Popen[bytes]] = []
    stop_requested_at: float | None = None

    def stop_all(_signum: int | None = None, _frame: object = None) -> None:
        nonlocal stop_requested_at
        if stop_requested_at is None:
            stop_requested_at = time.monotonic()
        for process in processes:
            if process.poll() is None:
                try:
                    os.killpg(process.pid, signal.SIGTERM)
                except ProcessLookupError:
                    pass

    signal.signal(signal.SIGINT, stop_all)
    signal.signal(signal.SIGTERM, stop_all)
    try:
        for command in ranks:
            processes.append(subprocess.Popen(command, start_new_session=True))
        # All ranks are required. Stop peers if one exits, including an early failure.
        while True:
            if stop_requested_at is not None and time.monotonic() - stop_requested_at >= 90:
                # vLLM can remain in its distributed shutdown after SIGTERM.
                # Escalate only the rank process groups created above.
                for process in processes:
                    if process.poll() is None:
                        try:
                            os.killpg(process.pid, signal.SIGKILL)
                        except ProcessLookupError:
                            pass
            for process in processes:
                code = process.poll()
                if code is not None:
                    stop_all()
                    for peer in processes:
                        if peer is not process:
                            peer.wait()
                    return code if code != 0 else 1
            time.sleep(0.5)
    except OSError as exc:
        print(f"Failed to launch rank: {exc}", file=sys.stderr)
        stop_all()
        return 1
    finally:
        for process in processes:
            if process.poll() is None:
                try:
                    os.killpg(process.pid, signal.SIGTERM)
                except ProcessLookupError:
                    pass
            process.wait()


if __name__ == "__main__":
    raise SystemExit(main())
