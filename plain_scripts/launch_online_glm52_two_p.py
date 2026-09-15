#!/usr/bin/env python3
"""Launch the GLM-5.2 one-P-one-D service topology.

The P service is one DP replica split across two 16-NPU hosts as PP2/TP16:
  python launch_online_glm52_two_p.py --role prefill --node-rank 0 --local-ip P0_IP --master-addr P0_IP --nic-name NIC
  python launch_online_glm52_two_p.py --role prefill --node-rank 1 --local-ip P1_IP --master-addr P0_IP --nic-name NIC

The D service is DP1/PP1/TP16 on one 16-NPU host:
  python launch_online_glm52_two_p.py --role decode --local-ip D_IP --nic-name NIC
"""

from __future__ import annotations

import argparse
import os
import signal
import subprocess
import sys
import time
from pathlib import Path


SCRIPT = Path(__file__).with_name("run_dp_template_glm52_two_p.sh")
LAYOUTS = {
    "prefill": {"dp": 1, "tp": 16, "pp": 2, "port": 18080},
    "decode": {"dp": 1, "tp": 16, "pp": 1, "port": 18082},
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--role", choices=LAYOUTS, required=True)
    parser.add_argument("--local-ip", required=True, help="This host's communication IP")
    parser.add_argument("--nic-name", required=True, help="Interface carrying the local IP")
    parser.add_argument("--node-rank", type=int, choices=(0, 1), help="P host index")
    parser.add_argument("--master-addr", help="P0 communication IP; required on both P hosts")
    parser.add_argument("--master-port", type=int, default=29500)
    parser.add_argument("--dp-address", help="DP coordinator IP")
    parser.add_argument("--dp-rpc-port", type=int, default=12321)
    parser.add_argument("--vllm-start-port", type=int, help="HTTP port for this role")
    parser.add_argument("--device-start", type=int, default=0, help="First local NPU ID")
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()

    layout = LAYOUTS[args.role]
    if args.role == "prefill":
        if args.node_rank is None or not args.master_addr:
            parser.error("P requires --node-rank 0|1 and --master-addr P0_IP")
        if args.node_rank == 0 and args.local_ip != args.master_addr:
            parser.error("P0 --local-ip must equal --master-addr")
        args.dp_address = args.dp_address or args.master_addr
        if args.dp_address != args.master_addr:
            parser.error("P --dp-address must equal --master-addr")
    else:
        if args.node_rank is not None or args.master_addr is not None:
            parser.error("D does not use --node-rank or --master-addr")
        args.dp_address = args.dp_address or args.local_ip

    if args.vllm_start_port is None:
        args.vllm_start_port = layout["port"]
    if args.device_start < 0 or args.device_start + layout["tp"] > 16:
        parser.error("this topology requires 16 local NPUs in the range 0..15")
    for port in (args.dp_rpc_port, args.master_port, args.vllm_start_port):
        if not 1 <= port <= 65535:
            parser.error("ports must be between 1 and 65535")
    return args


def main() -> int:
    args = parse_args()
    if not SCRIPT.is_file():
        print(f"Missing template: {SCRIPT}", file=sys.stderr)
        return 2

    layout = LAYOUTS[args.role]
    visible_devices = ",".join(
        map(str, range(args.device_start, args.device_start + layout["tp"]))
    )
    command = [
        "bash", str(SCRIPT), args.role, visible_devices,
        str(args.vllm_start_port), str(layout["dp"]), "0", args.dp_address,
        str(args.dp_rpc_port), str(layout["tp"]), str(layout["pp"]),
        args.local_ip, args.nic_name,
        str(args.node_rank if args.node_rank is not None else -1),
        args.master_addr or "-", str(args.master_port),
    ]
    print("Launching:", " ".join(command), flush=True)
    if args.dry_run:
        return 0

    process: subprocess.Popen[bytes] | None = None
    stop_requested_at: float | None = None

    def stop_process(_signum: int | None = None, _frame: object = None) -> None:
        nonlocal stop_requested_at
        if stop_requested_at is None:
            stop_requested_at = time.monotonic()
        if process is not None and process.poll() is None:
            try:
                os.killpg(process.pid, signal.SIGTERM)
            except ProcessLookupError:
                pass

    signal.signal(signal.SIGINT, stop_process)
    signal.signal(signal.SIGTERM, stop_process)
    try:
        process = subprocess.Popen(command, start_new_session=True)
        while process.poll() is None:
            if stop_requested_at is not None and time.monotonic() - stop_requested_at >= 90:
                try:
                    os.killpg(process.pid, signal.SIGKILL)
                except ProcessLookupError:
                    pass
            time.sleep(0.5)
        return process.returncode or 0
    except OSError as exc:
        print(f"Failed to launch service: {exc}", file=sys.stderr)
        stop_process()
        return 1
    finally:
        if process is not None and process.poll() is None:
            stop_process()
            process.wait()


if __name__ == "__main__":
    raise SystemExit(main())
