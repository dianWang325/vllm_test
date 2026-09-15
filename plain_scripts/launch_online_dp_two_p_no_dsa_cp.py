#!/usr/bin/env python3
"""Launch the DeepSeek two-P-host PD topology with DSA-CP disabled.

Run on P0 and P1, using P0's IP as --master-addr:
  python launch_online_dp_two_p_no_dsa_cp.py --role prefill --node-rank 0 --local-ip P0_IP --master-addr P0_IP --nic-name NIC
  python launch_online_dp_two_p_no_dsa_cp.py --role prefill --node-rank 1 --local-ip P1_IP --master-addr P0_IP --nic-name NIC
Run once on D:
  python launch_online_dp_two_p_no_dsa_cp.py --role decode --local-ip D_IP --nic-name NIC

P is DP2/PP2/TP8 on two 16-NPU hosts. D is DP2/PP1/TP8 on one
16-NPU host. This is a preserved copy of the two-P-host launcher; the
only service-feature change is that P does not enable DSA-CP.
"""

from __future__ import annotations

import argparse
import os
import signal
import subprocess
import sys
import time
from pathlib import Path


SCRIPT = Path(__file__).with_name("run_dp_template_two_p_no_dsa_cp.sh")
LAYOUTS = {
    "prefill": {"dp": 2, "tp": 8, "pp": 2, "port": 18080},
    "decode": {"dp": 2, "tp": 8, "pp": 1, "port": 18082},
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
    parser.add_argument("--vllm-start-port", type=int, help="First HTTP port for this role")
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
    if args.device_start < 0:
        parser.error("--device-start must be nonnegative")
    http_count = 1 if args.role == "prefill" else layout["dp"]
    http_ports = range(args.vllm_start_port, args.vllm_start_port + http_count)
    for port in (args.dp_rpc_port, args.master_port, *http_ports):
        if not 1 <= port <= 65535:
            parser.error("ports must be between 1 and 65535")
    return args


def main() -> int:
    args = parse_args()
    if not SCRIPT.is_file():
        print(f"Missing template: {SCRIPT}", file=sys.stderr)
        return 2

    layout = LAYOUTS[args.role]
    # P0 owns both DP engines behind the API. P1 starts one headless PP
    # stage for each DP rank so both eight-card groups are used.
    local_processes = 1 if args.role == "prefill" and args.node_rank == 0 else layout["dp"]
    devices_per_process = 16 if args.role == "prefill" and args.node_rank == 0 else layout["tp"]
    commands: list[list[str]] = []
    for rank in range(local_processes):
        start = args.device_start + rank * devices_per_process
        visible_devices = ",".join(map(str, range(start, start + devices_per_process)))
        if args.role == "prefill" and args.node_rank == 1:
            mp_node_rank = 1 + rank * 2
        else:
            mp_node_rank = args.node_rank if args.node_rank is not None else -1
        commands.append([
            "bash", str(SCRIPT), args.role, visible_devices,
            str(args.vllm_start_port + rank), str(layout["dp"]), str(rank),
            args.dp_address, str(args.dp_rpc_port), str(layout["tp"]),
            str(layout["pp"]), args.local_ip, args.nic_name, str(mp_node_rank),
            args.master_addr or "-", str(args.master_port),
        ])

    for command in commands:
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
        for command in commands:
            processes.append(subprocess.Popen(command, start_new_session=True))
        while True:
            if stop_requested_at is not None and time.monotonic() - stop_requested_at >= 90:
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
