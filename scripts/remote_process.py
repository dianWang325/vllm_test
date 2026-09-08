"""Run and stop one managed local process group using a verified pid file."""

from __future__ import annotations

import argparse
import json
import os
import select
import signal
import subprocess
import sys
import time
from pathlib import Path
from typing import Any


def _start_time(pid: int) -> str | None:
    try:
        stat = Path(f"/proc/{pid}/stat").read_text(encoding="utf-8")
    except OSError:
        return None
    end = stat.rfind(")")
    if end < 0:
        return None
    fields = stat[end + 2 :].split()
    return fields[19] if len(fields) > 19 else None


def _load(path: Path) -> dict[str, Any] | None:
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except FileNotFoundError:
        return None
    except (OSError, json.JSONDecodeError) as exc:
        raise RuntimeError(f"cannot read pid file {path}: {exc}") from exc
    if not isinstance(value, dict):
        raise RuntimeError(f"invalid pid file: {path}")
    return value


def _matches(record: dict[str, Any]) -> bool:
    pid = record.get("pid")
    pgid = record.get("pgid")
    started = record.get("start_time")
    if not isinstance(pid, int) or not isinstance(pgid, int) or pid <= 1 or pgid <= 1:
        return False
    if started != _start_time(pid):
        return False
    try:
        return os.getpgid(pid) == pgid
    except ProcessLookupError:
        return False


def _unlink_if_same(path: Path, pid: int) -> None:
    try:
        record = _load(path)
        if record is not None and record.get("pid") == pid:
            path.unlink(missing_ok=True)
    except (OSError, RuntimeError):
        pass


def run(
    pid_file: Path, command: list[str], *, stop_on_stdin_close: bool = False
) -> int:
    if command and command[0] == "--":
        command = command[1:]
    if not command:
        raise RuntimeError("run requires a command after --")
    existing = _load(pid_file)
    if existing is not None and _matches(existing):
        raise RuntimeError(
            f"managed process is already running with pid {existing['pid']}"
        )
    pid_file.unlink(missing_ok=True)
    child = subprocess.Popen(
        command, start_new_session=True,
        stdin=subprocess.DEVNULL if stop_on_stdin_close else None,
    )
    record = {
        "pid": child.pid,
        "pgid": os.getpgid(child.pid),
        "start_time": _start_time(child.pid),
        "command": command,
    }
    pid_file.parent.mkdir(parents=True, exist_ok=True)
    temporary = pid_file.with_name(f".{pid_file.name}.{os.getpid()}.tmp")
    temporary.write_text(json.dumps(record, ensure_ascii=False), encoding="utf-8")
    temporary.replace(pid_file)

    def terminate(_signum: int, _frame: object) -> None:
        if child.poll() is None:
            try:
                os.killpg(record["pgid"], signal.SIGTERM)
            except ProcessLookupError:
                pass

    signal.signal(signal.SIGTERM, terminate)
    signal.signal(signal.SIGINT, terminate)
    try:
        if stop_on_stdin_close:
            # Wait for child exit or controller EOF without a background reader.
            while child.poll() is None:
                readable, _, _ = select.select([sys.stdin], [], [], 0.5)
                if readable and not os.read(sys.stdin.fileno(), 4096):
                    terminate(signal.SIGTERM, None)
                    break
        return child.wait()
    finally:
        _unlink_if_same(pid_file, child.pid)


def stop(pid_file: Path, timeout_seconds: float) -> int:
    record = _load(pid_file)
    if record is None:
        return 0
    if not _matches(record):
        pid_file.unlink(missing_ok=True)
        return 0
    pgid = int(record["pgid"])
    os.killpg(pgid, signal.SIGTERM)
    deadline = time.monotonic() + timeout_seconds
    while time.monotonic() < deadline:
        if not _matches(record):
            pid_file.unlink(missing_ok=True)
            return 0
        time.sleep(0.5)
    raise RuntimeError(
        f"process group {pgid} did not stop after {timeout_seconds:g} seconds"
    )


def main() -> int:
    parser = argparse.ArgumentParser()
    subparsers = parser.add_subparsers(dest="action", required=True)
    run_parser = subparsers.add_parser("run")
    run_parser.add_argument("--pid-file", type=Path, required=True)
    run_parser.add_argument("--stop-on-stdin-close", action="store_true")
    run_parser.add_argument("command", nargs=argparse.REMAINDER)
    stop_parser = subparsers.add_parser("stop")
    stop_parser.add_argument("--pid-file", type=Path, required=True)
    stop_parser.add_argument("--timeout-seconds", type=float, default=120)
    args = parser.parse_args()
    if args.action == "run":
        return run(
            args.pid_file, args.command, stop_on_stdin_close=args.stop_on_stdin_close
        )
    return stop(args.pid_file, args.timeout_seconds)


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except (OSError, RuntimeError) as exc:
        print(f"managed process error: {exc}", file=sys.stderr)
        raise SystemExit(1) from exc
