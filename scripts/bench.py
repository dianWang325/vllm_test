"""Run AISBench performance requests and consolidate its statistics."""

from __future__ import annotations

import csv
import json
import os
import shutil
import signal
import subprocess
import tempfile
from pathlib import Path
from typing import Any, Callable, TextIO

from .config import ROOT


AISBENCH_CONFIG = (
    ROOT / "aisbench_workspace" / "configs" / "models" / "vllm_api" / "vllm_test.py"
)


class BenchError(RuntimeError):
    pass


def run_client(
    command: list[str], environment: dict[str, str], log: TextIO,
    check_alive: Callable[[], None],
) -> int:
    """Stop requests when any managed server rank exits, including headless ranks."""
    check_alive()
    log.flush()
    process = subprocess.Popen(
        command, stdout=log, stderr=subprocess.STDOUT, text=True,
        env=environment, start_new_session=True,
    )
    try:
        while True:
            check_alive()
            try:
                returncode = process.wait(timeout=1)
            except subprocess.TimeoutExpired:
                continue
            check_alive()
            return returncode
    except BaseException:
        if process.poll() is None:
            try:
                os.killpg(os.getpgid(process.pid), signal.SIGTERM)
            except ProcessLookupError:
                pass
            process.wait(timeout=120)
        raise


def build_server_runtime(server: dict[str, Any]) -> dict[str, Any]:
    arguments = server["arguments"]
    model_tag = server["model_tag"]
    return {
        "host": arguments["--host"],
        "port": int(arguments["--port"]),
        "tokenizer": arguments.get("--tokenizer", model_tag),
        "served_name": arguments.get("--served-model-name", model_tag),
        "trust_remote_code": (
            "--trust-remote-code" in arguments
            and arguments["--trust-remote-code"] is not False
        ),
    }


def _one(path: Path, pattern: str, required_directory: str) -> Path:
    matches = sorted(
        match for match in path.rglob(pattern) if required_directory in match.parts
    )
    if len(matches) != 1:
        raise BenchError(f"expected one {pattern} below {path}, found {len(matches)}")
    return matches[0]


def _runtime(
    server: dict[str, Any],
    settings: dict[str, Any],
    dataset: dict[str, Any],
) -> dict[str, Any]:
    return {
        "mode": "performance",
        "server": build_server_runtime(server),
        "settings": {
            "request_rate": settings["request_rate"],
            "retries": int(settings["retries"]),
            "max_output_tokens": int(dataset["output_length"]),
            "concurrency": int(settings["concurrency"]),
            "temperature": settings["temperature"],
            "repetition_penalty": settings["repetition_penalty"],
            "ignore_eos": bool(settings["ignore_eos"]),
            "stats": settings["stats"],
        },
        "dataset": {
            "abbr": "vtest_data",
            "path": dataset["path"],
            "meta_path": dataset["meta_path"],
        },
    }


def _invoke(
    executable: str,
    runtime: dict[str, Any],
    log_path: Path,
    check_alive: Callable[[], None],
) -> tuple[list[str], Path]:
    work_dir = Path(tempfile.mkdtemp(prefix="vtest-aisbench-"))
    command = [
        executable,
        str(AISBENCH_CONFIG),
        "--mode",
        "perf",
        "--work-dir",
        str(work_dir),
        "--num-warmups",
        "0",
        "--debug",
    ]
    environment = dict(os.environ)
    environment["VTEST_AISBENCH_RUNTIME"] = json.dumps(runtime, ensure_ascii=False)
    try:
        with log_path.open("a", encoding="utf-8") as log:
            log.write("$ " + subprocess.list2cmdline(command) + "\n")
            returncode = run_client(command, environment, log, check_alive)
        if returncode != 0:
            raise BenchError(f"AISBench performance command failed: {returncode}")
    except BaseException:
        shutil.rmtree(work_dir)
        raise
    return command, work_dir


def run_warmup(
    server: dict[str, Any],
    settings: dict[str, Any],
    dataset: dict[str, Any],
    log_path: Path,
    check_alive: Callable[[], None],
) -> list[str]:
    runtime = _runtime(server, settings, dataset)
    command, work_dir = _invoke(str(settings["executable"]), runtime, log_path, check_alive)
    shutil.rmtree(work_dir)
    return command


def run_performance(
    case_name: str,
    server: dict[str, Any],
    settings: dict[str, Any],
    dataset: dict[str, Any],
    run_dir: Path,
    check_alive: Callable[[], None],
) -> list[list[str]]:
    log_path = run_dir / f"{case_name}-bench.log"
    common_repeats: list[dict[str, Any]] = []
    detail_repeats: list[dict[str, Any]] = []
    csv_rows: list[dict[str, Any]] = []
    commands: list[list[str]] = []
    runtime = _runtime(server, settings, dataset)
    for repeat in range(1, int(settings["repeats"]) + 1):
        command, work_dir = _invoke(str(settings["executable"]), runtime, log_path, check_alive)
        commands.append(command)
        try:
            common_path = _one(work_dir, "vtest_data.json", "performances")
            csv_path = _one(work_dir, "vtest_data.csv", "performances")
            details_path = _one(
                work_dir, "vtest_data_details.jsonl", "performances"
            )
            common_repeats.append(
                {
                    "repeat": repeat,
                    "metrics": json.loads(common_path.read_text(encoding="utf-8")),
                }
            )
            with csv_path.open("r", encoding="utf-8", newline="") as stream:
                for row in csv.DictReader(stream):
                    csv_rows.append({"repeat": repeat, **row})
            with details_path.open("r", encoding="utf-8") as stream:
                details = [json.loads(line) for line in stream if line.strip()]
            detail_repeats.append({"repeat": repeat, "requests": details})
        finally:
            shutil.rmtree(work_dir)

    (run_dir / f"{case_name}-performance.json").write_text(
        json.dumps(
            {"case": case_name, "repeats": common_repeats},
            ensure_ascii=False,
            indent=2,
        )
        + "\n",
        encoding="utf-8",
    )
    (run_dir / f"{case_name}-performance-details.json").write_text(
        json.dumps(
            {"case": case_name, "repeats": detail_repeats},
            ensure_ascii=False,
            indent=2,
        )
        + "\n",
        encoding="utf-8",
    )
    csv_path = run_dir / f"{case_name}-performance.csv"
    if not csv_rows:
        raise BenchError("AISBench performance CSV contains no rows")
    with csv_path.open("w", encoding="utf-8", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(csv_rows[0]))
        writer.writeheader()
        writer.writerows(csv_rows)
    return commands
