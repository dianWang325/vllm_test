"""Run AISBench official GSM8K or GPQA accuracy evaluation."""

from __future__ import annotations

import csv
import json
import os
import shutil
import subprocess
import tempfile
from pathlib import Path
from typing import Any, Callable

from .bench import AISBENCH_CONFIG, build_server_runtime, run_client
from .config import relative_to_root


class AccuracyError(RuntimeError):
    pass


def _find_accuracy(value: Any) -> float:
    found: list[float] = []

    def visit(item: Any) -> None:
        if isinstance(item, dict):
            for key, child in item.items():
                if key == "accuracy" and isinstance(child, (int, float)):
                    found.append(float(child))
                else:
                    visit(child)
        elif isinstance(item, list):
            for child in item:
                visit(child)

    visit(value)
    if len(found) != 1:
        raise AccuracyError(f"expected one accuracy metric, found {len(found)}")
    return found[0]


def run_accuracy(
    case_name: str,
    server: dict[str, Any],
    settings: dict[str, Any],
    run_dir: Path,
    check_alive: Callable[[], None],
) -> list[str]:
    dataset_path = relative_to_root(settings["path"])
    if not dataset_path.is_file():
        raise AccuracyError(f"accuracy dataset does not exist: {dataset_path}")
    dataset_name = str(settings["dataset"])
    dataset_abbr = "gsm8k" if dataset_name == "gsm8k" else "GPQA_diamond"
    runtime = {
        "mode": "accuracy",
        "server": build_server_runtime(server),
        "settings": {
            "request_rate": settings["request_rate"],
            "retries": int(settings["retries"]),
            "max_output_tokens": int(settings["max_output_tokens"]),
            "concurrency": int(settings["concurrency"]),
            "generation_kwargs": settings["generation_kwargs"],
        },
        "dataset": {
            "name": dataset_name,
            "directory": str(dataset_path.parent),
            "filename": dataset_path.name,
        },
    }
    work_dir = Path(tempfile.mkdtemp(prefix="vtest-accuracy-"))
    command = [
        str(settings["executable"]),
        str(AISBENCH_CONFIG),
        "--mode",
        "all",
        "--work-dir",
        str(work_dir),
        "--num-warmups",
        str(settings["num_warmups"]),
        "--debug",
    ]
    if settings["dump_details"]:
        command.append("--dump-eval-details")
    environment = dict(os.environ)
    environment["VTEST_AISBENCH_RUNTIME"] = json.dumps(runtime, ensure_ascii=False)
    log_path = run_dir / f"{case_name}-accuracy.log"
    try:
        with log_path.open("w", encoding="utf-8") as log:
            log.write("$ " + subprocess.list2cmdline(command) + "\n")
            returncode = run_client(command, environment, log, check_alive)
        if returncode != 0:
            raise AccuracyError(f"AISBench accuracy command failed: {returncode}")
        matches = sorted(
            path
            for path in work_dir.rglob(f"{dataset_abbr}.json")
            if "results" in path.parts
        )
        if len(matches) != 1:
            raise AccuracyError(
                f"expected one AISBench result for {dataset_abbr}, found {len(matches)}"
            )
        raw_result = json.loads(matches[0].read_text(encoding="utf-8"))
        summary_tables: dict[str, list[dict[str, str]]] = {}
        for path in sorted(work_dir.rglob("*.csv")):
            with path.open("r", encoding="utf-8", newline="") as stream:
                summary_tables[str(path.relative_to(work_dir))] = list(
                    csv.DictReader(stream)
                )
        output = {
            "case": case_name,
            "dataset": dataset_name,
            "metrics": {"accuracy": _find_accuracy(raw_result)},
            "aisbench_result": raw_result,
            "aisbench_csv": summary_tables,
        }
        (run_dir / f"{case_name}-accuracy.json").write_text(
            json.dumps(output, ensure_ascii=False, indent=2) + "\n",
            encoding="utf-8",
        )
    finally:
        shutil.rmtree(work_dir)
    return command
