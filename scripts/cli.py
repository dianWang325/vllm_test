"""Public command-line interface for cases and suites."""

from __future__ import annotations

import argparse
import json
import shutil
import sys
from dataclasses import asdict
from datetime import datetime
from pathlib import Path
from typing import Any

from .accuracy import run_accuracy
from .bench import run_performance, run_warmup
from .config import ROOT, dump_yaml, list_entries, resolve_case, resolve_suite
from .data import assert_prefix_disjoint, generate_formal, generate_warmup
from .report import build_report, build_suite_comparison
from .runtime_info import build_run_id, read_runtime_info
from .server import VllmServer


def _now() -> str:
    return datetime.now().astimezone().isoformat(timespec="seconds")


def _write_json(path: Path, value: dict[str, Any]) -> None:
    path.write_text(
        json.dumps(value, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )


def _tokenizer(server: dict[str, Any]) -> str:
    return str(server["arguments"].get("--tokenizer", server["model_tag"]))


def _selected(
    kind: str,
    selection_name: str,
    run_id: str,
    runtime: dict[str, Any],
    cases: list[dict[str, Any]],
) -> dict[str, Any]:
    return {
        "run": {"run_id": run_id, **runtime},
        "selection": {
            "kind": kind,
            "name": selection_name,
            "cases": [case["name"] for case in cases],
        },
        "selected": {case["name"]: case["selected"] for case in cases},
        "effective": {case["name"]: case["effective"] for case in cases},
    }


def _execute_case(
    case: dict[str, Any],
    run_dir: Path,
    state: dict[str, Any],
    server_process: VllmServer,
) -> None:
    name = case["name"]
    effective = case["effective"]
    case_state = {"status": "running", "started_at": _now(), "commands": []}
    state["cases"][name] = case_state
    _write_json(run_dir / "run.json", state)
    if case["type"] == "performance":
        server = effective["server"]
        formal = generate_formal(
            case["definition"]["data"],
            effective["data"],
            _tokenizer(server),
            run_dir,
            name,
        )
        case_state["formal_data"] = formal
        if "warmup" in effective:
            warmup = generate_warmup(
                case["definition"]["warmup"],
                effective["warmup"],
                _tokenizer(server),
                run_dir,
                name,
            )
            assert_prefix_disjoint(
                formal["path"],
                warmup["path"],
                _tokenizer(server),
                int(server["arguments"]["--block-size"]),
            )
            case_state["warmup_data"] = warmup
            warmup_dataset = {
                **warmup,
                "output_length": int(effective["warmup"]["output_length"]),
            }
            warmup_log = run_dir / f"{name}-warmup.log"
            quiescence = server_process.wait_for_prefill_idle()
            if quiescence is not None:
                case_state["prefill_quiescence"] = quiescence
                _write_json(run_dir / "run.json", state)
            command = run_warmup(
                server, effective["warmup_bench"], warmup_dataset, warmup_log
            )
            case_state["commands"].append(command)
        formal_dataset = {
            **formal,
            "output_length": int(effective["data"]["output"]["length"]),
        }
        commands = run_performance(
            name, server, effective["bench"], formal_dataset, run_dir
        )
        case_state["commands"].extend(commands)
    else:
        server = effective["server"]
        if "warmup" in effective:
            warmup = generate_warmup(
                case["definition"]["warmup"],
                effective["warmup"],
                _tokenizer(server),
                run_dir,
                name,
            )
            case_state["warmup_data"] = warmup
            warmup_dataset = {
                **warmup,
                "output_length": int(effective["warmup"]["output_length"]),
            }
            warmup_log = run_dir / f"{name}-warmup.log"
            quiescence = server_process.wait_for_prefill_idle()
            if quiescence is not None:
                case_state["prefill_quiescence"] = quiescence
                _write_json(run_dir / "run.json", state)
            command = run_warmup(
                server, effective["warmup_bench"], warmup_dataset, warmup_log
            )
            case_state["commands"].append(command)
        command = run_accuracy(
            name, server, effective["accuracy"], run_dir
        )
        case_state["commands"].append(command)
    case_state["report"] = build_report(
        case["type"], name, effective["report"], run_dir
    )
    case_state["status"] = "completed"
    case_state["finished_at"] = _now()
    _write_json(run_dir / "run.json", state)


def _server_segments(
    cases: list[dict[str, Any]],
) -> list[list[dict[str, Any]]]:
    segments: list[list[dict[str, Any]]] = []
    for case in cases:
        if (
            not segments
            or segments[-1][0]["effective"]["server"]
            != case["effective"]["server"]
        ):
            segments.append([case])
        else:
            segments[-1].append(case)
    return segments


def execute(kind: str, name: str) -> list[Path]:
    suite: dict[str, Any] | None = None
    if kind == "case":
        cases = [resolve_case(name)]
    else:
        suite = resolve_suite(name)
        cases = suite["cases"]
    runtime_info = read_runtime_info()
    created_at = datetime.now().astimezone()
    runtime = asdict(runtime_info)
    by_type: dict[str, list[dict[str, Any]]] = {}
    for case in cases:
        by_type.setdefault(case["type"], []).append(case)

    run_ids = {
        run_type: build_run_id(runtime_info, run_type, created_at)
        for run_type in by_type
    }
    planned_dirs = {run_type: ROOT / "runs" / run_id for run_type, run_id in run_ids.items()}
    existing = [path for path in planned_dirs.values() if path.exists()]
    if existing:
        raise RuntimeError(f"run path already exists: {existing[0]}")
    suite_manifest: Path | None = None
    if kind == "suite":
        prefix = build_run_id(runtime_info, "performance", created_at).rsplit("_", 1)[0]
        suite_manifest = ROOT / "runs" / f"{prefix}_{name}_suite.yaml"
        if suite_manifest.exists():
            raise RuntimeError(f"suite manifest already exists: {suite_manifest}")

    run_dirs: dict[str, Path] = {}
    states: dict[str, dict[str, Any]] = {}
    for run_type, type_cases in by_type.items():
        run_id = run_ids[run_type]
        run_dir = planned_dirs[run_type]
        run_dir.mkdir(parents=True)
        run_dirs[run_type] = run_dir
        dump_yaml(
            run_dir / "selected.yaml",
            _selected(kind, name, run_id, runtime, type_cases),
        )
        state = {
            "run_id": run_id,
            "type": run_type,
            "status": "created",
            "selection": {"kind": kind, "name": name},
            "runtime": runtime,
            "started_at": _now(),
            "servers": [],
            "cases": {},
        }
        states[run_type] = state
        _write_json(run_dir / "run.json", state)

    if kind == "suite":
        manifest = {
            "suite": name,
            "created_at": created_at.isoformat(),
            "runs": {run_type: str(path) for run_type, path in run_dirs.items()},
        }
        if suite["definition"].get("comparison", False):
            manifest["comparison"] = str(
                run_dirs["performance"] / f"{name}-comparison.json"
            )
        dump_yaml(suite_manifest, manifest)

    error: BaseException | None = None
    try:
        for index, segment in enumerate(_server_segments(cases), start=1):
            log_name = f"server-{index:02d}.log"
            primary_dir = run_dirs[segment[0]["type"]]
            server = VllmServer(
                segment[0]["effective"]["server"], primary_dir / log_name
            )
            segment_error: BaseException | None = None
            try:
                command = server.start()
                record = {
                    "cases": [case["name"] for case in segment],
                    "command": command,
                    "log": log_name,
                }
                for run_type, state in states.items():
                    state["status"] = "running"
                    state["servers"].append(record)
                    _write_json(run_dirs[run_type] / "run.json", state)
                for case in segment:
                    try:
                        _execute_case(
                            case,
                            run_dirs[case["type"]],
                            states[case["type"]],
                            server,
                        )
                    except BaseException as case_exc:
                        case_state = states[case["type"]]["cases"][case["name"]]
                        case_state["status"] = "failed"
                        case_state["error"] = (
                            f"{type(case_exc).__name__}: {case_exc}"
                        )
                        case_state["finished_at"] = _now()
                        _write_json(
                            run_dirs[case["type"]] / "run.json",
                            states[case["type"]],
                        )
                        raise
            except BaseException as exc:
                segment_error = exc
            try:
                server.stop()
            except BaseException as stop_exc:
                if segment_error is None:
                    segment_error = stop_exc
            for run_dir in run_dirs.values():
                destination = run_dir / log_name
                if destination != server.log_path and server.log_path.exists():
                    shutil.copy2(server.log_path, destination)
            if segment_error is not None:
                raise segment_error

        if suite is not None and suite["definition"].get("comparison", False):
            comparison = build_suite_comparison(
                name,
                [case["name"] for case in cases],
                run_dirs["performance"],
            )
            states["performance"]["comparison"] = comparison
            _write_json(
                run_dirs["performance"] / "run.json", states["performance"]
            )
    except BaseException as exc:
        error = exc
        for run_type, state in states.items():
            state["status"] = "failed"
            state["error"] = f"{type(exc).__name__}: {exc}"
            state["finished_at"] = _now()
            _write_json(run_dirs[run_type] / "run.json", state)

    if error is not None:
        raise error
    for run_type, state in states.items():
        state["status"] = "completed"
        state["finished_at"] = _now()
        _write_json(run_dirs[run_type] / "run.json", state)
    return list(run_dirs.values())


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Minimal vLLM test framework")
    subparsers = parser.add_subparsers(dest="command", required=True)
    subparsers.add_parser("list", help="list configured cases and suites")
    run = subparsers.add_parser("run", help="run a case or suite")
    target = run.add_subparsers(dest="kind", required=True)
    for kind in ("case", "suite"):
        item = target.add_parser(kind)
        item.add_argument("name")
    info = subparsers.add_parser("runtime-info", help="show run identity inputs")
    info.add_argument("--run-type", choices=("performance", "accuracy"), required=True)
    return parser


def main() -> int:
    args = _parser().parse_args()
    try:
        if args.command == "list":
            for kind, entries in list_entries().items():
                print(f"{kind}:")
                for name, description in entries.items():
                    print(f"  {name}: {description}")
            return 0
        if args.command == "runtime-info":
            info = read_runtime_info()
            value = asdict(info)
            value["run_id"] = build_run_id(info, args.run_type)
            print(json.dumps(value, ensure_ascii=False, indent=2))
            return 0
        paths = execute(args.kind, args.name)
        for path in paths:
            print(path)
        return 0
    except Exception as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 1
