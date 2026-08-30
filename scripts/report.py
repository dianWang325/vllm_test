"""Build configured reports from structured AISBench result files only."""

from __future__ import annotations

import csv
import json
import re
from pathlib import Path
from statistics import mean
from typing import Any


class ReportError(RuntimeError):
    pass


NUMBER_WITH_UNIT = re.compile(r"^\s*(-?\d+(?:\.\d+)?)\s*(.*?)\s*$")


def _number(value: Any) -> tuple[float, str]:
    if isinstance(value, (int, float)):
        return float(value), ""
    if isinstance(value, str):
        match = NUMBER_WITH_UNIT.match(value)
        if match:
            return float(match.group(1)), match.group(2)
    raise ReportError(f"metric value is not numeric: {value!r}")


def _performance_values(run_dir: Path, case_name: str) -> list[dict[str, Any]]:
    common = json.loads(
        (run_dir / f"{case_name}-performance.json").read_text(encoding="utf-8")
    )
    csv_path = run_dir / f"{case_name}-performance.csv"
    with csv_path.open("r", encoding="utf-8", newline="") as stream:
        rows = list(csv.DictReader(stream))
    by_repeat: dict[int, dict[str, Any]] = {}
    for repeat in common["repeats"]:
        flat: dict[str, Any] = {}
        for metric, stages in repeat["metrics"].items():
            for stage, value in stages.items():
                flat[f"{metric}.{stage}"] = value
        by_repeat[int(repeat["repeat"])] = flat
    for row in rows:
        repeat = int(row["repeat"])
        metric = row["Performance Parameters"]
        stage = row["Stage"]
        for statistic, value in row.items():
            if statistic not in {"repeat", "Performance Parameters", "Stage", "N"}:
                by_repeat[repeat][f"{metric}.{stage}.{statistic}"] = value
    return [by_repeat[key] for key in sorted(by_repeat)]


def _accuracy_values(run_dir: Path, case_name: str) -> list[dict[str, Any]]:
    value = json.loads(
        (run_dir / f"{case_name}-accuracy.json").read_text(encoding="utf-8")
    )
    return [value["metrics"]]


def build_report(
    run_type: str,
    case_name: str,
    config: dict[str, Any],
    run_dir: Path,
) -> dict[str, Any]:
    repetitions = (
        _performance_values(run_dir, case_name)
        if run_type == "performance"
        else _accuracy_values(run_dir, case_name)
    )
    metrics: dict[str, Any] = {}
    for name in config["metrics"]:
        raw_values = [repeat[name] for repeat in repetitions if name in repeat]
        if len(raw_values) != len(repetitions):
            if config["required"]:
                raise ReportError(f"required metric is missing: {name}")
            continue
        converted = [_number(value) for value in raw_values]
        units = {unit for _, unit in converted}
        if len(units) != 1:
            raise ReportError(f"metric has inconsistent units: {name}: {sorted(units)}")
        values = [value for value, _ in converted]
        metrics[name] = {
            "values": values,
            "average": mean(values),
            "unit": next(iter(units)),
        }
    report = {"case": case_name, "type": run_type, "metrics": metrics}
    json_path = run_dir / f"{case_name}-report.json"
    json_path.write_text(
        json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    csv_path = run_dir / f"{case_name}-report.csv"
    with csv_path.open("w", encoding="utf-8", newline="") as stream:
        writer = csv.writer(stream)
        writer.writerow(["metric", "values", "average", "unit"])
        for name, value in metrics.items():
            writer.writerow(
                [name, json.dumps(value["values"]), value["average"], value["unit"]]
            )
    return report


def build_suite_comparison(
    suite_name: str,
    case_names: list[str],
    run_dir: Path,
) -> dict[str, Any]:
    reports = []
    for case_name in case_names:
        path = run_dir / f"{case_name}-report.json"
        report = json.loads(path.read_text(encoding="utf-8"))
        if report.get("case") != case_name or report.get("type") != "performance":
            raise ReportError(f"invalid performance report for case: {case_name}")
        reports.append(report)

    metric_names = list(reports[0]["metrics"])
    expected = set(metric_names)
    for report in reports[1:]:
        if set(report["metrics"]) != expected:
            raise ReportError(
                f"suite comparison metrics differ: {report['case']}"
            )

    metrics: dict[str, Any] = {}
    for metric_name in metric_names:
        units = {
            report["metrics"][metric_name]["unit"] for report in reports
        }
        if len(units) != 1:
            raise ReportError(
                f"suite comparison units differ: {metric_name}: {sorted(units)}"
            )
        metrics[metric_name] = {
            "unit": next(iter(units)),
            "cases": {
                report["case"]: {
                    "values": report["metrics"][metric_name]["values"],
                    "average": report["metrics"][metric_name]["average"],
                }
                for report in reports
            },
        }

    comparison = {
        "suite": suite_name,
        "type": "performance",
        "cases": case_names,
        "metrics": metrics,
    }
    json_path = run_dir / f"{suite_name}-comparison.json"
    json_path.write_text(
        json.dumps(comparison, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    csv_path = run_dir / f"{suite_name}-comparison.csv"
    with csv_path.open("w", encoding="utf-8", newline="") as stream:
        writer = csv.writer(stream)
        writer.writerow(["metric", "unit", *case_names])
        for metric_name, value in metrics.items():
            writer.writerow(
                [
                    metric_name,
                    value["unit"],
                    *(value["cases"][case_name]["average"] for case_name in case_names),
                ]
            )
    return comparison
