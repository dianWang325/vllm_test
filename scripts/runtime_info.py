"""
Collect runtime identity information and build test run IDs.

This module provides the following public interfaces:

- RuntimeInfo:
    Immutable runtime identity containing the hostname, normalized host alias,
    vllm-ascend branch, and commit ID.

- read_host_alias():
    Read the current hostname and derive a filesystem-safe host alias.

- read_vllm_ascend_info():
    Locate the installed editable vllm-ascend package through direct_url.json
    and read its current Git branch and commit.

- read_runtime_info():
    Collect host and vllm-ascend information into RuntimeInfo.

- build_run_id(runtime_info, run_type, created_at=None):
    Build a run ID in the following format:
    <host_alias>_<branch>_<MMDDHHMM>_<performance|accuracy>

- collect_run_info(run_type, created_at=None):
    Return the runtime information and generated run ID as a dictionary.

Command-line usage:

    python3 -m scripts.runtime_info --run-type performance
    python3 -m scripts.runtime_info --run-type accuracy

The command prints the collected runtime information and run ID as JSON.
Missing vllm-ascend, invalid direct_url.json, non-file installations,
detached Git HEAD, or Git command failures terminate the command immediately.
"""

from __future__ import annotations

import argparse
import json
import re
import socket
import subprocess
import sys
from dataclasses import asdict, dataclass
from datetime import datetime
from importlib.metadata import PackageNotFoundError, distribution
from pathlib import Path
from typing import Literal
from urllib.parse import unquote, urlparse
from urllib.request import url2pathname


PACKAGE_NAME = "vllm-ascend"
RUN_TYPES = ("performance", "accuracy")


class RuntimeInfoError(RuntimeError):
    pass


class VllmAscendNotFoundError(RuntimeInfoError):
    pass


@dataclass(frozen=True)
class RuntimeInfo:
    hostname: str
    host_alias: str
    branch: str
    commit: str


def _normalize_component(value: str, field_name: str) -> str:
    normalized = re.sub(r"[^A-Za-z0-9._-]+", "-", value.strip())
    normalized = re.sub(r"-+", "-", normalized).strip("-")

    if not normalized:
        raise RuntimeInfoError(f"{field_name} is empty after normalization")

    return normalized


def read_host_alias() -> tuple[str, str]:
    hostname = socket.gethostname().strip()
    if not hostname:
        raise RuntimeInfoError("socket.gethostname() returned an empty hostname")

    match = re.search(r"[.-]([^-\.]+)$", hostname)
    if match:
        suffix = re.sub(r"[^A-Za-z0-9]+", "-", match.group(1))
        suffix = re.sub(r"-+", "-", suffix).strip("-")

        if suffix:
            return hostname, f"server{suffix}"

    host_alias = re.sub(r"[^A-Za-z0-9]+", "-", hostname)
    host_alias = re.sub(r"-+", "-", host_alias).strip("-")

    if not host_alias:
        raise RuntimeInfoError(
            "The hostname is empty after normalization; "
            "unable to generate host_alias"
        )

    return hostname, host_alias


def _run_git(repo_path: Path, *args: str) -> str:
    try:
        result = subprocess.run(
            ["git", "-C", str(repo_path), *args],
            check=False,
            capture_output=True,
            text=True,
            timeout=10,
        )
    except FileNotFoundError as exc:
        raise RuntimeInfoError(
            "The git command is not available in the current environment"
        ) from exc
    except subprocess.TimeoutExpired as exc:
        raise RuntimeInfoError(
            f"Timed out while reading vllm-ascend Git information: {repo_path}"
        ) from exc

    if result.returncode != 0:
        message = result.stderr.strip() or result.stdout.strip()
        raise RuntimeInfoError(
            f"Failed to read vllm-ascend Git information: "
            f"git -C {repo_path} {' '.join(args)}; {message}"
        )

    return result.stdout.strip()


def _path_from_file_url(url: str) -> Path:
    parsed = urlparse(url)
    if parsed.scheme != "file":
        raise RuntimeInfoError(f"Expected a local file URL, got: {url}")

    path_text = unquote(parsed.path)
    if parsed.netloc and parsed.netloc != "localhost":
        path_text = f"//{parsed.netloc}{path_text}"

    source_path = Path(url2pathname(path_text)).resolve()
    if not source_path.is_dir():
        raise RuntimeInfoError(
            f"The vllm-ascend source path does not exist "
            f"or is not a directory: {source_path}"
        )

    return source_path


def read_vllm_ascend_info() -> tuple[str, str]:
    try:
        dist = distribution(PACKAGE_NAME)
    except PackageNotFoundError as exc:
        raise VllmAscendNotFoundError(
            f"The {PACKAGE_NAME} package was not found in the current "
            "environment; aborting the test run"
        ) from exc

    direct_url_text = dist.read_text("direct_url.json")
    if not direct_url_text:
        raise RuntimeInfoError(
            f"{PACKAGE_NAME} does not provide direct_url.json; "
            "unable to determine its branch"
        )

    try:
        direct_url = json.loads(direct_url_text)
    except json.JSONDecodeError as exc:
        raise RuntimeInfoError(
            f"{PACKAGE_NAME} direct_url.json is not valid JSON"
        ) from exc

    source_url = direct_url.get("url")
    if not isinstance(source_url, str) or not source_url.strip():
        raise RuntimeInfoError(
            f"{PACKAGE_NAME} direct_url.json does not contain a valid url"
        )

    source_path = _path_from_file_url(source_url)
    branch = _run_git(
        source_path,
        "symbolic-ref",
        "--quiet",
        "--short",
        "HEAD",
    )
    commit = _run_git(source_path, "rev-parse", "HEAD")

    return branch, commit


def read_runtime_info() -> RuntimeInfo:
    hostname, host_alias = read_host_alias()
    branch, commit = read_vllm_ascend_info()

    return RuntimeInfo(
        hostname=hostname,
        host_alias=_normalize_component(host_alias, "host_alias"),
        branch=branch,
        commit=commit,
    )


def build_run_id(
    runtime_info: RuntimeInfo,
    run_type: Literal["performance", "accuracy"],
    created_at: datetime | None = None,
) -> str:
    if run_type not in RUN_TYPES:
        raise RuntimeInfoError(
            f"run_type must be either performance or accuracy, got: {run_type}"
        )

    effective_time = created_at or datetime.now().astimezone()
    timestamp = effective_time.strftime("%m%d%H%M")
    branch = _normalize_component(runtime_info.branch, "vllm-ascend branch")

    return (
        f"{runtime_info.host_alias}_"
        f"{branch}_"
        f"{timestamp}_"
        f"{run_type}"
    )


def collect_run_info(
    run_type: Literal["performance", "accuracy"],
    created_at: datetime | None = None,
) -> dict[str, str]:
    runtime_info = read_runtime_info()
    result = asdict(runtime_info)
    result["run_id"] = build_run_id(
        runtime_info=runtime_info,
        run_type=run_type,
        created_at=created_at,
    )
    return result


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Read server and vllm-ascend information and generate a run ID"
        )
    )
    parser.add_argument(
        "--run-type",
        required=True,
        choices=RUN_TYPES,
        help="Type of test run",
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()

    try:
        result = collect_run_info(args.run_type)
    except RuntimeInfoError as exc:
        print(str(exc), file=sys.stderr)
        return 1

    print(json.dumps(result, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())