"""Create AISBench formal datasets and manual warmup datasets."""

from __future__ import annotations

import hashlib
import importlib
import json
import os
import random
import sys
from contextlib import contextmanager
from pathlib import Path
from typing import Any, Iterator
from importlib.metadata import version

from .config import ROOT


class DataError(RuntimeError):
    pass


def _canonical_hash(value: Any) -> str:
    payload = json.dumps(
        value, ensure_ascii=False, sort_keys=True, separators=(",", ":")
    ).encode("utf-8")
    return hashlib.sha256(payload).hexdigest()[:12]


def _file_hash(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()[:12]


def _meta(path: Path, request_count: int, output_length: int) -> Path:
    meta_path = Path(str(path) + ".meta.json")
    value = {
        "request_count": request_count,
        "sampling_mode": "default",
        "output_config": {
            "method": "percentage",
            "params": {"percentage_distribute": [[output_length, 1.0]]},
        },
    }
    meta_path.write_text(
        json.dumps(value, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    return meta_path


def _synthetic_config(input_config: dict[str, Any], count: int, output: int) -> dict[str, Any]:
    method = input_config["method"]
    if method == "fixed":
        input_method = "uniform"
        input_params = {
            "MinValue": int(input_config["length"]),
            "MaxValue": int(input_config["length"]),
        }
    elif method == "gaussian":
        input_method = "gaussian"
        input_params = {
            "Mean": int(input_config["mean"]),
            "Var": float(input_config["stddev"]) ** 2,
            "MinValue": int(input_config["min"]),
            "MaxValue": int(input_config["max"]),
        }
    else:
        raise DataError(f"unsupported AISBench synthetic input method: {method}")
    return {
        "Type": "string",
        "RequestCount": count,
        "StringConfig": {
            "Input": {"Method": input_method, "Params": input_params},
            "Output": {
                "Method": "uniform",
                "Params": {"MinValue": output, "MaxValue": output},
            },
        },
    }


def _load_synthetic(config: dict[str, Any], seed: int) -> list[dict[str, Any]]:
    import numpy as np
    import torch
    from ais_bench.benchmark.datasets import SyntheticDataset

    np.random.seed(seed)
    torch.manual_seed(seed)
    dataset = SyntheticDataset(
        config=config,
        reader_cfg={
            "input_columns": ["question", "max_out_len"],
            "output_column": "answer",
        },
    )
    return dataset.dataset["test"].to_list()


def _generate_formal_rows(config: dict[str, Any]) -> list[dict[str, str]]:
    input_config = config["input"]
    output_length = int(config["output"]["length"])
    count = int(config["request_count"])
    seed = int(config["seed"])
    rows: list[dict[str, str]] = []
    if input_config["method"] == "percentage":
        assigned = 0
        values = input_config["values"]
        for index, (length, ratio) in enumerate(values):
            part_count = (
                count - assigned
                if index == len(values) - 1
                else int(count * float(ratio))
            )
            assigned += part_count
            synthetic = _synthetic_config(
                {"method": "fixed", "length": int(length)},
                part_count,
                output_length,
            )
            rows.extend(
                {"question": item["question"]}
                for item in _load_synthetic(synthetic, seed + index)
            )
    else:
        synthetic = _synthetic_config(input_config, count, output_length)
        rows = [
            {"question": item["question"]}
            for item in _load_synthetic(synthetic, seed)
        ]
    if len(rows) != count:
        raise DataError(f"AISBench generated {len(rows)} rows, expected {count}")
    return rows


def _write_jsonl(path: Path, rows: list[dict[str, str]]) -> None:
    with path.open("w", encoding="utf-8") as stream:
        for row in rows:
            stream.write(json.dumps(row, ensure_ascii=False) + "\n")


def generate_formal(
    profile: str,
    config: dict[str, Any],
    tokenizer_path: str,
    run_dir: Path,
    case_name: str,
) -> dict[str, Any]:
    identity = {"profile": profile, "config": config, "tokenizer": tokenizer_path}
    config_hash = _canonical_hash(identity)
    if config["storage"] == "persistent":
        directory = ROOT / "aisbench_workspace" / "datasets" / "formal"
        matches = sorted(directory.glob(f"{profile}-{config_hash}-*.jsonl"))
        if len(matches) > 1:
            raise DataError(f"multiple persistent datasets match {profile}-{config_hash}")
        if matches:
            path = matches[0]
            expected_hash = path.stem.rsplit("-", 1)[-1]
            if _file_hash(path) != expected_hash:
                raise DataError(f"persistent dataset hash mismatch: {path}")
            meta_path = Path(str(path) + ".meta.json")
            if not meta_path.is_file():
                raise DataError(f"persistent dataset metadata is missing: {meta_path}")
            return {
                "generator": "AISBench SyntheticDataset",
                "generator_version": version("ais-bench-benchmark"),
                "storage": "persistent",
                "path": str(path),
                "meta_path": str(meta_path),
                "config_hash": config_hash,
                "content_hash": expected_hash,
                "reused": True,
            }
        temporary = directory / f".{profile}-{config_hash}.tmp"
        rows = _generate_formal_rows(config)
        _write_jsonl(temporary, rows)
        content_hash = _file_hash(temporary)
        path = directory / f"{profile}-{config_hash}-{content_hash}.jsonl"
        os.replace(temporary, path)
    elif config["storage"] == "per_run":
        rows = _generate_formal_rows(config)
        path = run_dir / f"{case_name}-formal.jsonl"
        _write_jsonl(path, rows)
        content_hash = _file_hash(path)
    else:
        raise DataError(f"invalid formal data storage: {config['storage']}")
    meta_path = _meta(path, int(config["request_count"]), int(config["output"]["length"]))
    return {
        "generator": "AISBench SyntheticDataset",
        "generator_version": version("ais-bench-benchmark"),
        "storage": config["storage"],
        "path": str(path),
        "meta_path": str(meta_path),
        "config_hash": config_hash,
        "content_hash": content_hash,
        "reused": False,
    }


@contextmanager
def _tool_import(tool_path: Path) -> Iterator[Any]:
    if not (tool_path / "generate_dataset.py").is_file():
        raise DataError(f"warmup generator does not exist: {tool_path}")
    previous_cwd = Path.cwd()
    sys.path.insert(0, str(tool_path))
    try:
        os.chdir(tool_path)
        module = importlib.import_module("generate_dataset")
        yield module
    finally:
        os.chdir(previous_cwd)
        sys.path.remove(str(tool_path))
        sys.modules.pop("generate_dataset", None)
        sys.modules.pop("data_picker", None)


def generate_warmup(
    profile: str,
    config: dict[str, Any],
    tokenizer_path: str,
    run_dir: Path,
    case_name: str,
) -> dict[str, Any]:
    tool_path = Path(config["tool_path"]).resolve()
    identity = {"profile": profile, "config": config, "tokenizer": tokenizer_path}
    config_hash = _canonical_hash(identity)
    directory = (
        ROOT / "aisbench_workspace" / "datasets" / "warmup"
        if config["storage"] == "persistent"
        else run_dir
    )
    if config["storage"] == "persistent":
        matches = sorted(directory.glob(f"{profile}-{config_hash}-*.jsonl"))
        if len(matches) > 1:
            raise DataError(f"multiple persistent warmup datasets match {profile}-{config_hash}")
        if matches:
            path = matches[0]
            expected_hash = path.stem.rsplit("-", 1)[-1]
            if _file_hash(path) != expected_hash:
                raise DataError(f"warmup dataset hash mismatch: {path}")
            meta_path = Path(str(path) + ".meta.json")
            if not meta_path.is_file():
                raise DataError(f"warmup metadata is missing: {meta_path}")
            return {
                "generator": "aisbench_auto_tools_prefix.create_dataset",
                "tool_path": str(tool_path),
                "storage": "persistent",
                "path": str(path),
                "meta_path": str(meta_path),
                "config_hash": config_hash,
                "content_hash": expected_hash,
                "reused": True,
            }
    random.seed(int(config["seed"]))
    with _tool_import(tool_path) as generator:
        samples = generator.create_dataset(
            tokenizer_path,
            int(config["input_length"]),
            int(config["request_count"]),
            0,
        )
    if samples is None or len(samples) != int(config["request_count"]):
        raise DataError("aisbench_auto_tools_prefix did not generate the requested warmup rows")
    temporary = directory / f".{profile}-{config_hash}.tmp"
    _write_jsonl(temporary, [{"question": sample} for sample in samples])
    content_hash = _file_hash(temporary)
    path = (
        directory / f"{profile}-{config_hash}-{content_hash}.jsonl"
        if config["storage"] == "persistent"
        else directory / f"{case_name}-warmup.jsonl"
    )
    os.replace(temporary, path)
    meta_path = _meta(path, int(config["request_count"]), int(config["output_length"]))
    return {
        "generator": "aisbench_auto_tools_prefix.create_dataset",
        "tool_path": str(tool_path),
        "storage": config["storage"],
        "path": str(path),
        "meta_path": str(meta_path),
        "config_hash": config_hash,
        "content_hash": content_hash,
        "reused": False,
    }


def assert_prefix_disjoint(
    formal_path: str, warmup_path: str, tokenizer_path: str, prefix_tokens: int
) -> None:
    from transformers import AutoTokenizer

    tokenizer = AutoTokenizer.from_pretrained(tokenizer_path, trust_remote_code=True)

    def prefixes(path: str) -> set[tuple[int, ...]]:
        result: set[tuple[int, ...]] = set()
        with Path(path).open("r", encoding="utf-8") as stream:
            for line in stream:
                question = json.loads(line)["question"]
                token_ids = tokenizer.encode(
                    question, add_special_tokens=False
                )[:prefix_tokens]
                result.add(tuple(token_ids))
        return result

    overlap = prefixes(formal_path) & prefixes(warmup_path)
    if overlap:
        raise DataError(
            f"warmup and formal datasets share {len(overlap)} token prefixes "
            f"at block size {prefix_tokens}"
        )
