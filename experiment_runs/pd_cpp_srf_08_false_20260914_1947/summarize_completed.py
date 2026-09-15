"""Summarize only the four completed runs after the user stopped round five."""

from __future__ import annotations

import csv
import json
from pathlib import Path


HERE = Path(__file__).resolve().parent
MANIFEST = json.loads((HERE / "manifest.json").read_text(encoding="utf-8-sig"))


def value(raw: str | int | float) -> float:
    return float(str(raw).split()[0])


def read_run(index: int, mode: str, dataset: str) -> dict:
    tag = f"{index:02d}_{mode}_{dataset}"
    folder = HERE / tag
    status = json.loads((folder / "status.json").read_text(encoding="utf-8-sig"))
    formal = json.loads((folder / "formal_metrics.json").read_text(encoding="utf-8-sig"))
    with (folder / "formal_metrics.csv").open(newline="", encoding="utf-8-sig") as stream:
        rows = {row["Performance Parameters"]: row for row in csv.DictReader(stream)}
    if status["status"] != "complete" or status["stop_errors"]:
        raise RuntimeError(f"{tag} did not finish cleanly")
    if (formal["Success Requests"]["total"], formal["Failed Requests"]["total"],
            formal["Total Generated Tokens"]["total"]) != (24, 0, 61440):
        raise RuntimeError(f"{tag} has invalid request or token counts")
    if mode == "enhanced":
        warmup = json.loads((folder / "warmup_metrics.json").read_text(encoding="utf-8-sig"))
        if (warmup["Success Requests"]["total"], warmup["Failed Requests"]["total"]) != (5, 0):
            raise RuntimeError(f"{tag} has invalid warmup counts")
    return {
        "index": index,
        "mode": mode,
        "dataset": dataset,
        "tag": tag,
        "p_host": status["p_host"],
        "d_host": status["d_host"],
        "ttft_ms": value(rows["TTFT"]["Average"]),
        "tpot_ms": value(rows["TPOT"]["Average"]),
        "total_tokens_per_s": value(formal["Total Token Throughput"]["total"]),
        "input_tokens_per_s": value(formal["Input Token Throughput"]["total"]),
        "output_tokens_per_s": value(formal["Output Token Throughput"]["total"]),
        "success_requests": 24,
        "failed_requests": 0,
        "generated_tokens": 61440,
        "explicit_warmup_success": 5 if mode == "enhanced" else 0,
    }


def api_args(path: Path) -> list[str]:
    marker = "non-default args: "
    return [line.split(marker, 1)[1] for line in path.read_text(encoding="utf-8").splitlines() if marker in line]


def main() -> None:
    runs = [read_run(i, *condition) for i, condition in enumerate(MANIFEST["schedule"][:4], 1)]
    by_condition = {(run["mode"], run["dataset"]): run for run in runs}
    if len(by_condition) != 4:
        raise RuntimeError("The first four runs do not cover all conditions")
    p_canonical = set()
    d_canonical = set()
    for run in runs:
        folder = HERE / run["tag"]
        p_args = api_args(folder / "prefill.log")
        d_args = api_args(folder / "decode.log")
        if len(p_args) != 1 or len(d_args) != 2:
            raise RuntimeError(f"{run['tag']} has incomplete service arguments")
        p = p_args[0]
        marker = ", 'additional_config': "
        if run["mode"] == "enhanced":
            if "'smooth_factor': 0.8, 'need_timing': False" not in p or "'short_request_first_config': {'enabled': True, 'threshold': 65546, 'long_max_wait_ms': 2000}" not in p:
                raise RuntimeError(f"{run['tag']} lacks requested CPP+SRF settings")
            p = p.split(marker, 1)[0] + "}"
        elif marker in p:
            raise RuntimeError(f"{run['tag']} unexpectedly enables CPP+SRF")
        p_canonical.add(p)
        d_canonical.add(tuple(sorted(d_args)))
    if len(p_canonical) != 1 or len(d_canonical) != 1:
        raise RuntimeError("Service arguments differ beyond the requested P configuration")
    keys = ("ttft_ms", "tpot_ms", "total_tokens_per_s", "input_tokens_per_s", "output_tokens_per_s")
    comparisons = {
        dataset: {
            key: {
                "absolute": by_condition[("enhanced", dataset)][key] - by_condition[("baseline", dataset)][key],
                "percent": (by_condition[("enhanced", dataset)][key] / by_condition[("baseline", dataset)][key] - 1) * 100,
            }
            for key in keys
        }
        for dataset in ("fixed", "variable")
    }
    result = {
        "run_id": MANIFEST["run_id"],
        "scope": "Four completed runs, one per condition; round five and all failed attempts excluded",
        "runs": runs,
        "comparisons": comparisons,
    }
    (HERE / "completed_summary.json").write_text(json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8")

    lines = [
        "# DeepSeek-V4-Flash A3 PD 实验：已完成轮次汇总",
        "",
        "按用户要求已停止实验。共 4 轮有效完成，每种条件 1 轮；第 5 轮在正式测试中途停止，所有未完成尝试均未计入。两台机器上的本次服务已退出，NPU 卡恢复空闲。",
        "",
        "模型：`/mnt/weight/DeepSeek-V4-Flash-w8a8-mtp`；P `80.5.9.129`，D `80.5.9.127`。P 为 DP1/TP8/PP2，D 为 DP2/TP8/PP1，均未开启投机解码。开启组的 P 配置为 CPP `enabled=true, smooth_factor=0.8, need_timing=false` 与 SRF `enabled=true, threshold=65546, long_max_wait_ms=2000`；关闭组两项均关闭。其余服务参数一致。",
        "",
        "每轮均重新启动 P、D 和代理。开启组正式测试前各完成 5/5 条显式预热。每轮正式测试 24/24 条成功、失败 0、生成 61,440 tokens。TTFT/TPOT 为 ais_bench 请求均值，吞吐为整轮 tokens/s。",
        "",
        "| 顺序 | CPP+SRF | 数据 | TTFT (ms) | TPOT (ms) | 总 tokens/s | 输入 tokens/s | 输出 tokens/s |",
        "| ---: | :--- | :--- | ---: | ---: | ---: | ---: | ---: |",
    ]
    for row in runs:
        lines.append(
            f"| {row['index']} | {'开' if row['mode'] == 'enhanced' else '关'} | {row['dataset']} "
            f"| {row['ttft_ms']:.2f} | {row['tpot_ms']:.2f} | {row['total_tokens_per_s']:.2f} "
            f"| {row['input_tokens_per_s']:.2f} | {row['output_tokens_per_s']:.2f} |"
        )
    lines += [
        "",
        "开启组相对关闭组的变化（正值表示数值更高；时延越低越好，吞吐越高越好）：",
        "",
        "| 数据 | TTFT | TPOT | 总 tokens/s | 输入 tokens/s | 输出 tokens/s |",
        "| :--- | ---: | ---: | ---: | ---: | ---: |",
    ]
    for dataset in ("fixed", "variable"):
        item = comparisons[dataset]
        lines.append(
            f"| {dataset} | {item['ttft_ms']['percent']:+.2f}% | {item['tpot_ms']['percent']:+.2f}% "
            f"| {item['total_tokens_per_s']['percent']:+.2f}% | {item['input_tokens_per_s']['percent']:+.2f}% "
            f"| {item['output_tokens_per_s']['percent']:+.2f}% |"
        )
    lines += [
        "",
        "这是每种条件单次测量的描述性对比，尚无计划中的第二轮重复，不能作为最终两轮均值或稳定性结论。",
        "",
    ]
    (HERE / "completed_summary.md").write_text("\n".join(lines), encoding="utf-8")
    print("Verified four complete runs, 96/96 formal requests, 10/10 explicit warmups, and service argument parity.")


if __name__ == "__main__":
    main()
