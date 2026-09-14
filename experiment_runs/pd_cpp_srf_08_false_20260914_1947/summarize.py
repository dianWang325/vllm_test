"""Summarize the eight completed AISBench runs without rerunning any service."""

from __future__ import annotations

import csv
import json
from collections import defaultdict
from pathlib import Path


HERE = Path(__file__).resolve().parent
MANIFEST = json.loads((HERE / "manifest.json").read_text(encoding="utf-8"))


def number(value: str | int | float) -> float:
    return float(str(value).split()[0])


def metric_rows(path: Path) -> dict[str, dict[str, str]]:
    with path.open(newline="", encoding="utf-8-sig") as stream:
        return {row["Performance Parameters"]: row for row in csv.DictReader(stream)}


def read_run(index: int, mode: str, dataset: str) -> dict:
    tag = f"{index:02d}_{mode}_{dataset}"
    folder = HERE / tag
    status = json.loads((folder / "status.json").read_text(encoding="utf-8"))
    if status["status"] != "complete":
        raise RuntimeError(f"{tag} is {status['status']}, not complete")
    totals = json.loads((folder / "formal_metrics.json").read_text(encoding="utf-8"))
    rows = metric_rows(folder / "formal_metrics.csv")
    if totals["Success Requests"]["total"] != 24 or totals["Failed Requests"]["total"] != 0:
        raise RuntimeError(f"{tag} has invalid request counts")
    result = {
        "run": index,
        "round": 1 if index <= 4 else 2,
        "mode": mode,
        "dataset": dataset,
        "p_host": status.get("p_host", MANIFEST["p_host"]),
        "d_host": status.get("d_host", MANIFEST["d_host"]),
        "ttft_ms": number(rows["TTFT"]["Average"]),
        "ttft_p90_ms": number(rows["TTFT"]["P90"]),
        "tpot_ms": number(rows["TPOT"]["Average"]),
        "tpot_p90_ms": number(rows["TPOT"]["P90"]),
        "total_tokens_per_s": number(totals["Total Token Throughput"]["total"]),
        "input_tokens_per_s": number(totals["Input Token Throughput"]["total"]),
        "output_tokens_per_s": number(totals["Output Token Throughput"]["total"]),
        "requests": totals["Success Requests"]["total"],
    }
    return result


def average(values: list[float]) -> float:
    return sum(values) / len(values)


def main() -> None:
    runs = [read_run(i, mode, dataset) for i, (mode, dataset) in enumerate(MANIFEST["schedule"], 1)]
    measures = ("ttft_ms", "tpot_ms", "total_tokens_per_s", "input_tokens_per_s", "output_tokens_per_s")
    grouped = defaultdict(list)
    for run in runs:
        grouped[(run["mode"], run["dataset"])].append(run)
    if any(len(group) != 2 for group in grouped.values()) or len(grouped) != 4:
        raise RuntimeError("Expected two runs for each of four combinations")
    means = {
        f"{mode}_{dataset}": {key: average([row[key] for row in group]) for key in measures}
        for (mode, dataset), group in grouped.items()
    }
    comparisons = {}
    for dataset in ("fixed", "variable"):
        baseline = means[f"baseline_{dataset}"]
        enhanced = means[f"enhanced_{dataset}"]
        comparisons[dataset] = {
            key: {
                "difference": enhanced[key] - baseline[key],
                "percent": (enhanced[key] / baseline[key] - 1) * 100,
            }
            for key in measures
        }
    output = {"runs": runs, "means": means, "comparisons": comparisons}
    (HERE / "summary.json").write_text(json.dumps(output, indent=2, ensure_ascii=False), encoding="utf-8")

    hosts = "、".join(f"P {p} / D {d}" for p, d in sorted({(row["p_host"], row["d_host"]) for row in runs}))
    lines = [
        "# DeepSeek-V4-Flash A3 PD：CPP 参数复测",
        "",
        f"主机：{hosts}。模型：`/mnt/weight/DeepSeek-V4-Flash-w8a8-mtp`。本次使用独立实验目录中的脚本和数据；vLLM 为 `a97dacb7`，vllm-ascend 为 `660c4582`。",
        "",
        f"服务通信使用 `{MANIFEST['p_nic']}`：P `{MANIFEST.get('p_comm_ip', MANIFEST['p_host'])}`，D `{MANIFEST.get('d_comm_ip', MANIFEST['d_host'])}`；代理仍通过管理地址连接 D。此前跨网段的管理网与数据网尝试均因 Mooncake/HCCL 传输失败中止，未完成正式轮次，不计入下表。",
        "",
        "P 为 DP1/TP8/PP2，D 为 DP2/TP8/PP1，均未开启投机解码。P 增强组 CPP 为 `enabled=true, smooth_factor=0.8, need_timing=false`，SRF 为 `enabled=true, threshold=65546, long_max_wait_ms=2000`；基线组两项关闭。P 两组显存利用率均为 0.85，HCCL_BUFFSIZE 均为 2560；D 显存利用率为 0.9，HCCL_BUFFSIZE 为 1024，全部轮次保持一致。",
        "",
        "每轮完整重启 P、D、代理。开启组在正式数据前均执行 5 条 `prefill_warmup.py` 预热；AISBench 自带的首请求预检查不计入下表。正式配置除代理端口改为 18090 外保持原值，每轮 24 条请求、batch size 4、每条输出 2560 tokens。八轮共 192/192 条正式请求成功，显式预热共 20/20 条成功。",
        "",
        "每项两轮；TTFT 与 TPOT 为 AISBench 请求均值，tokens/s 为整轮吞吐。",
        "",
        "| 轮次 | 顺序 | P CPP+SRF | 数据 | TTFT (ms) | TPOT (ms) | 总 tokens/s | 输入 tokens/s | 输出 tokens/s |",
        "| ---: | ---: | :--- | :--- | ---: | ---: | ---: | ---: | ---: |",
    ]
    for row in runs:
        lines.append(
            f"| {row['round']} | {row['run']} | {'开' if row['mode']=='enhanced' else '关'} | {row['dataset']} "
            f"| {row['ttft_ms']:.2f} | {row['tpot_ms']:.2f} | {row['total_tokens_per_s']:.2f} "
            f"| {row['input_tokens_per_s']:.2f} | {row['output_tokens_per_s']:.2f} |"
        )
    lines += [
        "", "两轮算术均值与开启相对关闭的变化：", "",
        "| 数据 | 配置 | TTFT (ms) | TPOT (ms) | 总 tokens/s | 输入 tokens/s | 输出 tokens/s |",
        "| :--- | :--- | ---: | ---: | ---: | ---: | ---: |",
    ]
    for dataset in ("fixed", "variable"):
        for mode in ("baseline", "enhanced"):
            row = means[f"{mode}_{dataset}"]
            lines.append(
                f"| {dataset} | {'开' if mode=='enhanced' else '关'} | {row['ttft_ms']:.2f} "
                f"| {row['tpot_ms']:.2f} | {row['total_tokens_per_s']:.2f} "
                f"| {row['input_tokens_per_s']:.2f} | {row['output_tokens_per_s']:.2f} |"
            )
        comp = comparisons[dataset]
        lines.append(
            f"| {dataset} | 开相对关 | {comp['ttft_ms']['percent']:+.2f}% "
            f"| {comp['tpot_ms']['percent']:+.2f}% "
            f"| {comp['total_tokens_per_s']['percent']:+.2f}% "
            f"| {comp['input_tokens_per_s']['percent']:+.2f}% "
            f"| {comp['output_tokens_per_s']['percent']:+.2f}% |"
        )
    lines += [
        "",
        "新参数增强组相对本轮基线：定长 TTFT {:+.2f}%、TPOT {:+.2f}%、总 tokens/s {:+.2f}%；变长 TTFT {:+.2f}%、TPOT {:+.2f}%、总 tokens/s {:+.2f}%。每项仅两轮，差异是描述性结果。".format(
            comparisons["fixed"]["ttft_ms"]["percent"], comparisons["fixed"]["tpot_ms"]["percent"], comparisons["fixed"]["total_tokens_per_s"]["percent"],
            comparisons["variable"]["ttft_ms"]["percent"], comparisons["variable"]["tpot_ms"]["percent"], comparisons["variable"]["total_tokens_per_s"]["percent"],
        ),
        "",
    ]
    old_path = HERE.parent / MANIFEST["comparison_run_id"] / "summary.json"
    if old_path.is_file():
        old = json.loads(old_path.read_text(encoding="utf-8"))
        lines += [
            "## 与上一轮增强组的历史参考",
            "",
            "上一轮增强组 CPP 为 `smooth_factor=1, need_timing=true`，SRF 相同。以下变化为本轮新增强组均值相对上一轮旧增强组均值；主机、通信网络和测试时间不同，只作历史参考，不能单独归因于 CPP 参数。",
            "",
            "| 数据 | TTFT 变化 | TPOT 变化 | 总 tokens/s 变化 |",
            "| :--- | ---: | ---: | ---: |",
        ]
        for dataset in ("fixed", "variable"):
            before = old["means"][f"enhanced_{dataset}"]
            after = means[f"enhanced_{dataset}"]
            lines.append(
                f"| {dataset} | {(after['ttft_ms']/before['ttft_ms']-1)*100:+.2f}% "
                f"| {(after['tpot_ms']/before['tpot_ms']-1)*100:+.2f}% "
                f"| {(after['total_tokens_per_s']/before['total_tokens_per_s']-1)*100:+.2f}% |"
            )
        lines.append("")
    (HERE / "summary.md").write_text("\n".join(lines), encoding="utf-8")


if __name__ == "__main__":
    main()
