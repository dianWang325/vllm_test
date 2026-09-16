#!/usr/bin/env python3
"""Stream the historical prefill datasets against the already-running GLM API.

This is a local-to-head fallback when the dedicated AISBench host cannot reach
the 80.5.9.x network. It does not manage the inference service.
"""

from __future__ import annotations

import argparse
import concurrent.futures
import json
import math
import statistics
import time
import urllib.error
import urllib.request
from pathlib import Path

from transformers import AutoTokenizer


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--warmup-data", type=Path, required=True)
    parser.add_argument("--fixed-data", type=Path, required=True)
    parser.add_argument("--tokenizer", required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--api-url", default="http://127.0.0.1:18080/v1/completions")
    parser.add_argument("--model", default="glm-5.2")
    parser.add_argument("--input-tokens", type=int, default=32768)
    parser.add_argument("--warmup-count", type=int, default=5)
    parser.add_argument("--warmup-output-tokens", type=int, default=1)
    parser.add_argument("--fixed-count", type=int, default=24)
    parser.add_argument("--fixed-output-tokens", type=int, default=2560)
    parser.add_argument("--fixed-concurrency", type=int, default=4)
    parser.add_argument("--timeout-seconds", type=int, default=3600)
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--warmup-only", action="store_true", help="Run only the prefill warmup stage")
    parser.add_argument("--fixed-only", action="store_true", help="Run only the fixed prefill stage")
    return parser.parse_args()


def load_questions(path: Path, count: int) -> list[str]:
    questions = []
    with path.open(encoding="utf-8") as stream:
        for line in stream:
            if line.strip():
                value = json.loads(line)
                questions.append(value["question"])
            if len(questions) == count:
                break
    if len(questions) != count:
        raise ValueError(f"{path}: wanted {count} questions, got {len(questions)}")
    return questions


def prepare_prompts(tokenizer, questions: list[str], target: int) -> tuple[list[str], list[dict]]:
    prompts = []
    metadata = []
    for index, question in enumerate(questions):
        original_ids = tokenizer.encode(question, add_special_tokens=False)
        selected_ids = original_ids[:target]
        prompt = tokenizer.decode(selected_ids, skip_special_tokens=False)
        round_trip_len = len(tokenizer.encode(prompt, add_special_tokens=False))
        if round_trip_len > target:
            selected_ids = selected_ids[: -(round_trip_len - target)]
            prompt = tokenizer.decode(selected_ids, skip_special_tokens=False)
            round_trip_len = len(tokenizer.encode(prompt, add_special_tokens=False))
        if round_trip_len > target:
            raise ValueError(f"prompt {index}: round-trip length exceeds target")
        prompts.append(prompt)
        metadata.append({
            "index": index,
            "original_tokens": len(original_ids),
            "prepared_tokens": round_trip_len,
            "truncated": len(original_ids) > len(selected_ids),
        })
    return prompts, metadata


def one_request(
    *,
    stage: str,
    index: int,
    prompt: str,
    prepared_tokens: int,
    output_tokens: int,
    api_url: str,
    model: str,
    timeout_seconds: int,
) -> dict:
    payload = {
        "model": model,
        "prompt": prompt,
        "max_tokens": output_tokens,
        "temperature": 0,
        "ignore_eos": True,
        "stream": True,
        "stream_options": {"include_usage": True},
    }
    body = json.dumps(payload, ensure_ascii=False).encode("utf-8")
    request = urllib.request.Request(
        api_url,
        data=body,
        headers={"Content-Type": "application/json"},
        method="POST",
    )
    started = time.perf_counter()
    first_token_at = None
    usage = None
    finish_reason = None
    chunks = []
    error = None
    try:
        opener = urllib.request.build_opener(urllib.request.ProxyHandler({}))
        with opener.open(request, timeout=timeout_seconds) as response:
            status = response.status
            for raw_line in response:
                if not raw_line.startswith(b"data:"):
                    continue
                data = raw_line[5:].strip()
                if data == b"[DONE]":
                    break
                if not data:
                    continue
                event = json.loads(data)
                if event.get("usage"):
                    usage = event["usage"]
                for choice in event.get("choices", []):
                    piece = choice.get("text") or ""
                    if piece:
                        if first_token_at is None:
                            first_token_at = time.perf_counter()
                        chunks.append(piece)
                    if choice.get("finish_reason"):
                        finish_reason = choice["finish_reason"]
    except urllib.error.HTTPError as exc:
        status = exc.code
        error = exc.read(2048).decode("utf-8", errors="replace")
    except Exception as exc:
        status = None
        error = repr(exc)
    ended = time.perf_counter()
    completion_tokens = usage.get("completion_tokens") if usage else None
    prompt_tokens = usage.get("prompt_tokens") if usage else None
    if completion_tokens is None:
        completion_tokens = 0
    latency = ended - started
    ttft = first_token_at - started if first_token_at is not None else None
    tpot = (
        (latency - ttft) / (completion_tokens - 1)
        if ttft is not None and completion_tokens > 1
        else None
    )
    return {
        "stage": stage,
        "index": index,
        "ok": status == 200 and error is None and completion_tokens > 0,
        "http_status": status,
        "error": error,
        "prepared_prompt_tokens": prepared_tokens,
        "server_prompt_tokens": prompt_tokens,
        "server_completion_tokens": completion_tokens,
        "latency_s": round(latency, 6),
        "ttft_s": round(ttft, 6) if ttft is not None else None,
        "tpot_s": round(tpot, 6) if tpot is not None else None,
        "finish_reason": finish_reason,
        "response_characters": sum(map(len, chunks)),
    }


def percentile(values: list[float], percent: int) -> float | None:
    if not values:
        return None
    ordered = sorted(values)
    position = (len(ordered) - 1) * percent / 100
    lower = math.floor(position)
    upper = math.ceil(position)
    return round(ordered[lower] + (ordered[upper] - ordered[lower]) * (position - lower), 6)


def summarize(results: list[dict], elapsed: float) -> dict:
    success = [item for item in results if item["ok"]]
    def distribution(key: str) -> dict:
        values = [item[key] for item in success if item[key] is not None]
        return {
            "mean": round(statistics.mean(values), 6) if values else None,
            "p50": percentile(values, 50),
            "p90": percentile(values, 90),
            "p95": percentile(values, 95),
            "max": round(max(values), 6) if values else None,
        }
    input_total = sum(item["server_prompt_tokens"] or 0 for item in success)
    output_total = sum(item["server_completion_tokens"] for item in success)
    return {
        "requests": len(results),
        "success": len(success),
        "failed": len(results) - len(success),
        "elapsed_s": round(elapsed, 6),
        "request_throughput_rps": round(len(success) / elapsed, 6),
        "input_tokens": input_total,
        "output_tokens": output_total,
        "input_throughput_tps": round(input_total / elapsed, 6),
        "output_throughput_tps": round(output_total / elapsed, 6),
        "latency_s": distribution("latency_s"),
        "ttft_s": distribution("ttft_s"),
        "tpot_s": distribution("tpot_s"),
    }


def run_stage(
    *,
    stage: str,
    prompts: list[str],
    prompt_metadata: list[dict],
    output_tokens: int,
    concurrency: int,
    args: argparse.Namespace,
    detail_stream,
) -> dict:
    results = []
    started = time.perf_counter()
    with concurrent.futures.ThreadPoolExecutor(max_workers=concurrency) as pool:
        futures = [
            pool.submit(
                one_request,
                stage=stage,
                index=index,
                prompt=prompt,
                prepared_tokens=prompt_metadata[index]["prepared_tokens"],
                output_tokens=output_tokens,
                api_url=args.api_url,
                model=args.model,
                timeout_seconds=args.timeout_seconds,
            )
            for index, prompt in enumerate(prompts)
        ]
        for future in concurrent.futures.as_completed(futures):
            item = future.result()
            results.append(item)
            print(json.dumps(item, ensure_ascii=False), file=detail_stream, flush=True)
            print(
                f"{stage} {len(results)}/{len(prompts)} "
                f"ok={item['ok']} prompt={item['server_prompt_tokens']} "
                f"output={item['server_completion_tokens']} "
                f"ttft={item['ttft_s']} latency={item['latency_s']}",
                flush=True,
            )
    return summarize(results, time.perf_counter() - started)


def main() -> None:
    args = parse_args()
    if args.warmup_only and args.fixed_only:
        raise ValueError("--warmup-only and --fixed-only are mutually exclusive")
    args.output_dir.mkdir(parents=True, exist_ok=True)
    tokenizer = AutoTokenizer.from_pretrained(args.tokenizer, trust_remote_code=True)
    if args.fixed_only:
        warmup, warmup_meta = [], []
    else:
        warmup, warmup_meta = prepare_prompts(
            tokenizer, load_questions(args.warmup_data, args.warmup_count), args.input_tokens
        )
    if args.warmup_only:
        fixed, fixed_meta = [], []
    else:
        fixed, fixed_meta = prepare_prompts(
            tokenizer, load_questions(args.fixed_data, args.fixed_count), args.input_tokens
        )
    manifest = {
        "api_url": args.api_url,
        "model": args.model,
        "tokenizer": args.tokenizer,
        "input_token_target": args.input_tokens,
        "warmup": {"count": args.warmup_count, "output_tokens": args.warmup_output_tokens, "concurrency": 1, "prompts": warmup_meta},
        "fixed": {"count": args.fixed_count, "output_tokens": args.fixed_output_tokens, "concurrency": args.fixed_concurrency, "prompts": fixed_meta},
        "warmup_data": str(args.warmup_data),
        "fixed_data": str(args.fixed_data),
    }
    (args.output_dir / "manifest.json").write_text(json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8")
    print(
        f"prepared warmup={len(warmup)} fixed={len(fixed)} "
        f"warmup_lengths={[m['prepared_tokens'] for m in warmup_meta]} "
        f"fixed_lengths={[m['prepared_tokens'] for m in fixed_meta]}",
        flush=True,
    )
    if args.dry_run:
        return
    with (args.output_dir / "requests.jsonl").open("w", encoding="utf-8") as details:
        if not args.fixed_only:
            warmup_summary = run_stage(
                stage="warmup",
                prompts=warmup,
                prompt_metadata=warmup_meta,
                output_tokens=args.warmup_output_tokens,
                concurrency=1,
                args=args,
                detail_stream=details,
            )
            (args.output_dir / "warmup_summary.json").write_text(
                json.dumps(warmup_summary, ensure_ascii=False, indent=2), encoding="utf-8"
            )
            print("warmup finished", flush=True)
            if args.warmup_only:
                (args.output_dir / "summary.json").write_text(
                    json.dumps({"warmup": warmup_summary}, ensure_ascii=False, indent=2),
                    encoding="utf-8",
                )
                return
        fixed_summary = run_stage(
            stage="fixed",
            prompts=fixed,
            prompt_metadata=fixed_meta,
            output_tokens=args.fixed_output_tokens,
            concurrency=args.fixed_concurrency,
            args=args,
            detail_stream=details,
        )
    (args.output_dir / "summary.json").write_text(
        json.dumps(
            {"fixed": fixed_summary} if args.fixed_only else {"warmup": warmup_summary, "fixed": fixed_summary},
            ensure_ascii=False,
            indent=2,
        ),
        encoding="utf-8",
    )
    print("fixed finished", flush=True)


if __name__ == "__main__":
    main()
