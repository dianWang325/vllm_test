#!/usr/bin/env python3
"""Verify an OpenAI-compatible vLLM server from inside its container."""

from __future__ import annotations

import argparse
import json
import sys
import time
import urllib.error
import urllib.request
from typing import Any


def http_json(
    method: str,
    url: str,
    payload: dict[str, Any] | None,
    timeout: int,
) -> tuple[int, str, float]:
    body = None
    headers = {"Accept": "application/json"}
    if payload is not None:
        body = json.dumps(payload, ensure_ascii=False).encode("utf-8")
        headers["Content-Type"] = "application/json"

    request = urllib.request.Request(
        url,
        data=body,
        headers=headers,
        method=method,
    )
    started = time.monotonic()
    try:
        with urllib.request.urlopen(request, timeout=timeout) as response:
            response_body = response.read().decode("utf-8", errors="replace")
            return response.status, response_body, time.monotonic() - started
    except urllib.error.HTTPError as error:
        response_body = error.read().decode("utf-8", errors="replace")
        return error.code, response_body, time.monotonic() - started


def display(label: str, status: int, body: str, elapsed: float) -> None:
    print(f"[{label}] HTTP {status}, elapsed={elapsed:.3f}s")
    if not body:
        return
    try:
        parsed = json.loads(body)
    except json.JSONDecodeError:
        print(body)
    else:
        print(json.dumps(parsed, ensure_ascii=False, indent=2))


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--base-url", default="http://127.0.0.1:10800")
    parser.add_argument("--model")
    parser.add_argument(
        "--prompt",
        default="Reply with exactly: inference service is working",
    )
    parser.add_argument("--max-tokens", type=int, default=32)
    parser.add_argument("--timeout", type=int, default=180)
    args = parser.parse_args()
    base_url = args.base_url.rstrip("/")

    try:
        status, body, elapsed = http_json(
            "GET", f"{base_url}/v1/models", None, timeout=15
        )
    except (urllib.error.URLError, TimeoutError) as error:
        print(f"[models] transport failure: {error}", file=sys.stderr)
        return 2
    display("models", status, body, elapsed)
    if status != 200:
        return 3

    try:
        discovered_model = json.loads(body)["data"][0]["id"]
    except (json.JSONDecodeError, KeyError, IndexError, TypeError):
        print("Unable to discover a model ID from /v1/models.", file=sys.stderr)
        return 4

    payload = {
        "model": args.model or discovered_model,
        "messages": [{"role": "user", "content": args.prompt}],
        "temperature": 0,
        "max_tokens": args.max_tokens,
        "stream": False,
    }
    try:
        status, body, elapsed = http_json(
            "POST",
            f"{base_url}/v1/chat/completions",
            payload,
            timeout=args.timeout,
        )
    except (urllib.error.URLError, TimeoutError) as error:
        print(f"[chat.completions] transport failure: {error}", file=sys.stderr)
        return 5
    display("chat.completions", status, body, elapsed)

    if status != 200:
        try:
            health_status, health_body, health_elapsed = http_json(
                "GET", f"{base_url}/health", None, timeout=10
            )
            display("health-after-failure", health_status, health_body, health_elapsed)
        except (urllib.error.URLError, TimeoutError) as error:
            print(f"[health-after-failure] transport failure: {error}")
        return 6

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
