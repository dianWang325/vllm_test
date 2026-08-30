"""Build, start, health-check, and stop one vLLM server process group."""

from __future__ import annotations

import json
import os
import signal
import socket
import subprocess
import time
import urllib.error
import urllib.request
from pathlib import Path
from typing import Any, TextIO


class ServerError(RuntimeError):
    pass


def _argument_value(value: Any) -> str:
    if isinstance(value, (dict, list)):
        return json.dumps(value, ensure_ascii=False, separators=(",", ":"))
    if isinstance(value, bool) or not isinstance(value, (str, int, float)):
        raise ServerError(
            "argument values must be null, a string, a number, a mapping, or a list"
        )
    return str(value)


def build_command(config: dict[str, Any]) -> list[str]:
    model_tag = config["model_tag"]
    if not isinstance(model_tag, str) or not model_tag:
        raise ServerError("model_tag must be a non-empty string")
    command = ["vllm", "serve", model_tag]
    arguments = config.get("arguments", {})
    if not isinstance(arguments, dict):
        raise ServerError("arguments must be a mapping")
    for name, value in arguments.items():
        if not isinstance(name, str) or not name:
            raise ServerError("argument names must be non-empty strings")
        if value is False:
            continue
        command.append(name)
        if value is not None and value is not True:
            command.append(_argument_value(value))
    return command


def build_environment(config: dict[str, Any]) -> dict[str, str]:
    environment = dict(os.environ)
    configured = config.get("environment", {})
    if not isinstance(configured, dict):
        raise ServerError("environment must be a mapping")
    for name, value in configured.items():
        if not isinstance(name, str) or not name:
            raise ServerError("environment names must be non-empty strings")
        if isinstance(value, bool) or not isinstance(value, (str, int, float)):
            raise ServerError("environment values must be strings or numbers")
        environment[name] = str(value)
    unset = config.get("unset_environment", [])
    if not isinstance(unset, list) or not all(
        isinstance(name, str) and name for name in unset
    ):
        raise ServerError("unset_environment must be a list of non-empty strings")
    for name in unset:
        environment.pop(name, None)
    return environment


def _port_is_free(host: str, port: int) -> bool:
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as sock:
        sock.settimeout(0.5)
        return sock.connect_ex((host, port)) != 0


def _healthy(config: dict[str, Any]) -> bool:
    arguments = config["arguments"]
    lifecycle = config["lifecycle"]
    url = (
        f"http://{arguments['--host']}:{arguments['--port']}"
        f"{lifecycle['health_path']}"
    )
    try:
        with urllib.request.urlopen(url, timeout=3) as response:
            return 200 <= response.status < 300
    except (OSError, urllib.error.HTTPError):
        return False


class VllmServer:
    def __init__(self, config: dict[str, Any], log_path: Path):
        self.config = config
        self.log_path = log_path
        self.process: subprocess.Popen[str] | None = None
        self._log: TextIO | None = None

    def start(self) -> list[str]:
        arguments = self.config["arguments"]
        lifecycle = self.config["lifecycle"]
        host = str(arguments["--host"])
        port = int(arguments["--port"])
        if not _port_is_free(host, port):
            raise ServerError(f"server port is already in use: {port}")
        command = build_command(self.config)
        self._log = self.log_path.open("w", encoding="utf-8")
        self.process = subprocess.Popen(
            command,
            stdout=self._log,
            stderr=subprocess.STDOUT,
            text=True,
            env=build_environment(self.config),
            start_new_session=True,
        )
        deadline = time.monotonic() + float(lifecycle["startup_timeout_seconds"])
        interval = float(lifecycle["health_interval_seconds"])
        while time.monotonic() < deadline:
            if self.process.poll() is not None:
                raise ServerError(
                    f"vLLM exited before becoming healthy: {self.process.returncode}"
                )
            if _healthy(self.config):
                return command
            time.sleep(interval)
        raise ServerError("vLLM health check timed out")

    def stop(self) -> None:
        try:
            if self.process is not None and self.process.poll() is None:
                try:
                    os.killpg(os.getpgid(self.process.pid), signal.SIGTERM)
                except ProcessLookupError:
                    return
                try:
                    self.process.wait(
                        timeout=float(self.config["lifecycle"]["shutdown_timeout_seconds"])
                    )
                except subprocess.TimeoutExpired as exc:
                    raise ServerError(
                        "vLLM did not stop after SIGTERM; forced or broad cleanup was not used"
                    ) from exc
        finally:
            if self._log is not None:
                self._log.close()
                self._log = None
