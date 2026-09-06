"""Build, start, health-check, and stop one vLLM server process group."""

from __future__ import annotations

import json
import math
import os
import re
import signal
import socket
import subprocess
import sys
import time
import urllib.error
import urllib.request
from datetime import datetime
from pathlib import Path
from typing import Any, TextIO

from .config import deep_merge, relative_to_root


class ServerError(RuntimeError):
    pass


_REQUEST_METRICS = {
    "vllm:num_requests_running": "running",
    "vllm:num_requests_waiting": "waiting",
}
_PROMETHEUS_SAMPLE = re.compile(
    r"^([^\s{]+)(?:\{.*\})?\s+([^\s]+)(?:\s+[^\s]+)?$"
)


def _parse_request_metrics(payload: str) -> dict[str, float]:
    totals = {label: 0.0 for label in _REQUEST_METRICS.values()}
    found: set[str] = set()
    for raw_line in payload.splitlines():
        line = raw_line.strip()
        if not line or line.startswith("#"):
            continue
        metric_name = line.split("{", 1)[0].split(None, 1)[0]
        if metric_name not in _REQUEST_METRICS:
            continue
        match = _PROMETHEUS_SAMPLE.match(line)
        if match is None or match.group(1) != metric_name:
            raise ServerError(f"invalid Prometheus sample for {metric_name}: {line!r}")
        try:
            value = float(match.group(2))
        except ValueError as exc:
            raise ServerError(
                f"invalid Prometheus value for {metric_name}: {match.group(2)!r}"
            ) from exc
        if not math.isfinite(value) or value < 0:
            raise ServerError(
                f"invalid Prometheus value for {metric_name}: {match.group(2)!r}"
            )
        totals[_REQUEST_METRICS[metric_name]] += value
        found.add(metric_name)
    missing = sorted(set(_REQUEST_METRICS) - found)
    if missing:
        raise ServerError(f"Prefill metrics are missing: {missing}")
    return totals


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


_PROXY_MANAGED_ARGUMENTS = {
    "--host",
    "--port",
    "--prefiller-hosts",
    "--prefiller-ports",
    "--decoder-hosts",
    "--decoder-ports",
}


def build_proxy_command(
    config: dict[str, Any],
    prefill_config: dict[str, Any],
    decode_config: dict[str, Any],
) -> list[str]:
    proxy = config["pd"]["proxy"]
    script = proxy.get("script")
    if not isinstance(script, str) or not script:
        raise ServerError("pd.proxy.script must be a non-empty string")
    script_path = relative_to_root(script)
    if not script_path.is_file():
        raise ServerError(f"proxy script does not exist: {script_path}")

    arguments = proxy.get("arguments", {})
    if not isinstance(arguments, dict):
        raise ServerError("pd.proxy.arguments must be a mapping")
    managed = sorted(_PROXY_MANAGED_ARGUMENTS.intersection(arguments))
    if managed:
        raise ServerError(
            "pd.proxy.arguments must not configure framework-managed arguments: "
            f"{managed}"
        )

    public_arguments = config["arguments"]
    prefill_arguments = prefill_config["arguments"]
    decode_arguments = decode_config["arguments"]
    command = [
        sys.executable,
        str(script_path),
        "--host",
        str(public_arguments["--host"]),
        "--port",
        str(public_arguments["--port"]),
        "--prefiller-hosts",
        _endpoint_host(prefill_config),
        "--prefiller-ports",
        str(prefill_arguments["--port"]),
        "--decoder-hosts",
        _endpoint_host(decode_config),
        "--decoder-ports",
        str(decode_arguments["--port"]),
    ]
    for name, value in arguments.items():
        if not isinstance(name, str) or not name:
            raise ServerError("proxy argument names must be non-empty strings")
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


def _endpoint_host(config: dict[str, Any]) -> str:
    host = config.get("endpoint_host", config["arguments"]["--host"])
    if not isinstance(host, str) or not host:
        raise ServerError("endpoint_host must be a non-empty string")
    return host


def _external(config: dict[str, Any]) -> bool:
    value = config.get("external", False)
    if not isinstance(value, bool):
        raise ServerError("external must be a boolean")
    return value


def _healthy(config: dict[str, Any]) -> bool:
    arguments = config["arguments"]
    lifecycle = config["lifecycle"]
    url = (
        f"http://{_endpoint_host(config)}:{arguments['--port']}"
        f"{lifecycle['health_path']}"
    )
    try:
        with urllib.request.urlopen(url, timeout=3) as response:
            return 200 <= response.status < 300
    except (OSError, urllib.error.HTTPError):
        return False


def _proxy_healthy(config: dict[str, Any]) -> bool:
    arguments = config["arguments"]
    lifecycle = config["lifecycle"]
    url = (
        f"http://{arguments['--host']}:{arguments['--port']}"
        f"{lifecycle['health_path']}"
    )
    try:
        with urllib.request.urlopen(url, timeout=3) as response:
            if not 200 <= response.status < 300:
                return False
            body = json.load(response)
    except (OSError, urllib.error.HTTPError, json.JSONDecodeError):
        return False
    return (
        isinstance(body, dict)
        and body.get("status") == "ok"
        and isinstance(body.get("prefill_instances"), int)
        and body["prefill_instances"] >= 1
        and isinstance(body.get("decode_instances"), int)
        and body["decode_instances"] >= 1
    )


class VllmServer:
    def __init__(self, config: dict[str, Any], log_path: Path):
        self.config = config
        self.log_path = log_path
        self.process: subprocess.Popen[str] | None = None
        self.processes: dict[str, subprocess.Popen[str]] = {}
        self._log: TextIO | None = None

    def _pd_enabled(self) -> bool:
        pd = self.config.get("pd")
        return isinstance(pd, dict) and (
            "prefill" in pd or "decode" in pd
        )

    def _build_pd_configs(
        self,
    ) -> tuple[dict[str, Any], dict[str, Any], dict[str, Any]]:
        pd = self.config.get("pd")
        if not isinstance(pd, dict):
            raise ServerError("pd must be a mapping")
        proxy = pd.get("proxy")
        if not isinstance(proxy, dict):
            raise ServerError("pd.proxy must be a mapping")
        for role in ("prefill", "decode"):
            if not isinstance(pd.get(role), dict):
                raise ServerError(f"pd.{role} must be a mapping")

        base = {key: value for key, value in self.config.items() if key != "pd"}
        prefill_config = deep_merge(base, pd["prefill"])
        decode_config = deep_merge(base, pd["decode"])
        proxy_lifecycle = deep_merge(
            self.config["lifecycle"], {"health_path": "/healthcheck"}
        )
        configured_lifecycle = proxy.get("lifecycle", {})
        if not isinstance(configured_lifecycle, dict):
            raise ServerError("pd.proxy.lifecycle must be a mapping")
        proxy_lifecycle = deep_merge(proxy_lifecycle, configured_lifecycle)
        proxy_config = {
            "arguments": {
                "--host": self.config["arguments"]["--host"],
                "--port": self.config["arguments"]["--port"],
            },
            "lifecycle": proxy_lifecycle,
            "environment": proxy.get("environment", {}),
            "unset_environment": proxy.get("unset_environment", []),
        }
        return prefill_config, decode_config, proxy_config

    def _check_pd_ports(
        self,
        prefill_config: dict[str, Any],
        decode_config: dict[str, Any],
        proxy_config: dict[str, Any],
    ) -> None:
        endpoints = {
            "prefill": (
                _endpoint_host(prefill_config),
                int(prefill_config["arguments"]["--port"]),
            ),
            "decode": (
                _endpoint_host(decode_config),
                int(decode_config["arguments"]["--port"]),
            ),
            "proxy": (
                str(proxy_config["arguments"]["--host"]),
                int(proxy_config["arguments"]["--port"]),
            ),
        }
        if len(set(endpoints.values())) != len(endpoints):
            raise ServerError("prefill, decode, and proxy endpoints must be distinct")
        configs = {
            "prefill": prefill_config,
            "decode": decode_config,
            "proxy": proxy_config,
        }
        for role, (host, port) in endpoints.items():
            if _external(configs[role]):
                continue
            if not _port_is_free(host, port):
                raise ServerError(f"{role} port is already in use: {port}")

    def _spawn(
        self,
        role: str,
        command: list[str],
        config: dict[str, Any],
    ) -> None:
        if self._log is None:
            raise ServerError("server log is not open")
        self._log.write(
            f"===== {role.upper()} =====\n"
            f"$ {subprocess.list2cmdline(command)}\n"
        )
        self._log.flush()
        self.processes[role] = subprocess.Popen(
            command,
            stdout=self._log,
            stderr=subprocess.STDOUT,
            text=True,
            env=build_environment(config),
            start_new_session=True,
        )

    def _wait_pd_backends_healthy(
        self,
        prefill_config: dict[str, Any],
        decode_config: dict[str, Any],
    ) -> None:
        configs = {
            "prefill": prefill_config,
            "decode": decode_config,
        }
        started_at = time.monotonic()
        deadlines = {
            role: started_at
            + float(config["lifecycle"]["startup_timeout_seconds"])
            for role, config in configs.items()
        }
        interval = min(
            float(config["lifecycle"]["health_interval_seconds"])
            for config in configs.values()
        )
        healthy = {"prefill": False, "decode": False}
        while not all(healthy.values()):
            now = time.monotonic()
            for role, config in configs.items():
                process = self.processes.get(role)
                if process is not None and process.poll() is not None:
                    raise ServerError(
                        f"{role} exited before becoming healthy: "
                        f"{process.returncode}"
                    )
                if not healthy[role]:
                    if now >= deadlines[role]:
                        raise ServerError(f"{role} health check timed out")
                    healthy[role] = _healthy(config)
            if not all(healthy.values()):
                time.sleep(interval)

    def _wait_proxy_healthy(self, proxy_config: dict[str, Any]) -> None:
        lifecycle = proxy_config["lifecycle"]
        deadline = time.monotonic() + float(
            lifecycle["startup_timeout_seconds"]
        )
        interval = float(lifecycle["health_interval_seconds"])
        while time.monotonic() < deadline:
            for role in ("prefill", "decode", "proxy"):
                process = self.processes.get(role)
                if process is not None and process.poll() is not None:
                    raise ServerError(
                        f"{role} exited before proxy became healthy: "
                        f"{process.returncode}"
                    )
            if _proxy_healthy(proxy_config):
                return
            time.sleep(interval)
        raise ServerError("proxy health check timed out")

    def wait_for_prefill_idle(self) -> dict[str, Any] | None:
        if not self._pd_enabled():
            return None
        prefill_config, _, _ = self._build_pd_configs()
        settings = prefill_config["lifecycle"].get("request_quiescence")
        if not isinstance(settings, dict):
            raise ServerError("lifecycle.request_quiescence must be a mapping")
        metrics_path = settings.get("metrics_path")
        if not isinstance(metrics_path, str) or not metrics_path.startswith("/"):
            raise ServerError("request quiescence metrics_path must start with /")

        def positive_number(name: str) -> float:
            value = settings.get(name)
            if (
                isinstance(value, bool)
                or not isinstance(value, (int, float))
                or value <= 0
            ):
                raise ServerError(f"request quiescence {name} must be positive")
            return float(value)

        request_timeout = positive_number("request_timeout_seconds")
        timeout = positive_number("timeout_seconds")
        interval = positive_number("poll_interval_seconds")
        required = settings.get("consecutive_idle_samples")
        if isinstance(required, bool) or not isinstance(required, int) or required <= 0:
            raise ServerError(
                "request quiescence consecutive_idle_samples must be a positive integer"
            )

        url = (
            f"http://{_endpoint_host(prefill_config)}:"
            f"{prefill_config['arguments']['--port']}{metrics_path}"
        )
        started_at = datetime.now().astimezone()
        started = time.monotonic()
        deadline = started + timeout
        consecutive = 0
        polls = 0
        last_metrics: dict[str, float] | None = None
        last_network_error: str | None = None
        while time.monotonic() < deadline:
            process = self.processes.get("prefill")
            if process is not None and process.poll() is not None:
                raise ServerError(
                    f"prefill exited while waiting for request quiescence: "
                    f"{process.returncode}"
                )
            polls += 1
            try:
                with urllib.request.urlopen(url, timeout=request_timeout) as response:
                    payload = response.read().decode("utf-8")
                last_metrics = _parse_request_metrics(payload)
                last_network_error = None
            except (OSError, UnicodeDecodeError, urllib.error.HTTPError) as exc:
                last_network_error = f"{type(exc).__name__}: {exc}"
                consecutive = 0
            else:
                if last_metrics["running"] == 0 and last_metrics["waiting"] == 0:
                    consecutive += 1
                    if consecutive >= required:
                        finished_at = datetime.now().astimezone()
                        return {
                            "endpoint": url,
                            "started_at": started_at.isoformat(timespec="seconds"),
                            "finished_at": finished_at.isoformat(timespec="seconds"),
                            "waited_seconds": round(time.monotonic() - started, 3),
                            "polls": polls,
                            "consecutive_idle_samples": consecutive,
                            "metrics": last_metrics,
                        }
                else:
                    consecutive = 0
            if time.monotonic() < deadline:
                time.sleep(interval)
        detail = (
            f"last metrics: {last_metrics}"
            if last_network_error is None
            else f"last metrics request error: {last_network_error}"
        )
        raise ServerError(
            f"prefill requests did not become quiescent within {timeout:g} seconds; "
            f"{detail}"
        )

    def _start_pd(self) -> dict[str, list[str]]:
        prefill_config, decode_config, proxy_config = self._build_pd_configs()
        self._check_pd_ports(prefill_config, decode_config, proxy_config)
        commands = {
            "prefill": build_command(prefill_config),
            "decode": build_command(decode_config),
            "proxy": build_proxy_command(
                self.config, prefill_config, decode_config
            ),
        }
        build_environment(prefill_config)
        build_environment(decode_config)
        build_environment(proxy_config)
        self._log = self.log_path.open("w", encoding="utf-8")
        try:
            for role, config in (
                ("prefill", prefill_config),
                ("decode", decode_config),
            ):
                if _external(config):
                    self._log.write(
                        f"===== {role.upper()} (EXTERNAL) =====\n"
                        f"endpoint: {_endpoint_host(config)}:"
                        f"{config['arguments']['--port']}\n"
                    )
                    self._log.flush()
                else:
                    self._spawn(role, commands[role], config)
            self._wait_pd_backends_healthy(prefill_config, decode_config)
            self._spawn("proxy", commands["proxy"], proxy_config)
            self._wait_proxy_healthy(proxy_config)
            return commands
        except BaseException:
            try:
                self.stop()
            except BaseException:
                pass
            raise

    def start(self) -> list[str] | dict[str, list[str]]:
        if self._pd_enabled():
            return self._start_pd()

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
        if self._pd_enabled():
            errors: list[str] = []
            try:
                timeout = float(
                    self.config["lifecycle"]["shutdown_timeout_seconds"]
                )
                for role in ("proxy", "decode", "prefill"):
                    process = self.processes.get(role)
                    if process is None or process.poll() is not None:
                        continue
                    try:
                        os.killpg(os.getpgid(process.pid), signal.SIGTERM)
                    except ProcessLookupError:
                        continue
                    except OSError as exc:
                        errors.append(f"{role}: {exc}")
                        continue
                    try:
                        process.wait(timeout=timeout)
                    except subprocess.TimeoutExpired:
                        errors.append(
                            f"{role} did not stop after SIGTERM; forced or broad "
                            "cleanup was not used"
                        )
            finally:
                if self._log is not None:
                    self._log.close()
                    self._log = None
            if errors:
                raise ServerError("; ".join(errors))
            return

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
