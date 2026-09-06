from __future__ import annotations

from pathlib import Path

import pytest

from scripts import cli
from scripts.config import (
    ConfigurationError,
    _derive_warmup_input_length,
    resolve_suite,
)
from scripts.pd_role import _role_config
from scripts.server import ServerError, VllmServer, _parse_request_metrics


BASELINE = [
    "deepseek_v4_pro_pd_baseline_fixed",
    "deepseek_v4_pro_pd_baseline_variable",
]
CPP = [
    "deepseek_v4_pro_pd_cpp_fixed",
    "deepseek_v4_pro_pd_cpp_variable",
]
SRF = [
    "deepseek_v4_pro_pd_srf_fixed",
    "deepseek_v4_pro_pd_srf_variable",
]
CPP_SRF = [
    "deepseek_v4_pro_pd_cpp_srf_fixed",
    "deepseek_v4_pro_pd_cpp_srf_variable",
]
ROTATIONS = {
    "deepseek_v4_pro_pd_performance": BASELINE + CPP + SRF + CPP_SRF,
    "deepseek_v4_pro_pd_performance_rotation_1": CPP + SRF + CPP_SRF + BASELINE,
    "deepseek_v4_pro_pd_performance_rotation_2": SRF + CPP_SRF + BASELINE + CPP,
    "deepseek_v4_pro_pd_performance_rotation_3": CPP_SRF + BASELINE + CPP + SRF,
}


def test_pro_suites_use_two_host_server_and_model_warmup() -> None:
    expected_cases = set(ROTATIONS["deepseek_v4_pro_pd_performance"])
    expected_decode = None
    for suite_name, expected_order in ROTATIONS.items():
        suite = resolve_suite(suite_name)
        cases = suite["cases"]
        assert [case["name"] for case in cases] == expected_order
        assert {case["name"] for case in cases} == expected_cases
        assert len(cli._server_segments(cases)) == 4
        for case in cases:
            effective = case["effective"]
            server = effective["server"]
            assert server["model_tag"] == "/mnt/share/DeepSeekV4-pro-0813-w4a8"
            assert server["arguments"]["--max-model-len"] == 1048576
            assert server["pd"]["prefill"]["endpoint_host"] == "80.5.9.127"
            assert server["pd"]["decode"]["endpoint_host"] == "80.5.9.128"
            assert server["pd"]["decode"]["external"] is True
            prefill = _role_config(server, "prefill")
            decode = _role_config(server, "decode")
            expected_decode = expected_decode or decode
            assert decode == expected_decode
            assert prefill["model_tag"] == "/mnt/share/DeepSeekV4-pro-0813-w4a8"
            assert decode["model_tag"] == "/mnt/weight/DeepSeekV4-pro-0813-w4a8"
            assert prefill["arguments"]["--data-parallel-size"] == 1
            assert prefill["arguments"]["--tensor-parallel-size"] == 8
            assert prefill["arguments"]["--pipeline-parallel-size"] == 2
            assert decode["arguments"]["--data-parallel-size"] == 1
            assert decode["arguments"]["--tensor-parallel-size"] == 8
            assert decode["arguments"]["--pipeline-parallel-size"] == 1
            prefill_topology = prefill["arguments"]["--kv-transfer-config"][
                "kv_connector_extra_config"
            ]
            decode_topology = decode["arguments"]["--kv-transfer-config"][
                "kv_connector_extra_config"
            ]
            assert prefill_topology == decode_topology
            assert prefill_topology == {
                "prefill": {"dp_size": 1, "tp_size": 8, "pp_size": 2},
                "decode": {"dp_size": 1, "tp_size": 8, "pp_size": 1},
            }
            assert prefill["environment"]["ASCEND_RT_VISIBLE_DEVICES"] == (
                "0,1,2,3,4,5,6,7,8,9,10,11,12,13,14,15"
            )
            assert decode["environment"]["ASCEND_RT_VISIBLE_DEVICES"] == (
                "0,1,2,3,4,5,6,7"
            )
            for role in ("prefill", "decode"):
                environment = server["pd"][role]["environment"]
                assert environment["GLOO_SOCKET_IFNAME"] == "enp194s0f0"
                assert environment["TP_SOCKET_IFNAME"] == "enp194s0f0"
                assert environment["HCCL_SOCKET_IFNAME"] == "enp194s0f0"
            assert effective["bench"]["concurrency"] == 8
            assert effective["warmup_bench"]["concurrency"] == 1
            assert effective["warmup"]["input_length"] == 1048575
            assert effective["warmup"]["output_length"] == 1


def test_flash_suite_keeps_existing_warmup_scope() -> None:
    suite = resolve_suite("deepseek_v4_flash_pd_performance")
    warmups = [case["definition"].get("warmup") for case in suite["cases"]]
    assert warmups == [
        None,
        None,
        "deepseek_v4_flash_w8a8_mtp_pd_max_len",
        "deepseek_v4_flash_w8a8_mtp_pd_max_len",
        None,
        None,
        "deepseek_v4_flash_w8a8_mtp_pd_max_len",
        "deepseek_v4_flash_w8a8_mtp_pd_max_len",
    ]
    for case in suite["cases"]:
        pd = case["effective"]["server"]["pd"]
        assert pd["prefill"]["endpoint_host"] == "80.5.9.127"
        assert pd["decode"]["endpoint_host"] == "80.5.9.128"
        assert case["effective"]["bench"]["concurrency"] == 4


def test_warmup_input_length_is_derived_from_model() -> None:
    effective = {
        "server": {"arguments": {"--max-model-len": 100}},
        "warmup": {
            "input_length": 10,
            "input_length_from_model_max": True,
            "output_length": 4,
        },
    }
    selected = {
        "warmup": {"profile": {"input_length_from_model_max": True}}
    }
    _derive_warmup_input_length("derived", effective, selected, {})
    assert effective["warmup"]["input_length"] == 96
    assert "input_length_from_model_max" not in effective["warmup"]


@pytest.mark.parametrize(
    ("arguments", "profile", "overrides"),
    [
        ({}, {"input_length_from_model_max": True}, {}),
        (
            {"--max-model-len": 100},
            {"input_length_from_model_max": True, "input_length": 99},
            {},
        ),
        (
            {"--max-model-len": 100},
            {"input_length_from_model_max": True},
            {"warmup": {"input_length": 99}},
        ),
    ],
)
def test_invalid_derived_warmup_configuration_is_rejected(
    arguments: dict, profile: dict, overrides: dict
) -> None:
    effective = {
        "server": {"arguments": arguments},
        "warmup": {
            "input_length": 10,
            "input_length_from_model_max": True,
            "output_length": 1,
        },
    }
    selected = {"warmup": {"profile": profile}}
    with pytest.raises(ConfigurationError):
        _derive_warmup_input_length("invalid", effective, selected, overrides)


def test_request_metrics_are_summed_across_labels() -> None:
    payload = """
# HELP vllm:num_requests_running running
vllm:num_requests_running{engine="0"} 1.0
vllm:num_requests_running{engine="1"} 2.0
vllm:num_requests_waiting{engine="0"} 3.0
vllm:num_requests_waiting{engine="1"} 4.0
"""
    assert _parse_request_metrics(payload) == {"running": 3.0, "waiting": 7.0}


def test_request_metrics_require_both_gauges() -> None:
    with pytest.raises(ServerError, match="missing"):
        _parse_request_metrics("vllm:num_requests_running 0\n")


class _Response:
    def __init__(self, payload: str):
        self.payload = payload

    def __enter__(self) -> "_Response":
        return self

    def __exit__(self, *_args: object) -> None:
        return None

    def read(self) -> bytes:
        return self.payload.encode("utf-8")


def _metrics(running: int, waiting: int) -> str:
    return (
        f'vllm:num_requests_running{{model_name="test"}} {running}.0\n'
        f'vllm:num_requests_waiting{{model_name="test"}} {waiting}.0\n'
    )


def test_prefill_idle_requires_three_consecutive_samples(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    config = resolve_suite("deepseek_v4_pro_pd_performance")["cases"][0][
        "effective"
    ]["server"]
    server = VllmServer(config, Path("unused.log"))
    payloads = iter(
        [
            _metrics(0, 0),
            _metrics(0, 0),
            _metrics(1, 0),
            _metrics(0, 0),
            _metrics(0, 0),
            _metrics(0, 0),
        ]
    )
    seen: list[tuple[str, float]] = []

    def urlopen(url: str, timeout: float) -> _Response:
        seen.append((url, timeout))
        return _Response(next(payloads))

    monkeypatch.setattr("scripts.server.urllib.request.urlopen", urlopen)
    monkeypatch.setattr("scripts.server.time.sleep", lambda _seconds: None)
    result = server.wait_for_prefill_idle()
    assert result is not None
    assert result["polls"] == 6
    assert result["metrics"] == {"running": 0.0, "waiting": 0.0}
    assert seen[-1] == ("http://80.5.9.127:18081/metrics", 3.0)


def test_prefill_idle_timeout_is_fatal(monkeypatch: pytest.MonkeyPatch) -> None:
    config = resolve_suite("deepseek_v4_pro_pd_performance")["cases"][0][
        "effective"
    ]["server"]
    settings = config["lifecycle"]["request_quiescence"]
    settings["timeout_seconds"] = 0.02
    settings["poll_interval_seconds"] = 0.01
    server = VllmServer(config, Path("unused.log"))
    clock = [0.0]

    monkeypatch.setattr(
        "scripts.server.urllib.request.urlopen",
        lambda *_args, **_kwargs: _Response(_metrics(1, 0)),
    )
    monkeypatch.setattr("scripts.server.time.monotonic", lambda: clock[0])
    monkeypatch.setattr(
        "scripts.server.time.sleep",
        lambda seconds: clock.__setitem__(0, clock[0] + seconds),
    )
    with pytest.raises(ServerError, match="did not become quiescent within"):
        server.wait_for_prefill_idle()


def test_prefill_exit_is_fatal() -> None:
    config = resolve_suite("deepseek_v4_pro_pd_performance")["cases"][0][
        "effective"
    ]["server"]
    server = VllmServer(config, Path("unused.log"))

    class Process:
        returncode = 17

        @staticmethod
        def poll() -> int:
            return 17

    server.processes["prefill"] = Process()  # type: ignore[assignment]
    with pytest.raises(ServerError, match="prefill exited.*17"):
        server.wait_for_prefill_idle()


def test_case_records_idle_check_before_warmup(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    case = resolve_suite("deepseek_v4_pro_pd_performance")["cases"][0]
    events: list[str] = []

    class Server:
        def wait_for_prefill_idle(self) -> dict:
            events.append("idle")
            return {"metrics": {"running": 0.0, "waiting": 0.0}}

    monkeypatch.setattr(
        cli,
        "generate_formal",
        lambda *_args: {"path": "formal", "meta_path": "formal.meta"},
    )
    monkeypatch.setattr(
        cli,
        "generate_warmup",
        lambda *_args: {"path": "warmup", "meta_path": "warmup.meta"},
    )
    monkeypatch.setattr(cli, "assert_prefix_disjoint", lambda *_args: None)

    def run_warmup(*_args: object) -> list[str]:
        events.append("warmup")
        return ["warmup"]

    def run_performance(*_args: object) -> list[list[str]]:
        events.append("performance")
        return []

    monkeypatch.setattr(cli, "run_warmup", run_warmup)
    monkeypatch.setattr(cli, "run_performance", run_performance)
    monkeypatch.setattr(cli, "build_report", lambda *_args: {})
    state = {"cases": {}}
    cli._execute_case(case, tmp_path, state, Server())  # type: ignore[arg-type]
    assert events == ["idle", "warmup", "performance"]
    assert state["cases"][case["name"]]["prefill_quiescence"]["metrics"] == {
        "running": 0.0,
        "waiting": 0.0,
    }
