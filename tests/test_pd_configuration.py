from __future__ import annotations

from pathlib import Path

import pytest

from scripts import cli
from scripts.config import (
    ConfigurationError,
    _derive_warmup_input_length,
    load_yaml,
    resolve_case,
    resolve_suite,
)
from scripts.data import generate_warmup
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


def test_block_size_matches_model_runtime_support() -> None:
    cases = load_yaml("cases")["cases"]
    for case_name in cases:
        server = resolve_case(case_name)["effective"]["server"]
        block_size = server["arguments"]["--block-size"]
        if server["model"] == "qwen3_30b_a3b_w8a8":
            assert block_size == 128
        else:
            assert block_size == 64


def test_non_pd_server_profiles_share_common_defaults() -> None:
    server = load_yaml("server")
    defaults = server["defaults"]
    assert defaults["arguments"]["--enable-expert-parallel"] is True
    assert "--max-num-seqs" not in defaults["arguments"]
    assert defaults["environment"]["ASCEND_RT_VISIBLE_DEVICES"] == (
        "0,1,2,3,4,5,6,7"
    )

    profile_names = {
        "non_pd_baseline",
        "non_pd_cpp",
        "non_pd_srf",
        "non_pd_cpp_srf",
    }
    assert profile_names <= set(server["profiles"])
    for name in profile_names:
        profile = server["profiles"][name]
        assert "--max-num-seqs" not in profile.get("arguments", {})
        assert "ASCEND_RT_VISIBLE_DEVICES" not in profile.get("environment", {})


def test_prefill_variable_suite_reuses_non_pd_profiles_and_scheduler_defaults() -> None:
    server = load_yaml("server")
    assert not any(name.endswith("_0830") for name in server["profiles"])
    suite = resolve_suite("prefill_variable_performance")
    assert suite["definition"]["comparison"] is True
    cases = {
        case["name"]: case for case in suite["cases"]
    }
    strategies = ("baseline", "cpp", "srf", "cpp_srf")
    assert list(cases) == [f"prefill_{strategy}_variable" for strategy in strategies]
    for strategy in strategies:
        case = cases[f"prefill_{strategy}_variable"]
        assert case["definition"]["server"] == f"non_pd_{strategy}"
        assert case["definition"]["data"] == "prefill_variable"
        assert case["effective"]["data"]["output"]["length"] == 1
        assert case["effective"]["bench"]["concurrency"] == 4
        profile = server["profiles"][f"non_pd_{strategy}"]
        assert "ASCEND_RT_VISIBLE_DEVICES" not in profile.get("environment", {})
        for argument in {
            "--data-parallel-size",
            "--pipeline-parallel-size",
            "--tensor-parallel-size",
            "--max-num-batched-tokens",
            "--enforce-eager",
            "--no-async-scheduling",
            "--no-enable-prefix-caching",
        }:
            assert argument not in profile.get("arguments", {})

        effective = case["effective"]["server"]
        assert effective["environment"]["ASCEND_RT_VISIBLE_DEVICES"] == (
            "0,1,2,3,4,5,6,7"
        )
        assert effective["arguments"].get("--data-parallel-size", 1) == 1
        assert effective["arguments"]["--pipeline-parallel-size"] == 2
        assert effective["arguments"]["--tensor-parallel-size"] == 4
        assert effective["arguments"]["--max-num-batched-tokens"] == 20480
        standalone = resolve_case(case["name"])["effective"]["server"]
        assert effective == standalone
        scheduler = effective["arguments"].get("--additional-config", {}).get(
            "scheduler_config", {}
        )
        if strategy == "baseline":
            assert scheduler == {}
        if "cpp" in strategy:
            assert scheduler["profiling_chunk_config"] == {
                "enabled": True, "smooth_factor": 1.0, "need_timing": True,
            }
        if "srf" in strategy:
            assert scheduler["short_request_first_config"] == (
                profile["arguments"]["--additional-config"]["scheduler_config"][
                    "short_request_first_config"
                ]
            )


def test_pd_profiles_only_define_common_service_structure() -> None:
    profiles = load_yaml("server")["profiles"]
    assert {"pd", "pd_two_host"} <= set(profiles)
    assert not {
        "pd_deepseek_v4_pro",
        "pd_deepseek_v4_flash_two_host",
        "pd_deepseek_v4_pro_two_host",
    } & set(profiles)

    deployment_arguments = {
        "--data-parallel-size",
        "--tensor-parallel-size",
        "--pipeline-parallel-size",
    }
    for profile_name in ("pd", "pd_two_host"):
        pd = profiles[profile_name]["pd"]
        for role_name in ("prefill", "decode"):
            role = pd[role_name]
            assert "endpoint_host" not in role
            assert "model_tag" not in role
            assert "environment" not in role
            assert not deployment_arguments & set(role["arguments"])

        prefill_transfer = pd["prefill"]["arguments"]["--kv-transfer-config"]
        decode_transfer = pd["decode"]["arguments"]["--kv-transfer-config"]
        assert prefill_transfer == {
            "kv_connector": "MooncakeConnectorV1",
            "kv_role": "kv_producer",
            "kv_port": "36000",
            "engine_id": "0",
        }
        assert decode_transfer == {
            "kv_connector": "MooncakeConnectorV1",
            "kv_role": "kv_consumer",
            "kv_port": "36100",
            "engine_id": "1",
        }

    assert profiles["pd_two_host"]["pd"]["decode"]["external"] is True


def test_decode_uses_tp_only_and_keeps_the_visible_device_count() -> None:
    single_host_case = resolve_suite("baseline_pd_0830")["cases"][0]
    assert single_host_case["definition"]["server"] == "pd"
    single_host = single_host_case["effective"]["server"]
    deployments = [
        (single_host, 8),
        (
            resolve_suite("deepseek_v4_pro_pd_performance")["cases"][0][
                "effective"
            ]["server"],
            16,
        ),
        (
            resolve_suite("deepseek_v4_flash_pd_performance")["cases"][0][
                "effective"
            ]["server"],
            8,
        ),
    ]
    for server, expected_tp in deployments:
        prefill = _role_config(server, "prefill")
        decode = _role_config(server, "decode")
        arguments = decode["arguments"]
        assert "--data-parallel-size" not in arguments
        assert arguments["--tensor-parallel-size"] == expected_tp
        assert len(
            decode["environment"]["ASCEND_RT_VISIBLE_DEVICES"].split(",")
        ) == expected_tp
        assert arguments["--kv-transfer-config"][
            "kv_connector_extra_config"
        ]["decode"] == {
            "dp_size": 1,
            "tp_size": expected_tp,
            "pp_size": 1,
        }
        assert prefill["arguments"]["--kv-transfer-config"][
            "kv_connector_extra_config"
        ] == arguments["--kv-transfer-config"]["kv_connector_extra_config"]

    prefill = _role_config(single_host, "prefill")
    decode = _role_config(single_host, "decode")
    assert prefill["arguments"]["--tensor-parallel-size"] == 4
    assert prefill["arguments"]["--pipeline-parallel-size"] == 2
    assert prefill["environment"]["ASCEND_RT_VISIBLE_DEVICES"] == "0,1,2,3,4,5,6,7"
    assert decode["environment"]["ASCEND_RT_VISIBLE_DEVICES"] == "8,9,10,11,12,13,14,15"
    for role, engine_id in ((prefill, "0"), (decode, "1")):
        assert role.get("external", False) is False
        transfer = role["arguments"]["--kv-transfer-config"]
        assert transfer["engine_id"] == engine_id
        assert not {"kv_rank", "kv_parallel_size", "kv_buffer_device"} & set(transfer)


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
            assert server["arguments"]["--max-model-len"] == 87295
            assert server["pd"]["prefill"]["endpoint_host"] == "80.5.9.127"
            assert server["pd"]["decode"]["endpoint_host"] == "80.5.17.122"
            assert server["pd"]["decode"]["external"] is True
            prefill = _role_config(server, "prefill")
            decode = _role_config(server, "decode")
            expected_decode = expected_decode or decode
            assert decode == expected_decode
            assert prefill["model_tag"] == "/mnt/share/DeepSeekV4-pro-0813-w4a8"
            assert decode["model_tag"] == (
                "/mnt/share/DeepSeekV4-pro-0813-w4a8"
            )
            assert prefill["arguments"]["--data-parallel-size"] == 1
            assert prefill["arguments"]["--tensor-parallel-size"] == 8
            assert prefill["arguments"]["--pipeline-parallel-size"] == 2
            assert "--data-parallel-size" not in decode["arguments"]
            assert decode["arguments"]["--tensor-parallel-size"] == 16
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
                "decode": {"dp_size": 1, "tp_size": 16, "pp_size": 1},
            }
            assert prefill["environment"]["ASCEND_RT_VISIBLE_DEVICES"] == (
                "0,1,2,3,4,5,6,7,8,9,10,11,12,13,14,15"
            )
            assert decode["environment"]["ASCEND_RT_VISIBLE_DEVICES"] == (
                "0,1,2,3,4,5,6,7,8,9,10,11,12,13,14,15"
            )
            prefill_environment = prefill["environment"]
            decode_environment = decode["environment"]
            for key in (
                "GLOO_SOCKET_IFNAME",
                "TP_SOCKET_IFNAME",
                "HCCL_SOCKET_IFNAME",
            ):
                assert prefill_environment[key] == "enp194s0f0"
                assert decode_environment[key] == "enp48s3u1u1"
            assert decode_environment["VLLM_HOST_IP"] == "80.5.17.122"
            assert decode_environment["HCCL_IF_IP"] == "80.5.17.122"
            assert "80.5.17.122" in decode_environment["NO_PROXY"].split(",")
            assert "80.5.17.122" in prefill_environment["NO_PROXY"].split(",")
            assert effective["bench"]["concurrency"] == 8
            assert effective["warmup_bench"]["concurrency"] == 1
            assert effective["warmup"]["input_length"] == 87294
            assert effective["warmup"]["output_length"] == 1


def test_flash_suite_uses_common_model_max_warmup() -> None:
    suite = resolve_suite("deepseek_v4_flash_pd_performance")
    warmups = [case["definition"].get("warmup") for case in suite["cases"]]
    assert warmups == ["model_max_len"] * 8
    assert len(cli._server_segments(suite["cases"])) == 4
    for case in suite["cases"]:
        effective = case["effective"]
        server = effective["server"]
        pd = server["pd"]
        assert pd["prefill"]["endpoint_host"] == "80.5.9.127"
        assert pd["decode"]["endpoint_host"] == "80.5.9.128"
        assert pd["decode"]["external"] is True
        prefill = _role_config(server, "prefill")
        decode = _role_config(server, "decode")
        assert prefill["arguments"]["--data-parallel-size"] == 1
        assert prefill["arguments"]["--tensor-parallel-size"] == 4
        assert prefill["arguments"]["--pipeline-parallel-size"] == 2
        assert "--data-parallel-size" not in decode["arguments"]
        assert decode["arguments"]["--tensor-parallel-size"] == 8
        assert decode["arguments"]["--pipeline-parallel-size"] == 1
        assert prefill["arguments"]["--kv-transfer-config"][
            "kv_connector_extra_config"
        ] == decode["arguments"]["--kv-transfer-config"][
            "kv_connector_extra_config"
        ]
        assert decode["arguments"]["--kv-transfer-config"][
            "kv_connector_extra_config"
        ]["decode"] == {"dp_size": 1, "tp_size": 8, "pp_size": 1}
        assert prefill["environment"]["ASCEND_RT_VISIBLE_DEVICES"] == (
            "0,1,2,3,4,5,6,7"
        )
        assert decode["environment"]["ASCEND_RT_VISIBLE_DEVICES"] == (
            "8,9,10,11,12,13,14,15"
        )
        assert effective["bench"]["concurrency"] == 4
        assert effective["warmup"]["input_length"] == 1048575
        assert effective["warmup"]["output_length"] == 1


def test_all_warmup_references_use_common_profile() -> None:
    warmup = load_yaml("data")["warmup"]
    assert "input_length" not in warmup["defaults"]
    assert set(warmup["profiles"]) == {"model_max_len"}

    cases = load_yaml("cases")["cases"]
    case_references = {
        definition["warmup"]
        for definition in cases.values()
        if "warmup" in definition
    }
    assert case_references == {"model_max_len"}
    for case_name, definition in cases.items():
        if "warmup" not in definition:
            continue
        effective = resolve_case(case_name)["effective"]
        max_model_len = effective["server"]["arguments"]["--max-model-len"]
        assert (
            effective["warmup"]["input_length"]
            + effective["warmup"]["output_length"]
            == max_model_len
        )

    suites = load_yaml("suites")["suites"]
    suite_references = [
        override["warmup"]
        for suite in suites.values()
        for override in suite.get("case_overrides", {}).values()
        if "warmup" in override
    ]
    assert suite_references
    assert set(suite_references) == {"model_max_len"}


def test_warmup_generation_does_not_require_git_metadata(tmp_path: Path) -> None:
    tool_path = tmp_path / "tool"
    tool_path.mkdir()
    (tool_path / "generate_dataset.py").write_text(
        "def create_dataset(_tokenizer, length, count, _offset):\n"
        "    return ['x' * length for _ in range(count)]\n",
        encoding="utf-8",
    )
    run_dir = tmp_path / "run"
    run_dir.mkdir()
    result = generate_warmup(
        "model_max_len",
        {
            "storage": "per_run",
            "tool_path": str(tool_path),
            "input_length": 8,
            "output_length": 1,
            "request_count": 2,
            "seed": 826,
        },
        "unused-tokenizer",
        run_dir,
        "test-case",
    )

    assert Path(result["path"]).is_file()
    assert result["reused"] is False
    assert "tool_revision" not in result


def test_accuracy_profiles_use_dataset_model_output_names() -> None:
    accuracy = load_yaml("accuracy")
    defaults = accuracy["defaults"]
    assert defaults["generation_kwargs"] == {
        "temperature": 1,
        "top_p": 0.95,
        "repetition_penalty": 1,
        "ignore_eos": False,
    }
    assert defaults["max_output_tokens"] == 32768
    assert defaults["concurrency"] == 128
    assert set(accuracy["profiles"]) == {
        "gsm8k_dsv4_flash_32k",
        "gpqa_dsv4_flash_32k",
    }
    for profile in accuracy["profiles"].values():
        assert profile["generation_kwargs"] == {
            "chat_template_kwargs": {"enable_thinking": True}
        }

    cases = load_yaml("cases")["cases"]
    references = {
        definition["accuracy"]
        for definition in cases.values()
        if "accuracy" in definition
    }
    assert references == set(accuracy["profiles"])
    expected_generation_kwargs = {
        **defaults["generation_kwargs"],
        "chat_template_kwargs": {"enable_thinking": True},
    }
    for case_name, definition in cases.items():
        if "accuracy" in definition:
            assert (
                resolve_case(case_name)["effective"]["accuracy"]["generation_kwargs"]
                == expected_generation_kwargs
            )


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
    assert result["p0"]["polls"] == 6
    assert result["p0"]["metrics"] == {"running": 0.0, "waiting": 0.0}
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
        def check_alive(self) -> None:
            pass

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
