from __future__ import annotations

import pytest

from scripts import cli
from scripts.config import ConfigurationError, resolve_suite


def test_runtime_overrides_isolate_non_pd_suite(monkeypatch: pytest.MonkeyPatch) -> None:
    cases = resolve_suite(
        "prefill_baseline_cpp_v2_runner_1_performance"
    )["cases"]
    monkeypatch.setenv("ASCEND_RT_VISIBLE_DEVICES", "8, 9,10,11,12,13,14,15")
    monkeypatch.setenv("VTEST_SERVER_PORT", "18090")

    cli._runtime_server_overrides(cases)

    for case in cases:
        server = case["effective"]["server"]
        assert server["environment"]["ASCEND_RT_VISIBLE_DEVICES"] == (
            "8,9,10,11,12,13,14,15"
        )
        assert server["arguments"]["--port"] == 18090


def test_runtime_overrides_reject_invalid_values(monkeypatch: pytest.MonkeyPatch) -> None:
    cases = resolve_suite(
        "prefill_baseline_cpp_v2_runner_1_performance"
    )["cases"]
    monkeypatch.setenv("ASCEND_RT_VISIBLE_DEVICES", "0,0")
    with pytest.raises(ConfigurationError, match="duplicate"):
        cli._runtime_server_overrides(cases)

    monkeypatch.setenv("ASCEND_RT_VISIBLE_DEVICES", "0,1")
    monkeypatch.setenv("VTEST_SERVER_PORT", "70000")
    with pytest.raises(ConfigurationError, match="between 1 and 65535"):
        cli._runtime_server_overrides(cases)


def test_run_tag_is_validated_and_inserted(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("VTEST_RUN_TAG", "cards8-15")
    tag = cli._runtime_run_tag()
    assert tag == "cards8-15"
    assert cli._tag_run_id("server97_dev_09091730_performance", tag) == (
        "server97_dev_09091730_cards8-15_performance"
    )

    monkeypatch.setenv("VTEST_RUN_TAG", "bad tag")
    with pytest.raises(ConfigurationError, match="VTEST_RUN_TAG"):
        cli._runtime_run_tag()
