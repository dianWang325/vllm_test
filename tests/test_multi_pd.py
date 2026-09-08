from __future__ import annotations

from copy import deepcopy
import io
import json
from pathlib import Path
import shlex
import subprocess
import threading
from unittest.mock import Mock

import pytest

from scripts.config import pd_role_nodes, resolve_suite
from scripts.pd_role import _role_config, _server_config
from scripts import server as serving
from scripts.server import ServerError, VllmServer, build_remote_command
from scripts.bench import run_client
from scripts import cli, remote_process
from scripts.runtime_info import RuntimeInfo


def test_node_overrides_follow_role_and_suite_overrides() -> None:
    server = resolve_suite("deepseek_v4_pro_pd_performance")["cases"][2]["effective"]["server"]
    server["pd"]["prefill"]["nodes"] = {
        "p0": {"arguments": {"--node-rank": 0}},
        "p1": {"arguments": {"--node-rank": 1, "--headless": True}},
    }
    original = deepcopy(server)
    nodes = pd_role_nodes(server, "prefill")
    assert list(nodes) == ["p0", "p1"]
    for node in nodes.values():
        assert "nodes" not in node and "pd" not in node
        assert node["arguments"]["--additional-config"]["scheduler_config"]["profiling_chunk_config"]["enabled"]
        assert node["arguments"]["--kv-transfer-config"]["kv_connector"] == "MooncakeConnectorV1"
    assert nodes["p1"]["arguments"]["--headless"] is True
    nodes["p0"]["environment"]["VLLM_HOST_IP"] = "changed"
    assert server == original
    assert nodes["p1"]["environment"]["VLLM_HOST_IP"] != "changed"
    with pytest.raises(RuntimeError, match="specify --node"):
        _role_config(server, "prefill")
    assert _role_config(server, "prefill", "p1")["arguments"]["--node-rank"] == 1


def test_manual_node_selection_keeps_suite_and_case_overrides() -> None:
    suite = "deepseek_v4_pro_pd_performance"
    case = "deepseek_v4_pro_pd_cpp_fixed"
    assert _server_config(case, suite) == resolve_suite(suite)["cases"][2]["effective"]["server"]
    with pytest.raises(RuntimeError, match="not in suite"):
        _server_config("baseline_smoke", suite)


@pytest.fixture
def multi_config(tmp_path: Path) -> dict:
    config = resolve_suite("deepseek_v4_pro_pd_performance")["cases"][0]["effective"]["server"]
    for role, names in (("prefill", ("p0", "p1")), ("decode", ("d0", "d1"))):
        config["pd"][role]["external"] = False
        config["pd"][role]["nodes"] = {
            name: {
                "endpoint_host": name,
                "remote": {"host": name, "container": "test", "workdir": "/repo"},
                "arguments": {"--headless": name == "p1"},
            }
            for name in names
        }
    script = tmp_path / "proxy.py"
    script.touch()
    config["pd"]["proxy"]["script"] = str(script)
    return config


def test_remote_command_quotes_config_and_tracks_controller() -> None:
    config = {
        "model_tag": "/model with spaces",
        "arguments": {"--additional-config": {"value": "a'b;$HOME"}},
        "environment": {"NAME": "spaces and 'quotes'"},
        "unset_environment": ["OTHER"],
        "remote": {"host": "user@host", "container": "test", "workdir": "/repo space"},
    }
    command = build_remote_command(config, "/tmp/test.pid")
    assert command[:-1] == ["ssh", "-T", "-o", "BatchMode=yes", "user@host"]
    remote = shlex.split(command[-1])
    assert remote[:7] == ["docker", "exec", "-i", "-w", "/repo space", "test", "python"]
    assert "--stop-on-stdin-close" in remote
    assert remote[remote.index("env") + 1:] == [
        "-u", "OTHER", "NAME=spaces and 'quotes'", "vllm", "serve", "/model with spaces",
        "--additional-config", '{"value":"a\'b;$HOME"}',
    ]
    config["unset_environment"].append("NAME")
    remote = shlex.split(build_remote_command(config, "/tmp/test.pid")[-1])
    assert not any(argument.startswith("NAME=") for argument in remote)


def test_proxy_lists_only_api_nodes_and_requires_both_decoders(multi_config: dict, tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    server = VllmServer(multi_config, tmp_path / "server.log")
    prefill, decode, proxy = server._build_pd_configs()
    command = serving.build_proxy_command(multi_config, prefill, decode)
    assert command[command.index("--prefiller-hosts"):command.index("--decoder-hosts")] == [
        "--prefiller-hosts", "p0", "--prefiller-ports", "18081",
    ]
    assert command[command.index("--decoder-hosts"):command.index("--workers")] == [
        "--decoder-hosts", "d0", "d1", "--decoder-ports", "18082", "18082",
    ]
    for count in (1, 2):
        response = io.StringIO(json.dumps({"status": "ok", "prefill_instances": 1, "decode_instances": count}))
        response.status = 200
        monkeypatch.setattr(serving.urllib.request, "urlopen", lambda *_args, **_kwargs: response)
        assert serving._proxy_healthy(proxy) is (count == 2)


def test_start_all_ranks_before_health_and_signal_all_before_wait(multi_config: dict, tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    events = []
    processes = []

    def popen(command, **kwargs):
        name = command[4] if command[0] == "ssh" else "proxy"
        process = Mock(pid=len(processes) + 1, returncode=None)
        process.poll.side_effect = lambda: process.returncode
        def signal_stop():
            events.append(("stop", name))
            process.returncode = 0
        process.stdin = Mock() if kwargs["stdin"] == subprocess.PIPE else None
        if process.stdin is not None:
            process.stdin.close.side_effect = signal_stop
        def wait(timeout):
            assert len([event for event in events if event[0] == "stop"]) == 5
            return 0
        process.wait.side_effect = wait
        process.signal_stop = signal_stop
        processes.append(process)
        events.append(("spawn", name))
        return process

    def healthy(config):
        assert len(processes) == 4
        events.append(("health", config["endpoint_host"]))
        return True

    monkeypatch.setattr(serving.subprocess, "Popen", popen)
    monkeypatch.setattr(serving, "_port_is_free", lambda *_args: True)
    monkeypatch.setattr(serving, "_healthy", healthy)
    monkeypatch.setattr(serving, "_proxy_healthy", lambda *_args: True)
    monkeypatch.setattr(serving.os, "getpgid", lambda pid: pid, raising=False)
    monkeypatch.setattr(serving.os, "killpg", lambda pid, _sig: processes[pid - 1].signal_stop(), raising=False)
    server = VllmServer(multi_config, tmp_path / "server.log")
    commands = server.start()
    assert list(commands) == ["prefill.p0", "prefill.p1", "decode.d0", "decode.d1", "proxy"]
    assert [name for event, name in events if event == "health"] == ["p0", "d0", "d1"]
    server.stop()
    assert [name for event, name in events if event == "stop"] == ["proxy", "d1", "d0", "p1", "p0"]
    assert len(server.node_records) == 5
    assert all((tmp_path / record["log"]).is_file() for record in server.node_records.values())


@pytest.mark.parametrize("name", ["prefill.p1", "decode.d1"])
def test_worker_exit_is_fatal_even_without_api(name: str, multi_config: dict, tmp_path: Path) -> None:
    server = VllmServer(multi_config, tmp_path / "server.log")
    process = Mock(returncode=17)
    process.poll.return_value = 17
    server.processes[name] = process
    with pytest.raises(ServerError, match=f"{name} exited: 17"):
        server.wait_for_prefill_idle()


def test_four_host_suite_topology_and_connector_are_explicit() -> None:
    suite = resolve_suite("deepseek_v4_pro_multi_pd_performance_4case")
    assert [len(segment) for segment in cli._server_segments(suite["cases"])] == [2, 2]
    topology = {"prefill": {"dp_size": 1, "tp_size": 16, "pp_size": 2}, "decode": {"dp_size": 2, "tp_size": 16, "pp_size": 1}}
    for index, case in enumerate(suite["cases"]):
        server = case["effective"]["server"]
        prefill = pd_role_nodes(server, "prefill")
        decode = pd_role_nodes(server, "decode")
        assert list(prefill) == ["p0", "p1"]
        assert list(decode) == ["d0", "d1"]
        assert case["effective"]["bench"]["concurrency"] == 8
        assert case["effective"]["bench"]["repeats"] == 1
        assert server["arguments"]["--max-model-len"] == 87295
        assert case["effective"]["warmup"]["input_length"] == 87294
        for role, nodes in (("prefill", prefill), ("decode", decode)):
            for rank, node in enumerate(nodes.values()):
                args = node["arguments"]
                assert not node.get("external", False)
                assert len(node["environment"]["ASCEND_RT_VISIBLE_DEVICES"].split(",")) == 16
                assert args["--distributed-executor-backend"] == "mp"
                assert args["--tensor-parallel-size"] == 16
                transfer = args["--kv-transfer-config"]
                assert transfer == {
                    "kv_connector": "MooncakeConnectorV1",
                    "kv_role": "kv_producer" if role == "prefill" else "kv_consumer",
                    "kv_port": "36000" if role == "prefill" else "36100",
                    "engine_id": "0" if role == "prefill" else "1",
                    "kv_connector_extra_config": topology,
                }
                if role == "prefill":
                    assert args["--nnodes"] == 2 and args["--node-rank"] == rank
                    assert args["--data-parallel-size"] == 1 and args["--pipeline-parallel-size"] == 2
                    assert args["--master-addr"] == "P0_IP"
                    assert args.get("--headless", False) is (rank == 1)
                    assert ("--additional-config" in args) is (index >= 2)
                    if index >= 2:
                        scheduler = args["--additional-config"]["scheduler_config"]
                        assert scheduler["profiling_chunk_config"]["enabled"]
                        assert scheduler["short_request_first_config"]["enabled"]
                else:
                    assert args["--data-parallel-size"] == 2 and args["--pipeline-parallel-size"] == 1
                    assert args["--data-parallel-rank"] == rank
                    assert args["--data-parallel-address"] == "D0_IP"
                    assert args["--enable-expert-parallel"] is True
                    assert not {"--nnodes", "--node-rank", "--headless", "--additional-config"} & args.keys()


def test_remote_supervisor_stops_on_controller_eof(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    stopped = threading.Event()
    child = Mock(pid=1234)
    child.poll.side_effect = lambda: 0 if stopped.is_set() else None
    def wait():
        assert stopped.wait(2), "controller EOF did not terminate the child"
        return 0
    child.wait.side_effect = wait
    popen = Mock(return_value=child)
    monkeypatch.setattr(remote_process.subprocess, "Popen", popen)
    monkeypatch.setattr(remote_process.os, "getpgid", lambda _pid: 1234, raising=False)
    monkeypatch.setattr(remote_process.os, "killpg", lambda *_args: stopped.set(), raising=False)
    monkeypatch.setattr(remote_process, "_start_time", lambda _pid: "test")
    monkeypatch.setattr(remote_process.select, "select", lambda *_args: ([remote_process.sys.stdin], [], []))
    monkeypatch.setattr(remote_process.sys, "stdin", Mock())
    monkeypatch.setattr(remote_process.os, "read", lambda *_args: b"")
    monkeypatch.setattr(remote_process.signal, "signal", lambda *_args: None)
    pid_file = tmp_path / "node.pid"
    assert remote_process.run(pid_file, ["vllm"], stop_on_stdin_close=True) == 0
    assert not pid_file.exists()
    assert popen.call_args.kwargs["stdin"] == subprocess.DEVNULL


def test_client_is_cancelled_when_worker_exits(monkeypatch: pytest.MonkeyPatch) -> None:
    process = Mock(pid=1234)
    process.poll.return_value = None
    process.wait.side_effect = [subprocess.TimeoutExpired("bench", 1), 0]
    monkeypatch.setattr("scripts.bench.subprocess.Popen", lambda *_args, **_kwargs: process)
    monkeypatch.setattr("scripts.bench.os.getpgid", lambda _pid: 1234, raising=False)
    killpg = Mock()
    monkeypatch.setattr("scripts.bench.os.killpg", killpg, raising=False)
    check_alive = Mock(side_effect=[None, None, ServerError("prefill.p1 exited: 17")])
    with pytest.raises(ServerError, match="prefill.p1 exited"):
        run_client(["bench"], {}, io.StringIO(), check_alive)
    killpg.assert_called_once()


def test_client_checks_worker_after_last_request(monkeypatch: pytest.MonkeyPatch) -> None:
    process = Mock()
    process.wait.return_value = 0
    process.poll.return_value = 0
    monkeypatch.setattr("scripts.bench.subprocess.Popen", lambda *_args, **_kwargs: process)
    check_alive = Mock(side_effect=[None, None, ServerError("decode.d1 exited: 17")])
    with pytest.raises(ServerError, match="decode.d1 exited"):
        run_client(["bench"], {}, io.StringIO(), check_alive)


@pytest.mark.parametrize("failure", [None, "startup", "case", "stop"])
def test_suite_restarts_all_nodes_and_records_failure(
    failure: str | None, tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
) -> None:
    suite_name = "deepseek_v4_pro_multi_pd_performance_4case"
    suite = resolve_suite(suite_name)
    for case in suite["cases"]:
        case["effective"]["server"]["pd"]["proxy"]["script"] = __file__
    monkeypatch.setattr(cli, "ROOT", tmp_path)
    monkeypatch.setattr(cli, "resolve_suite", lambda _name: suite)
    monkeypatch.setattr(cli, "read_runtime_info", lambda: RuntimeInfo("host", "host", "test", "abc"))
    monkeypatch.setattr(cli, "build_run_id", lambda *_args: "host_test_performance")
    monkeypatch.setattr(cli, "build_suite_comparison", lambda *_args: {"ok": True})
    monkeypatch.setattr(serving, "_port_is_free", lambda *_args: True)
    monkeypatch.setattr(serving, "_healthy", lambda *_args: True)
    monkeypatch.setattr(serving, "_proxy_healthy", lambda *_args: True)
    spawned = []
    stopped = []
    instances = []

    class Server(VllmServer):
        def _spawn(self, role, command, config):
            if self not in instances:
                instances.append(self)
            log = self.log_path.with_name(f"{self.log_path.stem}-{role}.log")
            log.touch()
            self.node_records[role] = {"log": log.name}
            spawned.append((len(instances), role))
            if failure == "startup" and role == "prefill.p1":
                raise ServerError("startup failure")
            process = Mock(returncode=None)
            process.poll.return_value = None
            def stop_node():
                stopped.append((len(instances), role))
                process.poll.return_value = 0
                process.returncode = 0
            process.stdin.close.side_effect = stop_node
            if failure == "stop" and role == "prefill.p0":
                process.wait.side_effect = subprocess.TimeoutExpired("node", 1)
            self.processes[role] = process

    def execute_case(case, _run_dir, state, _server):
        state["cases"][case["name"]] = {"status": "completed"}
        if failure == "case":
            raise ServerError("case failure")

    monkeypatch.setattr(cli, "VllmServer", Server)
    monkeypatch.setattr(cli, "_execute_case", execute_case)
    if failure is None:
        cli.execute("suite", suite_name)
        assert len(instances) == 2
        assert len(spawned) == len(stopped) == 10
        assert stopped[:5] == [(1, name) for name in ("proxy", "decode.d1", "decode.d0", "prefill.p1", "prefill.p0")]
    else:
        with pytest.raises(ServerError):
            cli.execute("suite", suite_name)
        assert len(instances) == 1
        assert len(stopped) == (1 if failure == "startup" else 5)
    state = json.loads((tmp_path / "runs" / "host_test_performance" / "run.json").read_text())
    assert state["status"] == ("completed" if failure is None else "failed")
    assert state["servers"][0]["nodes"]
    if failure == "stop":
        assert "prefill.p0" in state["servers"][0]["stop_error"]
