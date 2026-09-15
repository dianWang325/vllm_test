"""Check the temporary vllm-ascend mapping patch without importing vLLM."""

from __future__ import annotations

import os
import runpy
import sys
import types
from pathlib import Path


HERE = Path(__file__).resolve().parent
platforms = types.ModuleType("vllm.platforms")
platforms.current_platform = types.SimpleNamespace(
    device_control_id_to_physical_device_id=int
)
utils = types.ModuleType("vllm.v1.engine.utils")


def original(_env, rank, world, local, _ids=None):
    visible = [int(x) for x in os.environ[_env].split(",")]
    return visible[rank * world : rank * world + local]


utils.get_physical_gpu_ids_for_local_dp_rank = original
sys.modules["vllm"] = types.ModuleType("vllm")
sys.modules["vllm.platforms"] = platforms
sys.modules["vllm.v1"] = types.ModuleType("vllm.v1")
sys.modules["vllm.v1.engine"] = types.ModuleType("vllm.v1.engine")
sys.modules["vllm.v1.engine.utils"] = utils
runpy.run_path(str(HERE / "patched_dp_device_ids.py"))
mapping = utils.get_physical_gpu_ids_for_local_dp_rank

os.environ["ASCEND_RT_VISIBLE_DEVICES"] = ",".join(map(str, range(16)))
assert mapping("ASCEND_RT_VISIBLE_DEVICES", 0, 16, 8) == list(range(8))
assert mapping("ASCEND_RT_VISIBLE_DEVICES", 1, 16, 8) == list(range(8, 16))
os.environ["ASCEND_RT_VISIBLE_DEVICES"] = ",".join(map(str, range(8, 16)))
assert mapping("ASCEND_RT_VISIBLE_DEVICES", 1, 16, 8) == list(range(8, 16))
print("Full-list and pre-sharded DP device mapping passed")
