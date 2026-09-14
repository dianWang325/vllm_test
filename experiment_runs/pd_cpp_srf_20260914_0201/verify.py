"""Verify run completion, metric counts, warmups, and P/D configuration parity."""

from __future__ import annotations

import json
from collections import Counter
from pathlib import Path


HERE = Path(__file__).resolve().parent
MANIFEST = json.loads((HERE / "manifest.json").read_text(encoding="utf-8"))


def api_args(path: Path) -> list[str]:
    marker = "non-default args: "
    return [line.split(marker, 1)[1] for line in path.read_text(encoding="utf-8").splitlines() if marker in line]


def main() -> None:
    schedule = MANIFEST["schedule"]
    assert len(schedule) == 8
    assert Counter(map(tuple, schedule[:4])) == Counter(map(tuple, schedule[4:]))
    assert set(map(tuple, schedule[:4])) == {
        ("baseline", "fixed"), ("baseline", "variable"),
        ("enhanced", "fixed"), ("enhanced", "variable"),
    }
    p_canonical: set[str] = set()
    d_canonical: set[tuple[str, ...]] = set()
    for index, (mode, dataset) in enumerate(schedule, 1):
        tag = f"{index:02d}_{mode}_{dataset}"
        folder = HERE / tag
        status = json.loads((folder / "status.json").read_text(encoding="utf-8"))
        formal = json.loads((folder / "formal_metrics.json").read_text(encoding="utf-8"))
        assert status["status"] == "complete" and not status["stop_errors"], tag
        assert formal["Success Requests"]["total"] == 24, tag
        assert formal["Failed Requests"]["total"] == 0, tag
        assert formal["Total Generated Tokens"]["total"] == 24 * 2560, tag
        assert formal["Total Token Throughput"]["total"].endswith("token/s"), tag
        if mode == "enhanced":
            warmup = json.loads((folder / "warmup_metrics.json").read_text(encoding="utf-8"))
            assert warmup["Success Requests"]["total"] == 5, tag
            assert warmup["Failed Requests"]["total"] == 0, tag
        else:
            assert not (folder / "warmup_metrics.json").exists(), tag

        p_args = api_args(folder / "prefill.log")
        assert len(p_args) == 1, tag
        p = p_args[0]
        marker = ", 'additional_config': "
        if mode == "enhanced":
            assert marker in p, tag
            assert "'profiling_chunk_config': {'enabled': True" in p, tag
            assert "'short_request_first_config': {'enabled': True, 'threshold': 65546, 'long_max_wait_ms': 2000}" in p, tag
            p = p.split(marker, 1)[0] + "}"
        else:
            assert marker not in p, tag
        assert "'gpu_memory_utilization': 0.85" in p, tag
        p_canonical.add(p)

        d_args = api_args(folder / "decode.log")
        assert len(d_args) == 2, tag
        d_canonical.add(tuple(sorted(d_args)))
        assert "External prefix cache hit rate: 100.0%" in (folder / "decode.log").read_text(encoding="utf-8"), tag
    assert len(p_canonical) == 1, "P arguments differ beyond CPP+SRF"
    assert len(d_canonical) == 1, "D arguments differ between runs"
    print("Verified 8 complete runs, 192/192 formal requests, four 5/5 warmups, and P/D argument parity.")


if __name__ == "__main__":
    main()
