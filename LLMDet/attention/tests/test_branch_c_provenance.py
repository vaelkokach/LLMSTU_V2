"""Tests for the Branch-C provenance and GPU-selection layer.

These guard the evidence chain, not the science. If RUNS.jsonl can be torn by
concurrent writers, or an incomplete cache can pass for a finished one, then
every downstream number inherits that doubt.
"""

from __future__ import annotations

import importlib.util
import json
import sys
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import pytest

REPO = Path(__file__).resolve().parents[3]
MODULE = REPO / "tools/branch_c/provenance.py"

pytestmark = pytest.mark.skipif(not MODULE.exists(), reason="provenance module absent")


def _load():
    spec = importlib.util.spec_from_file_location("branch_c_provenance", MODULE)
    mod = importlib.util.module_from_spec(spec)
    sys.modules["branch_c_provenance"] = mod
    spec.loader.exec_module(mod)
    return mod


prov = _load()


def test_gpu_cap_cannot_be_exceeded():
    """The shared-box cap is a hard error, not a warning that gets ignored."""
    with pytest.raises(ValueError, match="shared"):
        prov.select_free_gpus(n=prov.MAX_CONCURRENT_GPUS + 1)


def test_busy_devices_are_not_selected(monkeypatch):
    """A device holding another user's job must never be handed out."""
    fake = [
        {"index": 0, "name": "A100", "memory_total_mib": 40960, "memory_used_mib": 4232, "utilization_pct": 0},
        {"index": 1, "name": "A100", "memory_total_mib": 40960, "memory_used_mib": 21, "utilization_pct": 0},
        {"index": 2, "name": "A100", "memory_total_mib": 40960, "memory_used_mib": 690, "utilization_pct": 0},
        {"index": 3, "name": "A100", "memory_total_mib": 40960, "memory_used_mib": 21, "utilization_pct": 0},
    ]
    monkeypatch.setattr(prov, "gpu_status", lambda: fake)
    got = prov.select_free_gpus(n=4)
    assert 0 not in got and 2 not in got, f"selected a busy device: {got}"
    assert set(got) == {1, 3}


def test_selection_raises_when_nothing_is_free(monkeypatch):
    """Refuse to share rather than silently double-book a device."""
    monkeypatch.setattr(
        prov,
        "gpu_status",
        lambda: [
            {"index": i, "name": "A100", "memory_total_mib": 40960, "memory_used_mib": 9000, "utilization_pct": 90}
            for i in range(4)
        ],
    )
    with pytest.raises(RuntimeError, match="no idle GPU"):
        prov.select_free_gpus(n=1)


def test_emptiest_device_first(monkeypatch):
    monkeypatch.setattr(
        prov,
        "gpu_status",
        lambda: [
            {"index": 0, "name": "A100", "memory_total_mib": 40960, "memory_used_mib": 400, "utilization_pct": 0},
            {"index": 1, "name": "A100", "memory_total_mib": 40960, "memory_used_mib": 21, "utilization_pct": 0},
        ],
    )
    assert prov.select_free_gpus(n=2) == [1, 0]


def test_concurrent_appends_do_not_tear(tmp_path):
    """Four writers, one file: every line must still parse."""
    path = tmp_path / "RUNS.jsonl"
    n = 40

    def write(i: int):
        prov.append_record(
            prov.RunRecord(
                run_id=f"run_{i}",
                arm="test_arm",
                command=["python", "-c", "pass"],
                seed=i,
                notes="x" * 500,  # long enough that a torn write would show
            ),
            path=path,
        )

    with ThreadPoolExecutor(max_workers=4) as ex:
        list(ex.map(write, range(n)))

    lines = [l for l in path.read_text().splitlines() if l.strip()]
    assert len(lines) == n, f"expected {n} lines, got {len(lines)}"
    for line in lines:
        json.loads(line)  # raises on a torn line
    assert len({json.loads(l)["run_id"] for l in lines}) == n


def test_records_carry_the_required_protocol_fields(tmp_path):
    """BRANCH_C_PROTOCOL.md section 9 names these explicitly."""
    path = tmp_path / "RUNS.jsonl"
    prov.append_record(
        prov.RunRecord(run_id="r", arm="a", command=["c"], seed=42, fold=0), path=path
    )
    rec = prov.read_records(path)[0]
    for key in (
        "run_id", "arm", "command", "seed", "fold", "git", "environment",
        "split_manifest_sha256", "feature_schema_sha256", "checkpoint_sha256",
        "wall_clock_s", "peak_memory_mib", "device", "complete",
    ):
        assert key in rec, f"provenance record is missing {key}"
    assert rec["git"]["commit"], "git commit not captured"


def test_incomplete_rows_are_not_read_as_evidence(tmp_path):
    path = tmp_path / "RUNS.jsonl"
    path.write_text(
        json.dumps({"run_id": "torn", "complete": False}) + "\n"
        + "{not valid json\n"
        + json.dumps({"run_id": "good", "complete": True}) + "\n"
    )
    recs = prov.read_records(path)
    assert [r["run_id"] for r in recs] == ["good"]


def test_sentinel_marks_completion(tmp_path):
    d = tmp_path / "cache"
    d.mkdir()
    assert not prov.is_complete(d)
    prov.write_sentinel(d, {"n_items": 3})
    assert prov.is_complete(d)
    payload = json.loads((d / "_COMPLETE.json").read_text())
    assert payload["n_items"] == 3 and payload["completed_at"]


def test_atomic_write_leaves_no_partial_file(tmp_path):
    p = tmp_path / "out.json"
    prov.atomic_write_json({"a": 1}, p)
    assert json.loads(p.read_text()) == {"a": 1}
    assert not list(tmp_path.glob("*.partial")), "temp file survived the write"


def test_sha256_of_missing_file_is_none_not_an_exception(tmp_path):
    assert prov.sha256_file(tmp_path / "nope.pth") is None


def test_git_state_captures_dirty_diff_hash():
    st = prov.git_state()
    assert st["commit"] and len(st["commit"]) == 40
    # dirty and diff_sha256 must agree with each other whatever the tree state
    assert bool(st["dirty"]) == (st["diff_sha256"] is not None)
