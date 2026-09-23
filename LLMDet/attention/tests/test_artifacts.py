"""Tests for the Branch-C reproducibility/release tooling
(artifacts.lock.json + tools/artifacts/* + bootstrap.py).

Every test in this file runs WITHOUT network access. Anything that would
otherwise touch the network (urllib.request.urlopen) is monkeypatched.

Scope, deliberately:
  * The schema/coverage tests validate the REAL artifacts.lock.json shipped
    at the repo root -- these should stay green as the lock file evolves.
  * The hardcoded-path tests validate the SCANNING MECHANISM against
    synthetic fixtures, plus assert that this task's own new files (bootstrap
    .py, pyproject.toml, tools/artifacts/*) are clean -- they do NOT assert
    the whole repository is clean. [internal notes, not included]
    RISK 4 already documents 6 pre-existing hardcoded-path files outside
    this agent's owned paths; asserting a repo-wide zero here would make this
    suite red for a problem this task was not scoped to fix. See
    tools/artifacts/ci_path_scan.py (a separate, non-pytest CI gate) for the
    repo-wide scan, which is expected to fail until RISK 4 is resolved.
"""
from __future__ import annotations

import json
import os
import re
import sys
import tarfile
import zipfile
from pathlib import Path
from unittest import mock

import pytest

REPO = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(REPO))

from tools.artifacts import (  # noqa: E402
    ArtifactError,
    ChecksumMismatch,
    PathTraversalError,
    atomic_install,
    cache_root,
    destination_path,
    download_to_temp,
    load_lock,
    safe_extract_tar,
    safe_extract_zip,
    scan_for_forbidden_paths,
    sha256_file,
    validate_lock_schema,
    verify_artifact,
)
from tools.artifacts.download import _refuse_reason, _resolve_one  # noqa: E402

LOCK_PATH = REPO / "artifacts.lock.json"
ATTENTION_RUNTIME_YAML = REPO / "LLMDet" / "configs" / "attention_runtime.yaml"
HEAD_POSE_PY = REPO / "LLMDet" / "attention" / "head_pose.py"
BENCH_FACE_BACKENDS_PY = REPO / "LLMDet" / "attention" / "bench_face_backends.py"


# ---------------------------------------------------------------------------
# 1. artifacts.lock.json schema validity + coverage of real runtime references
# ---------------------------------------------------------------------------


def test_lock_file_exists_and_is_valid_json():
    assert LOCK_PATH.exists(), "artifacts.lock.json must exist at the repo root"
    with open(LOCK_PATH) as f:
        json.load(f)  # raises if malformed


def test_lock_schema_passes_validation():
    lock = load_lock(LOCK_PATH)  # raises SchemaError on failure
    assert lock["schema"] == "llmstu_artifacts_lock.v1"
    assert len(lock["artifacts"]) > 0


def test_lock_schema_validator_actually_catches_problems():
    """The validator must be a real check, not a rubber stamp."""
    broken = {"schema": "x"}  # missing everything else
    problems = validate_lock_schema(broken)
    assert problems, "an obviously-incomplete lock object must fail validation"

    lock = json.loads(LOCK_PATH.read_text())
    mutated = json.loads(json.dumps(lock))  # deep copy
    mutated["artifacts"][0]["status"] = "not-a-real-status"
    problems = validate_lock_schema(mutated)
    assert any("invalid status" in p for p in problems)

    mutated2 = json.loads(json.dumps(lock))
    mutated2["artifacts"][0]["verified"] = True
    mutated2["artifacts"][0]["sha256"] = None
    problems2 = validate_lock_schema(mutated2)
    assert any("verified=true but sha256 is null" in p for p in problems2)


def test_every_artifact_id_is_unique():
    lock = load_lock(LOCK_PATH)
    ids = [a["id"] for a in lock["artifacts"]]
    assert len(ids) == len(set(ids))


def test_every_verified_artifact_has_a_real_looking_sha256():
    """Guards against ever re-introducing an invented/placeholder hash."""
    lock = load_lock(LOCK_PATH)
    hex_chars = set("0123456789abcdef")
    for a in lock["artifacts"]:
        if a["verified"]:
            sha = a["sha256"]
            assert sha and len(sha) == 64 and set(sha) <= hex_chars, (
                f"{a['id']}: verified=true must carry a real 64-hex-char sha256, got {sha!r}"
            )
        else:
            assert a["sha256"] is None, (
                f"{a['id']}: verified=false should record sha256=null, not a value that was "
                f"never actually checked"
            )


def _extract_yaml_scalar(text: str, key: str) -> str:
    m = re.search(rf'^\s*{re.escape(key)}:\s*"([^"]+)"', text, re.MULTILINE)
    assert m, f"could not find {key!r} in attention_runtime.yaml -- did the config change shape?"
    return m.group(1)


def _lock_destinations(lock: dict) -> set:
    return {a["destination"] for a in lock["artifacts"] if a.get("destination")}


def test_lock_covers_the_deployed_temporal_checkpoint():
    """LLMDet/configs/attention_runtime.yaml:temporal_checkpoint is what the
    live dashboard actually loads -- it must have a lock entry."""
    lock = load_lock(LOCK_PATH)
    yaml_text = ATTENTION_RUNTIME_YAML.read_text()
    rel = _extract_yaml_scalar(yaml_text, "temporal_checkpoint")
    expected_dest = f"LLMDet/{rel}"
    assert expected_dest in _lock_destinations(lock), (
        f"attention_runtime.yaml's temporal_checkpoint ({rel!r}) has no matching "
        f"artifacts.lock.json destination -- the deployed model would be unresolvable "
        f"via bootstrap.py"
    )


def test_lock_covers_the_deployed_calibration_file():
    lock = load_lock(LOCK_PATH)
    yaml_text = ATTENTION_RUNTIME_YAML.read_text()
    rel = _extract_yaml_scalar(yaml_text, "calibration")
    expected_dest = f"LLMDet/{rel}"
    assert expected_dest in _lock_destinations(lock)


def test_lock_covers_the_deployed_detector_checkpoint():
    lock = load_lock(LOCK_PATH)
    yaml_text = ATTENTION_RUNTIME_YAML.read_text()
    rel = _extract_yaml_scalar(yaml_text, "checkpoint_path")
    expected_dest = f"LLMDet/{rel}"
    assert expected_dest in _lock_destinations(lock)
    # This one is trained directly on restricted footage -- must never be
    # silently marked public/redistribution-permitted.
    entry = next(a for a in lock["artifacts"] if a["destination"] == expected_dest)
    assert entry["status"] == "restricted"
    assert entry["redistribution"]["permitted"] != "yes"


@pytest.mark.parametrize(
    "class_marker",
    ["face_landmarker.task", "face_detection_full_range.tflite"],
)
def test_lock_covers_every_mediapipe_default_model_in_head_pose_py(class_marker):
    lock = load_lock(LOCK_PATH)
    text = HEAD_POSE_PY.read_text()
    m = re.search(
        rf'DEFAULT_MODEL\s*=\s*"\.\./huggingface/mediapipe/({re.escape(class_marker)})"', text
    )
    assert m, f"head_pose.py no longer references {class_marker!r} the way this test expects"
    expected_dest = f"huggingface/mediapipe/{class_marker}"
    assert expected_dest in _lock_destinations(lock)


def test_lock_covers_the_benchmark_only_mediapipe_asset():
    """blaze_face_short_range.tflite isn't in the live deployed path, but
    bench_face_backends.py (real, committed code) uses it -- still needs a
    lock entry so `research`/`hpc` profiles can reproduce the benchmark."""
    lock = load_lock(LOCK_PATH)
    text = BENCH_FACE_BACKENDS_PY.read_text()
    assert "blaze_face_short_range.tflite" in text
    assert "huggingface/mediapipe/blaze_face_short_range.tflite" in _lock_destinations(lock)


def test_lock_covers_clip_model_referenced_by_features_py():
    lock = load_lock(LOCK_PATH)
    ids = {a["id"] for a in lock["artifacts"]}
    assert "model.clip_vit_base_patch32" in ids
    features_py = (REPO / "LLMDet" / "attention" / "features.py").read_text()
    assert "openai/clip-vit-base-patch32" in features_py


def test_every_profile_referenced_by_an_artifact_is_declared():
    lock = load_lock(LOCK_PATH)
    declared = set(lock["profiles"])
    for a in lock["artifacts"]:
        for p in a.get("profiles", []):
            assert p in declared, f"{a['id']} references undeclared profile {p!r}"


def test_demo_profile_is_a_subset_of_every_other_profile_by_intent():
    """Not a strict subset requirement structurally, but every artifact the
    demo profile needs should also be usable (present in the superset
    profiles) so 'research'/'full-data'/'hpc' never regress something demo
    already guarantees."""
    lock = load_lock(LOCK_PATH)
    demo_ids = {a["id"] for a in lock["artifacts"] if "demo" in a.get("profiles", [])}
    for profile in ("research", "full-data", "hpc"):
        ids = {a["id"] for a in lock["artifacts"] if profile in a.get("profiles", [])}
        missing = demo_ids - ids
        assert not missing, f"profile {profile!r} is missing demo artifact(s): {missing}"


# ---------------------------------------------------------------------------
# 2. Checksum rejection of a corrupted file
# ---------------------------------------------------------------------------


def test_verify_artifact_rejects_wrong_sha256(tmp_path):
    good_content = b"the real bytes\n" * 100
    p = tmp_path / "thing.bin"
    p.write_bytes(good_content)
    artifact = {
        "bytes": len(good_content),
        "sha256": "0" * 64,  # deliberately wrong
    }
    result = verify_artifact(p, artifact)
    assert not result.ok
    assert "sha256 mismatch" in result.reason


def test_verify_artifact_rejects_corrupted_after_truncation(tmp_path):
    content = os.urandom(4096)
    p = tmp_path / "thing.bin"
    p.write_bytes(content)
    real_sha = sha256_file(p)
    artifact = {"bytes": len(content), "sha256": real_sha}
    assert verify_artifact(p, artifact).ok

    # Simulate corruption (e.g. a truncated/interrupted write that somehow
    # still landed at the destination).
    p.write_bytes(content[:-10])
    result = verify_artifact(p, artifact)
    assert not result.ok
    assert "size mismatch" in result.reason


def test_verify_artifact_accepts_correct_file(tmp_path):
    content = os.urandom(2048)
    p = tmp_path / "thing.bin"
    p.write_bytes(content)
    artifact = {"bytes": len(content), "sha256": sha256_file(p)}
    result = verify_artifact(p, artifact)
    assert result.ok
    assert result.actual_sha256 == artifact["sha256"]


def test_verify_artifact_never_claims_ok_without_a_sha256_on_record(tmp_path):
    """verified=false / sha256=null artifacts must never be reported as
    verified, even if a file of the right size happens to exist."""
    content = os.urandom(1024)
    p = tmp_path / "thing.bin"
    p.write_bytes(content)
    artifact = {"bytes": len(content), "sha256": None}
    result = verify_artifact(p, artifact)
    assert not result.ok


# ---------------------------------------------------------------------------
# 3. Interrupted-download resume (network fully mocked)
# ---------------------------------------------------------------------------


class _FakeResponse:
    def __init__(self, data: bytes, length):
        self._data = data
        self._pos = 0
        self.length = length

    def read(self, n=-1):
        if n < 0:
            n = len(self._data) - self._pos
        chunk = self._data[self._pos:self._pos + n]
        self._pos += len(chunk)
        return chunk

    def __enter__(self):
        return self

    def __exit__(self, *a):
        return False


def test_download_resume_sends_range_header_and_completes_file(tmp_path):
    full_content = b"0123456789" * 100_000  # 1,000,000 bytes
    already_have = full_content[:400_000]

    partial = tmp_path / "partial.bin"
    partial.write_bytes(already_have)

    captured_requests = []

    def fake_urlopen(req, timeout=None):
        captured_requests.append(req)
        rng = req.headers.get("Range")
        if rng:
            m = re.match(r"bytes=(\d+)-", rng)
            start = int(m.group(1))
            return _FakeResponse(full_content[start:], len(full_content) - start)
        return _FakeResponse(full_content, len(full_content))

    with mock.patch("urllib.request.urlopen", side_effect=fake_urlopen):
        result_path = download_to_temp(
            "https://example.invalid/file.bin",
            tmp_path,
            expected_bytes=len(full_content),
            resume_from=partial,
        )

    assert result_path.read_bytes() == full_content
    assert len(captured_requests) == 1
    assert captured_requests[0].headers.get("Range") == "bytes=400000-"


def test_download_without_prior_partial_sends_no_range_header(tmp_path):
    full_content = b"abcdef" * 1000

    def fake_urlopen(req, timeout=None):
        assert "Range" not in req.headers
        return _FakeResponse(full_content, len(full_content))

    with mock.patch("urllib.request.urlopen", side_effect=fake_urlopen):
        result_path = download_to_temp(
            "https://example.invalid/file.bin", tmp_path, expected_bytes=len(full_content)
        )
    assert result_path.read_bytes() == full_content


def test_download_network_error_raises_artifact_error_not_silent_failure(tmp_path):
    import urllib.error

    def fake_urlopen(req, timeout=None):
        raise urllib.error.URLError("simulated network failure")

    with mock.patch("urllib.request.urlopen", side_effect=fake_urlopen):
        with pytest.raises(ArtifactError):
            download_to_temp("https://example.invalid/file.bin", tmp_path)


# ---------------------------------------------------------------------------
# 4. Atomic installation
# ---------------------------------------------------------------------------


def test_atomic_install_moves_file_into_place(tmp_path):
    src = tmp_path / "src.bin"
    src.write_bytes(b"payload")
    dest = tmp_path / "nested" / "dir" / "dest.bin"
    atomic_install(src, dest)
    assert dest.read_bytes() == b"payload"
    assert not src.exists()  # os.replace moves, does not copy


def test_atomic_install_overwrites_existing_destination_atomically(tmp_path):
    dest = tmp_path / "dest.bin"
    dest.write_bytes(b"old content")
    src = tmp_path / "src.bin"
    src.write_bytes(b"new content")
    atomic_install(src, dest)
    assert dest.read_bytes() == b"new content"


def test_atomic_install_leaves_no_partial_file_visible_at_the_destination_name(tmp_path, monkeypatch):
    """Even if os.replace is simulated as failing (cross-device link, the
    real-world trigger for the fallback path), the destination must end up
    either fully old or fully new -- never a half-written file under the
    real destination name."""
    dest = tmp_path / "dest.bin"
    dest.write_bytes(b"old")
    src = tmp_path / "src.bin"
    src.write_bytes(b"brand new content, longer than old")

    real_replace = os.replace
    call_count = {"n": 0}

    def flaky_replace(a, b):
        call_count["n"] += 1
        if call_count["n"] == 1:
            raise OSError("simulated EXDEV (cross-device link)")
        return real_replace(a, b)

    monkeypatch.setattr(os, "replace", flaky_replace)
    atomic_install(src, dest)
    assert dest.read_bytes() == b"brand new content, longer than old"
    # no stray temp files left behind in the destination directory
    leftovers = [p for p in tmp_path.iterdir() if p.name not in ("dest.bin",)]
    assert leftovers == []


# ---------------------------------------------------------------------------
# 5. Safe archive extraction / path-traversal rejection
# ---------------------------------------------------------------------------


def test_safe_extract_zip_normal_case(tmp_path):
    zip_path = tmp_path / "good.zip"
    with zipfile.ZipFile(zip_path, "w") as zf:
        zf.writestr("a/b.txt", "hello")
        zf.writestr("c.txt", "world")
    dest = tmp_path / "out"
    safe_extract_zip(zip_path, dest)
    assert (dest / "a" / "b.txt").read_text() == "hello"
    assert (dest / "c.txt").read_text() == "world"


def test_safe_extract_zip_rejects_dotdot_traversal(tmp_path):
    zip_path = tmp_path / "evil.zip"
    with zipfile.ZipFile(zip_path, "w") as zf:
        zf.writestr("../../evil.txt", "pwned")
    dest = tmp_path / "out"
    dest.mkdir()
    with pytest.raises(PathTraversalError):
        safe_extract_zip(zip_path, dest)
    # Nothing should have been written outside dest, or even inside it.
    assert not (tmp_path / "evil.txt").exists()
    assert list(dest.iterdir()) == []


def test_safe_extract_zip_rejects_absolute_path_member(tmp_path):
    zip_path = tmp_path / "evil_abs.zip"
    with zipfile.ZipFile(zip_path, "w") as zf:
        zi = zipfile.ZipInfo("/etc/evil.txt")
        zf.writestr(zi, "pwned")
    dest = tmp_path / "out"
    dest.mkdir()
    with pytest.raises(PathTraversalError):
        safe_extract_zip(zip_path, dest)


def test_safe_extract_tar_normal_case(tmp_path):
    tar_path = tmp_path / "good.tar"
    inner = tmp_path / "inner.txt"
    inner.write_text("hello tar")
    with tarfile.open(tar_path, "w") as tf:
        tf.add(inner, arcname="a/inner.txt")
    dest = tmp_path / "out"
    safe_extract_tar(tar_path, dest)
    assert (dest / "a" / "inner.txt").read_text() == "hello tar"


def test_safe_extract_tar_rejects_dotdot_traversal(tmp_path):
    tar_path = tmp_path / "evil.tar"
    inner = tmp_path / "inner.txt"
    inner.write_text("pwned")
    with tarfile.open(tar_path, "w") as tf:
        tf.add(inner, arcname="../../evil.txt")
    dest = tmp_path / "out"
    dest.mkdir()
    with pytest.raises(PathTraversalError):
        safe_extract_tar(tar_path, dest)
    assert not (tmp_path / "evil.txt").exists()


def test_safe_extract_tar_rejects_symlink_escaping_destination(tmp_path):
    tar_path = tmp_path / "evil_symlink.tar"
    with tarfile.open(tar_path, "w") as tf:
        info = tarfile.TarInfo(name="link_to_outside")
        info.type = tarfile.SYMTYPE
        info.linkname = "../../../etc/passwd"
        tf.addfile(info)
    dest = tmp_path / "out"
    dest.mkdir()
    with pytest.raises(PathTraversalError):
        safe_extract_tar(tar_path, dest)


# ---------------------------------------------------------------------------
# 6. Offline behaviour
# ---------------------------------------------------------------------------


def _public_http_artifact(**overrides):
    art = {
        "id": "test.fixture",
        "status": "public",
        "bytes": 4,
        "sha256": "9f86d081884c7d659a2feaa0c55ad015a3bf4f1b2b0b822cd15d6c15b0f00a08",
        "source": {"kind": "http", "url": "https://example.invalid/fixture.bin"},
        "destination": "fixture.bin",
        "extract": {"method": "none"},
        "producer": None,
        "purpose": "test fixture",
        "license": {"spdx": "MIT"},
    }
    art.update(overrides)
    return art


def _lock_with(artifact, cache_root_dir):
    return {
        "cache_root_env": "LLMSTU_CACHE_ROOT_TEST_UNUSED",
        "cache_root_default": str(cache_root_dir),
        "artifacts": [artifact],
    }


def test_offline_flag_refuses_public_artifact_without_touching_network(tmp_path):
    artifact = _public_http_artifact()
    lock = _lock_with(artifact, tmp_path)

    with mock.patch("tools.artifacts.download.download_to_temp") as fake_dl:
        ok = _resolve_one(lock, artifact, offline=True, no_download=False, dry_run=False)
    assert ok is False
    fake_dl.assert_not_called()


def test_no_download_flag_refuses_public_artifact_without_touching_network(tmp_path):
    artifact = _public_http_artifact()
    lock = _lock_with(artifact, tmp_path)

    with mock.patch("tools.artifacts.download.download_to_temp") as fake_dl:
        ok = _resolve_one(lock, artifact, offline=False, no_download=True, dry_run=False)
    assert ok is False
    fake_dl.assert_not_called()


def test_restricted_status_never_fetched_even_online(tmp_path):
    """status=restricted must refuse regardless of offline/no_download."""
    artifact = _public_http_artifact(status="restricted")
    reason = _refuse_reason(artifact, offline=False, no_download=False)
    assert reason is not None
    assert "restricted" in reason


def test_manual_status_never_fetched_even_online(tmp_path):
    artifact = _public_http_artifact(status="manual", source={"kind": "unpublished"})
    reason = _refuse_reason(artifact, offline=False, no_download=False)
    assert reason is not None
    assert "manual" in reason


def test_public_status_with_network_available_has_no_refusal_reason():
    artifact = _public_http_artifact(status="public")
    assert _refuse_reason(artifact, offline=False, no_download=False) is None


def test_already_verified_file_short_circuits_without_network(tmp_path):
    """If the file is already present and correct, resolve must not touch
    the network at all -- verified-present is a strictly local check."""
    content = b"test"  # sha256 of this matches the fixture's hash above... compute for real:
    real_sha = sha256_file_from_bytes = __import__("hashlib").sha256(content).hexdigest()
    artifact = _public_http_artifact(sha256=real_sha, bytes=len(content))
    dest = tmp_path / artifact["destination"]
    dest.write_bytes(content)
    lock = _lock_with(artifact, tmp_path)

    with mock.patch("tools.artifacts.download.download_to_temp") as fake_dl:
        ok = _resolve_one(lock, artifact, offline=False, no_download=False, dry_run=False)
    assert ok is True
    fake_dl.assert_not_called()


# ---------------------------------------------------------------------------
# 7. Absence of hard-coded developer absolute paths
# ---------------------------------------------------------------------------


def test_scan_mechanism_detects_a_planted_hardcoded_path(tmp_path):
    bad = tmp_path / "bad.py"
    bad.write_text('ROOT = "/home/jovyan/Computer_vision/whatever"\n')
    hits = scan_for_forbidden_paths([bad])
    assert len(hits) == 1
    assert "bad.py" in hits[0]


def test_scan_mechanism_is_silent_on_a_clean_file(tmp_path):
    clean = tmp_path / "clean.py"
    clean.write_text('ROOT = Path(__file__).resolve().parents[2]\n')
    assert scan_for_forbidden_paths([clean]) == []


def test_scan_mechanism_reports_correct_line_number(tmp_path):
    bad = tmp_path / "bad.py"
    bad.write_text("line1\nline2\nROOT = '/home/jovyan/x'\nline4\n")
    hits = scan_for_forbidden_paths([bad])
    assert len(hits) == 1
    assert ":3:" in hits[0]


@pytest.mark.parametrize(
    "relpath",
    [
        "bootstrap.py",
        "pyproject.toml",
        "tools/artifacts/download.py",
        "tools/artifacts/upload.py",
        # tools/artifacts/__init__.py and tools/artifacts/ci_path_scan.py are
        # intentionally NOT in this list: both necessarily contain the
        # literal string "/home/jovyan" as data/documentation (the former
        # defines the forbidden-substring constant, the latter's docstring
        # names it), and are excluded by tools/artifacts/ci_path_scan.py's
        # own EXCLUDE_FILES for the same reason -- see
        # test_ci_path_scan_self_exclusions_are_exactly_the_self_referential_files
        # below for the check that keeps this in sync.
    ],
)
def test_new_reproducibility_tooling_has_no_hardcoded_developer_paths(relpath):
    """This task's own deliverables must not introduce the smell they exist
    to detect."""
    path = REPO / relpath
    assert path.exists(), f"expected deliverable missing: {relpath}"
    hits = scan_for_forbidden_paths([path])
    assert hits == [], f"{relpath} contains a hard-coded developer path: {hits}"


def test_ci_path_scan_self_exclusions_are_exactly_the_self_referential_files():
    """ci_path_scan.py's EXCLUDE_FILES must be exactly the files that
    genuinely need to contain the literal string as data/documentation --
    not a dumping ground for files someone didn't want to clean up."""
    import importlib.util

    spec = importlib.util.spec_from_file_location(
        "ci_path_scan", REPO / "tools" / "artifacts" / "ci_path_scan.py"
    )
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)

    assert set(mod.EXCLUDE_FILES) == {
        "tools/artifacts/__init__.py",
        "tools/artifacts/ci_path_scan.py",
    }
    for relpath in mod.EXCLUDE_FILES:
        path = REPO / relpath
        assert path.exists()
        hits = scan_for_forbidden_paths([path])
        assert hits, (
            f"{relpath} is in EXCLUDE_FILES but no longer actually contains the "
            f"forbidden substring -- the exclusion is stale and should be removed"
        )


def test_artifacts_lock_json_itself_has_no_hardcoded_developer_paths():
    hits = scan_for_forbidden_paths([LOCK_PATH])
    assert hits == [], f"artifacts.lock.json contains a hard-coded developer path: {hits}"


def test_ci_workflow_references_the_path_scan_script():
    """Loose coupling check: if ci_path_scan.py gets renamed, ci.yml should
    not silently stop calling it.

    The workflow lives at one of two paths. Its home is .github/workflows/ci.yml,
    but pushing there needs a Personal Access Token carrying the `workflow` scope,
    which this project's token lacks, so it is staged at ci/github-actions/ci.yml
    instead (see that directory's README for the activation steps). Accept either,
    and fail if it has gone missing altogether — the point of the test is that the
    scan keeps being invoked, not where the file sits.
    """
    candidates = [REPO / ".github" / "workflows" / "ci.yml",
                  REPO / "ci" / "github-actions" / "ci.yml"]
    found = [c for c in candidates if c.is_file()]
    assert found, f"CI workflow not found at any of {[str(c) for c in candidates]}"
    assert "ci_path_scan.py" in found[0].read_text()
