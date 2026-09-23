"""Shared resolver/verifier core for artifacts.lock.json.

Used by ``bootstrap.py`` (repo root), ``tools/artifacts/download.py`` (the
CLI), ``tools/artifacts/upload.py`` (the publisher), and
``LLMDet/attention/tests/test_artifacts.py``.

Design constraints this module exists to satisfy (see docs/REPRODUCIBILITY.md
and BRANCH_C reproducibility brief):

* Network access is never implicit. Every function that can touch the network
  takes an explicit ``offline``/``allow_network`` argument; nothing in this
  module reaches out on import.
* Never silently substitute a different file for the one the lock file names,
  and never treat an unverified file as good. ``verify_artifact`` is the only
  function allowed to say "this file is fine", and it only ever does that by
  actually re-hashing bytes on disk.
* Extraction must reject path traversal (``..`` segments, absolute paths,
  symlinks pointing outside the destination) before writing anything.
* Credentials are read from environment variables only and must never be
  logged, printed, or embedded in an exception message.
"""

from __future__ import annotations

import hashlib
import json
import os
import shutil
import stat
import tarfile
import tempfile
import zipfile
from dataclasses import dataclass
from pathlib import Path
from typing import Callable, Dict, Iterable, List, Optional, Sequence, Tuple

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------

#: tools/artifacts/__init__.py -> tools/artifacts -> tools -> repo root
REPO_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_LOCK_PATH = REPO_ROOT / "artifacts.lock.json"

CACHE_ROOT_ENV_DEFAULT = "LLMSTU_CACHE_ROOT"

#: Substrings that must never appear in committed runtime code/configs.
#: Used by both the CI scan and test_artifacts.py.
FORBIDDEN_PATH_SUBSTRINGS: Tuple[str, ...] = ("/home/jovyan",)


class ArtifactError(Exception):
    """Base class for all artifact-tooling errors."""


class SchemaError(ArtifactError):
    """artifacts.lock.json does not conform to the expected schema."""


class ChecksumMismatch(ArtifactError):
    """A file's sha256 (or size) does not match the lock entry."""


class PathTraversalError(ArtifactError):
    """An archive member would write outside its destination directory."""


class UnresolvableArtifact(ArtifactError):
    """The artifact cannot be fetched automatically (manual/restricted, or
    offline with nothing cached)."""


# ---------------------------------------------------------------------------
# Schema
# ---------------------------------------------------------------------------

REQUIRED_TOP_KEYS = (
    "schema",
    "cache_root_env",
    "cache_root_default",
    "profiles",
    "artifacts",
)

REQUIRED_ARTIFACT_KEYS = (
    "id",
    "version",
    "purpose",
    "owner",
    "type",
    "source",
    "upstream",
    "license",
    "redistribution",
    "status",
    "bytes",
    "sha256",
    "verified",
    "destination",
    "profiles",
    "extract",
    "producer",
    "deprecated",
)

VALID_STATUS = {"public", "restricted", "manual"}
VALID_REDISTRIBUTION_PERMITTED = {"yes", "no", "pending_ethics_review"}
VALID_EXTRACT_METHODS = {"none", "zip", "tar"}


def validate_lock_schema(lock: dict) -> List[str]:
    """Return a list of human-readable problems; empty list means valid.

    Deliberately dependency-free (no ``jsonschema``) so this can run in a
    bare Python environment before any project dependency is installed --
    that is the whole point of ``bootstrap.py``.
    """
    problems: List[str] = []

    if not isinstance(lock, dict):
        return ["top level is not a JSON object"]

    for key in REQUIRED_TOP_KEYS:
        if key not in lock:
            problems.append(f"missing required top-level key: {key!r}")

    profiles = lock.get("profiles")
    profile_names = set()
    if isinstance(profiles, dict):
        profile_names = set(profiles.keys())
    elif profiles is not None:
        problems.append("'profiles' must be an object mapping profile name -> metadata")

    artifacts = lock.get("artifacts")
    if not isinstance(artifacts, list):
        problems.append("'artifacts' must be a list")
        artifacts = []

    seen_ids = set()
    for i, art in enumerate(artifacts):
        prefix = f"artifacts[{i}]"
        if not isinstance(art, dict):
            problems.append(f"{prefix} is not an object")
            continue

        for key in REQUIRED_ARTIFACT_KEYS:
            if key not in art:
                problems.append(f"{prefix} ({art.get('id', '?')}) missing key: {key!r}")

        art_id = art.get("id")
        if not art_id or not isinstance(art_id, str):
            problems.append(f"{prefix} has no valid string 'id'")
        elif art_id in seen_ids:
            problems.append(f"duplicate artifact id: {art_id!r}")
        else:
            seen_ids.add(art_id)

        status = art.get("status")
        if status is not None and status not in VALID_STATUS:
            problems.append(f"{prefix} ({art_id}) invalid status {status!r}, must be one of {sorted(VALID_STATUS)}")

        redis = art.get("redistribution")
        if isinstance(redis, dict):
            permitted = redis.get("permitted")
            if permitted is not None and permitted not in VALID_REDISTRIBUTION_PERMITTED:
                problems.append(
                    f"{prefix} ({art_id}) invalid redistribution.permitted {permitted!r}, "
                    f"must be one of {sorted(VALID_REDISTRIBUTION_PERMITTED)}"
                )
        elif "redistribution" in art:
            problems.append(f"{prefix} ({art_id}) 'redistribution' must be an object")

        sha256 = art.get("sha256")
        verified = art.get("verified")
        if verified is True and not sha256:
            problems.append(f"{prefix} ({art_id}) verified=true but sha256 is null/empty -- not allowed")
        if sha256 is not None and not isinstance(sha256, str):
            problems.append(f"{prefix} ({art_id}) sha256 must be a string or null")
        elif isinstance(sha256, str) and len(sha256) != 64:
            problems.append(f"{prefix} ({art_id}) sha256 must be 64 hex chars, got length {len(sha256)}")

        art_profiles = art.get("profiles")
        if isinstance(art_profiles, list):
            if profile_names:
                unknown = [p for p in art_profiles if p not in profile_names]
                if unknown:
                    problems.append(f"{prefix} ({art_id}) references undeclared profile(s): {unknown}")
        elif "profiles" in art:
            problems.append(f"{prefix} ({art_id}) 'profiles' must be a list")

        extract = art.get("extract")
        if isinstance(extract, dict):
            method = extract.get("method")
            if method is not None and method not in VALID_EXTRACT_METHODS:
                problems.append(f"{prefix} ({art_id}) invalid extract.method {method!r}")
        elif "extract" in art:
            problems.append(f"{prefix} ({art_id}) 'extract' must be an object")

        dest = art.get("destination")
        if isinstance(dest, str) and (".." in Path(dest).parts or Path(dest).is_absolute()):
            problems.append(f"{prefix} ({art_id}) destination looks unsafe (absolute or contains '..'): {dest!r}")

    return problems


def load_lock(path: Path = DEFAULT_LOCK_PATH) -> dict:
    """Load and schema-validate artifacts.lock.json. Raises SchemaError."""
    with open(path, "r", encoding="utf-8") as f:
        lock = json.load(f)
    problems = validate_lock_schema(lock)
    if problems:
        raise SchemaError(
            f"{path} failed schema validation ({len(problems)} problem(s)):\n  "
            + "\n  ".join(problems)
        )
    return lock


# ---------------------------------------------------------------------------
# Cache root / destination resolution
# ---------------------------------------------------------------------------


def cache_root(lock: dict, env: Optional[Dict[str, str]] = None) -> Path:
    """Resolve the cache root: env override, else the lock's own default.

    The default is the repository root itself (see the note in
    artifacts.lock.json) because existing runtime code hardcodes destination
    paths relative to the repo, e.g. LLMDet/attention/head_pose.py's
    ``DEFAULT_MODEL = "../huggingface/mediapipe/..."``.
    """
    env = os.environ if env is None else env
    env_name = lock.get("cache_root_env", CACHE_ROOT_ENV_DEFAULT)
    override = env.get(env_name)
    if override:
        return Path(override).expanduser().resolve()
    default = lock.get("cache_root_default", ".")
    root = (REPO_ROOT / default) if not Path(default).is_absolute() else Path(default)
    return root.resolve()


def destination_path(lock: dict, artifact: dict, env: Optional[Dict[str, str]] = None) -> Optional[Path]:
    dest = artifact.get("destination")
    if dest is None:
        return None
    return cache_root(lock, env) / dest


def artifacts_for_profile(lock: dict, profile: str) -> List[dict]:
    known = lock.get("profiles", {})
    if profile not in known:
        raise ArtifactError(f"unknown profile {profile!r}; known profiles: {sorted(known)}")
    return [a for a in lock["artifacts"] if profile in a.get("profiles", [])]


def find_artifact(lock: dict, artifact_id: str) -> dict:
    for a in lock["artifacts"]:
        if a["id"] == artifact_id:
            return a
    raise ArtifactError(f"no artifact with id {artifact_id!r}")


# ---------------------------------------------------------------------------
# Hashing / verification
# ---------------------------------------------------------------------------


def sha256_file(path: Path, chunk_size: int = 1 << 20) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        while True:
            chunk = f.read(chunk_size)
            if not chunk:
                break
            h.update(chunk)
    return h.hexdigest()


@dataclass
class VerifyResult:
    ok: bool
    reason: str
    actual_bytes: Optional[int] = None
    actual_sha256: Optional[str] = None


def verify_artifact(path: Path, artifact: dict) -> VerifyResult:
    """Verify a file already on disk against its lock entry.

    Never returns ok=True unless bytes AND sha256 (when the lock entry has a
    sha256 at all) actually match. If the lock entry itself has no sha256
    (verified=false, sha256=null), the strongest this function can say is
    "present and size-plausible" -- it will not claim the content is correct,
    because we were told not to invent a hash.
    """
    if not path.exists():
        return VerifyResult(False, "file does not exist")
    if path.is_dir():
        return VerifyResult(False, "path is a directory, not a file")

    actual_bytes = path.stat().st_size
    expected_bytes = artifact.get("bytes")
    if expected_bytes is not None and actual_bytes != expected_bytes:
        return VerifyResult(False, f"size mismatch: expected {expected_bytes}, got {actual_bytes}", actual_bytes)

    expected_sha256 = artifact.get("sha256")
    if not expected_sha256:
        return VerifyResult(
            False,
            "lock entry has no sha256 on record (verified=false) -- cannot confirm content, "
            "only that a file of the expected size is present",
            actual_bytes,
        )

    actual_sha256 = sha256_file(path)
    if actual_sha256 != expected_sha256:
        return VerifyResult(
            False,
            f"sha256 mismatch: expected {expected_sha256}, got {actual_sha256}",
            actual_bytes,
            actual_sha256,
        )
    return VerifyResult(True, "sha256 and size match", actual_bytes, actual_sha256)


# ---------------------------------------------------------------------------
# Safe extraction
# ---------------------------------------------------------------------------


def _is_within(base: Path, target: Path) -> bool:
    try:
        target.resolve().relative_to(base.resolve())
        return True
    except ValueError:
        return False


def safe_extract_zip(zip_path: Path, dest_dir: Path) -> None:
    dest_dir = dest_dir.resolve()
    with zipfile.ZipFile(zip_path) as zf:
        for member in zf.infolist():
            member_path = dest_dir / member.filename
            if not _is_within(dest_dir, member_path):
                raise PathTraversalError(
                    f"zip member {member.filename!r} would extract outside {dest_dir}"
                )
            # Reject absolute paths and any '..' segment explicitly too, in
            # case _is_within's resolve() is fooled by a not-yet-existing path.
            if Path(member.filename).is_absolute() or ".." in Path(member.filename).parts:
                raise PathTraversalError(f"zip member {member.filename!r} looks like a traversal attempt")
        # Only extract after every member has been checked.
        zf.extractall(dest_dir)


def safe_extract_tar(tar_path: Path, dest_dir: Path) -> None:
    dest_dir = dest_dir.resolve()
    with tarfile.open(tar_path) as tf:
        members = tf.getmembers()
        for member in members:
            if Path(member.name).is_absolute() or ".." in Path(member.name).parts:
                raise PathTraversalError(f"tar member {member.name!r} looks like a traversal attempt")
            member_path = dest_dir / member.name
            if not _is_within(dest_dir, member_path):
                raise PathTraversalError(f"tar member {member.name!r} would extract outside {dest_dir}")
            if member.issym() or member.islnk():
                link_target = (dest_dir / member.name).parent / member.linkname
                if not _is_within(dest_dir, link_target):
                    raise PathTraversalError(f"tar member {member.name!r} is a link escaping {dest_dir}")
        # Python 3.12+ tarfile supports filter="data"; on 3.11 we've already
        # done the equivalent checks by hand above, so extract unfiltered.
        try:
            tf.extractall(dest_dir, filter="data")  # type: ignore[call-arg]
        except TypeError:
            tf.extractall(dest_dir)


def safe_extract(archive_path: Path, dest_dir: Path, method: str) -> None:
    dest_dir.mkdir(parents=True, exist_ok=True)
    if method == "zip":
        safe_extract_zip(archive_path, dest_dir)
    elif method == "tar":
        safe_extract_tar(archive_path, dest_dir)
    elif method == "none":
        raise ArtifactError("safe_extract called with method='none'; nothing to extract")
    else:
        raise ArtifactError(f"unknown extract method: {method!r}")


# ---------------------------------------------------------------------------
# Atomic install
# ---------------------------------------------------------------------------


def atomic_install(tmp_path: Path, dest_path: Path) -> None:
    """Move tmp_path to dest_path atomically (same-filesystem rename).

    dest_path's parent directories are created as needed. If tmp_path and
    dest_path are on different filesystems, falls back to copy+fsync+rename
    of the copy (still atomic from the destination directory's point of view).
    """
    dest_path.parent.mkdir(parents=True, exist_ok=True)
    try:
        os.replace(tmp_path, dest_path)
    except OSError:
        # Different filesystems (e.g. EXDEV): os.replace can't do a plain
        # rename. Copy+fsync+rename-of-the-copy is still atomic from the
        # destination directory's point of view, and to match os.replace's
        # own move (not copy) semantics, the original tmp_path is removed
        # once its bytes are safely installed.
        fd, tmp_name = tempfile.mkstemp(dir=str(dest_path.parent))
        try:
            with os.fdopen(fd, "wb") as out_f, open(tmp_path, "rb") as in_f:
                shutil.copyfileobj(in_f, out_f)
                out_f.flush()
                os.fsync(out_f.fileno())
            os.replace(tmp_name, dest_path)
            tmp_path.unlink(missing_ok=True)
        finally:
            if os.path.exists(tmp_name):
                os.remove(tmp_name)


# ---------------------------------------------------------------------------
# Network download (resumable), only ever called with offline=False
# ---------------------------------------------------------------------------


def download_to_temp(
    url: str,
    tmp_dir: Path,
    expected_bytes: Optional[int] = None,
    progress_cb: Optional[Callable[[int, Optional[int]], None]] = None,
    timeout: float = 60.0,
    resume_from: Optional[Path] = None,
) -> Path:
    """Download url into a temp file inside tmp_dir, returning its path.

    Supports resuming an interrupted download: if ``resume_from`` points at a
    partial file, a Range request continues from its current size. Callers
    are responsible for verifying the result afterwards -- this function
    never claims correctness, only that bytes were written.
    """
    import urllib.request
    import urllib.error

    tmp_dir.mkdir(parents=True, exist_ok=True)
    fd, tmp_name = tempfile.mkstemp(dir=str(tmp_dir), suffix=".part")
    os.close(fd)
    tmp_path = Path(tmp_name)

    existing_bytes = 0
    if resume_from is not None and resume_from.exists():
        existing_bytes = resume_from.stat().st_size
        shutil.copyfile(resume_from, tmp_path)

    headers = {}
    if existing_bytes:
        headers["Range"] = f"bytes={existing_bytes}-"

    req = urllib.request.Request(url, headers=headers)
    mode = "ab" if existing_bytes else "wb"
    try:
        with urllib.request.urlopen(req, timeout=timeout) as resp:
            total = resp.length
            if total is not None and existing_bytes:
                total += existing_bytes
            written = existing_bytes
            with open(tmp_path, mode) as out_f:
                while True:
                    chunk = resp.read(1 << 20)
                    if not chunk:
                        break
                    out_f.write(chunk)
                    written += len(chunk)
                    if progress_cb:
                        progress_cb(written, total if total else expected_bytes)
    except urllib.error.URLError as e:
        raise ArtifactError(f"download failed for {url}: {e}") from e

    return tmp_path


# ---------------------------------------------------------------------------
# Hard-coded developer-path scan (used by CI and test_artifacts.py)
# ---------------------------------------------------------------------------


def scan_for_forbidden_paths(
    paths: Iterable[Path],
    forbidden: Sequence[str] = FORBIDDEN_PATH_SUBSTRINGS,
) -> List[str]:
    """Return "path:lineno: matched-substring" strings for every hit."""
    hits: List[str] = []
    for p in paths:
        try:
            text = p.read_text(errors="ignore")
        except (OSError, UnicodeDecodeError):
            continue
        for lineno, line in enumerate(text.splitlines(), start=1):
            for needle in forbidden:
                if needle in line:
                    hits.append(f"{p}:{lineno}: contains {needle!r}")
    return hits


# ---------------------------------------------------------------------------
# Credentials (env-var only, never echoed)
# ---------------------------------------------------------------------------

#: name -> env var, per upload backend. Kept in one place so download.py/
#: upload.py/tests agree on exactly which vars are credentials and must never
#: be printed.
CREDENTIAL_ENV_VARS: Dict[str, str] = {
    "github": "GITHUB_TOKEN",
    "hf": "HF_TOKEN",
    "zenodo": "ZENODO_TOKEN",
}


def get_credential(backend: str) -> str:
    """Read a credential from its environment variable. Raises if unset.

    Never returns a default, never logs the value, and the exception message
    below deliberately names only the *variable*, not any value.
    """
    env_var = CREDENTIAL_ENV_VARS.get(backend)
    if env_var is None:
        raise ArtifactError(f"unknown upload backend {backend!r}; known: {sorted(CREDENTIAL_ENV_VARS)}")
    value = os.environ.get(env_var)
    if not value:
        raise ArtifactError(
            f"backend {backend!r} needs credential env var {env_var} to be set; refusing to proceed without it"
        )
    return value


def redact(text: str, secrets: Iterable[str]) -> str:
    """Defence in depth: strip any known secret value out of a string before
    it is ever logged or included in an error message."""
    for s in secrets:
        if s:
            text = text.replace(s, "***REDACTED***")
    return text
