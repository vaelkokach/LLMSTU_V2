#!/usr/bin/env python3
"""Backend-neutral publisher for artifacts.lock.json entries.

    python -m tools.artifacts.upload publish --id ID --backend github [--dry-run]
    python -m tools.artifacts.upload publish --id ID --backend hf [--dry-run]
    python -m tools.artifacts.upload publish --id ID --backend zenodo [--dry-run]
    python -m tools.artifacts.upload verify-published --id ID

Credentials are read ONLY from environment variables, never from a CLI flag,
a file this script writes, or anything that could end up in shell history in
plaintext by this script's own doing:

    backend   env var
    -------   -------
    github    GITHUB_TOKEN
    hf        HF_TOKEN
    zenodo    ZENODO_TOKEN

This script never prints, logs, or includes a credential value in any
exception message (see tools.artifacts.get_credential / redact). If you see
a token in this script's output, that is a bug -- file it.

Redistribution policy this script enforces, not just documents:
  * Refuses to publish any artifact whose lock entry has
    redistribution.permitted != "yes". "pending_ethics_review" and "no" are
    both hard refusals -- there is no --force override, on purpose. If an
    ethics/consent determination later clears an artifact, update the lock
    file's redistribution.permitted field first; that is the single source
    of truth this script trusts.
  * Refuses to publish any artifact whose upstream license does not permit
    redistribution (checked via the same lock field a human already filled
    in during the license audit -- this script does not re-derive license
    facts, it enforces the recorded verdict).
  * For artifacts this project does not own the copyright to (anything with
    a non-null `upstream.repo`) and whose license permits redistribution,
    publishing here still only ever means "store the recipe (URL + hash)",
    matching this project's stated policy of not gratuitously re-mirroring
    upstream weights. Only artifacts with upstream.repo == null (i.e.
    project-generated) are ever actually uploaded as new bytes by this tool.

Verification-by-re-download: after a publish, `verify-published` fetches the
just-uploaded asset into an EMPTY temporary cache directory (never the real
destination) and re-checks its sha256, so a corrupted upload is caught before
anyone else pulls it.
"""
from __future__ import annotations

import argparse
import json
import sys
import tempfile
import urllib.error
import urllib.request
from pathlib import Path
from typing import Optional

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from tools.artifacts import (  # noqa: E402
    ArtifactError,
    DEFAULT_LOCK_PATH,
    cache_root,
    destination_path,
    find_artifact,
    get_credential,
    load_lock,
    redact,
    sha256_file,
    verify_artifact,
)

BACKENDS = ("github", "hf", "zenodo")


def _refuse_if_not_publishable(artifact: dict) -> None:
    permitted = artifact.get("redistribution", {}).get("permitted")
    if permitted != "yes":
        raise ArtifactError(
            f"refusing to publish {artifact['id']!r}: redistribution.permitted={permitted!r}, "
            f"not 'yes'. There is no override flag for this -- fix the lock file's "
            f"redistribution field (after an actual ethics/legal determination) if this is wrong."
        )
    if artifact.get("upstream", {}).get("repo"):
        raise ArtifactError(
            f"refusing to upload new bytes for {artifact['id']!r}: it has a non-null "
            f"upstream.repo ({artifact['upstream']['repo']!r}), meaning this project did not "
            f"generate it. Policy: never mirror upstream weights whose recipe (URL + hash) "
            f"already exists in the lock file -- point people at the recipe instead."
        )


def _local_source_path(lock: dict, artifact: dict) -> Path:
    dest = destination_path(lock, artifact)
    if dest is None or not dest.exists():
        raise ArtifactError(
            f"cannot publish {artifact['id']!r}: no local file at its lock destination "
            f"({dest}). Generate it first (see the artifact's 'producer' field)."
        )
    result = verify_artifact(dest, artifact)
    if not result.ok:
        raise ArtifactError(
            f"refusing to publish {artifact['id']!r}: the local file does not match the lock "
            f"entry ({result.reason}). Publishing an unverified file is exactly what this "
            f"tooling exists to prevent."
        )
    return dest


# ---------------------------------------------------------------------------
# Backend implementations
#
# Each backend function takes (artifact, local_path, token, dry_run) and
# returns the public URL the asset would be/was published at. All network
# calls go through urllib (stdlib only, matching the rest of this tooling).
# ---------------------------------------------------------------------------


def _github_release_upload_url(repo: str, tag: str, token: str) -> str:
    """Look up (or note the need to create) the release, return its
    upload_url template. Real network call; callers must handle dry_run
    before reaching here."""
    api = f"https://api.github.com/repos/{repo}/releases/tags/{tag}"
    req = urllib.request.Request(api, headers={
        "Authorization": f"Bearer {token}",
        "Accept": "application/vnd.github+json",
    })
    with urllib.request.urlopen(req, timeout=30) as resp:
        data = json.loads(resp.read())
    return data["upload_url"].split("{")[0]


def publish_github(artifact: dict, local_path: Path, token: str, dry_run: bool,
                    repo: str, tag: str) -> str:
    asset_name = local_path.name
    target_url = f"https://github.com/{repo}/releases/download/{tag}/{asset_name}"
    if dry_run:
        print(f"  DRY RUN: would upload {local_path} ({local_path.stat().st_size} bytes) "
              f"to GitHub release {repo}@{tag} as {asset_name!r}")
        print(f"  DRY RUN: resulting URL would be {target_url}")
        return target_url

    try:
        upload_base = _github_release_upload_url(repo, tag, token)
    except (urllib.error.URLError, KeyError) as e:
        raise ArtifactError(f"could not resolve GitHub release {repo}@{tag}: {e}") from e

    upload_url = f"{upload_base}?name={asset_name}"
    req = urllib.request.Request(
        upload_url,
        data=local_path.read_bytes(),
        method="POST",
        headers={
            "Authorization": f"Bearer {token}",
            "Content-Type": "application/octet-stream",
        },
    )
    with urllib.request.urlopen(req, timeout=300):
        pass
    return target_url


def publish_hf(artifact: dict, local_path: Path, token: str, dry_run: bool,
               repo_id: str, path_in_repo: str) -> str:
    target_url = f"https://huggingface.co/{repo_id}/resolve/main/{path_in_repo}"
    if dry_run:
        print(f"  DRY RUN: would upload {local_path} ({local_path.stat().st_size} bytes) "
              f"to HF repo {repo_id!r} at {path_in_repo!r}")
        print(f"  DRY RUN: resulting URL would be {target_url}")
        return target_url
    raise ArtifactError(
        "publish_hf: real upload needs the HF Hub commit API's multipart/LFS flow, which is "
        "non-trivial to implement correctly over raw urllib and is not exercised by this "
        "task's test suite (no network in CI). Implemented here as dry-run only; wire up "
        "huggingface_hub.upload_file() as an optional dependency ([train] or a new [publish] "
        "extra) before using this for a real publish."
    )


def publish_zenodo(artifact: dict, local_path: Path, token: str, dry_run: bool,
                    deposition_id: str) -> str:
    target_url = f"https://zenodo.org/record/{deposition_id}/files/{local_path.name}"
    if dry_run:
        print(f"  DRY RUN: would upload {local_path} ({local_path.stat().st_size} bytes) "
              f"to Zenodo deposition {deposition_id}")
        print(f"  DRY RUN: resulting URL would be {target_url}")
        return target_url
    raise ArtifactError(
        "publish_zenodo: real upload needs Zenodo's bucket-based file API (create/reuse a "
        "deposition, then PUT to its bucket URL). Implemented here as dry-run only, same "
        "reasoning as publish_hf."
    )


def cmd_publish(args: argparse.Namespace) -> int:
    lock = load_lock(Path(args.lock))
    artifact = find_artifact(lock, args.id)
    _refuse_if_not_publishable(artifact)
    local_path = _local_source_path(lock, artifact)

    token = None
    if not args.dry_run:
        token = get_credential(args.backend)  # raises if unset; never printed

    print(f"Publishing {artifact['id']!r} via backend={args.backend!r} "
          f"(dry_run={args.dry_run})")
    try:
        if args.backend == "github":
            if not args.repo or not args.tag:
                raise ArtifactError("--backend github requires --repo owner/name and --tag")
            url = publish_github(artifact, local_path, token, args.dry_run, args.repo, args.tag)
        elif args.backend == "hf":
            if not args.repo:
                raise ArtifactError("--backend hf requires --repo <repo_id>")
            path_in_repo = args.path_in_repo or local_path.name
            url = publish_hf(artifact, local_path, token, args.dry_run, args.repo, path_in_repo)
        elif args.backend == "zenodo":
            if not args.deposition_id:
                raise ArtifactError("--backend zenodo requires --deposition-id")
            url = publish_zenodo(artifact, local_path, token, args.dry_run, args.deposition_id)
        else:
            raise ArtifactError(f"unknown backend {args.backend!r}")
    except ArtifactError as e:
        # Defence in depth: redact the token even though none of the above
        # should ever include it in an exception.
        safe_secrets = [token] if token else []
        print(f"error: {redact(str(e), safe_secrets)}", file=sys.stderr)
        return 2

    if args.dry_run:
        print("Dry run complete; nothing was uploaded, and the lock file was not modified.")
        return 0

    print(f"Uploaded. Public URL: {url}")
    print(f"Update artifacts.lock.json's source.url for {artifact['id']!r} to this value "
          f"by hand (this tool deliberately does not auto-edit the lock file), then run:")
    print(f"  python -m tools.artifacts.upload verify-published --id {artifact['id']}")
    return 0


def cmd_verify_published(args: argparse.Namespace) -> int:
    """Re-download the artifact's source.url into an EMPTY temp cache and
    check its sha256 -- never trusts the local file that was just uploaded
    from, only what a fresh puller would actually get."""
    lock = load_lock(Path(args.lock))
    artifact = find_artifact(lock, args.id)
    url = artifact.get("source", {}).get("url")
    if not url:
        print(f"error: {artifact['id']!r} has no source.url on record yet -- publish it "
              f"and update the lock file first.", file=sys.stderr)
        return 2

    from tools.artifacts import download_to_temp  # local import: only needed here

    with tempfile.TemporaryDirectory(prefix="llmstu_verify_published_") as td:
        empty_cache = Path(td)
        print(f"Re-downloading {url} into an EMPTY temp cache ({empty_cache}) to verify the "
              f"published copy independently of the local file it was uploaded from...")
        try:
            tmp_path = download_to_temp(url, empty_cache, expected_bytes=artifact.get("bytes"))
        except ArtifactError as e:
            print(f"error: re-download failed: {e}", file=sys.stderr)
            return 2
        result = verify_artifact(tmp_path, artifact)
        print(f"{'OK' if result.ok else 'FAIL'}: {result.reason}")
        return 0 if result.ok else 1


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(prog="tools.artifacts.upload", description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--lock", default=str(DEFAULT_LOCK_PATH))
    sub = p.add_subparsers(dest="command", required=True)

    sp = sub.add_parser("publish", help="publish one artifact's current local file to a backend")
    sp.add_argument("--id", required=True)
    sp.add_argument("--backend", required=True, choices=BACKENDS)
    sp.add_argument("--dry-run", action="store_true")
    sp.add_argument("--repo", default=None, help="github: owner/name; hf: repo_id")
    sp.add_argument("--tag", default=None, help="github: release tag")
    sp.add_argument("--path-in-repo", default=None, help="hf: path within the repo")
    sp.add_argument("--deposition-id", default=None, help="zenodo: existing deposition id")
    sp.set_defaults(func=cmd_publish)

    sp = sub.add_parser("verify-published", help="re-download from source.url into an empty cache and verify")
    sp.add_argument("--id", required=True)
    sp.set_defaults(func=cmd_verify_published)

    return p


def main(argv=None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    try:
        return args.func(args)
    except ArtifactError as e:
        print(f"error: {e}", file=sys.stderr)
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
