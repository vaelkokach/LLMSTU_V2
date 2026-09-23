#!/usr/bin/env python3
"""Resolver CLI for artifacts.lock.json.

    python -m tools.artifacts.download list [--profile P]
    python -m tools.artifacts.download verify [--profile P] [--id ID]
    python -m tools.artifacts.download resolve --profile P [--dry-run] [--offline] [--no-download]
    python -m tools.artifacts.download repair --id ID [--dry-run] [--offline]
    python -m tools.artifacts.download offline --profile P

Every subcommand that could touch the network respects two independent
flags, on purpose:

  --offline       "I have no network right now." Never attempts a request;
                   reports what is missing instead of fetching it.
  --no-download   "I might have network, but I do not want you to use it
                   automatically." Same refusal as --offline, kept as a
                   separate flag because the intent is different (a CI box
                   with network but a policy against pulling large files
                   mid-job, vs. an actually air-gapped machine).

Artifacts whose lock entry has status="restricted" are NEVER auto-fetched by
either flag's absence -- see artifacts.lock.json's status_values note.
Artifacts with status="manual" are never auto-fetched either, because by
definition no fetch path exists for them yet (source.kind is "unpublished"
or "manual"); this CLI prints the producer command / manual instructions.
"""
from __future__ import annotations

import argparse
import shutil
import sys
from pathlib import Path
from typing import Optional

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from tools.artifacts import (  # noqa: E402
    ArtifactError,
    DEFAULT_LOCK_PATH,
    UnresolvableArtifact,
    artifacts_for_profile,
    atomic_install,
    cache_root,
    destination_path,
    download_to_temp,
    find_artifact,
    load_lock,
    safe_extract,
    validate_lock_schema,
    verify_artifact,
)


def _fmt_bytes(n: Optional[int]) -> str:
    if n is None:
        return "unknown size"
    for unit in ("B", "KB", "MB", "GB", "TB"):
        if n < 1024:
            return f"{n:.0f}{unit}"
        n /= 1024
    return f"{n:.1f}PB"


def _print_preflight(artifact: dict) -> None:
    """Print size/source/licence before any fetch, per bootstrap's contract."""
    src = artifact.get("source", {})
    lic = artifact.get("license", {})
    print(f"  id:      {artifact['id']}")
    print(f"  purpose: {artifact.get('purpose', '')[:100]}")
    print(f"  size:    {_fmt_bytes(artifact.get('bytes'))}")
    print(f"  source:  {src.get('kind')} {src.get('url') or src.get('repo_id') or ''}")
    print(f"  licence: {lic.get('spdx')}")
    print(f"  status:  {artifact.get('status')}")


def cmd_list(args: argparse.Namespace) -> int:
    lock = load_lock(Path(args.lock))
    artifacts = artifacts_for_profile(lock, args.profile) if args.profile else lock["artifacts"]
    root = cache_root(lock)
    print(f"cache_root: {root}")
    print(f"{'id':<45} {'status':<10} {'verified':<9} {'bytes':>12}  destination")
    for a in artifacts:
        dest = destination_path(lock, a) or Path("(managed by huggingface_hub cache)")
        present = dest.exists() if a.get("destination") else None
        marker = "" if present is None else ("[on disk]" if present else "[missing]")
        print(
            f"{a['id']:<45} {a['status']:<10} {str(a['verified']):<9} "
            f"{str(a.get('bytes') or '-'):>12}  {dest} {marker}"
        )
    return 0


def cmd_verify(args: argparse.Namespace) -> int:
    lock = load_lock(Path(args.lock))
    if args.id:
        artifacts = [find_artifact(lock, args.id)]
    elif args.profile:
        artifacts = artifacts_for_profile(lock, args.profile)
    else:
        artifacts = lock["artifacts"]

    failures = 0
    for a in artifacts:
        dest = destination_path(lock, a)
        if dest is None:
            print(f"SKIP  {a['id']} (no fixed destination -- managed externally)")
            continue
        result = verify_artifact(dest, a)
        status = "OK  " if result.ok else "FAIL"
        print(f"{status}  {a['id']}: {result.reason}")
        if not result.ok:
            failures += 1
    return 1 if failures else 0


def _refuse_reason(artifact: dict, offline: bool, no_download: bool) -> Optional[str]:
    if artifact["status"] == "restricted":
        return (
            "status=restricted: this artifact is gated by an unresolved ethics/consent "
            "question [internal notes, not included] RISK 1) and/or requires "
            "institutional data access. It is never auto-fetched, regardless of network "
            "availability. If you already hold access, place the file yourself at the "
            "destination shown above."
        )
    if artifact["status"] == "manual":
        return (
            "status=manual: no automatic fetch path exists for this artifact yet "
            f"(source.kind={artifact.get('source', {}).get('kind')!r}). "
            + (f"Producer: {artifact['producer']}" if artifact.get("producer") else "No producer command on record.")
        )
    if offline or no_download:
        return "offline/--no-download requested: not attempting a network fetch."
    return None


def _resolve_one(lock: dict, artifact: dict, offline: bool, no_download: bool, dry_run: bool) -> bool:
    """Returns True if the artifact ends up verified-present, False otherwise."""
    print("-" * 72)
    _print_preflight(artifact)

    dest = destination_path(lock, artifact)
    if dest is None:
        print("  -> no fixed destination (e.g. managed by huggingface_hub); nothing for "
              "this CLI to place. See destination_note in the lock entry.")
        return True

    if dest.exists():
        result = verify_artifact(dest, artifact)
        if result.ok:
            print(f"  -> already present and verified ({result.reason})")
            return True
        print(f"  -> present but INVALID: {result.reason}")
        if not artifact.get("sha256"):
            print("  -> lock entry has no sha256 on record; cannot auto-repair by re-verifying "
                  "content, only by re-fetching from source (if any) or accepting the risk.")

    refuse = _refuse_reason(artifact, offline, no_download)
    if refuse:
        print(f"  -> NOT FETCHING: {refuse}")
        return False

    src = artifact.get("source", {})
    url = src.get("url")
    if src.get("kind") != "http" or not url:
        print(f"  -> source.kind={src.get('kind')!r} has no automatic fetch implemented by this CLI "
              f"(huggingface_hub artifacts are pre-warmed by 'transformers'/'huggingface_hub' "
              f"itself, not by this script).")
        return False

    if dry_run:
        print(f"  -> DRY RUN: would download {url} -> {dest}")
        return False

    print(f"  -> downloading {url}")
    tmp_dir = dest.parent / ".artifact_tmp"
    tmp_path = download_to_temp(url, tmp_dir, expected_bytes=artifact.get("bytes"))
    try:
        result = verify_artifact(tmp_path, artifact)
        if not result.ok:
            raise ArtifactError(
                f"downloaded file for {artifact['id']} failed verification: {result.reason}. "
                f"Refusing to install it, and refusing to substitute any other file."
            )
        extract = artifact.get("extract", {"method": "none"})
        if extract.get("method", "none") != "none":
            extract_dir = dest if dest.suffix == "" else dest.parent
            safe_extract(tmp_path, extract_dir, extract["method"])
            tmp_path.unlink(missing_ok=True)
        else:
            atomic_install(tmp_path, dest)
        print(f"  -> installed and verified: {dest}")
        return True
    finally:
        if tmp_path.exists():
            tmp_path.unlink(missing_ok=True)
        if tmp_dir.exists() and not any(tmp_dir.iterdir()):
            tmp_dir.rmdir()


def cmd_resolve(args: argparse.Namespace) -> int:
    lock = load_lock(Path(args.lock))
    artifacts = artifacts_for_profile(lock, args.profile)
    print(f"Profile {args.profile!r}: {len(artifacts)} artifact(s). cache_root={cache_root(lock)}")
    ok_count = 0
    for a in artifacts:
        if _resolve_one(lock, a, args.offline, args.no_download, args.dry_run):
            ok_count += 1
    print("-" * 72)
    print(f"{ok_count}/{len(artifacts)} artifact(s) verified-present for profile {args.profile!r}.")
    if ok_count < len(artifacts) and not args.dry_run:
        print("Some artifacts are missing or gated. This is expected for restricted/manual "
              "artifacts on a fresh clone -- see docs/REPRODUCIBILITY.md for what each profile "
              "actually guarantees.")
    return 0 if (args.dry_run or ok_count == len(artifacts)) else 1


def cmd_repair(args: argparse.Namespace) -> int:
    lock = load_lock(Path(args.lock))
    artifact = find_artifact(lock, args.id)
    ok = _resolve_one(lock, artifact, args.offline, False, args.dry_run)
    return 0 if ok else 1


def cmd_offline(args: argparse.Namespace) -> int:
    """Report what --profile needs and what is already satisfiable offline."""
    lock = load_lock(Path(args.lock))
    artifacts = artifacts_for_profile(lock, args.profile)
    missing = []
    for a in artifacts:
        dest = destination_path(lock, a)
        if dest is None:
            continue
        result = verify_artifact(dest, a)
        if not result.ok:
            missing.append((a, result))
    if not missing:
        print(f"Profile {args.profile!r} is fully satisfied offline.")
        return 0
    print(f"Profile {args.profile!r} is MISSING {len(missing)} artifact(s) with no network access:")
    for a, result in missing:
        print(f"  - {a['id']}: {result.reason} (status={a['status']})")
    return 1


def cmd_validate_schema(args: argparse.Namespace) -> int:
    import json

    with open(args.lock, "r", encoding="utf-8") as f:
        lock = json.load(f)
    problems = validate_lock_schema(lock)
    if problems:
        print(f"{args.lock}: {len(problems)} schema problem(s):")
        for p in problems:
            print(f"  - {p}")
        return 1
    print(f"{args.lock}: schema OK ({len(lock['artifacts'])} artifacts).")
    return 0


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(prog="tools.artifacts.download", description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--lock", default=str(DEFAULT_LOCK_PATH), help="path to artifacts.lock.json")
    sub = p.add_subparsers(dest="command", required=True)

    sp = sub.add_parser("list", help="list artifacts and their on-disk status")
    sp.add_argument("--profile", default=None)
    sp.set_defaults(func=cmd_list)

    sp = sub.add_parser("verify", help="verify checksums of already-installed artifacts")
    sp.add_argument("--profile", default=None)
    sp.add_argument("--id", default=None)
    sp.set_defaults(func=cmd_verify)

    sp = sub.add_parser("resolve", help="fetch+verify+install every artifact a profile needs")
    sp.add_argument("--profile", required=True)
    sp.add_argument("--dry-run", action="store_true")
    sp.add_argument("--offline", action="store_true")
    sp.add_argument("--no-download", action="store_true")
    sp.set_defaults(func=cmd_resolve)

    sp = sub.add_parser("repair", help="re-fetch+re-verify a single artifact")
    sp.add_argument("--id", required=True)
    sp.add_argument("--dry-run", action="store_true")
    sp.add_argument("--offline", action="store_true")
    sp.set_defaults(func=cmd_repair)

    sp = sub.add_parser("offline", help="report what a profile is missing with no network")
    sp.add_argument("--profile", required=True)
    sp.set_defaults(func=cmd_offline)

    sp = sub.add_parser("validate-schema", help="validate artifacts.lock.json's own schema")
    sp.set_defaults(func=cmd_validate_schema)

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
