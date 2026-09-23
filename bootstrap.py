#!/usr/bin/env python3
"""One-command artifact bootstrap.

    python bootstrap.py --profile demo
    python bootstrap.py --profile research --dry-run
    python bootstrap.py --profile full-data --offline
    python bootstrap.py --profile hpc --no-download

This script does NOT install Python packages (see pyproject.toml /
``pip install -e .[demo]`` for that) and never runs pip. It only resolves,
verifies, and installs the model weights / checkpoints / config files that
``artifacts.lock.json`` says a profile needs, using the shared resolver in
``tools/artifacts``.

Contract (see docs/REPRODUCIBILITY.md for the human-facing version):
  * Prints size, source, and licence for every artifact before fetching it.
  * Downloads to a temp file first; only an artifact that passes size+sha256
    verification is ever installed at its real destination, and the install
    is atomic (os.replace).
  * Never substitutes a different file for the one named in the lock, and
    never proceeds with a file it could not verify.
  * --offline and --no-download both suppress every network access; the
    difference is intent (see tools/artifacts/download.py's module docstring).
  * status="restricted" and status="manual" artifacts are NEVER auto-fetched,
    with or without network access -- this script prints exactly why and
    what a human needs to do instead (obtain access, or run the recorded
    producer command).

Deliberately runnable with a bare ``python3`` and nothing pip-installed:
this module only imports the standard library plus ``tools.artifacts``,
which is itself stdlib-only.
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(REPO_ROOT))

from tools.artifacts import ArtifactError, DEFAULT_LOCK_PATH, load_lock  # noqa: E402
from tools.artifacts.download import cmd_resolve  # noqa: E402

PROFILES = ("demo", "research", "full-data", "hpc")


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument("--profile", required=True, choices=PROFILES)
    parser.add_argument("--lock", default=str(DEFAULT_LOCK_PATH))
    parser.add_argument("--dry-run", action="store_true",
                         help="print what would happen; fetch/install nothing")
    parser.add_argument("--offline", action="store_true",
                         help="never touch the network; report what is missing")
    parser.add_argument("--no-download", action="store_true",
                         help="same refusal as --offline, distinct intent (see module docstring)")
    args = parser.parse_args(argv)

    try:
        lock = load_lock(Path(args.lock))
    except ArtifactError as e:
        print(f"bootstrap: {e}", file=sys.stderr)
        print(
            "bootstrap: artifacts.lock.json failed to load/validate -- this is a hard stop. "
            "A broken lock file must never be treated as 'no artifacts needed'.",
            file=sys.stderr,
        )
        return 2

    print(f"LLMSTU bootstrap -- profile={args.profile!r}")
    print(f"Repository: {REPO_ROOT}")
    if args.offline or args.no_download:
        print("Network access: DISABLED for this run.")
    print()

    rc = cmd_resolve(args)

    print()
    if rc == 0:
        print(f"Profile {args.profile!r} is fully resolved. See docs/REPRODUCIBILITY.md for "
              f"exactly what this profile does and does not let you reproduce.")
    else:
        print(
            f"Profile {args.profile!r} is INCOMPLETE (see the gaps printed above). "
            f"This is expected for profiles that touch the restricted LLMSTU corpus on a "
            f"fresh clone -- see docs/REPRODUCIBILITY.md. It is NOT expected for the 'demo' "
            f"profile with network access and no --offline/--no-download; if that failed, "
            f"something is actually broken."
        )
    return rc


if __name__ == "__main__":
    raise SystemExit(main())
