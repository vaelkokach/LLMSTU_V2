"""What the Space must hold for every registry model to be selectable — and
whether a given copy of it actually does.

The Space fetches its weights at runtime from a private artifact repo
(``deploy/hf_space_live/app.py``), and the registry it then builds is derived
from whatever landed. That makes "are all the models in the Space the best ones,
and up to date?" a question about a remote repo — and the HPC has no outbound
network, so nothing here can answer it by asking the Hub.

Instead this emits the exact file list the Space needs, with sizes and sha256,
and verifies a local copy against it. Run ``--emit`` on the HPC, carry the JSON
to a machine with network, and run ``--verify`` against a snapshot of the repo.

Three failure modes it is meant to catch, all of which have a silent form:

**A model in the registry with no checkpoint in the Space.** The dropdown is
built from the run records that arrived, so a missing checkpoint usually means a
missing entry — the model is not offered and nothing says it was expected.

**A model with a checkpoint but no calibration.** ``session_replay.load_model``
refuses to run it (deliberately: an uncalibrated model never abstains and would
look more decisive than a calibrated one). That refusal surfaces when a visitor
picks it, not at boot. Ten of the twenty-four live-capable variants had no
calibration file until they were fitted.

**A stale registry snapshot.** ``runtime/model_registry.json`` is a derived
cache; if it is older than the sweeps it describes, it lists models that no
longer exist and omits ones that do.

    python tools/dashboard/artifact_manifest.py --emit artifacts_manifest.json
    python tools/dashboard/artifact_manifest.py --verify artifacts_manifest.json \
        --root /path/to/snapshot_of_the_artifact_repo

When the Hub is reachable, ``--check-remote`` answers the question directly
instead of via a carried manifest, and ``--push`` uploads whatever is missing as
one commit:

    python tools/dashboard/artifact_manifest.py --check-remote
    HF_TOKEN=hf_...write python tools/dashboard/artifact_manifest.py --push

Two tokens, deliberately
------------------------
Reading the repo and writing to it are different privileges, and every action
here except ``--push`` only reads. So:

``--check-remote``, and every read
    the ordinary resolution -- ``HF_TOKEN``, then the active profile from
    ``huggingface-cli login``. A read token is enough and is what should
    normally be active.

``--push``
    **``HF_TOKEN_W`` only**, falling back to ``HF_TOKEN`` if that is unset. The
    write token is never consulted by anything that does not write, so a routine
    ``--check-remote`` cannot be the thing that leaks or misuses it.

    HF_TOKEN_W=hf_...write python tools/dashboard/artifact_manifest.py --push

The token's role is checked BEFORE any upload starts. A read token otherwise
fails with a 403 naming the preupload endpoint, which reads like a permissions
problem with the repo rather than with the token -- and it fails after the files
have been hashed and staged, so the wait is wasted before the message is wrong.

``huggingface-cli auth list`` / ``auth switch`` store several named tokens and
choose which is active, but only one is active at a time. That is fine for reads
and the wrong shape for this: the env var keeps the write credential scoped to
the single command that needs it instead of making it the global default.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(Path(__file__).parent))

import model_registry as MR                                     # noqa: E402

#: Paths the Space needs that are not per-model. Each is repo-relative, exactly
#: as ``app.py`` links it into place.
FIXED = [
    ("LLMDet/work_dirs/thesis/runtime/model_registry.json",
     "derived registry snapshot"),
    ("LLMDet/configs/attention_runtime.yaml", "deployment config"),
]

#: Trees the Space needs whole. Listed rather than hashed file by file: the
#: detector is 4.29 GB and the session cache is ~900 JPEGs, so a per-file digest
#: would cost more than it tells anyone.
TREES = [
    ("tools/dashboard/sessions", "precomputed session caches"),
    ("huggingface", "detector text encoder + LMM + mediapipe (restricted)"),
]


def sha256(p: Path, chunk: int = 1 << 20) -> str:
    h = hashlib.sha256()
    with p.open("rb") as fh:
        for blk in iter(lambda: fh.read(chunk), b""):
            h.update(blk)
    return h.hexdigest()


def build(with_hashes: bool = True) -> dict:
    entries = MR.scan()
    files, models = [], []

    for e in entries:
        if not e.live_capable:
            # Not offered live, so the Space has no use for the weights. Recorded
            # anyway, as `needed: false`, so a reader can tell "deliberately
            # absent" from "lost".
            models.append({"variant_id": e.variant_id, "needed": False,
                           "reason": e.blocked_reason})
            continue
        cal = (REPO / "LLMDet" / "work_dirs" / "thesis" / "runtime" / "dashboard"
               / f"{e.variant_id.replace('/', '__')}.json")
        row = {
            "variant_id": e.variant_id, "needed": True,
            "taxonomy": e.taxonomy, "n_classes": e.n_classes,
            "coverage": e.coverage, "val_macro_f1": e.val.get("macro_f1"),
            "comparable_group": e.comparable_group,
            "replay_capable": e.replay_capable,
            "checkpoint": e.checkpoint,
            "calibration": str(cal.relative_to(REPO)),
            "calibration_present": cal.exists(),
        }
        models.append(row)
        for rel in (e.checkpoint, row["calibration"]):
            files.append(rel)

    # Every registry entry also needs its run_record and eval metrics, or it
    # does not appear in the Space's registry at all.
    for e in entries:
        if not e.live_capable:
            continue
        d = (REPO / e.checkpoint).parent.parent
        for rel in ("run_record.json", "eval_val/metrics.json"):
            files.append(str((d / rel).relative_to(REPO)))
    for rel, _ in FIXED:
        files.append(rel)

    out_files = []
    for rel in sorted(set(files)):
        p = REPO / rel
        rec = {"path": rel, "present": p.exists()}
        if p.exists():
            rec["bytes"] = p.stat().st_size
            if with_hashes:
                rec["sha256"] = sha256(p)
        out_files.append(rec)

    trees = []
    for rel, what in TREES:
        p = REPO / rel
        n = sum(1 for _ in p.rglob("*") if _.is_file()) if p.is_dir() else 0
        nbytes = sum(f.stat().st_size for f in p.rglob("*") if f.is_file()) \
            if p.is_dir() else 0
        trees.append({"path": rel, "what": what, "present": p.is_dir(),
                      "n_files": n, "bytes": nbytes})

    default = MR.default_entry(entries)
    return {
        "generated_by": "tools/dashboard/artifact_manifest.py",
        "registry_default": default.variant_id,
        "n_variants": len(entries),
        "n_live_capable": sum(1 for e in entries if e.live_capable),
        "models": models,
        "files": out_files,
        "trees": trees,
    }


def verify(man: dict, root: Path) -> int:
    """Check a copy of the artifact repo against the manifest. Returns exit code."""
    bad = 0
    print(f"verifying {root}")
    print(f"  manifest expects {man['n_live_capable']} live-capable models "
          f"of {man['n_variants']} variants; default {man['registry_default']}")

    missing_cal = [m for m in man["models"]
                   if m.get("needed") and not m.get("calibration_present")]
    if missing_cal:
        bad += len(missing_cal)
        print(f"\n  {len(missing_cal)} model(s) have NO calibration even at the "
              f"source. The Space will refuse to run these when selected:")
        for m in missing_cal:
            print(f"    {m['variant_id']}")
        print("    fix: python tools/dashboard/calibrate_registry.py")

    absent, changed = [], []
    for f in man["files"]:
        if not f.get("present"):
            continue                    # absent at the source; reported above
        p = root / f["path"]
        if not p.exists():
            absent.append(f["path"])
            continue
        if p.stat().st_size != f.get("bytes"):
            changed.append((f["path"], "size"))
        elif "sha256" in f and sha256(p) != f["sha256"]:
            changed.append((f["path"], "sha256"))

    if absent:
        bad += len(absent)
        print(f"\n  {len(absent)} file(s) MISSING from {root}:")
        for r in absent[:40]:
            print(f"    {r}")
        if len(absent) > 40:
            print(f"    ... and {len(absent) - 40} more")
    if changed:
        bad += len(changed)
        print(f"\n  {len(changed)} file(s) DIFFER:")
        for r, why in changed[:40]:
            print(f"    {r}  ({why})")

    for t in man["trees"]:
        p = root / t["path"]
        n = sum(1 for _ in p.rglob("*") if _.is_file()) if p.is_dir() else 0
        if n < t["n_files"]:
            bad += 1
            print(f"\n  tree {t['path']}: {n} files, manifest has "
                  f"{t['n_files']} ({t['what']})")

    print("\nOK — the copy matches the manifest" if not bad
          else f"\n{bad} problem(s). Upload the missing/differing paths.")
    return 1 if bad else 0


def read_token():
    """Token for read-only work. Never the write one unless it is all there is."""
    import os
    return (os.environ.get("HF_TOKEN")
            or os.environ.get("HUGGING_FACE_HUB_TOKEN") or None)


def write_token():
    """Token for ``--push``. HF_TOKEN_W first, so the write credential does not
    have to be the machine's default identity."""
    import os
    return (os.environ.get("HF_TOKEN_W")
            or os.environ.get("HF_WRITE_TOKEN")
            or os.environ.get("HF_TOKEN")
            or os.environ.get("HUGGING_FACE_HUB_TOKEN") or None)


def assert_can_write(token=None) -> str:
    """Fail before uploading if this credential cannot. Returns the user name.

    whoami reports the token's role, so the check costs one request and
    converts a post-staging 403 into a sentence naming the actual problem.
    """
    from huggingface_hub import HfApi
    try:
        who = HfApi().whoami(token=token)
    except Exception as e:
        raise SystemExit(
            f"could not authenticate to the Hub ({type(e).__name__}: "
            f"{str(e)[:120]}). Set HF_TOKEN_W to a write token, or run "
            f"`huggingface-cli login`.")
    role = (who.get("auth", {}).get("accessToken", {}) or {}).get("role")
    name = (who.get("auth", {}).get("accessToken", {}) or {}).get("displayName", "?")
    if role != "write":
        raise SystemExit(
            f"the token in use ({name!r}) has role {role!r}, not 'write', so the "
            f"upload would fail with a 403 on the preupload endpoint after "
            f"staging every file.\n"
            f"  Set HF_TOKEN_W to a token with WRITE access to the artifact "
            f"repo:\n"
            f"      HF_TOKEN_W=hf_...  python {Path(__file__).name} --push\n"
            f"  Note the write access must be on the MODEL REPO that holds the "
            f"artifacts, not on the Space.")
    return who.get("name", "?")


def _remote_files(repo: str, token=None) -> set:
    from huggingface_hub import HfApi
    return set(HfApi().list_repo_files(repo, repo_type="model", token=token))


def check_remote(man: dict, repo: str, token=None) -> tuple:
    """(missing paths, {variant: what it is missing}) against the live repo.

    The Space builds its registry from whatever is in this repo, so a model
    whose files never landed is simply not offered — no error, no empty entry,
    nothing that says it was expected. That is the failure this answers.
    """
    have = _remote_files(repo, token)
    missing = [f["path"] for f in man["files"]
               if f.get("present") and f["path"] not in have]
    bad = {}
    for m in man["models"]:
        if not m.get("needed"):
            continue
        gaps = [k for k in ("checkpoint", "calibration") if m[k] not in have]
        d = "/".join(m["checkpoint"].split("/")[:-2])
        gaps += [e for e in ("run_record.json", "eval_val/metrics.json")
                 if f"{d}/{e}" not in have]
        if gaps:
            bad[m["variant_id"]] = gaps
    return missing, bad


def report_remote(man: dict, repo: str, token=None) -> int:
    missing, bad = check_remote(man, repo, token)
    need = [m for m in man["models"] if m.get("needed")]
    print(f"{repo}")
    print(f"  live-capable models in this checkout : {len(need)}")
    print(f"  models the Space CANNOT offer        : {len(bad)}")
    if bad:
        print()
        for k, v in sorted(bad.items()):
            print(f"    {k:44} missing {', '.join(v)}")
    nbytes = sum(Path(REPO / p).stat().st_size for p in missing
                 if (REPO / p).exists())
    print(f"\n  {len(missing)} file(s) to upload, {nbytes / 1e6:.1f} MB")
    return 1 if bad else 0


def push(man: dict, repo: str, token=None, message=None) -> int:
    """Upload everything the repo is missing, as one commit."""
    from huggingface_hub import CommitOperationAdd, HfApi
    who = assert_can_write(token)          # before anything is staged
    print(f"authenticated as {who} with a write token")
    missing, bad = check_remote(man, repo, token)
    if not missing:
        print("nothing to upload — the repo already has every file")
        return 0
    nbytes = sum((REPO / p).stat().st_size for p in missing)
    print(f"uploading {len(missing)} files ({nbytes / 1e6:.1f} MB) to {repo}")
    ops = [CommitOperationAdd(path_in_repo=p, path_or_fileobj=str(REPO / p))
           for p in missing]
    info = HfApi().create_commit(
        repo_id=repo, repo_type="model", operations=ops, token=token,
        commit_message=message or (
            f"Add {len(bad)} model variant(s) the Space was missing"))
    print("committed:", getattr(info, "oid", info))
    return 0


def main() -> int:
    ap = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--emit", type=Path, help="write the manifest JSON")
    ap.add_argument("--verify", type=Path, help="a manifest to check against")
    ap.add_argument("--root", type=Path, default=REPO,
                    help="copy of the artifact repo to verify (default: this repo)")
    ap.add_argument("--no-hashes", action="store_true",
                    help="sizes only; much faster over 564 MB of checkpoints")
    ap.add_argument("--repo", default="CHANGE_ME/your-dashboard-artifacts",
                    help="the artifact repo --check-remote/--push talk to")
    ap.add_argument("--check-remote", action="store_true",
                    help="ask the Hub which models the Space is missing")
    ap.add_argument("--push", action="store_true",
                    help="upload the missing files as one commit (WRITE token)")
    args = ap.parse_args()

    if args.verify:
        return verify(json.loads(args.verify.read_text()), args.root)

    if args.check_remote or args.push:
        man = build(with_hashes=False)
        if args.push:
            return push(man, args.repo, write_token())
        return report_remote(man, args.repo, read_token())

    man = build(with_hashes=not args.no_hashes)
    print(f"registry default : {man['registry_default']}")
    print(f"variants         : {man['n_variants']} "
          f"({man['n_live_capable']} live-capable)")
    need = [m for m in man["models"] if m.get("needed")]
    nocal = [m for m in need if not m.get("calibration_present")]
    total = sum(f.get("bytes", 0) for f in man["files"])
    print(f"files the Space needs : {len(man['files'])}, {total / 1e6:.0f} MB")
    print(f"models with no calibration : {len(nocal)}"
          + (f" -> {[m['variant_id'] for m in nocal]}" if nocal else ""))
    for t in man["trees"]:
        print(f"tree {t['path']:32} {t['n_files']:6d} files "
              f"{t['bytes'] / 1e9:7.2f} GB "
              + ("" if t["present"] else "  MISSING"))
    print()
    print(f"{'variant':46} {'tax':17} {'cls':>3} {'cov':>5} {'cal':>4}")
    print("-" * 82)
    for m in need:
        print(f"{m['variant_id']:46} {m['taxonomy']:17} {m['n_classes']:3d} "
              f"{m['coverage']:5.3f} {'yes' if m['calibration_present'] else 'NO':>4}")

    if args.emit:
        args.emit.parent.mkdir(parents=True, exist_ok=True)
        args.emit.write_text(json.dumps(man, indent=2))
        print(f"\nwritten: {args.emit}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
