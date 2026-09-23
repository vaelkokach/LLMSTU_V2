#!/usr/bin/env python
"""Hugging Face Space entrypoint.

Fetches the artifacts the dashboard needs from a private model repo, then starts
`tools/dashboard/server.py` on the port Spaces routes to.

Why weights are fetched at runtime rather than baked into the image:

* `artifacts.lock.json` marks the detector and the LLMSTU corpus **restricted**,
  gated on an unresolved ethics/consent question. Baking them into image layers
  would put them in every copy of the Space and in its build cache, where they
  cannot be revoked. Fetching under a token keeps the access decision in one
  place — the model repo's own permissions.
* Temporal checkpoints are ~50 MB each and there are 13 deployable ones. Layers
  that large make every rebuild slow for no benefit, since they change far less
  often than the code.

Environment (set these as Space secrets/variables, not in the Dockerfile):

    HF_TOKEN         read token for the private artifact repo   (secret)
    ARTIFACT_REPO    e.g. "CHANGE_ME/your-dashboard-artifacts"    (variable)
    ARTIFACT_TYPE    "model" (default) or "dataset"             (variable)
    DASHBOARD_MODEL  registry id, default arch/mstcn_556_hp     (variable)
                     CPU phase-1 default, argued in README_DEPLOY.md: best
                     test macro-F1 among deployable variants and ~4.6x faster
                     to replay on CPU. The GPU Space (deploy/hf_space_live)
                     defaults to the DEPLOYED checkpoint instead, because
                     neither half of that argument applies there.
    SESSION          session dir name under sessions/, default 0325
"""

from __future__ import annotations

import os
import sys
from pathlib import Path

HOME = Path(__file__).resolve().parent
PORT = int(os.environ.get("PORT", "7860"))

# Where server.py expects things, relative to the repo root it is run from.
SESSIONS_DIR = HOME / "tools" / "dashboard" / "sessions"
WORKDIRS = HOME / "LLMDet" / "work_dirs"


def fetch_artifacts() -> bool:
    """Pull checkpoints + session cache from the private repo. True if usable."""
    repo = os.environ.get("ARTIFACT_REPO", "").strip()
    if not repo:
        print("[app] ARTIFACT_REPO not set — starting in replay-only mode "
              "(cue logs render; no model switching).", flush=True)
        return False

    token = os.environ.get("HF_TOKEN") or os.environ.get("HUGGING_FACE_HUB_TOKEN")
    if not token:
        print("[app] ARTIFACT_REPO is set but HF_TOKEN is not. A private repo "
              "cannot be read without it; add it as a Space secret.", flush=True)
        return False

    from huggingface_hub import snapshot_download

    print(f"[app] fetching artifacts from {repo} ...", flush=True)
    local = snapshot_download(
        repo_id=repo,
        repo_type=os.environ.get("ARTIFACT_TYPE", "model"),
        token=token,
        local_dir=str(HOME / "artifacts"),
    )
    print(f"[app] artifacts at {local}", flush=True)

    # The archive mirrors the repo layout, so link the two trees into place
    # rather than copying: a Space's disk is small and these are the big files.
    src = Path(local)
    for rel, dest in (("tools/dashboard/sessions", SESSIONS_DIR),
                      ("LLMDet/work_dirs", WORKDIRS)):
        s = src / rel
        if not s.exists():
            print(f"[app] note: {rel} not present in the artifact repo", flush=True)
            continue
        dest.parent.mkdir(parents=True, exist_ok=True)
        if dest.exists() or dest.is_symlink():
            continue
        try:
            dest.symlink_to(s, target_is_directory=True)
        except OSError:
            import shutil
            shutil.copytree(s, dest)
        print(f"[app] {rel} -> {dest}", flush=True)
    return True


def main() -> int:
    have_artifacts = fetch_artifacts()

    session = os.environ.get("SESSION", "0325")
    session_dir = SESSIONS_DIR / session
    model = os.environ.get("DASHBOARD_MODEL", "arch/mstcn_556_hp")

    argv = [sys.argv[0], "--host", "0.0.0.0", "--port", str(PORT)]
    if have_artifacts and session_dir.exists():
        argv += ["--session", str(session_dir), "--model", model]
        print(f"[app] session {session_dir} | model {model}", flush=True)
    else:
        # Falls back to a tracked cue log so the Space always renders something
        # rather than failing to boot. Model switching is unavailable here: a
        # recorded cue log stores decisions, not features.
        replay = HOME / "tools" / "dashboard" / "demo_session.jsonl"
        argv += ["--replay", str(replay)]
        print(f"[app] replay mode: {replay}", flush=True)

    sys.path.insert(0, str(HOME / "tools" / "dashboard"))
    sys.argv = argv
    os.chdir(HOME)

    import runpy
    runpy.run_path(str(HOME / "tools" / "dashboard" / "server.py"),
                   run_name="__main__")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
