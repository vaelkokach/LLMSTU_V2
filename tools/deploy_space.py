#!/usr/bin/env python
"""Push dashboard/app changes to the live Hugging Face Space.

Only the named files, never the whole tree. `upload_folder` over the repo root
would carry `grounding_data/` and `work_dirs/` — student-derived material that
`artifacts.lock.json` marks restricted — into a repo whose visibility is one
click away from public. Naming each file is the control.

    python tools/deploy_space.py tools/dashboard/server.py ...
    python tools/deploy_space.py --all      # every file the Space ships

The Space rebuilds on commit; `--wait` polls until it is RUNNING again.
"""
import argparse
import os
import subprocess
import sys
import time
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
SPACE = "CHANGE_ME/your-space"

#: Everything the image COPYs, as <repo path> -> <path in the Space>. app.py and
#: the Docker context live under deploy/hf_space_live/ here but at the root
#: there, which is exactly the mapping that is easy to get wrong by hand.
ROOT_FILES = {
    "deploy/hf_space_live/app.py": "app.py",
    "deploy/hf_space_live/Dockerfile": "Dockerfile",
    "deploy/hf_space_live/requirements.txt": "requirements.txt",
    "deploy/hf_space_live/README.md": "README.md",
}


def space_path(p):
    """Repo-relative path -> path inside the Space repo.

    ``as_posix``, not ``str``: on Windows ``str`` gives backslashes, which the
    Hub stores literally -- a push then creates a root file named
    ``tools\\dashboard\\server.py`` beside the real one, leaves the real one
    unchanged, and still reports success. It also meant no ROOT_FILES key could
    ever match, so app.py would have landed under ``deploy\\hf_space_live``.
    """
    rel = Path(p).resolve().relative_to(REPO).as_posix()
    return ROOT_FILES.get(rel, rel)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("files", nargs="*")
    ap.add_argument("--all", action="store_true",
                    help="every root file plus tools/dashboard")
    ap.add_argument("-m", "--message", default="")
    ap.add_argument("--wait", action="store_true")
    a = ap.parse_args()

    from huggingface_hub import CommitOperationAdd, HfApi
    api = HfApi()

    paths = [REPO / f for f in a.files]
    if a.all:
        paths += [REPO / f for f in ROOT_FILES]
        paths += sorted((REPO / "tools" / "dashboard").glob("*.py"))
        paths += [REPO / "tools" / "dashboard" / "index.html"]
    paths = sorted({p.resolve() for p in paths})
    if not paths:
        ap.error("nothing to push")

    # Compile every .py before pushing. A syntax error costs a full rebuild to
    # discover (the Space boots, crashes, and reports RUNTIME_ERROR minutes
    # later), and it is free to catch here.
    for p in paths:
        if p.suffix == ".py" and p.exists():
            try:
                compile(p.read_text(), str(p), "exec")
            except SyntaxError as e:
                sys.exit(f"refusing to push, {p.name} does not compile:\n"
                         f"  line {e.lineno}: {e.text and e.text.rstrip()}\n"
                         f"  {e.msg}")

    ops = []
    for p in paths:
        if not p.exists():
            sys.exit(f"missing: {p}")
        ops.append(CommitOperationAdd(path_in_repo=space_path(p),
                                      path_or_fileobj=str(p)))
        print(f"  {p.relative_to(REPO)} -> {ops[-1].path_in_repo}")

    msg = a.message or f"Update {len(ops)} file(s) from the working tree"
    try:
        sha = subprocess.run(["git", "rev-parse", "--short", "HEAD"], cwd=REPO,
                             capture_output=True, text=True).stdout.strip()
        if sha:
            msg += f" ({sha})"
    except Exception:
        pass
    api.create_commit(SPACE, repo_type="space", operations=ops,
                      commit_message=msg)
    print(f"pushed: {msg}")

    if a.wait:
        last = None
        for _ in range(60):
            raw = (api.space_info(SPACE).runtime.raw or {})
            st, hw = raw.get("stage"), (raw.get("hardware") or {}).get("current")
            if (st, hw) != last:
                print(f"  {time.strftime('%H:%M:%S')} {st} hw={hw}"
                      + (f"  ERR {raw.get('errorMessage')}"
                         if raw.get("errorMessage") else ""), flush=True)
                last = (st, hw)
            if st in ("RUNNING", "RUNTIME_ERROR", "BUILD_ERROR"):
                return 0 if st == "RUNNING" else 1
            time.sleep(20)
        print("timed out waiting")
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())
