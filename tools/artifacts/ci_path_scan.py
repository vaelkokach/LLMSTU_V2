#!/usr/bin/env python3
"""CI gate: no hard-coded '/home/jovyan' in committed runtime code/configs.

Scope (deliberately narrower than "every file containing the string",
matching [internal notes, not included] RISK 4's own classification
of "blocking" vs "non-blocking"):

  INCLUDED  -- code/config a reproducer would actually execute or load:
    LLMDet/attention/**/*.py   (excluding attention/tests, which are test
                                 fixtures, not runtime code)
    LLMDet/configs/**/*.py, **/*.yaml
    LLMDet/matching/**/*.py
    LLMDet/tools/**/*.py
    grounding_data/llmstu_tools/**/*.py
    tools/**/*.py              (dashboard + this artifacts tooling itself)
    bootstrap.py, pyproject.toml

  EXCLUDED on purpose:
    LLMDet/work_dirs/**        -- generated training-run logs/configs, not
                                   code a new user runs; force-added per
                                   .gitignore's own comment, not authored here
    LLMDet/mmdet/**, LLMDet/llava/**  -- vendored upstream trees, not ours to
                                   fix; a hardcoded path there is an upstream
                                   concern, not a Branch-C reproducibility gap
    *.md, *.log, *.json, *.jsonl, *.ipynb -- docs/logs/data, not executed code

RISK 4 already identified 6 files inside INCLUDED scope with hardcoded
`/home/jovyan/...` defaults (LLMDet/matching/evaluate_matching.py,
LLMDet/matching/run_matching_experiment.py, and four scripts under
grounding_data/llmstu_tools/). This script is expected to FAIL against
those until someone fixes them -- see the big comment at the top of
.github/workflows/ci.yml. Fixing them is out of scope for the
Reproducibility/Release agent (those files are owned by Branches A's
data-prep tooling), so this scan reports, it does not repair.
"""
from __future__ import annotations

import subprocess
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from tools.artifacts import FORBIDDEN_PATH_SUBSTRINGS, scan_for_forbidden_paths  # noqa: E402

REPO_ROOT = Path(__file__).resolve().parents[2]

INCLUDE_GLOBS = (
    "LLMDet/configs/*.py",
    "LLMDet/configs/*.yaml",
    "LLMDet/matching/*.py",
    "LLMDet/tools/*.py",
    "grounding_data/llmstu_tools/*.py",
    "tools/dashboard/*.py",
    "tools/artifacts/*.py",
    "bootstrap.py",
    "pyproject.toml",
)

EXCLUDE_DIR_PARTS = ("work_dirs", "mmdet", "llava", "__pycache__")

#: tools/artifacts/__init__.py DEFINES the forbidden-substring constant this
#: very scan uses, and this file's own module docstring above NAMES that same
#: literal string for documentation purposes -- both necessarily contain
#: "/home/jovyan" as data/documentation, not as a hardcoded path bug.
#: Excluded by name, not by pattern, so this stays a conscious, visible
#: exception rather than a silent blind spot.
EXCLUDE_FILES = ("tools/artifacts/__init__.py", "tools/artifacts/ci_path_scan.py")


def _git_tracked_files() -> list[Path]:
    out = subprocess.run(
        ["git", "ls-files"], cwd=REPO_ROOT, capture_output=True, text=True, check=True
    ).stdout
    return [REPO_ROOT / line for line in out.splitlines() if line]


def _matches_include(path: Path) -> bool:
    rel = path.relative_to(REPO_ROOT)
    if rel.as_posix() in EXCLUDE_FILES:
        return False
    if any(part in EXCLUDE_DIR_PARTS for part in rel.parts):
        return False
    if rel.parts[:2] == ("LLMDet", "attention") and "tests" not in rel.parts:
        return rel.suffix == ".py"
    for pattern in INCLUDE_GLOBS:
        if rel.match(pattern):
            return True
    return False


def main() -> int:
    tracked = _git_tracked_files()
    scoped = [p for p in tracked if p.is_file() and _matches_include(p)]
    hits = scan_for_forbidden_paths(scoped, FORBIDDEN_PATH_SUBSTRINGS)
    if hits:
        print(f"FAIL: {len(hits)} hard-coded path reference(s) found in "
              f"{len(scoped)} scanned runtime file(s):")
        for h in hits:
            print(f"  {h}")
        print(
            "\nSee [internal notes, not included] RISK 4 for the known set and the "
            "smallest fix for each. This CI job is expected to fail until those are fixed; "
            "it exists to stop the count from growing, not to claim it is already zero."
        )
        return 1
    print(f"OK: no hard-coded path references in {len(scoped)} scanned runtime file(s).")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
