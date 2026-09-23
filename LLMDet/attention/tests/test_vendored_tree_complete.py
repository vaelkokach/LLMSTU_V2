"""The repo must be self-contained and self-consistent in a fresh clone.

These do not test behaviour. They test that files which exist on a working
machine are actually *tracked*, because the repo has now been bitten three times
by the same thing: `.gitignore` carried an unanchored directory rule, git
patterns without a leading slash match at any depth, and source directories were
silently excluded from every clone:

    LLMDet/mmdet/datasets/                    (53 files)     `datasets/`
    LLMDet/configs/_base_/datasets/           (coco_detection.py)
    LLMDet/ram/data/                          (9 files)      `data/`

None was noticeable locally — on a machine that had once run the pipeline the
directories are simply there, untracked — so the first two surfaced only when a
Hugging Face Space built from the repo. One died at `import mmdet.datasets`, the
other at a FileNotFoundError for a `_base_` config, inside a background analysis
job, minutes into a GPU run.

The third never reached a clone: `test_no_unanchored_pattern_hides_vendored_source`
caught `data/` first. It would have thrown FileNotFoundError on
`ram/data/ram_tag_list.txt`, which `ram/models/{ram,ram_plus,tag2text}.py` read
at construction.

Cheap, offline, no imports of the heavy stack.
"""
from __future__ import annotations

import ast
from pathlib import Path

import pytest

REPO = Path(__file__).resolve().parents[3]
CONFIGS = REPO / "LLMDet" / "configs"
MMDET = REPO / "LLMDet" / "mmdet"

#: Directory names that occur inside source trees but never hold source. Editor
#: and tool droppings, so they are neither policed nor walked. `.ipynb_checkpoints`
#: matters most here: Jupyter mirrors any config opened in the browser as
#: `<name>-checkpoint.py`, keeping its `_base_` paths verbatim, so the copies
#: inherit from siblings that were never mirrored. They are untracked, they are
#: not what a clone builds from, and left in scope they fail
#: test_every_config_base_resolves for a reason that has nothing to do with what
#: it is guarding.
NEVER_SOURCE = {"__pycache__", ".pytest_cache", ".mypy_cache",
                ".ipynb_checkpoints", ".venv", "node_modules", ".git"}


def _is_source_path(path: Path) -> bool:
    return not (set(path.parts) & NEVER_SOURCE)


def _base_targets(cfg: Path):
    """The paths a config's `_base_ = ...` refers to, resolved."""
    try:
        tree = ast.parse(cfg.read_text(encoding="utf-8", errors="replace"))
    except SyntaxError:
        return []
    out = []
    for node in ast.walk(tree):
        if not (isinstance(node, ast.Assign)
                and any(getattr(t, "id", "") == "_base_" for t in node.targets)):
            continue
        value = node.value
        if isinstance(value, ast.Constant):
            items = [value]
        elif isinstance(value, (ast.List, ast.Tuple)):
            items = [e for e in value.elts if isinstance(e, ast.Constant)]
        else:
            continue
        out += [(cfg.parent / e.value).resolve()
                for e in items if isinstance(e.value, str)]
    return out


@pytest.mark.skipif(not CONFIGS.is_dir(), reason="configs tree absent")
def test_every_config_base_resolves():
    """No config may inherit from a file that is not in the repo."""
    missing = []
    for cfg in sorted(CONFIGS.rglob("*.py")):
        if not _is_source_path(cfg):
            continue
        for target in _base_targets(cfg):
            if not target.exists():
                missing.append(f"{cfg.relative_to(REPO).as_posix()} -> {target}")
    assert not missing, (
        "config(s) inherit from files missing in this clone:\n  "
        + "\n  ".join(missing)
        + "\n\nIf they exist on your machine but not in `git ls-files`, check "
          ".gitignore for an unanchored directory pattern."
    )


@pytest.mark.skipif(not MMDET.is_dir(), reason="vendored mmdet absent")
@pytest.mark.parametrize("sub", ["datasets", "models", "apis", "structures",
                                 "evaluation", "engine", "utils",
                                 "visualization"])
def test_mmdet_subpackage_present(sub):
    """register_all_modules() imports these; a missing one is ModuleNotFoundError."""
    pkg = MMDET / sub / "__init__.py"
    assert pkg.is_file(), (
        f"LLMDet/mmdet/{sub}/ is missing from this clone. It is imported by "
        f"mmdet.utils.setup_env.register_all_modules(), which the detector "
        f"calls on first use."
    )


RAM = REPO / "LLMDet" / "ram"

#: Read by `ram/models/{ram,ram_plus,tag2text}.py` as `{CONFIG_PATH}/data/...`
#: in their constructor defaults, so an absent file is a FileNotFoundError at
#: model construction rather than at import — later, and further from the cause.
RAM_DATA_FILES = ["__init__.py", "dataset.py", "randaugment.py", "utils.py",
                  "ram_tag_list.txt", "ram_tag_list_chinese.txt",
                  "ram_tag_list_threshold.txt", "tag_list.txt",
                  "tag2text_ori_tag_list.txt"]


@pytest.mark.skipif(not RAM.is_dir(), reason="vendored ram absent")
@pytest.mark.parametrize("name", RAM_DATA_FILES)
def test_ram_data_file_present(name):
    """`ram/data/` is vendored source and tag vocabulary, not a data folder.

    Its name is the entire problem: a `.gitignore` reading `data/` excludes it,
    and nothing complains until something constructs a RAM tagger.
    """
    assert (RAM / "data" / name).is_file(), (
        f"LLMDet/ram/data/{name} is missing from this clone. "
        f"ram_plus() opens the tag lists from this directory on construction; "
        f"check .gitignore for an unanchored `data/`."
    )


@pytest.mark.skipif(not (REPO / ".gitignore").is_file(), reason="no .gitignore")
def test_no_unanchored_pattern_hides_vendored_source():
    """A bare `foo/` rule matches at any depth — keep it away from source trees.

    Checked by name rather than by running git, so it holds in a checkout with
    no git available. Only directory names that actually occur inside the
    vendored trees are policed; unanchored rules for build noise are fine.
    """
    vendored_dirnames = set()
    for tree in (MMDET, CONFIGS, REPO / "LLMDet" / "llava", REPO / "LLMDet" / "ram"):
        if tree.is_dir():
            vendored_dirnames |= {d.name for d in tree.rglob("*")
                                  if d.is_dir() and _is_source_path(d)}

    offenders = []
    for i, raw in enumerate((REPO / ".gitignore").read_text(
            encoding="utf-8").splitlines(), start=1):
        line = raw.strip()
        if not line or line.startswith("#") or line.startswith("!"):
            continue
        if line.startswith("/") or "/" in line.rstrip("/"):
            continue                      # anchored, or already a path
        name = line.rstrip("/")
        if name in vendored_dirnames:
            offenders.append(f".gitignore:{i}: {line!r} also matches "
                             f"a directory inside a vendored tree")

    assert not offenders, "\n".join(offenders) + (
        "\n\nAnchor it (`/name/`) or spell the data path out in full."
    )


# --------------------------------------------------------------------------
# The deployed calibration is published twice, under two names, for two
# resolvers. Keep them from drifting apart.
# --------------------------------------------------------------------------

RUNTIME = REPO / "LLMDet" / "work_dirs" / "thesis" / "runtime"
DEPLOYED_CAL = RUNTIME / "mstcn_553_ff_thresholds.json"
PINNED_CAL = RUNTIME / "dashboard" / "ff_det__mstcn_553_facefound@s42.json"

#: Everything except the provenance key this repo adds to the copy.
_CAL_NUMBERS = ("temperature", "display_threshold", "alert_threshold",
                "display_coverage", "alert_coverage",
                "display_selective_accuracy", "alert_selective_accuracy",
                "fitted_on")


@pytest.mark.skipif(not (DEPLOYED_CAL.is_file() and PINNED_CAL.is_file()),
                    reason="calibration artifacts absent")
def test_pinned_calibration_matches_the_deployed_one():
    """The dashboard's pinned s42 entry must carry s42's own thresholds.

    attention_runtime.yaml deploys mstcn_553_ff_s42; the dashboard resolves
    calibration by variant id, so the same numbers are published under
    ff_det__mstcn_553_facefound@s42.json. If these two ever disagree, the live
    Space is abstaining at thresholds nobody fitted for the checkpoint it runs
    — the exact failure mode load_model() refuses to allow by default.
    """
    import json
    a = json.loads(DEPLOYED_CAL.read_text(encoding="utf-8"))
    b = json.loads(PINNED_CAL.read_text(encoding="utf-8"))
    differing = {k: (a.get(k), b.get(k)) for k in _CAL_NUMBERS
                 if a.get(k) != b.get(k)}
    assert not differing, (
        f"{PINNED_CAL.name} has drifted from {DEPLOYED_CAL.name}: {differing}")
    assert "mstcn_553_ff_s42" in b["fitted_on"], (
        f"the pinned calibration claims to be s42 but was fitted on "
        f"{b['fitted_on']}")
