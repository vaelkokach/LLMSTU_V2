#!/usr/bin/env python3
"""Package the images the gold manifest names, so annotation can run off-HPC.

`serve.py` needs pixels, and the pixels live only on the HPC. This copies
exactly the crops the manifest lists -- not the corpus -- into a tarball whose
paths are relative to the repo root, so extracting it on another machine drops
them exactly where `serve.py` already looks (`grounding_data/LLMSTU/crops/`).

    python tools/gold_annotator/make_offline_bundle.py \\
        --manifest grounding_data/llmstu_tools/outputs/gold_candidates.jsonl \\
        --out ~/gold_offline_bundle.tar.gz

`--with-frames` adds each crop's full source frame, which is what the `f` key
shows for occlusion and context calls. Those are 2812x1050 and dominate the
size, so it is opt-in and the script prints both totals before building. They
are copied at NATIVE RESOLUTION and never downscaled: a resized frame is the
exact failure this project spent a branch diagnosing, and a human occlusion
call is not worth introducing a second copy of the corpus at the wrong scale.

The bundle contains student images. `grounding_data/` is gitignored on both
ends, so extracting into the repo cannot commit them by accident -- but the
tarball itself should live outside any repo and be deleted once the gold set
is back.
"""
from __future__ import annotations

import argparse
import json
import shutil
import sys
import tarfile
import tempfile
from pathlib import Path

REPO = Path(__file__).resolve().parents[2]
CROPS_ROOT = REPO / "grounding_data" / "LLMSTU" / "crops"
FRAMES_ROOT = REPO / "grounding_data" / "stu_img" / "frames"

#: The annotator itself, for --with-tool. Listed explicitly rather than globbed:
#: the previous hand-assembled bundle shipped __pycache__ and
#: .ipynb_checkpoints, which is noise at best and a stale second copy of the
#: vocabulary at worst.
TOOL_FILES = ("serve.py", "index.html", "vocab.py", "README.md",
              "compute_agreement.py", "finalize_gold.py")


def human(n: int) -> str:
    for unit in ("B", "KB", "MB", "GB"):
        if n < 1024 or unit == "GB":
            return f"{n:.1f} {unit}"
        n /= 1024.0


def plan(records: list, with_frames: bool):
    """(items, missing) where each item is (src, arcname).

    Missing files are collected rather than skipped: a bundle quietly short of
    200 crops would show up as blank images halfway through an annotation
    session, which is the worst possible time to find out.
    """
    items, missing, seen = [], [], set()
    for r in records:
        fn = r.get("file_name")
        if not fn:
            missing.append("<record with no file_name>")
            continue
        src = CROPS_ROOT / fn
        arc = f"grounding_data/LLMSTU/crops/{fn}"
        if arc in seen:
            continue
        seen.add(arc)
        (items if src.is_file() else missing).append((src, arc) if src.is_file() else str(src))

    if with_frames:
        for r in records:
            sf = r.get("src_frame")
            if not sf:
                continue
            src = FRAMES_ROOT / sf
            arc = f"grounding_data/stu_img/frames/{sf}"
            if arc in seen:
                continue
            seen.add(arc)
            (items if src.is_file() else missing).append(
                (src, arc) if src.is_file() else str(src))
    return items, missing


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--manifest", type=Path,
                    default=REPO / "grounding_data/llmstu_tools/outputs/gold_candidates.jsonl")
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--with-frames", action="store_true",
                    help="also include each crop's full source frame (the `f` "
                         "key view). Large -- see the size report.")
    ap.add_argument("--with-tool", action="store_true",
                    help="also include the annotator itself, so the bundle is "
                         "self-contained and needs only Python 3")
    ap.add_argument("--start-here", type=Path, default=None,
                    help="a README to place at the root of the bundle")
    ap.add_argument("--dry-run", action="store_true",
                    help="report sizes and missing files, write nothing")
    args = ap.parse_args()

    if not args.manifest.is_file():
        print(f"manifest not found: {args.manifest}", file=sys.stderr)
        return 2
    records = [json.loads(l) for l in args.manifest.open(encoding="utf-8") if l.strip()]
    print(f"manifest: {len(records)} records from {args.manifest}")

    crops, missing_c = plan(records, with_frames=False)
    crop_bytes = sum(s.stat().st_size for s, _ in crops)
    print(f"  crops   : {len(crops):>6} files, {human(crop_bytes)}")

    items, missing = crops, missing_c
    if args.with_frames:
        items, missing = plan(records, with_frames=True)
        frame_bytes = sum(s.stat().st_size for s, _ in items) - crop_bytes
        n_frames = len(items) - len(crops)
        print(f"  frames  : {n_frames:>6} files, {human(frame_bytes)}")
    print(f"  TOTAL   : {len(items):>6} files, "
          f"{human(sum(s.stat().st_size for s, _ in items))}")

    if missing:
        print(f"\n  MISSING : {len(missing)} files not on disk. The bundle would "
              f"be short and those items would render blank mid-session:",
              file=sys.stderr)
        for m in missing[:10]:
            print(f"      {m}", file=sys.stderr)
        if len(missing) > 10:
            print(f"      ... and {len(missing) - 10} more", file=sys.stderr)
        print("  Fix the paths or re-sample the manifest before bundling.",
              file=sys.stderr)
        return 3

    if args.dry_run:
        print("\ndry run: nothing written")
        return 0

    args.out = args.out.expanduser()
    args.out.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory() as td:
        # The manifest travels with the pixels so the receiving machine cannot
        # annotate one sample against another sample's images.
        staged = Path(td) / "grounding_data/llmstu_tools/outputs"
        staged.mkdir(parents=True)
        shutil.copy2(args.manifest, staged / "gold_candidates.jsonl")
        with tarfile.open(args.out, "w:gz") as tar:
            tar.add(staged / "gold_candidates.jsonl",
                    arcname="grounding_data/llmstu_tools/outputs/gold_candidates.jsonl")
            if args.with_tool:
                here = Path(__file__).resolve().parent
                for name in TOOL_FILES:
                    f = here / name
                    if f.exists():
                        tar.add(f, arcname=f"tools/gold_annotator/{name}")
                    else:
                        print(f"  note: {name} not found, omitted", file=sys.stderr)
            if args.start_here and args.start_here.exists():
                tar.add(args.start_here, arcname="START_HERE.md")
            for i, (src, arc) in enumerate(items, 1):
                tar.add(src, arcname=arc)
                if i % 250 == 0:
                    print(f"    {i}/{len(items)}", flush=True)

    print(f"\nwrote {args.out}  ({human(args.out.stat().st_size)})")
    print("Extract at the REPO ROOT on the other machine:")
    print(f"    tar -xzf {args.out.name} -C <repo root>")
    print("Then: cd tools/gold_annotator && python serve.py "
          "--manifest ../../grounding_data/llmstu_tools/outputs/gold_candidates.jsonl "
          "--annotator wael")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
