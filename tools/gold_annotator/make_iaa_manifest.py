#!/usr/bin/env python3
"""Draw the shared subset that makes inter-annotator agreement measurable.

`[internal notes, not included]` threat C is single-annotator labelling, and the
second gold set did not close it. `Gold_annotation_wael` (1,000 crops) and
`event_gold_bundle/gold_annotations_Admin.jsonl` (984) overlap on **3 crops**,
because the candidate manifest was regenerated between the rounds. Two disjoint
samples give two independent ceilings and no kappa at all --
`compute_agreement.py` returns kappa = 1.000 over n = 2, which measures nothing.

Agreement needs the SAME crops labelled twice. This draws them.

Three decisions, each of which would invalidate the measurement if made the
other way:

**The manifest carries the PSEUDO-labels, not the first annotator's.** The gold
tool pre-fills whatever the manifest holds, so shipping wael's labels would make
the second pass a review of the first and the kappa a measure of how persuasive
the pre-fill is. Pseudo-labels are what the first pass saw, so the second pass
is drawn under identical conditions -- which is the comparison kappa is supposed
to be about. (The pre-fill remains an optimism caveat for BOTH passes; it is
recorded in `measure_ceiling.py` and does not change between them.)

**Stratified on the first annotator's cue class, not on the pseudo-label's.**
The point is to sample the human label space evenly, and `phone_use`,
`turned_to_peer` and `uncertain` are 3-5% of the corpus each. An unstratified
draw of 250 would land ~8 `uncertain` crops and the per-class agreement for the
classes the thesis most needs would rest on single digits. Stratifying on the
pseudo-label instead would oversample where the pseudo-labeller is confident,
which is exactly the wrong place.

**Every drawn crop is already in the first gold set.** Anything else cannot be
joined and is wasted annotation effort.

### The kappa this produces is an UPPER bound, and here is why

Both passes are pre-filled from the same pseudo-labels, and the first annotator
accepted the pre-filled value on **89.1% of the 250 x 6 categorical fields**
measured here (`pseudo_edited` was true on only 261 of their 1,000 crops). A
second annotator working from the same prefill will also accept most of them, so
some of the agreement measured is two people deferring to one model rather than
two people seeing the same thing.

That is the price of keeping the conditions identical to the first pass, and the
alternative is worse: annotating blind would measure a different task than the
one the first pass performed, and the ceiling in `measure_ceiling.py` carries
exactly the same caveat for exactly the same reason.

So report it as what it is -- **an upper bound on independent agreement, under
the same pre-fill conditions as the first pass**. ``--blind`` strips the label
fields from the manifest and bounds it from the other side; annotating both is
the only way to get a range, and it doubles the work.

    python tools/gold_annotator/make_iaa_manifest.py --n 250
    # then, as the second annotator:
    python tools/gold_annotator/serve.py \\
        --manifest tools/gold_annotator/gold_candidates_iaa.jsonl
    python tools/gold_annotator/compute_agreement.py \\
        Gold_annotation_wael/gold_annotations_wael.jsonl \\
        <the second annotator's gold_annotations_*.jsonl>
"""
from __future__ import annotations

import argparse
import json
import random
import sys
from collections import Counter, defaultdict
from pathlib import Path

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO / "LLMDet"))
sys.path.insert(0, str(Path(__file__).resolve().parent))

from attention.taxonomy import CUE_CLASSES, map_record          # noqa: E402


def load_jsonl(p: Path) -> list:
    return [json.loads(l) for l in p.open() if l.strip()]


def main() -> int:
    ap = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--first", type=Path,
                    default=REPO / "Gold_annotation_wael" / "gold_annotations_wael.jsonl",
                    help="the gold file already annotated")
    ap.add_argument("--pseudo", type=Path,
                    default=REPO / "grounding_data" / "llmstu_tools" / "outputs"
                            / "gold_candidates.jsonl",
                    help="the manifest the first pass worked from")
    ap.add_argument("--out", type=Path,
                    default=Path(__file__).parent / "gold_candidates_iaa.jsonl")
    ap.add_argument("--n", type=int, default=250,
                    help="target size; the draw is per-class and may fall short "
                         "for a class the first annotator rarely used")
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--blind", action="store_true",
                    help="strip the label fields so nothing is pre-filled. "
                         "Measures independent agreement rather than agreement "
                         "under a shared prefill -- a LOWER bound, and not the "
                         "condition the first pass ran under")
    args = ap.parse_args()

    first = [r for r in load_jsonl(args.first) if r.get("status") == "ok"]
    pseudo = {r["file_name"]: r for r in load_jsonl(args.pseudo)}

    # face_kpts is measured on the crop and is not collected by the gold tool;
    # the cue rules need it. Same join measure_ceiling.py makes.
    joinable = []
    for r in first:
        src = pseudo.get(r["file_name"])
        if src is None:
            continue
        rec = dict(r)
        rec.setdefault("face_kpts", src.get("face_kpts", 3))
        joinable.append(rec)

    if not joinable:
        print(f"none of {args.first.name} joins {args.pseudo.name} on file_name. "
              f"The first pass was annotated from a different manifest.",
              file=sys.stderr)
        return 2

    by_cue = defaultdict(list)
    for r in joinable:
        by_cue[CUE_CLASSES[map_record(r)]].append(r["file_name"])

    rng = random.Random(args.seed)
    per = max(1, args.n // len(CUE_CLASSES))
    picked: list[str] = []
    short = {}
    for cue in CUE_CLASSES:
        pool = sorted(by_cue.get(cue, []))
        rng.shuffle(pool)
        take = pool[:per]
        picked += take
        if len(take) < per:
            short[cue] = (len(take), per)

    # Spend whatever the rare classes could not fill on the common ones, so the
    # target size is met without under-sampling anything.
    if len(picked) < args.n:
        rest = sorted(set(r["file_name"] for r in joinable) - set(picked))
        rng.shuffle(rest)
        picked += rest[:args.n - len(picked)]

    picked = sorted(set(picked))
    rows = [pseudo[f] for f in picked]
    if args.blind:
        from vocab import ALL_LABEL_FIELDS
        # `caption` and `model_confidence` must go too. index.html:103 renders
        # the caption under the crop, and the captions STATE THE LABEL in plain
        # English -- "looking down at a mobile phone held in both hands",
        # "raising their hand", "writing on a desk". Stripping `activity` while
        # still displaying that is not blinding, it is blinding one field and
        # leaving the answer on screen.
        drop = set(ALL_LABEL_FIELDS) | {"caption", "model_confidence"}
        rows = [{k: v for k, v in r.items() if k not in drop} for r in rows]
        print(f"BLIND: stripped {len(drop)} fields (labels, caption, "
              f"model_confidence); nothing will be pre-filled or described.")
    args.out.write_text("".join(json.dumps(r) + "\n" for r in rows), encoding="utf-8")

    hist = Counter(CUE_CLASSES[map_record(
        {**pseudo[f], **{k: v for k, v in next(
            r for r in joinable if r["file_name"] == f).items()}})] for f in picked)
    print(f"first pass      : {len(first)} ok, {len(joinable)} joinable")
    print(f"drawn           : {len(rows)} crops, stratified on the FIRST "
          f"annotator's cue class (seed {args.seed})")
    for cue in CUE_CLASSES:
        print(f"  {cue:16}{hist.get(cue, 0):5d}   (available {len(by_cue.get(cue, [])):5d})")
    for cue, (got, want) in short.items():
        print(f"  note: {cue} could only supply {got} of {want}")
    print(f"\nwritten: {args.out}")
    print("BLIND: no pre-fill and no caption. This measures independent "
          "agreement -- a LOWER bound -- and is NOT the condition the first "
          "pass ran under." if args.blind else
          "It carries the PSEUDO-labels, so the second pass sees what the "
          "first saw. Agreement from it is an UPPER bound; see the module "
          "docstring.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
