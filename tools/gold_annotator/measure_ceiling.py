#!/usr/bin/env python3
"""How good are the pseudo-labels? The number every result in this project lacks.

Every macro-F1 in [internal notes, not included] is measured against Qwen3.5-27B's pseudo-labels, and
nobody has ever measured how well those agree with a human on the six cue
classes. Without it there is no denominator: [value removed] could be 60% or 95% of what
is achievable, and the two readings imply completely different next steps.

This scores the PSEUDO-LABELS AS IF THEY WERE A MODEL, against human gold, with
exactly the metric the models are scored with (``thesis_eval.metrics``), so the
two numbers sit on the same axis.

    python tools/gold_annotator/measure_ceiling.py \\
        --human tools/gold_annotator/gold_annotations_wael.jsonl \\
        --pseudo grounding_data/llmstu_tools/outputs/gold_candidates.jsonl \\
        --out outputs/label_ceiling.json

It reports the ceiling under BOTH cue rule versions. That is a model-free test
of the v2 repair: if dropping ``attention_target`` is right, pseudo-human
agreement should go UP, and it should go up without any training run being
involved. If it goes down, v2 is wrong and no amount of model tuning will hide
that.

Two properties of the gold set bound what may be claimed and are printed with
every number:

  * The annotation tool PRE-FILLS the pseudo-label, so the annotator is
    correcting rather than labelling blind. Agreement is therefore an
    OPTIMISTIC estimate of the pseudo-label's accuracy.
  * ``rejected`` items are dropped here, so the ceiling is conditional on the
    crop being readable. The reject rate is reported alongside, because the
    models are trained and scored on those crops too.
"""
from __future__ import annotations

import argparse
import json
import sys
from collections import Counter
from pathlib import Path

import numpy as np

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO / "LLMDet"))

from attention.taxonomy import CUE_CLASSES, RULESETS, map_record  # noqa: E402
from attention.thesis_eval import metrics as M  # noqa: E402


def load_jsonl(path: Path) -> list:
    return [json.loads(l) for l in path.open(encoding="utf-8") if l.strip()]


def bootstrap_macro_f1(y_true, y_pred, n_boot: int, seed: int) -> dict:
    """Percentile CI over items.

    Clustered on nothing: the gold sample is one crop per item, drawn
    stratified across videos, so an item-level resample is the honest unit
    here. If a future gold set has several crops per track, cluster on track.
    """
    rng = np.random.default_rng(seed)
    n = len(y_true)
    vals = []
    for _ in range(n_boot):
        idx = rng.integers(0, n, n)
        cm = M.confusion_matrix(y_true[idx], y_pred[idx], len(CUE_CLASSES))
        vals.append(M.macro_f1(cm))
    lo, hi = np.percentile(vals, [2.5, 97.5])
    return {"mean": float(np.mean(vals)), "ci95": [float(lo), float(hi)]}


#: Fields the cue rules read that the annotation tool does NOT collect, and so
#: must be recovered from the crop's own record. Currently just one.
MEASURED_FIELDS = ("face_kpts",)


def join_measured_fields(human: list, pseudo: dict) -> dict:
    """Copy measured-but-unannotated fields onto each human record, in place.

    ``face_kpts`` is how many face keypoints the detector found on the crop. It
    is a MEASUREMENT of the image, not an annotator judgement, and the gold tool
    never asks for it — so a human record arrives without it, ``map_record``
    falls back to its default of 3, and the ``uncertain`` gate
    (``occluded AND face_kpts <= 2``) can never fire on the human side.

    It fires on the pseudo side, which carries the field. Scoring one against the
    other with the gate live on only one of them does not measure the
    pseudo-labeller: it measures the asymmetry. Human ``uncertain`` is
    systematically under-counted and every pseudo ``uncertain`` over an occluded
    crop is scored as a false positive it did not commit.

    Taking the value from the pseudo record leaks nothing — it is the same
    detector output for the same crop that the corpus was built from, and it is
    not one of the fields the annotator could disagree with.

    Returns a report of how many labels the join actually moved, so the
    correction is visible rather than assumed.
    """
    from attention.taxonomy import CUE_CLASSES as _C
    before = [map_record(h, "v1") for h in human]
    filled = Counter()
    for h in human:
        src = pseudo.get(h["file_name"], {})
        for f in MEASURED_FIELDS:
            if f not in h and f in src:
                h[f] = src[f]
                filled[f] += 1
    after = [map_record(h, "v1") for h in human]
    moved = Counter(f"{_C[a]} -> {_C[b]}" for a, b in zip(before, after) if a != b)
    return {"filled": dict(filled), "n_labels_changed": int(sum(moved.values())),
            "transitions": dict(moved)}


def score(human: list, pseudo: dict, ruleset: str, n_boot: int, seed: int) -> dict:
    """Pseudo-labels scored against human gold under one cue rule version."""
    y_true = np.array([map_record(h, ruleset) for h in human], dtype=np.int64)
    y_pred = np.array([map_record(pseudo[h["file_name"]], ruleset) for h in human],
                      dtype=np.int64)
    cm = M.confusion_matrix(y_true, y_pred, len(CUE_CLASSES))
    prf = M.per_class_prf(cm)
    support = Counter(int(v) for v in y_true)
    return {
        "ruleset": ruleset,
        "n_items": int(len(y_true)),
        "accuracy": float(M.accuracy(cm)),
        "macro_f1": float(M.macro_f1(cm)),
        "balanced_accuracy": float(M.balanced_accuracy(cm)),
        "macro_f1_bootstrap": bootstrap_macro_f1(y_true, y_pred, n_boot, seed),
        "per_class": {
            c: {"precision": float(prf["precision"][i]),
                "recall": float(prf["recall"][i]),
                "f1": float(prf["f1"][i]),
                "support_human": support.get(i, 0)}
            for i, c in enumerate(CUE_CLASSES)},
        "confusion_matrix": cm.tolist(),
        "class_order": list(CUE_CLASSES),
    }


def field_agreement(human: list, pseudo: dict) -> dict:
    """Per-field exact agreement, for reading WHERE the pseudo-label is wrong."""
    from vocab import ALL_LABEL_FIELDS
    out = {}
    for f in ALL_LABEL_FIELDS:
        hits = sum(h.get(f) == pseudo[h["file_name"]].get(f) for h in human)
        out[f] = {"agreement": hits / max(len(human), 1), "n": len(human)}
    return out


def render(r: dict) -> str:
    o = [f"\n{'=' * 72}",
         f"LABEL CEILING -- pseudo-labels scored against human gold",
         f"{'=' * 72}",
         f"gold items matched : {r['n_matched']}",
         f"rejected as unusable by the annotator: {r['n_rejected']} "
         f"({r['reject_rate']:.1%}) -- excluded below, but the models are "
         f"trained and scored on them"]
    if r["n_unmatched_human"]:
        o.append(f"human items with no pseudo record: {r['n_unmatched_human']} "
                 f"(excluded)")
    o.append(f"\npseudo-label field-exact on all 10 fields: "
             f"{r['all_fields_exact']:.1%} of matched items")
    o.append(f"\n{'field':<22}{'agreement':>11}")
    for f, v in sorted(r["field_agreement"].items(), key=lambda kv: kv[1]["agreement"]):
        o.append(f"{f:<22}{v['agreement']:>10.1%}")

    for rs in RULESETS:
        s = r["by_ruleset"][rs]
        b = s["macro_f1_bootstrap"]
        o.append(f"\n--- ruleset {rs} ---")
        o.append(f"  THE CEILING: macro-F1 {s['macro_f1']:.4f}  "
                 f"95% CI [{b['ci95'][0]:.4f}, {b['ci95'][1]:.4f}]   "
                 f"accuracy {s['accuracy']:.4f}")
        o.append(f"  {'class':<20}{'P':>8}{'R':>8}{'F1':>8}{'support':>9}")
        for c in CUE_CLASSES:
            p = s["per_class"][c]
            o.append(f"  {c:<20}{p['precision']:>8.3f}{p['recall']:>8.3f}"
                     f"{p['f1']:>8.3f}{p['support_human']:>9}")

    d = (r["by_ruleset"]["v2"]["macro_f1"] - r["by_ruleset"]["v1"]["macro_f1"])
    o.append(f"\nv2 - v1 on pseudo-human agreement: {d:+.4f}")
    o.append("  This is a MODEL-FREE test of the rule repair. Positive means "
             "dropping\n  attention_target makes the pseudo-labels agree with a "
             "human better,\n  which is evidence for v2 that no training run is "
             "involved in.")
    o.append("\nHow to read the ceiling: a trained model's macro-F1 against the "
             "PSEUDO-\nlabels cannot be meaningfully compared to 1.0. Compare it "
             "to this number.\nCaveat: the tool pre-fills pseudo-labels, so this "
             "is an OPTIMISTIC estimate\nof pseudo-label accuracy, and it is "
             "conditional on the crop being readable.")
    return "\n".join(o)


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--human", type=Path, required=True,
                    help="gold_annotations_<name>.jsonl from serve.py")
    ap.add_argument("--pseudo", type=Path, required=True,
                    help="the manifest the annotator worked from, carrying the "
                         "original pseudo-label fields (gold_candidates.jsonl)")
    ap.add_argument("--out", type=Path, default=None)
    ap.add_argument("--n-boot", type=int, default=2000)
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()

    sys.path.insert(0, str(Path(__file__).resolve().parent))
    pseudo = {r["file_name"]: r for r in load_jsonl(args.pseudo)}
    raw = load_jsonl(args.human)
    rejected = [h for h in raw if h.get("status") == "rejected"]
    usable = [h for h in raw if h.get("status") not in ("rejected",)]
    unmatched = [h for h in usable if h["file_name"] not in pseudo]
    human = [h for h in usable if h["file_name"] in pseudo]
    join_report = join_measured_fields(human, pseudo)
    if not human:
        print("no human item joins to a pseudo record on file_name. The gold "
              "file was probably annotated from a different manifest than "
              "--pseudo; pass the manifest serve.py was launched with.",
              file=sys.stderr)
        return 2

    from vocab import ALL_LABEL_FIELDS
    exact = sum(all(h.get(f) == pseudo[h["file_name"]].get(f)
                    for f in ALL_LABEL_FIELDS) for h in human)

    report = {
        "n_matched": len(human),
        "n_rejected": len(rejected),
        "reject_rate": len(rejected) / max(len(raw), 1),
        "n_unmatched_human": len(unmatched),
        # What join_measured_fields recovered, and what it changed. Reported
        # because a correction nobody can see is indistinguishable from a bug.
        "measured_field_join": join_report,
        "all_fields_exact": exact / max(len(human), 1),
        "field_agreement": field_agreement(human, pseudo),
        "by_ruleset": {rs: score(human, pseudo, rs, args.n_boot, args.seed)
                       for rs in RULESETS},
        "human_file": str(args.human), "pseudo_file": str(args.pseudo),
        "caveats": [
            "the annotation tool pre-fills pseudo-labels, so agreement is an "
            "optimistic estimate of pseudo-label accuracy",
            "rejected crops are excluded, so the ceiling is conditional on the "
            "crop being readable",
            "one annotator unless a second gold file is scored separately; run "
            "compute_agreement.py for kappa before quoting this as THE ceiling",
            "face_kpts is joined from the pseudo manifest by file_name: the gold "
            "tool does not collect it, and without it the `uncertain` gate "
            "(occluded AND face_kpts <= 2) fires on the pseudo side only",
        ],
    }
    print(render(report))
    if args.out:
        args.out.parent.mkdir(parents=True, exist_ok=True)
        args.out.write_text(json.dumps(report, indent=2), encoding="utf-8")
        print(f"\nwrote {args.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
