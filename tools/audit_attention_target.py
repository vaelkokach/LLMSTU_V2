#!/usr/bin/env python
"""Corpus-wide re-measurement of the ``attention_target`` defect.

Everything in ``taxonomy.RULESET_V2_RATIONALE`` was measured on the 1,000-crop
*stratified* gold-candidate sample, which oversamples rare activities and is
therefore not a corpus prevalence. This script re-runs the identical
measurements over the full label set so the numbers can be cited.

It reads labels and writes one JSON report. It does not touch sequences,
features, checkpoints or the split manifest, and it needs no GPU.

    python tools/audit_attention_target.py \
        --labels grounding_data/llmstu_tools/outputs/labels_tracked.jsonl \
        --out outputs/attention_target_audit.json

Optionally pass ``--human`` to run the same measurements over a human-annotated
file (``event_gold_bundle/gold_annotations_Admin.jsonl``). That answers the
question the pseudo-label audit cannot: is this a VLM artefact, or is it built
into the annotation vocabulary?
"""
from __future__ import annotations

import argparse
import json
import sys
from collections import Counter, defaultdict
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / "LLMDet"))

from attention.taxonomy import (  # noqa: E402
    CUE_CLASSES, CUE_TO_ID, candidate_set, cue_conditions, map_record,
)

#: The disjuncts of each v1 rule that reads ``attention_target``, so the report
#: says WHICH condition carried the class rather than only that the class fired.
DISJUNCTS = {
    "looking_away": {
        "gaze==away_or_window": lambda r: r.get("gaze_direction") == "away_or_window",
        "activity==looking_away": lambda r: r.get("activity") == "looking_away",
        "target==distracted": lambda r: r.get("attention_target") == "distracted",
    },
    "turned_to_peer": {
        "activity==talking_to_peer": lambda r: r.get("activity") == "talking_to_peer",
        "talking==True": lambda r: bool(r.get("talking")),
        "gaze==peer": lambda r: r.get("gaze_direction") == "peer",
        "target==peer": lambda r: r.get("attention_target") == "peer",
    },
}


def load(path: Path, status_ok_only: bool) -> list:
    """Read one jsonl of label records.

    ``status_ok_only`` keeps the human tool's ``status == "ok"`` rows. The
    human file also carries ``rejected`` rows (crops the annotator judged
    unusable); those have no meaningful cue label and are counted separately
    rather than mapped.
    """
    recs, rejected = [], 0
    with path.open(encoding="utf-8") as fh:
        for line in fh:
            line = line.strip()
            if not line:
                continue
            r = json.loads(line)
            if status_ok_only and r.get("status") not in (None, "ok"):
                rejected += 1
                continue
            recs.append(r)
    return recs, rejected


def target_given_gaze(recs: list) -> dict:
    """How much of ``attention_target`` is already determined by gaze?

    Reported as the accuracy of the best constant prediction per gaze value --
    i.e. one minus the irreducible error of the map gaze -> target. A high
    number means the field is a recoding, not an observation.
    """
    ct = defaultdict(Counter)
    for r in recs:
        ct[r.get("gaze_direction", "unknown")][r.get("attention_target", "unknown")] += 1
    total = majority = 0
    rows = {}
    for gaze, row in sorted(ct.items()):
        n = sum(row.values())
        top, top_n = row.most_common(1)[0]
        total += n
        majority += top_n
        rows[gaze] = {"n": n, "modal_target": top, "modal_share": top_n / n,
                      "counts": dict(row)}
    return {"predictable_share": majority / max(total, 1), "n": total, "by_gaze": rows}


def rule_drivers(recs: list, cue: str) -> dict:
    """Which disjunct of ``cue``'s v1 rule actually fires, and alone how often."""
    parts = DISJUNCTS[cue]
    fired = [r for r in recs if dict(cue_conditions(r, "v1"))[cue]]
    each, alone = Counter(), Counter()
    for r in fired:
        hits = [k for k, fn in parts.items() if fn(r)]
        each.update(hits)
        alone["+".join(sorted(hits))] += 1
    return {
        "n_fired": len(fired),
        "share_of_corpus": len(fired) / max(len(recs), 1),
        "by_disjunct": {k: {"n": v, "share_of_fired": v / max(len(fired), 1)}
                        for k, v in each.most_common()},
        "by_exact_combination": {k: v for k, v in alone.most_common(12)},
    }


def distracted_alone(recs: list) -> dict:
    """The subpopulation the repair is about.

    Records where the v1 ``looking_away`` rule fires ONLY because
    ``attention_target == distracted`` -- no gaze evidence, no activity
    evidence. Their gaze/activity composition says what they physically are,
    and their v1 label says where precedence actually sends them.
    """
    sub = [r for r in recs
           if r.get("attention_target") == "distracted"
           and r.get("gaze_direction") != "away_or_window"
           and r.get("activity") != "looking_away"]
    n = max(len(sub), 1)
    return {
        "n": len(sub),
        "share_of_corpus": len(sub) / max(len(recs), 1),
        "gaze_direction": {k: v / n for k, v in
                           Counter(r.get("gaze_direction") for r in sub).most_common()},
        "activity": {k: v / n for k, v in
                     Counter(r.get("activity") for r in sub).most_common()},
        "v1_label_after_precedence": {k: v / n for k, v in
                                      Counter(CUE_CLASSES[map_record(r, "v1")]
                                              for r in sub).most_common()},
    }


def ruleset_stats(recs: list, ruleset: str) -> dict:
    """Everything that characterises one rule version's target."""
    labels = [map_record(r, ruleset) for r in recs]
    cands = [candidate_set(r, ruleset) for r in recs]
    n = max(len(recs), 1)
    hist = Counter(labels)

    hd = CUE_TO_ID["head_down"]
    la = CUE_TO_ID["looking_away"]
    hd_rows = [c for lab, c in zip(labels, cands) if lab == hd]
    hd_amb = sum(la in c for c in hd_rows)

    la_rows = [r for r, lab in zip(recs, labels) if lab == la]
    la_pure = sum(r.get("gaze_direction") == "away_or_window" for r in la_rows)

    true_by_rule = sum(dict(cue_conditions(r, ruleset))["looking_away"] for r in recs)
    as_label = hist[la]

    return {
        "class_counts": {CUE_CLASSES[i]: hist[i] for i in range(len(CUE_CLASSES))},
        "class_shares": {CUE_CLASSES[i]: hist[i] / n for i in range(len(CUE_CLASSES))},
        "multi_fire_rate": sum(len(c) > 1 for c in cands) / n,
        "head_down_ambiguous_with_looking_away": {
            "n_head_down": len(hd_rows), "n_ambiguous": hd_amb,
            "rate": hd_amb / max(len(hd_rows), 1)},
        "looking_away_purity": {
            "n_labelled": len(la_rows), "n_gaze_away_or_window": la_pure,
            "rate": la_pure / max(len(la_rows), 1)},
        "looking_away_undercrediting": {
            "true_by_rule": true_by_rule, "allowed_to_be_label": as_label,
            "ratio": true_by_rule / max(as_label, 1)},
    }


def compare(recs: list) -> dict:
    v1 = [map_record(r, "v1") for r in recs]
    v2 = [map_record(r, "v2") for r in recs]
    moved = Counter((CUE_CLASSES[a], CUE_CLASSES[b])
                    for a, b in zip(v1, v2) if a != b)
    n = max(len(recs), 1)
    return {
        "n_records": len(recs),
        "n_label_changed": sum(a != b for a, b in zip(v1, v2)),
        "label_change_rate": sum(a != b for a, b in zip(v1, v2)) / n,
        "transitions_v1_to_v2": {f"{a} -> {b}": c for (a, b), c in moved.most_common()},
    }


def audit(recs: list, rejected: int) -> dict:
    return {
        "n_records": len(recs),
        "n_rejected_unusable": rejected,
        "rejected_rate": rejected / max(len(recs) + rejected, 1),
        "attention_target_given_gaze": target_given_gaze(recs),
        "rule_drivers": {c: rule_drivers(recs, c) for c in DISJUNCTS},
        "looking_away_from_distracted_alone": distracted_alone(recs),
        "v1": ruleset_stats(recs, "v1"),
        "v2": ruleset_stats(recs, "v2"),
        "v1_vs_v2": compare(recs),
    }


def render(name: str, a: dict) -> str:
    out = [f"\n{'=' * 70}", f"{name}: {a['n_records']:,} records"]
    if a["n_rejected_unusable"]:
        out.append(f"  rejected as unusable by the annotator: "
                   f"{a['n_rejected_unusable']:,} ({a['rejected_rate']:.1%})")

    tg = a["attention_target_given_gaze"]
    out.append(f"\nattention_target predictable from gaze_direction alone: "
               f"{tg['predictable_share']:.1%}")
    out.append(f"  {'gaze':<20}{'n':>9}  modal attention_target")
    for gaze, row in sorted(tg["by_gaze"].items(), key=lambda kv: -kv[1]["n"]):
        out.append(f"  {gaze:<20}{row['n']:>9}  {row['modal_target']:<12}"
                   f" {row['modal_share']:.1%}")

    for cue, rd in a["rule_drivers"].items():
        out.append(f"\nv1 {cue} rule fires on {rd['n_fired']:,} "
                   f"({rd['share_of_corpus']:.1%}); driven by:")
        for k, v in rd["by_disjunct"].items():
            out.append(f"  {k:<28}{v['n']:>9}  {v['share_of_fired']:.1%} of fired")

    da = a["looking_away_from_distracted_alone"]
    out.append(f"\nlooking_away fired by target==distracted ALONE: {da['n']:,} "
               f"({da['share_of_corpus']:.1%} of corpus)")
    for field in ("gaze_direction", "activity", "v1_label_after_precedence"):
        top = list(da[field].items())[:4]
        out.append(f"  {field:<28}" + ", ".join(f"{k} {v:.1%}" for k, v in top))

    out.append(f"\n{'metric':<44}{'v1':>13}{'v2':>13}")
    for label, path in (
        ("multi-fire rate", ("multi_fire_rate",)),
        ("head_down ambiguous w/ looking_away",
         ("head_down_ambiguous_with_looking_away", "rate")),
        ("looking_away purity (gaze==away_or_window)",
         ("looking_away_purity", "rate")),
    ):
        vals = []
        for v in ("v1", "v2"):
            x = a[v]
            for k in path:
                x = x[k]
            vals.append(x)
        out.append(f"{label:<44}{vals[0]:>12.1%}{vals[1]:>13.1%}")
    r1 = a["v1"]["looking_away_undercrediting"]
    r2 = a["v2"]["looking_away_undercrediting"]
    out.append(f"{'looking_away under-crediting ratio':<44}"
               f"{r1['ratio']:>12.2f}x{r2['ratio']:>12.2f}x")

    out.append(f"\n{'class':<20}{'v1':>10}{'v2':>10}{'delta':>10}")
    for c in CUE_CLASSES:
        n1 = a["v1"]["class_counts"][c]
        n2 = a["v2"]["class_counts"][c]
        out.append(f"{c:<20}{n1:>10,}{n2:>10,}{n2 - n1:>+10,}")
    cmp_ = a["v1_vs_v2"]
    out.append(f"\nhard labels changed by the repair: {cmp_['n_label_changed']:,}"
               f" / {cmp_['n_records']:,} = {cmp_['label_change_rate']:.2%}")
    for k, v in list(cmp_["transitions_v1_to_v2"].items())[:8]:
        out.append(f"  {k:<44}{v:>9,}")
    return "\n".join(out)


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--labels", type=Path, required=True,
                    help="pseudo-label jsonl (labels_tracked.jsonl) or a directory "
                         "of shard_*.jsonl")
    ap.add_argument("--human", type=Path, default=None,
                    help="human-annotated jsonl to run the same audit over")
    ap.add_argument("--out", type=Path, default=None, help="write the JSON report here")
    args = ap.parse_args()

    paths = (sorted(args.labels.glob("*.jsonl")) if args.labels.is_dir()
             else [args.labels])
    if not paths:
        print(f"no jsonl found at {args.labels}", file=sys.stderr)
        return 2
    recs, rejected = [], 0
    for p in paths:
        r, rej = load(p, status_ok_only=True)
        recs.extend(r)
        rejected += rej
    if not recs:
        print(f"no records parsed from {args.labels}", file=sys.stderr)
        return 2

    report = {"pseudo": audit(recs, rejected),
              "source": {"labels": str(args.labels), "n_files": len(paths)}}
    print(render(f"PSEUDO-LABELS  {args.labels}", report["pseudo"]))

    if args.human:
        hrecs, hrej = load(args.human, status_ok_only=True)
        report["human"] = audit(hrecs, hrej)
        report["source"]["human"] = str(args.human)
        print(render(f"HUMAN LABELS  {args.human}", report["human"]))
        print("\nIf the two agree on `gaze==down -> distracted`, the defect is in "
              "the annotation vocabulary, not in the pseudo-labeller.")
        print("Caveat: the annotation tool PRE-FILLS pseudo-labels, so human "
              "agreement is partly acceptance rather than independent judgement.")

    if args.out:
        args.out.parent.mkdir(parents=True, exist_ok=True)
        args.out.write_text(json.dumps(report, indent=2), encoding="utf-8")
        print(f"\nwrote {args.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
