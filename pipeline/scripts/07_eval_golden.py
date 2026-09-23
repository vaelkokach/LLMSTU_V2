#!/usr/bin/env python
"""Score the Qwen pseudo-labels against your hand-corrected golden.json.

Reads golden.json (exported from labeling/index.html), compares each field's
pseudo value to the golden value on the crops YOU reviewed, and reports per-field
accuracy + example disagreements.

  python scripts/07_eval_golden.py --golden labeling/golden.json

Only crops with status 'verified' or 'edited' count (i.e. ones you actually
looked at). 'verified' means the pseudo-label was already correct; 'edited' means
you changed it, so those are exactly where the pseudo-labeler was wrong.
"""
import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from llmstu import report, schema


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--golden", default="labeling/golden.json")
    ap.add_argument("--out", default="./work/eval")
    ap.add_argument("--all", action="store_true",
                    help="score every item, not just reviewed (verified/edited) ones")
    args = ap.parse_args()

    data = json.load(open(args.golden))
    items = data["items"]
    reviewed = items if args.all else [
        it for it in items if it.get("status") in ("verified", "edited")]
    if not reviewed:
        raise SystemExit("No reviewed crops found. Label some in labeling/index.html "
                         "(press 'v' to verify or edit fields), then Export golden.json.")

    # score enum + bool fields (skip free-text caption + model_confidence)
    fields = [f for f, s in schema.FIELDS.items() if s["type"] in ("enum", "bool")]
    scores, disagreements, accs = [], {}, []
    for f in fields:
        n = correct = 0
        bad = []
        for it in reviewed:
            g = it.get("golden", {})
            p = it.get("pseudo", {})
            if f not in g:
                continue
            n += 1
            if str(g[f]) == str(p.get(f)):
                correct += 1
            else:
                bad.append({"crop": it.get("crop_path", it.get("id")),
                            "pseudo": p.get(f), "golden": g[f]})
        acc = correct / n if n else 0.0
        scores.append({"field": f, "accuracy": acc, "n": n})
        disagreements[f] = bad
        if n:
            accs.append(acc)

    overall = sum(accs) / len(accs) if accs else 0.0
    meta = {"n_reviewed": len(reviewed), "overall": overall}

    # per-group (representative vs hard) overall accuracy, if groups are present
    def _group_overall(rows):
        gaccs = []
        for f in fields:
            pairs = [(it["pseudo"].get(f), it["golden"].get(f))
                     for it in rows if f in it.get("golden", {})]
            if pairs:
                gaccs.append(sum(str(p) == str(g) for p, g in pairs) / len(pairs))
        return (sum(gaccs) / len(gaccs)) if gaccs else 0.0

    groups = {}
    for gname in ("representative", "hard"):
        grows = [it for it in reviewed if it.get("group") == gname]
        if grows:
            groups[gname] = {"n": len(grows), "overall": _group_overall(grows)}
    meta["groups"] = groups

    # console summary
    print(f"\nGolden eval — {len(reviewed)} reviewed crops | overall macro-acc {overall*100:.1f}%")
    for gname, g in groups.items():
        print(f"  {gname:<15} {g['overall']*100:5.1f}%  (n={g['n']})")
    print(f"{'field':<20} {'acc':>6}  n")
    for s in scores:
        print(f"{s['field']:<20} {s['accuracy']*100:5.0f}% {s['n']:>4}")

    out = Path(args.out); out.mkdir(parents=True, exist_ok=True)
    (out / "golden_scores.json").write_text(json.dumps(
        {"meta": meta, "scores": scores, "disagreements": disagreements}, indent=1))
    rp = report.golden_report(scores, disagreements, meta, out / "golden_report.html")
    print(f"\n[eval] json  -> {out/'golden_scores.json'}")
    print(f"[eval] html  -> {rp}")


if __name__ == "__main__":
    main()
