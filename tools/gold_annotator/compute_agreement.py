#!/usr/bin/env python3
"""Inter-annotator agreement: Cohen's kappa + percentage agreement per field.

Usage: python compute_agreement.py gold_annotations_A.jsonl gold_annotations_B.jsonl

Only items annotated by BOTH annotators are compared. Items rejected by either
annotator are excluded from the field-level comparison but their reject-decision
agreement is reported separately.
"""
import json
import sys
from collections import Counter

sys.path.insert(0, __file__.rsplit("/", 1)[0])
from vocab import ALL_LABEL_FIELDS  # noqa: E402


def load(path):
    out = {}
    with open(path) as fh:
        for line in fh:
            rec = json.loads(line)
            out[rec["file_name"]] = rec
    return out


def cohens_kappa(pairs):
    """pairs: list of (label_a, label_b). Returns (kappa, percent_agreement)."""
    n = len(pairs)
    if n == 0:
        return float("nan"), float("nan")
    po = sum(1 for a, b in pairs if a == b) / n
    ca = Counter(a for a, _ in pairs)
    cb = Counter(b for _, b in pairs)
    pe = sum(ca[k] * cb[k] for k in set(ca) | set(cb)) / (n * n)
    if pe == 1.0:
        return 1.0, po
    return (po - pe) / (1 - pe), po


def main():
    if len(sys.argv) != 3:
        raise SystemExit(__doc__)
    a, b = load(sys.argv[1]), load(sys.argv[2])
    common = sorted(set(a) & set(b))
    if not common:
        raise SystemExit("no overlapping file_names between the two annotation files")

    reject_pairs = [(a[f]["status"] == "rejected", b[f]["status"] == "rejected") for f in common]
    k_rej, po_rej = cohens_kappa(reject_pairs)

    usable = [f for f in common if a[f]["status"] != "rejected" and b[f]["status"] != "rejected"]
    print(f"overlapping items: {len(common)}   usable (neither rejected): {len(usable)}")
    print(f"reject decision:   kappa={k_rej:.3f}  agreement={po_rej:.1%}")
    print()
    print(f"{'field':<20} {'kappa':>7} {'agree':>8}   n")
    kappas = []
    for field in ALL_LABEL_FIELDS:
        pairs = [(a[f][field], b[f][field]) for f in usable]
        k, po = cohens_kappa(pairs)
        kappas.append(k)
        print(f"{field:<20} {k:>7.3f} {po:>7.1%}   {len(pairs)}")
    mean_k = sum(kappas) / len(kappas)
    print(f"\nmean kappa over {len(kappas)} fields: {mean_k:.3f}")
    if mean_k < 0.6:
        print("NOTE: kappa < 0.6 suggests the taxonomy is ambiguous for these fields "
              "— consider simplifying the label set before blaming the model (to-do item 9).")


if __name__ == "__main__":
    main()
