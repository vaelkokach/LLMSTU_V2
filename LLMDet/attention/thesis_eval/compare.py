"""Paired video-level bootstrap between any two prediction archives.

Used for comparisons that cross sweep roots — most importantly the architecture
question (temporal transformer vs MS-TCN vs ASRF at an identical feature block),
which the within-sweep contrast machinery in ``aggregate.py`` cannot express.

The archives must be frame-aligned (same split, same sequence order), which is
guaranteed because ``run_eval`` emits sequences in manifest order; the tool
verifies it and refuses otherwise rather than silently comparing misaligned
rows.

    python -m attention.thesis_eval.compare \
        --a "asrf_556=work_dirs/thesis/arch/asrf_556_hp_s42/eval_val/predictions.npz" \
        --b "transformer_556=work_dirs/thesis/ladder/transformer_556_hp_s42/eval_val/predictions.npz" \
        --out work_dirs/thesis/tables/arch_vs_transformer.json
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Dict, List

import numpy as np

from attention.taxonomy import CUE_CLASSES
from attention.thesis_eval import EVALUATOR_VERSION
from attention.thesis_eval import bootstrap as B
from attention.thesis_eval import metrics as M


def load(path: Path):
    z = np.load(path, allow_pickle=False)
    return z["y"], z["pred"], z["video_id"], z["seq_key"]


def compare_pair(a_path: Path, b_path: Path, n_boot: int, seed: int) -> Dict:
    ya, preda, vida, ka = load(a_path)
    yb, predb, vidb, kb = load(b_path)
    if not (np.array_equal(ya, yb) and np.array_equal(ka, kb)):
        raise SystemExit(f"{a_path} and {b_path} are not frame-aligned")

    def stats(pred):
        def inner(idx):
            cm = M.confusion_matrix(ya[idx], pred[idx])
            prf = M.per_class_prf(cm)
            out = {"accuracy": M.accuracy(cm),
                   "balanced_accuracy": M.balanced_accuracy(cm),
                   "macro_f1": M.macro_f1(cm),
                   "weighted_f1": M.weighted_f1(cm)}
            for c, name in enumerate(CUE_CLASSES):
                out[f"f1::{name}"] = float(prf["f1"][c])
            return out
        return inner

    return B.paired_cluster_bootstrap_multi(
        vida, stats(preda), stats(predb), n_boot=n_boot, seed=seed)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--a", action="append", required=True, help="NAME=path (repeatable)")
    ap.add_argument("--b", action="append", required=True, help="NAME=path (repeatable)")
    ap.add_argument("--out", required=True)
    ap.add_argument("--n-boot", type=int, default=2000)
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()

    a_items = [e.split("=", 1) for e in args.a]
    b_items = [e.split("=", 1) for e in args.b]
    if len(a_items) != len(b_items):
        raise SystemExit("--a and --b must be given the same number of times "
                         "(they are paired positionally, seed by seed)")

    per_seed: Dict[str, Dict] = {}
    for (na, pa), (nb, pb) in zip(a_items, b_items):
        per_seed[f"{na}_vs_{nb}"] = compare_pair(Path(pa), Path(pb), args.n_boot, args.seed)

    keys = list(next(iter(per_seed.values())))
    agg = {}
    for k in keys:
        d = [v[k]["difference"] for v in per_seed.values()]
        agg[k] = {
            "mean_difference": float(np.mean(d)),
            "sd_difference": float(np.std(d, ddof=1)) if len(d) > 1 else 0.0,
            "n_pairs": len(d),
            "n_pairs_significant": sum(1 for v in per_seed.values()
                                       if v[k]["significant_at_alpha"]),
            "per_pair": {n: {"difference": v[k]["difference"],
                             "ci": [v[k]["ci_low"], v[k]["ci_high"]],
                             "p": v[k]["p_value_two_sided"],
                             "significant": v[k]["significant_at_alpha"]}
                         for n, v in per_seed.items()},
        }

    out = {"evaluator_version": EVALUATOR_VERSION,
           "note": ("Positive difference favours the --a system. Significance is a "
                    "paired video-level cluster bootstrap at alpha=0.05; a difference "
                    "is only reported as real when it holds in every seed pair."),
           "aggregate": agg}
    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    Path(args.out).write_text(json.dumps(out, indent=2))

    print(f"{'metric':24s} {'mean Δ':>9} {'sd':>7} {'sig pairs':>10}")
    for k in ["macro_f1", "balanced_accuracy", "accuracy"] + \
             [f"f1::{c}" for c in CUE_CLASSES]:
        a = agg[k]
        print(f"{k:24s} {a['mean_difference']:+9.4f} {a['sd_difference']:7.4f} "
              f"{a['n_pairs_significant']:>6}/{a['n_pairs']}")
    print(f"\nwritten: {args.out}")


if __name__ == "__main__":
    main()
