"""Aggregate the sweep into thesis Table A, seed summaries and paired tests.

Reads every ``eval_<split>/metrics.json`` and ``predictions.npz`` under a sweep
root and produces:

* per-configuration seed summary (mean / sd / normal-approx CI over seeds);
* the single-seed cluster-bootstrap interval, reported next to it — the two
  quantify different uncertainties (data vs training) and neither substitutes
  for the other;
* **paired** video-level bootstrap on every adjacent pair of the ladder, giving
  a CI on the difference and a two-sided bootstrap p-value. This is what
  decides whether 556 -> 570 is a result or noise.

Paired comparisons use the seed-matched runs (seed 42 vs seed 42, …) evaluated
on the identical frame ordering, so the same cluster draw indexes the same
frames in both systems.

Usage (from LLMDet/):
    python -m attention.thesis_eval.aggregate --root work_dirs/thesis/ladder \
        --split val --out work_dirs/thesis/tables
"""

from __future__ import annotations

import argparse
import csv
import json
from collections import defaultdict
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np

from attention.taxonomy import CUE_CLASSES
from attention.thesis_eval import EVALUATOR_VERSION
from attention.thesis_eval import bootstrap as B
from attention.thesis_eval import metrics as M

LADDER_ORDER = ["552_base", "556_hp", "563_expr", "563_dyn", "570_full"]
LADDER_LABEL = {
    "552_base": "base (CLIP + geometry + colour + posture)",
    "556_hp": "+ head pose (4)",
    "563_expr": "+ head pose + facial expression (7)",
    "563_dyn": "+ head pose + body-language/gaze dynamics (7)",
    "570_full": "+ head pose + expression + dynamics (14)",
}
#: Adjacent contrasts that isolate exactly one feature family.
CONTRASTS: List[Tuple[str, str, str]] = [
    ("556_hp", "552_base", "head pose"),
    ("563_expr", "556_hp", "facial expression (isolated)"),
    ("563_dyn", "556_hp", "body-language/gaze dynamics (isolated)"),
    ("570_full", "556_hp", "expression + dynamics (combined, historic contrast)"),
    ("570_full", "563_dyn", "expression given dynamics"),
    ("570_full", "563_expr", "dynamics given expression"),
]


def discover(root: Path, split: str) -> Dict[str, List[Dict]]:
    """{config_key: [run dicts]} where config_key is model+feature_config.

    The cue-label set is part of the key. Two runs on the same architecture and
    the same feature config but different label RULE versions are measured
    against different targets; pooling them into one mean-over-seeds would
    silently average two numbers that answer different questions. Runs on the
    stored labels key as before, so existing result trees group unchanged.
    """
    out: Dict[str, List[Dict]] = defaultdict(list)
    for d in sorted(root.iterdir()):
        m = d / f"eval_{split}" / "metrics.json"
        if not m.exists():
            continue
        res = json.loads(m.read_text())
        key = f"{res['model']}::{res['feature_config']}"
        cue = res.get("cue_labels") or ""
        if cue:
            key += f"::{Path(cue).stem}"
        out[key].append({"metrics": res, "dir": d,
                         "predictions": d / f"eval_{split}" / "predictions.npz"})
    return out


def load_arrays(path: Path):
    z = np.load(path, allow_pickle=False)
    return z["probs"], z["y"], z["pred"], z["video_id"], z["seq_key"]


def paired_compare(a_pred: Path, b_pred: Path, n_boot: int, seed: int) -> Optional[Dict]:
    """Paired video-level bootstrap of (A - B) on macro-F1, balanced accuracy
    and accuracy. Returns None if the two archives are not frame-aligned."""
    pa, ya, preda, vida, ka = load_arrays(a_pred)
    pb, yb, predb, vidb, kb = load_arrays(b_pred)
    if len(ya) != len(yb) or not np.array_equal(ya, yb) or not np.array_equal(ka, kb):
        return None
    out = {}
    for name, fn in (("macro_f1", M.macro_f1),
                     ("balanced_accuracy", M.balanced_accuracy),
                     ("accuracy", M.accuracy)):
        out[name] = B.paired_cluster_bootstrap(
            vida,
            lambda idx, f=fn: f(M.confusion_matrix(ya[idx], preda[idx])),
            lambda idx, f=fn: f(M.confusion_matrix(yb[idx], predb[idx])),
            n_boot=n_boot, seed=seed)
    for c, cname in enumerate(CUE_CLASSES):
        out[f"f1::{cname}"] = B.paired_cluster_bootstrap(
            vida,
            lambda idx, c=c: float(M.per_class_prf(M.confusion_matrix(ya[idx], preda[idx]))["f1"][c]),
            lambda idx, c=c: float(M.per_class_prf(M.confusion_matrix(yb[idx], predb[idx]))["f1"][c]),
            n_boot=max(500, n_boot // 2), seed=seed)
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", required=True)
    ap.add_argument("--split", default="val")
    ap.add_argument("--out", required=True)
    ap.add_argument("--n-boot", type=int, default=2000)
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()

    root = Path(args.root)
    runs = discover(root, args.split)
    if not runs:
        raise SystemExit(f"no eval_{args.split}/metrics.json under {root}")

    summary: Dict = {"evaluator_version": EVALUATOR_VERSION, "root": str(root),
                     "split": args.split, "configs": {}}
    for key, group in sorted(runs.items()):
        group.sort(key=lambda g: g["metrics"]["seed"])
        met = [g["metrics"] for g in group]
        row = {
            "model": met[0]["model"], "feature_config": met[0]["feature_config"],
            "cue_labels": met[0].get("cue_labels", ""),
            "cue_labels_trained_on": met[0].get("cue_labels_trained_on", ""),
            "input_dim": met[0]["input_dim"], "seeds": [m["seed"] for m in met],
            "n_frames": met[0]["n_frames"], "n_videos": met[0]["n_videos"],
            "over_seeds": {k: B.seed_summary([m[k] for m in met]) for k in
                           ("accuracy", "balanced_accuracy", "macro_f1", "weighted_f1",
                            "macro_auprc", "macro_auroc", "ece", "brier", "nll")},
            "per_class_f1_over_seeds": {
                c: B.seed_summary([m["per_class"][c]["f1"] for m in met])
                for c in CUE_CLASSES},
            "per_class_auprc_over_seeds": {
                c: B.seed_summary([m["per_class"][c]["auprc"] for m in met])
                for c in CUE_CLASSES},
            "seed42_bootstrap": met[0].get("bootstrap"),
            "runs": [str(g["dir"]) for g in group],
        }
        summary["configs"][key] = row

    # paired contrasts, seed-matched
    summary["contrasts"] = {}
    for a_cfg, b_cfg, label in CONTRASTS:
        ka, kb = f"transformer::{a_cfg}", f"transformer::{b_cfg}"
        if ka not in runs or kb not in runs:
            continue
        per_seed = {}
        for ga in runs[ka]:
            s = ga["metrics"]["seed"]
            gb = next((g for g in runs[kb] if g["metrics"]["seed"] == s), None)
            if gb is None:
                continue
            cmp_ = paired_compare(ga["predictions"], gb["predictions"],
                                  args.n_boot, args.seed)
            if cmp_ is not None:
                per_seed[str(s)] = cmp_
        deltas = [summary["configs"][ka]["over_seeds"]["macro_f1"]["mean"]
                  - summary["configs"][kb]["over_seeds"]["macro_f1"]["mean"]]
        summary["contrasts"][f"{a_cfg}_minus_{b_cfg}"] = {
            "label": label, "a": a_cfg, "b": b_cfg,
            "delta_macro_f1_seed_means": deltas[0],
            "paired_bootstrap_per_seed": per_seed,
            "n_seeds_significant_macro_f1": sum(
                1 for v in per_seed.values() if v["macro_f1"]["significant_at_alpha"]),
            "n_seeds": len(per_seed),
        }

    out = Path(args.out); out.mkdir(parents=True, exist_ok=True)
    (out / f"table_a_{args.split}.json").write_text(json.dumps(summary, indent=2))

    # ---- Table A (csv + markdown) ----
    def fmt(v, n=4):
        return "n/a" if v is None or (isinstance(v, float) and not np.isfinite(v)) else f"{v:.{n}f}"

    rows = []
    for key in sorted(summary["configs"],
                      key=lambda k: (k.split("::")[0],
                                     LADDER_ORDER.index(k.split("::")[1])
                                     if k.split("::")[1] in LADDER_ORDER else 99)):
        c = summary["configs"][key]
        o = c["over_seeds"]
        bs = c.get("seed42_bootstrap") or {}
        rows.append({
            "model": c["model"], "dims": c["input_dim"],
            "added_features": LADDER_LABEL.get(c["feature_config"], c["feature_config"]),
            "split": args.split, "n_seeds": o["macro_f1"]["n"],
            "accuracy": fmt(o["accuracy"]["mean"]),
            "accuracy_sd": fmt(o["accuracy"]["std"]),
            "balanced_accuracy": fmt(o["balanced_accuracy"]["mean"]),
            "balanced_accuracy_sd": fmt(o["balanced_accuracy"]["std"]),
            "macro_f1": fmt(o["macro_f1"]["mean"]),
            "macro_f1_sd": fmt(o["macro_f1"]["std"]),
            "macro_f1_seed_ci": f"[{fmt(o['macro_f1'].get('ci_low'))}, {fmt(o['macro_f1'].get('ci_high'))}]",
            "macro_f1_bootstrap_ci_seed42":
                f"[{fmt(bs.get('macro_f1', {}).get('ci_low'))}, {fmt(bs.get('macro_f1', {}).get('ci_high'))}]",
            "macro_auprc": fmt(o["macro_auprc"]["mean"]),
            "ece": fmt(o["ece"]["mean"]),
            "brier": fmt(o["brier"]["mean"]),
            "nll": fmt(o["nll"]["mean"]),
        })
    with open(out / f"table_a_{args.split}.csv", "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        w.writeheader()
        w.writerows(rows)

    md = [f"### Table A — visible-cue classification ({args.split} split)", "",
          "| model | dims | added feature block | seeds | accuracy | balanced acc | macro-F1 | macro-F1 95% CI (seeds) | macro-F1 95% CI (video bootstrap, seed 42) | macro-AUPRC | ECE |",
          "|---|---|---|---|---|---|---|---|---|---|---|"]
    for r in rows:
        md.append(f"| {r['model']} | {r['dims']} | {r['added_features']} | {r['n_seeds']} | "
                  f"{r['accuracy']} ± {r['accuracy_sd']} | {r['balanced_accuracy']} ± {r['balanced_accuracy_sd']} | "
                  f"**{r['macro_f1']} ± {r['macro_f1_sd']}** | {r['macro_f1_seed_ci']} | "
                  f"{r['macro_f1_bootstrap_ci_seed42']} | {r['macro_auprc']} | {r['ece']} |")
    md += ["", "### Per-class F1 (mean ± sd over seeds)", "",
           "| config | " + " | ".join(CUE_CLASSES) + " |",
           "|---|" + "---|" * len(CUE_CLASSES)]
    for key in sorted(summary["configs"],
                      key=lambda k: LADDER_ORDER.index(k.split("::")[1])
                      if k.split("::")[1] in LADDER_ORDER else 99):
        c = summary["configs"][key]
        md.append(f"| {key} | " + " | ".join(
            f"{fmt(c['per_class_f1_over_seeds'][cl]['mean'], 3)} ± {fmt(c['per_class_f1_over_seeds'][cl]['std'], 3)}"
            for cl in CUE_CLASSES) + " |")

    md += ["", "### Isolated feature contrasts (paired video-level bootstrap)", "",
           "| contrast | Δ macro-F1 (seed means) | seeds with a significant Δ | verdict |",
           "|---|---|---|---|"]
    for name, c in summary["contrasts"].items():
        n_sig, n = c["n_seeds_significant_macro_f1"], c["n_seeds"]
        verdict = ("supported" if n and n_sig == n else
                   "mixed" if n_sig else "not supported (within noise)")
        md.append(f"| {c['label']} ({c['a']} − {c['b']}) | {c['delta_macro_f1_seed_means']:+.4f} | "
                  f"{n_sig}/{n} | {verdict} |")
    (out / f"table_a_{args.split}.md").write_text("\n".join(md) + "\n")

    print("\n".join(md))
    print(f"\nwritten: {out}/table_a_{args.split}.{{json,csv,md}}")


if __name__ == "__main__":
    main()
