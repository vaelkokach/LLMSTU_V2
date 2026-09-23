"""CMOSE — a **separate** four-level ordinal engagement experiment.

This is deliberately not part of the visible-cue model. CMOSE labels an
*internal state* (engagement) on short online-coaching clips; the project's own
taxonomy labels *observable cues* in a computer laboratory and refuses to infer
mental state. Merging the two label sets would destroy exactly the distinction
the thesis is built on. The two tasks are trained, evaluated and reported apart,
and their numbers are never placed in the same table.

What this file measures
-----------------------
1. A four-level ordinal engagement classifier over CMOSE's released 1024-d I3D
   clip embeddings, under the dataset's **own published split**.
2. The identical model under a **subject-disjoint** split.

(2) exists because of a property of the release that the paper does not
foreground and that this project is obliged to check, having spent a month
recovering from a leaked split of its own:

    103 subjects; **101 of them appear in more than one official split**.

Clip names encode the subject (``videoX_Y_personZ``), so this is verifiable in
three lines. The published 70/20/10 segment split therefore lets a model see the
same person — same face, same webcam, same room, same session — in training and
at test. The gap between (1) and (2) is the size of that effect, measured.

Metrics are the ordinal family the engagement literature uses, not the nominal
family used for the six visible cues: accuracy, average (balanced) accuracy,
macro-F1, MAE over the ordered levels, quadratic weighted kappa, and Spearman
rank correlation.

    python -m attention.thesis_eval.cmose --root ../grounding_data/external/CMOSE \
        --out work_dirs/thesis/cmose
"""

from __future__ import annotations

import argparse
import json
import re
from collections import Counter, defaultdict
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

#: Ordered engagement levels. The order is the whole point: MAE and quadratic
#: weighted kappa are only meaningful if 0 < 1 < 2 < 3 is the true scale.
LEVELS = ["Highly Disengage", "Disengage", "Engage", "Highly Engage"]
LEVEL_TO_ID = {l: i for i, l in enumerate(LEVELS)}


def subject_of(clip: str) -> str:
    """``videoX_Y_personZ`` -> ``vX_pZ``. A subject is a person within a session."""
    m = re.match(r"video(\d+)_(\d+)_person(\d+)$", clip)
    if not m:
        raise ValueError(f"unparseable CMOSE clip name: {clip}")
    return f"v{m.group(1)}_p{m.group(3)}"


def load(root: Path):
    """Returns clips, I3D features, ordinal labels, official split, subject, agreement.

    295 of the 12,197 released clips carry an **empty** ``embeds`` list. They are
    dropped, not zero-filled: a zero vector is a valid input the model would
    learn from, and it would be indistinguishable from a genuine all-zero
    embedding — the same "missing looks like a real value" trap that made the
    OpenCV head-pose backend useless [internal notes, not included].
    """
    d = json.load(open(root / "final_data_1.json"))
    clips_all = sorted(d)
    clips = [c for c in clips_all if len(d[c]["embeds"]) == 1024]
    n_dropped = len(clips_all) - len(clips)
    if n_dropped:
        print(f"CMOSE: dropped {n_dropped}/{len(clips_all)} clips with empty I3D "
              f"embeddings ({n_dropped / len(clips_all):.1%})")
    X = np.stack([np.asarray(d[c]["embeds"], dtype=np.float32) for c in clips])
    y = np.array([LEVEL_TO_ID[d[c]["label"]] for c in clips], dtype=np.int64)
    split = np.array([d[c]["split"] for c in clips])
    subj = np.array([subject_of(c) for c in clips])
    agree = np.array([float(d[c]["agreement"]) for c in clips], dtype=np.float32)
    return clips, X, y, split, subj, agree, n_dropped


def official_split(split: np.ndarray):
    """CMOSE's own split; 'unlabel' is what the release uses for validation."""
    return (split == "train"), (split == "unlabel"), (split == "test")


def subject_disjoint_split(subj: np.ndarray, y: np.ndarray, seed: int = 0,
                           frac=(0.70, 0.15, 0.15)):
    """Assign whole subjects to train/val/test, greedily balancing label mix.

    Same policy as `llmstu_tools/make_splits.py` uses for videos: sort by size
    descending and give each subject to whichever split is furthest below its
    target share. Deterministic given the seed.
    """
    subjects = sorted(set(subj))
    counts = {s: int((subj == s).sum()) for s in subjects}
    order = sorted(subjects, key=lambda s: (-counts[s], s))
    total = len(subj)
    sizes = {"train": 0, "val": 0, "test": 0}
    targets = dict(zip(("train", "val", "test"), frac))
    assign: Dict[str, str] = {}
    for s in order:
        pick = max(targets, key=lambda k: targets[k] - sizes[k] / max(total, 1))
        assign[s] = pick
        sizes[pick] += counts[s]
    m = np.array([assign[s] for s in subj])
    return (m == "train"), (m == "val"), (m == "test")


class MLP(nn.Module):
    """Deliberately small. The point is the split comparison, not a new SOTA."""

    def __init__(self, in_dim: int, n_classes: int = 4, hidden: int = 512,
                 dropout: float = 0.3):
        super().__init__()
        self.net = nn.Sequential(
            nn.LayerNorm(in_dim),
            nn.Linear(in_dim, hidden), nn.GELU(), nn.Dropout(dropout),
            nn.Linear(hidden, hidden // 2), nn.GELU(), nn.Dropout(dropout),
            nn.Linear(hidden // 2, n_classes))

    def forward(self, x):
        return self.net(x)


# --------------------------------------------------------------------------
# ordinal metrics
# --------------------------------------------------------------------------

def confusion(y_true, y_pred, k=4):
    return np.bincount(y_true * k + y_pred, minlength=k * k).reshape(k, k)


def quadratic_weighted_kappa(y_true, y_pred, k: int = 4) -> float:
    """Cohen's kappa with quadratic penalties — the standard ordinal-agreement
    statistic. Confusing level 0 with level 3 costs 9x confusing 0 with 1."""
    O = confusion(y_true, y_pred, k).astype(np.float64)
    w = (np.arange(k)[:, None] - np.arange(k)[None, :]) ** 2 / (k - 1) ** 2
    hist_t = np.bincount(y_true, minlength=k).astype(np.float64)
    hist_p = np.bincount(y_pred, minlength=k).astype(np.float64)
    E = np.outer(hist_t, hist_p)
    E = E * O.sum() / max(E.sum(), 1e-12)
    denom = (w * E).sum()
    return float(1.0 - (w * O).sum() / denom) if denom > 0 else float("nan")


def spearman(a: np.ndarray, b: np.ndarray) -> float:
    def rank(v):
        order = np.argsort(v, kind="stable")
        r = np.empty(len(v), dtype=np.float64)
        s = v[order]
        i = 0
        while i < len(s):
            j = i
            while j + 1 < len(s) and s[j + 1] == s[i]:
                j += 1
            r[order[i:j + 1]] = 0.5 * (i + j) + 1.0
            i = j + 1
        return r
    ra, rb = rank(a.astype(np.float64)), rank(b.astype(np.float64))
    ra -= ra.mean(); rb -= rb.mean()
    den = np.sqrt((ra ** 2).sum() * (rb ** 2).sum())
    return float((ra * rb).sum() / den) if den > 0 else float("nan")


def ordinal_metrics(y_true: np.ndarray, y_pred: np.ndarray, k: int = 4) -> Dict:
    cm = confusion(y_true, y_pred, k)
    tp = np.diag(cm).astype(np.float64)
    fp = cm.sum(0) - tp
    fn = cm.sum(1) - tp
    with np.errstate(divide="ignore", invalid="ignore"):
        prec = np.where(tp + fp > 0, tp / (tp + fp), 0.0)
        rec = np.where(tp + fn > 0, tp / (tp + fn), 0.0)
        f1 = np.where(prec + rec > 0, 2 * prec * rec / (prec + rec), 0.0)
    present = cm.sum(1) > 0
    return {
        "n": int(len(y_true)),
        "accuracy": float(np.trace(cm) / cm.sum()),
        "average_accuracy": float(rec[present].mean()),   # CMOSE's "average accuracy"
        "macro_f1": float(f1[present].mean()),
        "mae": float(np.abs(y_true - y_pred).mean()),
        "quadratic_weighted_kappa": quadratic_weighted_kappa(y_true, y_pred, k),
        "spearman": spearman(y_true, y_pred),
        "per_class_f1": {LEVELS[i]: float(f1[i]) for i in range(k)},
        "per_class_recall": {LEVELS[i]: float(rec[i]) for i in range(k)},
        "support": {LEVELS[i]: int(cm[i].sum()) for i in range(k)},
        "confusion_matrix": cm.tolist(),
        "level_order": LEVELS,
    }


def train_eval(X, y, tr, va, te, seed: int, device, epochs: int = 60,
               lr: float = 1e-3, bs: int = 256) -> Dict:
    torch.manual_seed(seed)
    np.random.seed(seed)
    mu, sd = X[tr].mean(0), X[tr].std(0) + 1e-6      # statistics from TRAIN only
    Z = torch.tensor((X - mu) / sd, device=device)
    Y = torch.tensor(y, device=device)
    idx = {k: torch.tensor(np.flatnonzero(m), device=device) for k, m in
           (("train", tr), ("val", va), ("test", te))}

    hist = np.bincount(y[tr], minlength=4).astype(np.float32)
    w = 1.0 / np.sqrt(np.maximum(hist, 1.0))
    w = torch.tensor(w / w.sum() * 4, device=device)

    model = MLP(X.shape[1]).to(device)
    opt = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=0.01)
    g = torch.Generator(device="cpu"); g.manual_seed(seed)

    best, best_state = -1.0, None
    for ep in range(epochs):
        model.train()
        perm = idx["train"][torch.randperm(len(idx["train"]), generator=g).to(device)]
        for i in range(0, len(perm), bs):
            b = perm[i:i + bs]
            opt.zero_grad(set_to_none=True)
            F.cross_entropy(model(Z[b]), Y[b], weight=w).backward()
            opt.step()
        model.eval()
        with torch.no_grad():
            pv = model(Z[idx["val"]]).argmax(-1).cpu().numpy()
        # selection on average accuracy: with 69% of clips in one level,
        # plain accuracy would select a majority-class collapse.
        m = ordinal_metrics(y[va], pv)["average_accuracy"]
        if m > best:
            best = m
            best_state = {k: v.detach().clone() for k, v in model.state_dict().items()}
    model.load_state_dict(best_state)
    model.eval()
    out = {}
    with torch.no_grad():
        for name, m in (("val", va), ("test", te)):
            p = model(Z[idx[name]]).argmax(-1).cpu().numpy()
            out[name] = ordinal_metrics(y[m], p)
    # majority-class control on the same test frames
    maj = int(Counter(y[tr].tolist()).most_common(1)[0][0])
    out["test_majority_control"] = ordinal_metrics(y[te], np.full(te.sum(), maj))
    out["selection_metric"] = "val average_accuracy"
    out["best_val_average_accuracy"] = float(best)
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--seeds", default="42,43,44")
    ap.add_argument("--epochs", type=int, default=60)
    ap.add_argument("--device", default="cuda:0")
    args = ap.parse_args()

    root = Path(args.root)
    clips, X, y, split, subj, agree, n_dropped = load(root)
    device = torch.device(args.device if torch.cuda.is_available() else "cpu")

    # --- the leakage audit, reported whether or not anyone asked ---
    bysub = defaultdict(set)
    for c, s in zip(clips, split):
        bysub[subject_of(c)].add(s)
    leaked = sorted(s for s, v in bysub.items() if len(v) > 1)
    audit = {
        "n_clips_used": len(clips), "n_clips_dropped_empty_i3d": n_dropped,
        "n_subjects": len(bysub),
        "official_split_sizes": {k: int(v) for k, v in Counter(split).items()},
        "label_distribution": {LEVELS[i]: int((y == i).sum()) for i in range(4)},
        "subjects_in_more_than_one_official_split": len(leaked),
        "fraction_of_subjects_leaked": len(leaked) / len(bysub),
        "annotator_agreement_min_max": [float(agree.min()), float(agree.max())],
        "note": ("CMOSE's released split assigns clips, not subjects. The same "
                 "person, webcam and session therefore appear in train and test. "
                 "This is not a defect of the dataset — the paper reports a "
                 "random segment split — but any number computed under it "
                 "measures something weaker than subject-level generalisation."),
    }
    print(f"CMOSE: {audit['n_clips_used']} clips, {audit['n_subjects']} subjects, "
          f"{audit['subjects_in_more_than_one_official_split']} subjects leak "
          f"across the official split")

    seeds = [int(s) for s in args.seeds.split(",")]
    results = {"audit": audit, "protocols": {}}
    for proto in ("official", "subject_disjoint"):
        if proto == "official":
            tr, va, te = official_split(split)
        else:
            tr, va, te = subject_disjoint_split(subj, y)
            # verify the property we are claiming
            assert not (set(subj[tr]) & set(subj[te])), "subject leaked into test"
            assert not (set(subj[tr]) & set(subj[va])), "subject leaked into val"
        runs = [train_eval(X, y, tr, va, te, s, device, args.epochs) for s in seeds]
        agg = {}
        for metric in ("accuracy", "average_accuracy", "macro_f1", "mae",
                       "quadratic_weighted_kappa", "spearman"):
            v = np.array([r["test"][metric] for r in runs], dtype=float)
            agg[metric] = {"mean": float(v.mean()),
                           "std": float(v.std(ddof=1)) if len(v) > 1 else 0.0,
                           "values": v.tolist()}
        results["protocols"][proto] = {
            "split_sizes": {"train": int(tr.sum()), "val": int(va.sum()), "test": int(te.sum())},
            "n_subjects": {"train": len(set(subj[tr])), "val": len(set(subj[va])),
                           "test": len(set(subj[te]))},
            "subject_overlap_train_test": len(set(subj[tr]) & set(subj[te])),
            "test_over_seeds": agg,
            "per_seed": {str(s): r for s, r in zip(seeds, runs)},
            "majority_control": runs[0]["test_majority_control"],
        }
        a = agg
        print(f"\n[{proto}] test over {len(seeds)} seeds "
              f"({results['protocols'][proto]['subject_overlap_train_test']} subjects "
              f"shared train/test)")
        print(f"  accuracy         {a['accuracy']['mean']:.4f} ± {a['accuracy']['std']:.4f}")
        print(f"  average accuracy {a['average_accuracy']['mean']:.4f} ± {a['average_accuracy']['std']:.4f}")
        print(f"  macro-F1         {a['macro_f1']['mean']:.4f} ± {a['macro_f1']['std']:.4f}")
        print(f"  MAE (ordinal)    {a['mae']['mean']:.4f} ± {a['mae']['std']:.4f}")
        print(f"  QWK              {a['quadratic_weighted_kappa']['mean']:.4f} ± {a['quadratic_weighted_kappa']['std']:.4f}")
        print(f"  Spearman         {a['spearman']['mean']:.4f} ± {a['spearman']['std']:.4f}")

    out = Path(args.out); out.mkdir(parents=True, exist_ok=True)
    (out / "cmose_results.json").write_text(json.dumps(results, indent=2))
    print(f"\nwritten: {out}/cmose_results.json")


if __name__ == "__main__":
    main()
