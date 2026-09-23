"""Confidence calibration for assignment scores (to-do item 28).

Fits logistic / isotonic calibrators mapping raw compatibility components to
P(assignment correct), and reports AUROC, AUPRC, ECE, Brier. Intended to be
fit on the gold *calibration* split and evaluated on held-out data; until the
gold set exists it can be exercised on pseudo-gold assignments produced by
run_matching_experiment.py (clearly a demo, not a thesis number).

Input format: jsonl, one assignment per line, with numeric feature keys and a
boolean "correct" key (as written by run_matching_experiment.py --dump).
"""

import argparse
import json
from pathlib import Path
from typing import List, Sequence

import numpy as np
from sklearn.isotonic import IsotonicRegression
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import average_precision_score, brier_score_loss, roc_auc_score

DEFAULT_FEATURES = ["clip_sim_z", "margin", "spatial", "det_conf", "n_boxes"]


def expected_calibration_error(y: np.ndarray, p: np.ndarray,
                               n_bins: int = 10) -> float:
    bins = np.linspace(0.0, 1.0, n_bins + 1)
    ece = 0.0
    for lo, hi in zip(bins[:-1], bins[1:]):
        mask = (p >= lo) & (p < hi) if hi < 1.0 else (p >= lo) & (p <= hi)
        if mask.sum() == 0:
            continue
        ece += mask.mean() * abs(y[mask].mean() - p[mask].mean())
    return float(ece)


def load_assignments(path: Path, features: Sequence[str]) -> tuple:
    X, y = [], []
    with open(path) as fh:
        for line in fh:
            rec = json.loads(line)
            X.append([float(rec[f]) for f in features])
            y.append(int(rec["correct"]))
    return np.asarray(X, dtype=np.float64), np.asarray(y, dtype=np.int64)


def evaluate_probs(y: np.ndarray, p: np.ndarray) -> dict:
    return {
        "auroc": float(roc_auc_score(y, p)),
        "auprc": float(average_precision_score(y, p)),
        "ece": expected_calibration_error(y, p),
        "brier": float(brier_score_loss(y, p)),
        "base_rate": float(y.mean()),
        "n": int(len(y)),
    }


def fit_and_evaluate(X: np.ndarray, y: np.ndarray, seed: int = 0,
                     test_frac: float = 0.5) -> dict:
    """Split, fit logistic + isotonic-on-logistic, report test metrics."""
    rng = np.random.RandomState(seed)
    idx = rng.permutation(len(y))
    n_test = int(len(y) * test_frac)
    te, tr = idx[:n_test], idx[n_test:]

    logit = LogisticRegression(max_iter=1000)
    logit.fit(X[tr], y[tr])
    p_tr = logit.predict_proba(X[tr])[:, 1]
    p_te = logit.predict_proba(X[te])[:, 1]

    iso = IsotonicRegression(out_of_bounds="clip")
    iso.fit(p_tr, y[tr])
    p_te_iso = iso.predict(p_te)

    return {
        "logistic": evaluate_probs(y[te], p_te),
        "logistic+isotonic": evaluate_probs(y[te], p_te_iso),
        "coefficients": dict(zip(DEFAULT_FEATURES, logit.coef_[0].tolist())),
    }


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("assignments", type=Path,
                    help="jsonl from run_matching_experiment.py --dump")
    ap.add_argument("--features", nargs="+", default=DEFAULT_FEATURES)
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()

    X, y = load_assignments(args.assignments, args.features)
    res = fit_and_evaluate(X, y, seed=args.seed)
    print(json.dumps(res, indent=2))


if __name__ == "__main__":
    main()
