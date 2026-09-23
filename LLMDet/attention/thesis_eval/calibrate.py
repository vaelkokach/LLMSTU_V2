"""Post-hoc calibration and selective prediction for instructor-facing alerts.

The dashboard currently treats every cue prediction as equally reliable. It
should not: an alert that fires on a 0.35-confidence `phone_use` frame costs the
instructor's attention, and the taxonomy's founding principle — report what is
visible, never assert a mental state — extends naturally to *not reporting* what
the model cannot see clearly.

Two independent pieces:

**Temperature scaling** (Guo et al.). A single scalar T divides the logits;
fitted by minimising NLL on the **validation** split only. It cannot change any
argmax, therefore cannot change accuracy, macro-F1 or the confusion matrix — it
only rescales confidence. That property is asserted in the tests.

**Selective prediction.** Frames whose calibrated confidence falls below a
threshold are mapped to `uncertain` (abstention) rather than forced into a cue.
The coverage–risk curve reports, at each retained coverage, the error on the
retained frames — and, because the deployment question is about episodes rather
than frames, the alert threshold is chosen on validation and then held fixed.

Fitting is on stored probability archives, so this never re-runs a model.

Usage (from LLMDet/):
    python -m attention.thesis_eval.calibrate \
        --val-predictions  work_dirs/thesis/ladder/<run>/eval_val/predictions.npz \
        --eval-predictions work_dirs/thesis/ladder/<run>/eval_val/predictions.npz \
        --out work_dirs/thesis/calibration/<run>
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Dict, Optional, Sequence

import numpy as np

from attention.taxonomy import CUE_CLASSES
from attention.thesis_eval import EVALUATOR_VERSION
from attention.thesis_eval import metrics as M

UNCERTAIN = CUE_CLASSES.index("uncertain")


def _logits_from_probs(probs: np.ndarray) -> np.ndarray:
    """Recover logits up to an additive per-row constant.

    Softmax is shift-invariant, so ``log p`` is a valid logit vector: applying
    temperature T to ``log p`` gives exactly the same distribution as applying T
    to the original logits. Storing probabilities rather than logits therefore
    loses nothing for calibration.
    """
    return np.log(np.clip(probs, 1e-12, 1.0))


def temperature_nll(T: float, logits: np.ndarray, y: np.ndarray) -> float:
    z = logits / max(T, 1e-6)
    z = z - z.max(1, keepdims=True)
    logp = z - np.log(np.exp(z).sum(1, keepdims=True))
    return float(-logp[np.arange(len(y)), y].mean())


def fit_temperature(probs: np.ndarray, y: np.ndarray,
                    lo: float = 0.05, hi: float = 10.0, iters: int = 60) -> float:
    """Golden-section search on NLL. Convex in log T in practice, and a 1-D
    search avoids depending on an optimiser's convergence settings."""
    logits = _logits_from_probs(probs)
    gr = (np.sqrt(5.0) - 1.0) / 2.0
    a, b = np.log(lo), np.log(hi)
    c, d = b - gr * (b - a), a + gr * (b - a)
    fc, fd = temperature_nll(np.exp(c), logits, y), temperature_nll(np.exp(d), logits, y)
    for _ in range(iters):
        if fc < fd:
            b, d, fd = d, c, fc
            c = b - gr * (b - a)
            fc = temperature_nll(np.exp(c), logits, y)
        else:
            a, c, fc = c, d, fd
            d = a + gr * (b - a)
            fd = temperature_nll(np.exp(d), logits, y)
    return float(np.exp((a + b) / 2.0))


def apply_temperature(probs: np.ndarray, T: float) -> np.ndarray:
    z = _logits_from_probs(probs) / max(T, 1e-6)
    z = z - z.max(1, keepdims=True)
    e = np.exp(z)
    return e / e.sum(1, keepdims=True)


def fit_vector_scaling(probs: np.ndarray, y: np.ndarray, iters: int = 400,
                       lr: float = 0.05) -> Dict[str, list]:
    """Per-class scale + bias on the logits (Guo et al.'s vector scaling).

    More expressive than a single temperature and therefore more prone to
    overfitting a small validation set; reported alongside so the choice between
    them is made on measured ECE rather than assumed.
    """
    import torch
    z = torch.tensor(_logits_from_probs(probs), dtype=torch.float64)
    t = torch.tensor(y, dtype=torch.long)
    w = torch.ones(probs.shape[1], dtype=torch.float64, requires_grad=True)
    b = torch.zeros(probs.shape[1], dtype=torch.float64, requires_grad=True)
    opt = torch.optim.LBFGS([w, b], lr=lr, max_iter=iters)

    def closure():
        opt.zero_grad()
        loss = torch.nn.functional.cross_entropy(z * w + b, t)
        loss.backward()
        return loss
    opt.step(closure)
    return {"w": w.detach().tolist(), "b": b.detach().tolist()}


def apply_vector_scaling(probs: np.ndarray, params: Dict[str, list]) -> np.ndarray:
    z = _logits_from_probs(probs) * np.asarray(params["w"]) + np.asarray(params["b"])
    z = z - z.max(1, keepdims=True)
    e = np.exp(z)
    return e / e.sum(1, keepdims=True)


def coverage_risk_curve(probs: np.ndarray, y: np.ndarray,
                        thresholds: Optional[Sequence[float]] = None) -> list:
    """Selective-prediction curve.

    At each confidence threshold: what fraction of frames is retained, and what
    is the error rate on the retained ones. ``abstain_error`` additionally
    reports the error when abstentions are re-labelled `uncertain` rather than
    dropped, which is what the dashboard actually does.
    """
    if thresholds is None:
        thresholds = np.round(np.arange(0.0, 1.0, 0.02), 2)
    conf = probs.max(1)
    pred = probs.argmax(1)
    correct = pred == y
    rows = []
    for t in thresholds:
        keep = conf >= t
        n = int(keep.sum())
        abst = pred.copy()
        abst[~keep] = UNCERTAIN
        rows.append({
            "threshold": float(t),
            "coverage": float(n / len(y)),
            "n_retained": n,
            "selective_error": float(1.0 - correct[keep].mean()) if n else float("nan"),
            "selective_accuracy": float(correct[keep].mean()) if n else float("nan"),
            "selective_macro_f1": M.macro_f1(M.confusion_matrix(y[keep], pred[keep])) if n else float("nan"),
            "abstain_accuracy": float((abst == y).mean()),
            "abstain_macro_f1": M.macro_f1(M.confusion_matrix(y, abst)),
        })
    return rows


def area_under_risk_coverage(rows: Sequence[Dict]) -> float:
    """AURC: mean selective error weighted by coverage step (lower is better)."""
    pts = sorted(((r["coverage"], r["selective_error"]) for r in rows
                  if np.isfinite(r["selective_error"])), key=lambda p: p[0])
    if len(pts) < 2:
        return float("nan")
        # trapezoid over coverage
    c = np.array([p[0] for p in pts]); e = np.array([p[1] for p in pts])
    return float(np.trapz(e, c) / max(c[-1] - c[0], 1e-9))


def calibration_block(probs: np.ndarray, y: np.ndarray) -> Dict:
    return {
        "ece": M.expected_calibration_error(probs, y),
        "classwise_ece": M.classwise_ece(probs, y),
        "brier": M.brier_score(probs, y),
        "nll": M.negative_log_likelihood(probs, y),
        "accuracy": float((probs.argmax(1) == y).mean()),
        "macro_f1": M.macro_f1(M.confusion_matrix(y, probs.argmax(1))),
        "reliability": M.reliability_bins(probs.max(1),
                                          (probs.argmax(1) == y).astype(float)),
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--val-predictions", required=True,
                    help="archive used to FIT the calibrator (validation only)")
    ap.add_argument("--eval-predictions", required=True,
                    help="archive the fitted calibrator is APPLIED to")
    ap.add_argument("--out", required=True)
    ap.add_argument("--vector-scaling", action="store_true")
    args = ap.parse_args()

    v = np.load(args.val_predictions, allow_pickle=False)
    e = np.load(args.eval_predictions, allow_pickle=False)
    vp, vy = v["probs"].astype(np.float64), v["y"]
    ep, ey = e["probs"].astype(np.float64), e["y"]

    T = fit_temperature(vp, vy)
    ep_T = apply_temperature(ep, T)

    res: Dict = {
        "evaluator_version": EVALUATOR_VERSION,
        "fit_on": args.val_predictions, "applied_to": args.eval_predictions,
        "temperature": T,
        "before": calibration_block(ep, ey),
        "after_temperature": calibration_block(ep_T, ey),
    }
    # temperature scaling is argmax-preserving; assert it rather than assume it
    assert np.array_equal(ep.argmax(1), ep_T.argmax(1)), \
        "temperature scaling changed a prediction — it must not"
    res["argmax_preserved"] = True

    if args.vector_scaling:
        params = fit_vector_scaling(vp, vy)
        ep_V = apply_vector_scaling(ep, params)
        res["vector_scaling_params"] = params
        res["after_vector_scaling"] = calibration_block(ep_V, ey)
        res["vector_scaling_changed_predictions"] = int(
            (ep_V.argmax(1) != ep.argmax(1)).sum())

    res["coverage_risk_uncalibrated"] = coverage_risk_curve(ep, ey)
    res["coverage_risk_calibrated"] = coverage_risk_curve(ep_T, ey)
    res["aurc_uncalibrated"] = area_under_risk_coverage(res["coverage_risk_uncalibrated"])
    res["aurc_calibrated"] = area_under_risk_coverage(res["coverage_risk_calibrated"])

    out = Path(args.out); out.mkdir(parents=True, exist_ok=True)
    (out / "calibration.json").write_text(json.dumps(res, indent=2))
    import csv
    with open(out / "coverage_risk.csv", "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["threshold", "coverage", "selective_error", "selective_macro_f1",
                    "abstain_accuracy", "abstain_macro_f1"])
        for r in res["coverage_risk_calibrated"]:
            w.writerow([r["threshold"], f"{r['coverage']:.4f}",
                        f"{r['selective_error']:.4f}", f"{r['selective_macro_f1']:.4f}",
                        f"{r['abstain_accuracy']:.4f}", f"{r['abstain_macro_f1']:.4f}"])

    b, a = res["before"], res["after_temperature"]
    print(f"T = {T:.4f}")
    print(f"  ECE   {b['ece']:.4f} -> {a['ece']:.4f}")
    print(f"  Brier {b['brier']:.4f} -> {a['brier']:.4f}")
    print(f"  NLL   {b['nll']:.4f} -> {a['nll']:.4f}")
    print(f"  accuracy unchanged: {b['accuracy']:.4f} / {a['accuracy']:.4f}")
    print(f"  AURC  {res['aurc_uncalibrated']:.4f} -> {res['aurc_calibrated']:.4f}")


if __name__ == "__main__":
    main()
