"""Frame-level classification and calibration metrics.

Every quantity is computed from a confusion matrix or from the raw probability
array, never from a mean of per-batch means. That specific error inflated the
March pipeline's accuracy by 2.6 points [internal notes, not included] and the DDP trainer's
per-shard averaging misordered the 556/570 ladder [internal notes, not included], so the
implementations here take the *whole* prediction array at once.

Class ordering is always ``attention.taxonomy.CUE_CLASSES``; nothing in this
module infers an ordering from the data.
"""

from __future__ import annotations

from typing import Dict, List, Optional

import numpy as np

from attention.taxonomy import CUE_CLASSES

NUM_CLASSES = len(CUE_CLASSES)


def confusion_matrix(y_true: np.ndarray, y_pred: np.ndarray,
                     num_classes: int = NUM_CLASSES) -> np.ndarray:
    """``cm[t, p]`` — rows are truth, columns are prediction."""
    k = (y_true >= 0) & (y_true < num_classes) & (y_pred >= 0) & (y_pred < num_classes)
    return np.bincount(y_true[k] * num_classes + y_pred[k],
                       minlength=num_classes ** 2).reshape(num_classes, num_classes)


def per_class_prf(cm: np.ndarray) -> Dict[str, np.ndarray]:
    tp = np.diag(cm).astype(np.float64)
    fp = cm.sum(0) - tp
    fn = cm.sum(1) - tp
    with np.errstate(divide="ignore", invalid="ignore"):
        p = np.where(tp + fp > 0, tp / (tp + fp), 0.0)
        r = np.where(tp + fn > 0, tp / (tp + fn), 0.0)
        f = np.where(p + r > 0, 2 * p * r / (p + r), 0.0)
    return {"precision": p, "recall": r, "f1": f, "support": cm.sum(1)}


def macro_f1(cm: np.ndarray, present_only: bool = True) -> float:
    """Mean F1 over classes. ``present_only`` averages over classes with
    non-zero *support*, matching the historic evaluator so the new numbers can
    be compared with the archived ones."""
    prf = per_class_prf(cm)
    keep = prf["support"] > 0 if present_only else np.ones(len(prf["f1"]), bool)
    return float(prf["f1"][keep].mean()) if keep.any() else 0.0


def weighted_f1(cm: np.ndarray) -> float:
    prf = per_class_prf(cm)
    sup = prf["support"].astype(np.float64)
    return float((prf["f1"] * sup).sum() / sup.sum()) if sup.sum() else 0.0


def balanced_accuracy(cm: np.ndarray) -> float:
    """Mean per-class recall — the CMOSE-style 'average accuracy'."""
    prf = per_class_prf(cm)
    keep = prf["support"] > 0
    return float(prf["recall"][keep].mean()) if keep.any() else 0.0


def accuracy(cm: np.ndarray) -> float:
    return float(np.trace(cm) / cm.sum()) if cm.sum() else 0.0


def _average_precision(pos: np.ndarray, score: np.ndarray) -> float:
    """Step-wise AP (identical to sklearn's ``average_precision_score``)."""
    order = np.argsort(-score, kind="stable")
    pos = pos[order]
    tp = np.cumsum(pos)
    prec = tp / np.arange(1, len(pos) + 1)
    n_pos = pos.sum()
    return float((prec * pos).sum() / n_pos) if n_pos else float("nan")


def _auroc(pos: np.ndarray, score: np.ndarray) -> float:
    """One-vs-rest AUROC via the rank statistic, ties averaged."""
    n_pos, n_neg = int(pos.sum()), int((1 - pos).sum())
    if n_pos == 0 or n_neg == 0:
        return float("nan")
    order = np.argsort(score, kind="stable")
    ranks = np.empty(len(score), dtype=np.float64)
    s = score[order]
    i = 0
    while i < len(s):
        j = i
        while j + 1 < len(s) and s[j + 1] == s[i]:
            j += 1
        ranks[order[i:j + 1]] = 0.5 * (i + j) + 1.0
        i = j + 1
    return float((ranks[pos == 1].sum() - n_pos * (n_pos + 1) / 2) / (n_pos * n_neg))


def ranking_metrics(probs: np.ndarray, y_true: np.ndarray,
                    num_classes: int = NUM_CLASSES) -> Dict[str, List[float]]:
    """Per-class AUPRC and one-vs-rest AUROC; NaN where a class is absent."""
    auprc, auroc = [], []
    for c in range(num_classes):
        pos = (y_true == c).astype(np.int64)
        if pos.sum() == 0 or pos.sum() == len(pos):
            auprc.append(float("nan"))
            auroc.append(float("nan"))
            continue
        auprc.append(_average_precision(pos, probs[:, c]))
        auroc.append(_auroc(pos, probs[:, c]))
    return {"auprc": auprc, "auroc": auroc}


# --------------------------------------------------------------------------
# calibration
# --------------------------------------------------------------------------

def reliability_bins(conf: np.ndarray, correct: np.ndarray, n_bins: int = 15):
    """Equal-width confidence bins. Returns per-bin (lo, hi, n, conf, acc)."""
    edges = np.linspace(0.0, 1.0, n_bins + 1)
    # right-closed bins so conf == 1.0 lands in the last bin rather than
    # opening an empty one past the end.
    idx = np.clip(np.digitize(conf, edges[1:-1], right=True), 0, n_bins - 1)
    rows = []
    for b in range(n_bins):
        m = idx == b
        rows.append({
            "bin_lo": float(edges[b]), "bin_hi": float(edges[b + 1]),
            "count": int(m.sum()),
            "mean_confidence": float(conf[m].mean()) if m.any() else float("nan"),
            "accuracy": float(correct[m].mean()) if m.any() else float("nan"),
        })
    return rows


def expected_calibration_error(probs: np.ndarray, y_true: np.ndarray,
                               n_bins: int = 15) -> float:
    conf = probs.max(1)
    correct = (probs.argmax(1) == y_true).astype(np.float64)
    rows = reliability_bins(conf, correct, n_bins)
    n = len(y_true)
    return float(sum(r["count"] / n * abs(r["accuracy"] - r["mean_confidence"])
                     for r in rows if r["count"] > 0))


def classwise_ece(probs: np.ndarray, y_true: np.ndarray, n_bins: int = 15,
                  num_classes: int = NUM_CLASSES,
                  class_names: Optional[List[str]] = None) -> Dict[str, float]:
    """Static-calibration error: ECE of each class's probability channel
    against that class's empirical frequency, then averaged."""
    names = class_names or CUE_CLASSES
    per = {}
    for c in range(num_classes):
        conf = probs[:, c]
        correct = (y_true == c).astype(np.float64)
        rows = reliability_bins(conf, correct, n_bins)
        n = len(y_true)
        per[names[c]] = float(
            sum(r["count"] / n * abs(r["accuracy"] - r["mean_confidence"])
                for r in rows if r["count"] > 0))
    per["macro"] = float(np.mean(list(per.values())))
    return per


def brier_score(probs: np.ndarray, y_true: np.ndarray,
                num_classes: int = NUM_CLASSES) -> float:
    """Multiclass Brier score: mean squared error against the one-hot target.
    Range [0, 2]; lower is better."""
    onehot = np.zeros_like(probs)
    onehot[np.arange(len(y_true)), y_true] = 1.0
    return float(((probs - onehot) ** 2).sum(1).mean())


def negative_log_likelihood(probs: np.ndarray, y_true: np.ndarray) -> float:
    p = np.clip(probs[np.arange(len(y_true)), y_true], 1e-12, 1.0)
    return float(-np.log(p).mean())


# --------------------------------------------------------------------------
# aggregate
# --------------------------------------------------------------------------

def frame_metrics(probs: np.ndarray, y_true: np.ndarray,
                  y_pred: Optional[np.ndarray] = None,
                  num_classes: int = NUM_CLASSES,
                  n_bins: int = 15,
                  class_names: Optional[List[str]] = None) -> Dict:
    """The complete frame-level metric block for one model on one split.

    ``class_names`` must match ``num_classes`` and the id order the labels use.
    It defaults to the 6 cue classes; a coarser taxonomy passes its own, so the
    per-class block is keyed by names that mean what they say rather than by
    six labels silently reused for two or three classes.
    """
    names = class_names or CUE_CLASSES
    if len(names) != num_classes:
        raise ValueError(f"{len(names)} class names for {num_classes} classes")
    if len(y_true) and int(np.min(y_true)) < 0:
        # IGNORE_INDEX (-100) reaching here means abstained frames were not
        # filtered. Caught explicitly because the natural symptom is an
        # IndexError from brier_score's one-hot, which points at the wrong
        # place -- and a metric that happened not to index by label would have
        # quietly averaged over frames the model was never asked to predict.
        raise ValueError(
            f"{int((np.asarray(y_true) < 0).sum())} of {len(y_true)} labels are "
            f"negative (IGNORE_INDEX). Filter abstained frames before scoring; "
            f"run_eval.predict() does this on `y != IGNORE_INDEX`.")
    if y_pred is None:
        y_pred = probs.argmax(1)
    cm = confusion_matrix(y_true, y_pred, num_classes)
    prf = per_class_prf(cm)
    rank = ranking_metrics(probs, y_true, num_classes)
    finite = lambda v: [x for x in v if not np.isnan(x)]
    conf = probs.max(1)
    correct = (y_pred == y_true).astype(np.float64)
    return {
        "n_frames": int(len(y_true)),
        "accuracy": accuracy(cm),
        "balanced_accuracy": balanced_accuracy(cm),
        "macro_precision": float(prf["precision"][prf["support"] > 0].mean()),
        "macro_recall": float(prf["recall"][prf["support"] > 0].mean()),
        "macro_f1": macro_f1(cm),
        "weighted_f1": weighted_f1(cm),
        "macro_auprc": float(np.mean(finite(rank["auprc"]))) if finite(rank["auprc"]) else float("nan"),
        "macro_auroc": float(np.mean(finite(rank["auroc"]))) if finite(rank["auroc"]) else float("nan"),
        "ece": expected_calibration_error(probs, y_true, n_bins),
        "classwise_ece": classwise_ece(probs, y_true, n_bins, num_classes, names),
        "brier": brier_score(probs, y_true, num_classes),
        "nll": negative_log_likelihood(probs, y_true),
        "per_class": {
            names[c]: {
                "precision": float(prf["precision"][c]),
                "recall": float(prf["recall"][c]),
                "f1": float(prf["f1"][c]),
                "auprc": rank["auprc"][c],
                "auroc": rank["auroc"][c],
                "support": int(prf["support"][c]),
            } for c in range(num_classes)
        },
        "confusion_matrix": cm.tolist(),
        "class_order": list(names),
        "reliability": reliability_bins(conf, correct, n_bins),
    }
