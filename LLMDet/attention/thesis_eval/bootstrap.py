"""Cluster bootstrap confidence intervals.

Frames inside one (video, seat) track are massively autocorrelated — the cue
persists for tens of consecutive seconds by construction, which is the entire
premise of the event layer. Resampling *frames* would therefore treat ~271k
highly dependent observations as independent and produce intervals roughly an
order of magnitude too narrow.

Resampling unit is the **cluster**: a video (default) or a track. Comparisons
between two models evaluated on the same clusters use **paired** resampling —
the same cluster draw is applied to both systems, so the interval is on the
*difference*, which is what a claim like "556 beats 570 by [value removed]" needs.
"""

from __future__ import annotations

from typing import Callable, Dict, List, Optional, Sequence, Tuple

import numpy as np


def cluster_bootstrap(
    cluster_ids: np.ndarray,
    statistic: Callable[[np.ndarray], float],
    n_boot: int = 2000,
    alpha: float = 0.05,
    seed: int = 0,
) -> Dict[str, float]:
    """Percentile bootstrap CI for ``statistic(row_index_selection)``.

    ``statistic`` receives an index array into the original rows and returns a
    scalar. Clusters are drawn with replacement; all rows of a drawn cluster
    are included (and a cluster drawn twice contributes its rows twice).
    """
    uniq, inv = np.unique(cluster_ids, return_inverse=True)
    rows_by_cluster = [np.flatnonzero(inv == c) for c in range(len(uniq))]
    rng = np.random.default_rng(seed)
    point = float(statistic(np.arange(len(cluster_ids))))
    vals = np.empty(n_boot, dtype=np.float64)
    for b in range(n_boot):
        draw = rng.integers(0, len(uniq), size=len(uniq))
        idx = np.concatenate([rows_by_cluster[d] for d in draw])
        vals[b] = statistic(idx)
    finite = vals[np.isfinite(vals)]
    lo, hi = (np.percentile(finite, [100 * alpha / 2, 100 * (1 - alpha / 2)])
              if finite.size else (np.nan, np.nan))
    return {
        "point": point,
        "ci_low": float(lo), "ci_high": float(hi),
        "boot_mean": float(finite.mean()) if finite.size else float("nan"),
        "boot_std": float(finite.std(ddof=1)) if finite.size > 1 else float("nan"),
        "n_clusters": int(len(uniq)), "n_boot": int(n_boot), "alpha": alpha,
    }


def cluster_bootstrap_multi(
    cluster_ids: np.ndarray,
    statistics: Callable[[np.ndarray], Dict[str, float]],
    n_boot: int = 2000,
    alpha: float = 0.05,
    seed: int = 0,
) -> Dict[str, Dict[str, float]]:
    """Bootstrap many statistics from **one** set of resamples.

    Bootstrapping each metric with its own draws would be both slower and
    subtly wrong for comparisons across metrics: the intervals would come from
    different resamples of the same data. One draw, all statistics.
    """
    uniq, inv = np.unique(cluster_ids, return_inverse=True)
    rows_by_cluster = [np.flatnonzero(inv == c) for c in range(len(uniq))]
    rng = np.random.default_rng(seed)
    point = statistics(np.arange(len(cluster_ids)))
    names = list(point)
    acc = {k: np.empty(n_boot, dtype=np.float64) for k in names}
    for b in range(n_boot):
        draw = rng.integers(0, len(uniq), size=len(uniq))
        idx = np.concatenate([rows_by_cluster[d] for d in draw])
        vals = statistics(idx)
        for k in names:
            acc[k][b] = vals[k]
    out = {}
    for k in names:
        finite = acc[k][np.isfinite(acc[k])]
        lo, hi = (np.percentile(finite, [100 * alpha / 2, 100 * (1 - alpha / 2)])
                  if finite.size else (np.nan, np.nan))
        out[k] = {"point": float(point[k]), "ci_low": float(lo), "ci_high": float(hi),
                  "boot_mean": float(finite.mean()) if finite.size else float("nan"),
                  "boot_std": float(finite.std(ddof=1)) if finite.size > 1 else float("nan"),
                  "n_clusters": int(len(uniq)), "n_boot": int(n_boot), "alpha": alpha}
    return out


def paired_cluster_bootstrap_multi(
    cluster_ids: np.ndarray,
    statistics_a: Callable[[np.ndarray], Dict[str, float]],
    statistics_b: Callable[[np.ndarray], Dict[str, float]],
    n_boot: int = 2000,
    alpha: float = 0.05,
    seed: int = 0,
) -> Dict[str, Dict[str, float]]:
    """Paired version of :func:`cluster_bootstrap_multi`: one shared cluster
    draw feeds both systems, and the interval is on the difference."""
    uniq, inv = np.unique(cluster_ids, return_inverse=True)
    rows_by_cluster = [np.flatnonzero(inv == c) for c in range(len(uniq))]
    rng = np.random.default_rng(seed)
    full = np.arange(len(cluster_ids))
    pa, pb = statistics_a(full), statistics_b(full)
    names = list(pa)
    acc = {k: np.empty(n_boot, dtype=np.float64) for k in names}
    for b in range(n_boot):
        draw = rng.integers(0, len(uniq), size=len(uniq))
        idx = np.concatenate([rows_by_cluster[d] for d in draw])
        va, vb = statistics_a(idx), statistics_b(idx)
        for k in names:
            acc[k][b] = va[k] - vb[k]
    out = {}
    for k in names:
        finite = acc[k][np.isfinite(acc[k])]
        lo, hi = (np.percentile(finite, [100 * alpha / 2, 100 * (1 - alpha / 2)])
                  if finite.size else (np.nan, np.nan))
        p = (2.0 * min((finite <= 0).mean(), (finite >= 0).mean())
             if finite.size else float("nan"))
        out[k] = {"difference": float(pa[k] - pb[k]),
                  "a": float(pa[k]), "b": float(pb[k]),
                  "ci_low": float(lo), "ci_high": float(hi),
                  "boot_std": float(finite.std(ddof=1)) if finite.size > 1 else float("nan"),
                  "p_value_two_sided": float(min(1.0, p)),
                  "significant_at_alpha": bool(np.isfinite(lo) and np.isfinite(hi)
                                               and (lo > 0 or hi < 0)),
                  "n_clusters": int(len(uniq)), "n_boot": int(n_boot), "alpha": alpha}
    return out


def paired_cluster_bootstrap(
    cluster_ids: np.ndarray,
    statistic_a: Callable[[np.ndarray], float],
    statistic_b: Callable[[np.ndarray], float],
    n_boot: int = 2000,
    alpha: float = 0.05,
    seed: int = 0,
) -> Dict[str, float]:
    """CI on ``stat_a - stat_b`` under a shared cluster resample.

    Also reports a two-sided bootstrap p-value: the fraction of resamples whose
    difference falls on the opposite side of zero from the point estimate,
    doubled. This is the test the thesis needs before calling a [value removed] macro-F1
    gap real.
    """
    uniq, inv = np.unique(cluster_ids, return_inverse=True)
    rows_by_cluster = [np.flatnonzero(inv == c) for c in range(len(uniq))]
    rng = np.random.default_rng(seed)
    full = np.arange(len(cluster_ids))
    point = float(statistic_a(full) - statistic_b(full))
    diffs = np.empty(n_boot, dtype=np.float64)
    for b in range(n_boot):
        draw = rng.integers(0, len(uniq), size=len(uniq))
        idx = np.concatenate([rows_by_cluster[d] for d in draw])
        diffs[b] = statistic_a(idx) - statistic_b(idx)
    finite = diffs[np.isfinite(diffs)]
    lo, hi = (np.percentile(finite, [100 * alpha / 2, 100 * (1 - alpha / 2)])
              if finite.size else (np.nan, np.nan))
    if finite.size:
        p = 2.0 * min((finite <= 0).mean(), (finite >= 0).mean())
    else:
        p = float("nan")
    return {
        "difference": point, "ci_low": float(lo), "ci_high": float(hi),
        "boot_std": float(finite.std(ddof=1)) if finite.size > 1 else float("nan"),
        "p_value_two_sided": float(min(1.0, p)),
        "significant_at_alpha": bool(np.isfinite(lo) and np.isfinite(hi)
                                     and (lo > 0 or hi < 0)),
        "n_clusters": int(len(uniq)), "n_boot": int(n_boot), "alpha": alpha,
    }


def seed_summary(values: Sequence[float]) -> Dict[str, float]:
    """Mean / sd / normal-approximation 95% CI over independent seeds.

    Reported alongside — not instead of — the bootstrap interval: the bootstrap
    quantifies *data* uncertainty, the seed spread quantifies *training*
    uncertainty, and this project's epoch-to-epoch validation macro-F1 swings by
    ~0.05, so the second is not negligible.
    """
    v = np.asarray([x for x in values if np.isfinite(x)], dtype=np.float64)
    if v.size == 0:
        return {"mean": float("nan"), "std": float("nan"), "n": 0}
    sd = float(v.std(ddof=1)) if v.size > 1 else 0.0
    half = 1.96 * sd / np.sqrt(v.size) if v.size > 1 else 0.0
    return {"mean": float(v.mean()), "std": sd, "n": int(v.size),
            "ci_low": float(v.mean() - half), "ci_high": float(v.mean() + half),
            "values": [float(x) for x in v]}
