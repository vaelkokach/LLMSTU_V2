"""Student-unit parsing and the four matching strategies.

A "student unit" is one student's whole description (action + emotion / full
caption) treated as an atomic item, so all of a student's phrases move to the
same box together (to-do item 26: match at student-unit level, not phrase
level).

All matchers return a list of (unit_idx, box_idx) pairs over the *presented*
orderings; unmatched units/boxes are simply absent from the list.
"""

import re
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple

import numpy as np
from scipy.optimize import linear_sum_assignment

# Matches "Student 3: typing intently, focused." in the old whole-frame captions.
_UNIT_RE = re.compile(r"student\s+(\d+)\s*:\s*([^.]*?)(?:\.|$)", re.IGNORECASE)


@dataclass
class StudentUnit:
    """One student's description parsed out of a whole-frame caption."""

    index: int                      # the N in "Student N" (1-based as written)
    text: str                       # full unit text ("typing intently, focused")
    action: str = ""
    emotion: str = ""
    format_valid: bool = True       # parsed cleanly into action + emotion
    meta: Dict = field(default_factory=dict)


def parse_student_units(caption: str) -> List[StudentUnit]:
    """Parse "Student N: action, emotion." units from a whole-frame caption."""
    units = []
    for m in _UNIT_RE.finditer(caption):
        idx = int(m.group(1))
        text = m.group(2).strip()
        parts = [p.strip() for p in text.split(",")]
        action = parts[0] if parts else ""
        emotion = parts[1] if len(parts) > 1 else ""
        valid = bool(action) and bool(emotion) and len(action.split()) <= 5
        units.append(StudentUnit(index=idx, text=text, action=action,
                                 emotion=emotion, format_valid=valid))
    return units


# ---------------------------------------------------------------------------
# Strategies
# ---------------------------------------------------------------------------

def match_ordinal(n_units: int, n_boxes: int) -> List[Tuple[int, int]]:
    """Unit k -> box k in the order boxes were presented (detector output
    order). This is the current pipeline baseline (jsonl_formatter.py:72)."""
    return [(k, k) for k in range(min(n_units, n_boxes))]


def match_ordinal_lr(n_units: int, boxes: np.ndarray) -> List[Tuple[int, int]]:
    """Unit k -> k-th box sorted by x-center (left-to-right reading order).

    boxes: (M, 4) xyxy in presented order. Returns pairs indexed against the
    presented order.
    """
    xc = (boxes[:, 0] + boxes[:, 2]) / 2.0
    order = np.argsort(xc, kind="stable")
    return [(k, int(order[k])) for k in range(min(n_units, len(order)))]


def match_hungarian(cost: np.ndarray,
                    reject_cost: Optional[float] = None
                    ) -> List[Tuple[int, int]]:
    """Globally optimal assignment on an (N_units, M_boxes) cost matrix.

    If reject_cost is given, the matrix is padded with dustbin rows/columns at
    that cost so a unit may stay unassigned when every real cost is worse.
    """
    n, m = cost.shape
    if reject_cost is None:
        r, c = linear_sum_assignment(cost)
        return [(int(i), int(j)) for i, j in zip(r, c)]
    size = n + m  # enough dustbins for everything to opt out
    padded = np.full((size, size), reject_cost, dtype=np.float64)
    padded[:n, :m] = cost
    padded[n:, m:] = 0.0  # dustbin-to-dustbin is free
    r, c = linear_sum_assignment(padded)
    return [(int(i), int(j)) for i, j in zip(r, c) if i < n and j < m]


def sinkhorn_log(cost: np.ndarray, eps: float = 0.1,
                 n_iters: int = 200) -> np.ndarray:
    """Balanced entropic OT in log domain; returns the transport plan P."""
    n, m = cost.shape
    log_a = np.full(n, -np.log(n))
    log_b = np.full(m, -np.log(m))
    K = -cost / eps  # log kernel
    f = np.zeros(n)
    g = np.zeros(m)
    for _ in range(n_iters):
        # f_i = eps*(log a_i - logsumexp_j (K_ij + g_j/eps))
        f = eps * (log_a - _logsumexp(K + g[None, :] / eps, axis=1))
        g = eps * (log_b - _logsumexp(K + f[:, None] / eps, axis=0))
    logP = K + f[:, None] / eps + g[None, :] / eps
    return np.exp(logP)


def _logsumexp(x: np.ndarray, axis: int) -> np.ndarray:
    mx = np.max(x, axis=axis, keepdims=True)
    out = mx.squeeze(axis) + np.log(np.sum(np.exp(x - mx), axis=axis))
    return out


def match_sinkhorn(cost: np.ndarray, eps: float = 0.1, n_iters: int = 200,
                   dustbin_cost: Optional[float] = None,
                   ) -> Tuple[List[Tuple[int, int]], np.ndarray]:
    """Entropic OT matching with dustbin padding for the unbalanced case.

    The cost matrix is padded to square with dustbin cells (cost =
    dustbin_cost, default: cost mean + 1 std) so extra units/boxes can absorb
    mass. The soft plan is discretized to a hard assignment by running
    Hungarian on -P (a valid global rounding of the plan). Returns (pairs,
    plan restricted to real cells).
    """
    n, m = cost.shape
    if dustbin_cost is None:
        dustbin_cost = float(cost.mean() + cost.std())
    size = max(n, m)
    padded = np.full((size, size), dustbin_cost, dtype=np.float64)
    padded[:n, :m] = cost
    P = sinkhorn_log(padded, eps=eps, n_iters=n_iters)
    r, c = linear_sum_assignment(-P)
    pairs = [(int(i), int(j)) for i, j in zip(r, c) if i < n and j < m]
    return pairs, P[:n, :m]
