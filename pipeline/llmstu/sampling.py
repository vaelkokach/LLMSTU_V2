"""Golden-set selection: stratified representative set + low-confidence hard subset.

Shared by scripts/03_sample_eval.py (local crops) and scripts/14_golden_from_hf.py
(crops pulled from the Hub) so both sample identically.
"""

from __future__ import annotations

import random
from collections import defaultdict
from typing import Dict, List, Optional, Tuple

from . import schema


def _conf(r: Dict):
    c = r.get(schema.MODEL_CONFIDENCE_FIELD)
    return c if isinstance(c, (int, float)) else None


def select(rows: List[Dict], n_total: int, fields: List[str],
           min_per_class: int, hard_frac: float,
           seed: int = 13) -> List[Tuple[int, str]]:
    """Return [(row_index, group)] where group is 'representative' or 'hard'."""
    random.seed(seed)
    idx = list(range(len(rows)))
    n_hard = int(hard_frac * n_total)
    n_repr = n_total - n_hard

    # representative: per-class floor across each stratify field, then random fill
    selected = set()
    for f in fields:
        buckets = defaultdict(list)
        for i in idx:
            buckets[str(rows[i].get(f))].append(i)
        for _, items in buckets.items():
            random.shuffle(items)
            selected.update(items[:min_per_class])
    remaining = [i for i in idx if i not in selected]
    random.shuffle(remaining)
    while len(selected) < n_repr and remaining:
        selected.add(remaining.pop())
    repr_idx = set(selected)

    # hard: lowest model_confidence not already picked
    scored = sorted(((_conf(rows[i]), i) for i in idx
                     if i not in repr_idx and _conf(rows[i]) is not None),
                    key=lambda x: x[0])
    hard_idx = set(i for _, i in scored[:n_hard])
    if len(hard_idx) < n_hard:
        pool = [i for i in idx if i not in repr_idx and i not in hard_idx]
        random.shuffle(pool)
        while len(hard_idx) < n_hard and pool:
            hard_idx.add(pool.pop())

    picks = ([(i, "representative") for i in repr_idx]
             + [(i, "hard") for i in hard_idx])
    random.shuffle(picks)
    return picks
