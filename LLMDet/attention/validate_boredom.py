"""Validate facial-expression → BOREDOM against DIPSER expert labels.

`Thesis_Topic.md` names "facial expressions (e.g., boredom, perplexity,
curiosity)" as a key indicator. Those are **academic** emotions (Pekrun's
framework), not the 7 basic Ekman expressions any off-the-shelf FER model
predicts. Claiming a FER model "detects boredom" without evidence would be
exactly the kind of unfounded assertion this project has spent days removing.

DIPSER labels emotion with the 9-category academic scheme, expert-annotated:

    1 Boredom   2 Despair  3 Shame   4 Anxiety  5 Anger
    6 Relief    7 Pride    8 Hope    9 Enjoyment

Code 1 is Boredom — one of the three the thesis names, with real ground truth.
This script measures how well basic facial-expression probabilities predict
expert-labelled boredom, giving an honest, citable number instead of a claim.

Uses DIPSER's OWN per-frame emotion probabilities from metadata (the same
7-class basic-expression family our FER model outputs), so it runs on metadata
alone — no images, no GPU — and isolates the *construct* question (do basic
expressions carry boredom signal?) from our model's domain transfer.

Reports per-expression AUROC plus a logistic combination, with a permutation
test because the classes are heavily imbalanced.

Usage:
    python -m attention.validate_boredom \
        --dipser-root ../grounding_data/external/DIPSER \
        --window 15 --out work_dirs/boredom_validation.json
"""
import argparse
import json
import shutil
import tempfile
import zipfile
from pathlib import Path

import numpy as np

BOREDOM = 1
EMO_KEYS = ["angry", "disgust", "fear", "happy", "sad", "surprise", "neutral"]


def parse_ts(s):
    import re
    p = re.split(r"[:_]", s)
    if len(p) < 3:
        return float("nan")
    return (int(p[0]) * 3600 + int(p[1]) * 60 + int(p[2])
            + (int(p[3]) / 1e6 if len(p) > 3 else 0))


def auroc(scores, labels):
    """Rank-based AUROC; ties handled by average ranks."""
    s = np.asarray(scores, float)
    y = np.asarray(labels, int)
    n1, n0 = int(y.sum()), int((1 - y).sum())
    if n1 == 0 or n0 == 0:
        return float("nan")
    order = np.argsort(s)
    ranks = np.empty(len(s), float)
    ranks[order] = np.arange(1, len(s) + 1)
    # average ranks within tie groups
    _, inv, cnt = np.unique(s, return_inverse=True, return_counts=True)
    sums = np.zeros(len(cnt))
    np.add.at(sums, inv, ranks)
    ranks = (sums / cnt)[inv]
    return float((ranks[y == 1].sum() - n1 * (n1 + 1) / 2) / (n1 * n0))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dipser-root", required=True)
    ap.add_argument("--window", type=float, default=15.0)
    ap.add_argument("--max-subjects", type=int, default=26)
    ap.add_argument("--out", required=True)
    args = ap.parse_args()

    X, Y, subj = [], [], []
    for zp in sorted(Path(args.dipser_root).glob("*.zip"))[:args.max_subjects]:
        tmp = Path(tempfile.mkdtemp(prefix="bore_"))
        try:
            try:
                zipfile.ZipFile(zp).close()
            except zipfile.BadZipFile:
                print(f"  {zp.stem}: corrupt archive, skipped")
                continue
            with zipfile.ZipFile(zp) as z:
                z.extractall(tmp, members=[n for n in z.namelist()
                                           if n.startswith(("metadata/", "labels/"))])
            labels = []
            for lf in (tmp / "labels").glob("*.json"):
                for r in json.load(lf.open()):
                    if "emotion" in r and "datetime" in r:
                        t = parse_ts(r["datetime"])
                        if not np.isnan(t):
                            labels.append((t, int(r["emotion"])))
            ts, probs = [], []
            for mf in sorted((tmp / "metadata").glob("*.json")):
                t = parse_ts(mf.stem)
                if np.isnan(t):
                    continue
                try:
                    md = json.load(mf.open())
                    face = ((md or {}).get("person") or {}).get("face") or {}
                    pe = (face.get("emotion") or {}).get("probability_emotion")
                except (AttributeError, TypeError, json.JSONDecodeError):
                    continue
                if not pe:
                    continue
                v = np.array([float(pe.get(k, 0.0)) for k in EMO_KEYS], float)
                s = v.sum()
                probs.append(v / s if s > 0 else v)
                ts.append(t)
            if not ts or not labels:
                continue
            ts = np.asarray(ts)
            probs = np.stack(probs)
            got = 0
            for lt, emo in labels:
                sel = np.abs(ts - lt) <= args.window
                if sel.sum() < 3:
                    continue
                X.append(probs[sel].mean(0))
                Y.append(1 if emo == BOREDOM else 0)
                subj.append(zp.stem)
                got += 1
            print(f"  {zp.stem}: {len(ts)} frames, {got} paired")
        finally:
            shutil.rmtree(tmp, ignore_errors=True)

    if len(X) < 20:
        raise SystemExit(f"only {len(X)} paired observations")
    X = np.stack(X)
    Y = np.asarray(Y, int)
    print(f"\npaired observations {len(Y)}  boredom {Y.sum()} "
          f"({Y.mean():.1%})  other {len(Y)-Y.sum()}")

    per = {}
    for j, k in enumerate(EMO_KEYS):
        per[k] = auroc(X[:, j], Y)

    # Logistic combination, subject-wise split so it is not scored on the
    # subjects it was fitted on.
    uniq = sorted(set(subj))
    half = set(uniq[::2])
    tr = np.array([s in half for s in subj])
    comb = float("nan")
    if tr.sum() > 10 and (~tr).sum() > 10 and Y[tr].sum() > 2 and Y[~tr].sum() > 2:
        Xt = np.c_[X[tr], np.ones(tr.sum())]
        w = np.zeros(Xt.shape[1])
        for _ in range(300):          # plain gradient ascent; tiny problem
            p = 1 / (1 + np.exp(-Xt @ w))
            w += 0.5 * Xt.T @ (Y[tr] - p) / len(Y[tr])
        comb = auroc(np.c_[X[~tr], np.ones((~tr).sum())] @ w, Y[~tr])

    best = max((v for v in per.values() if not np.isnan(v)), default=float("nan"))
    rng = np.random.default_rng(0)
    key = max(per, key=lambda k: abs(per[k] - 0.5) if not np.isnan(per[k]) else -1)
    null = np.array([auroc(X[:, EMO_KEYS.index(key)], rng.permutation(Y))
                     for _ in range(2000)])
    p_val = float((np.abs(null - 0.5) >= abs(per[key] - 0.5)).mean())

    out = {"n": int(len(Y)), "n_boredom": int(Y.sum()),
           "boredom_rate": float(Y.mean()), "window_s": args.window,
           "auroc_per_expression": per,
           "auroc_logistic_combination_heldout_subjects": comb,
           "most_discriminative": key, "perm_p_for_that": p_val,
           "note": ("Basic (Ekman) facial-expression probabilities from DIPSER "
                    "metadata vs EXPERT academic-emotion labels (1 = Boredom, "
                    "Pekrun framework). AUROC 0.5 = chance.")}
    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    Path(args.out).write_text(json.dumps(out, indent=2))

    print("\n=== Basic expression -> expert BOREDOM (AUROC, 0.5 = chance) ===")
    for k in EMO_KEYS:
        bar = "" if np.isnan(per[k]) else ("  <-- most discriminative"
                                           if k == key else "")
        print(f"  {k:9s} {per[k]:.3f}{bar}")
    print(f"\n  logistic combination (held-out subjects): {comb:.3f}")
    print(f"  permutation p for {key}: {p_val:.4f}")
    print(f"\nwritten: {args.out}")


if __name__ == "__main__":
    main()
