#!/usr/bin/env python
"""Is the visual encoder what limits cue accuracy? A linear probe says.

Embeds the SAME crops with two encoders, fits multinomial logistic regression to
the cue label, and compares. If a bigger encoder's representation is not more
linearly separable for these cues, swapping it in is unlikely to repay a corpus
rebuild -- and the prior here is that it will not: the head stream, a whole extra
CLIP pass, is worth about +[value removed] against [value removed] for the frame rate.

**Scored on the HUMAN gold set**, not the pseudo-labels: the LLMSTU labels were
made by a Qwen3.5-VL teacher, and a probe scored against them would reward
agreeing with that teacher rather than reading the image.

**The dimension control matters.** SigLIP2-so400m is 1152-dim against CLIP
ViT-B/32's 512, and a higher-dimensional representation can win a linear probe on
capacity alone. This project has been bitten by exactly that shape of artefact
before (a y_frac AUROC of [value removed] that was a default value). So three numbers are
reported per encoder: raw, PCA-reduced to a common dimension, and the
regularisation swept by cross-validation. If it only wins raw, it has not won.
"""
import argparse, json, sys, time
from collections import Counter
from pathlib import Path

import numpy as np

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / "LLMDet"))
GOLD = REPO / "Gold_annotation_wael" / "gold_annotations_wael.jsonl"
CROPS = REPO / "grounding_data" / "LLMSTU" / "crops"


def load_gold(taxonomy, limit):
    from attention import taxonomy as TX
    if taxonomy == "cue9":
        classes, mapper = list(TX.CUE9_CLASSES), TX.map_record_cue9
    else:
        classes, mapper = list(TX.CUE_CLASSES), TX.map_record
    rows, skip = [], Counter()
    for line in GOLD.read_text().splitlines():
        rec = json.loads(line)
        if rec.get("status") != "ok":
            skip["status"] += 1; continue
        p = CROPS / rec["file_name"]
        if not p.exists():
            skip["missing crop"] += 1; continue
        y = mapper(rec)
        if classes[y] == "uncertain":
            skip["gold uncertain"] += 1; continue
        rows.append((p, y))
        if limit and len(rows) >= limit: break
    return classes, rows, skip


def embed(model_id, paths, device, batch=16):
    import torch
    from PIL import Image
    import transformers
    t0 = time.time()
    if "clip" in model_id:
        m = transformers.CLIPModel.from_pretrained(model_id).to(device).eval()
        proc = transformers.CLIPProcessor.from_pretrained(model_id)
        fn = lambda i: m.get_image_features(**i)
    else:
        # Pick the class from the checkpoint's OWN model_type, not from whatever
        # the installed transformers happens to expose. `siglip2-so400m-patch14
        # -384` is the fixed-resolution variant and declares model_type
        # "siglip", i.e. the v1 architecture; only the `-naflex` checkpoints use
        # the Siglip2* classes, whose patch embedding is a Linear rather than a
        # Conv2d. Guessing by availability loads the wrong one and fails with a
        # shape mismatch of [1152,3,14,14] against [1152,588].
        mt = transformers.AutoConfig.from_pretrained(model_id).model_type
        cls = getattr(transformers, "Siglip2VisionModel") if mt == "siglip2" \
            else getattr(transformers, "SiglipVisionModel")
        m = cls.from_pretrained(model_id).to(device).eval()
        proc = transformers.AutoProcessor.from_pretrained(model_id)
        fn = lambda i: m(**i).pooler_output
    out = []
    for s in range(0, len(paths), batch):
        ims = [Image.open(p).convert("RGB") for p in paths[s:s + batch]]
        with torch.inference_mode():
            i = proc(images=ims, return_tensors="pt").to(device)
            i = {k: (v.to(next(m.parameters()).dtype)
                     if hasattr(v, "is_floating_point") and v.is_floating_point()
                     else v) for k, v in i.items()}
            out.append(fn(i).float().cpu().numpy())
    del m
    torch.cuda.empty_cache()
    e = np.concatenate(out)
    print(f"  {model_id}: {e.shape} in {time.time()-t0:.0f}s")
    return e


def probe(X, y, classes, seed=0, pca_dim=None):
    from sklearn.decomposition import PCA
    from sklearn.linear_model import LogisticRegression
    from sklearn.model_selection import StratifiedKFold
    from sklearn.preprocessing import StandardScaler
    from sklearn.metrics import f1_score
    X = np.asarray(X, dtype=np.float64)
    if pca_dim and pca_dim < X.shape[1]:
        X = PCA(n_components=pca_dim, random_state=seed).fit_transform(X)
    best = (-1, None)
    for C in (0.003, 0.01, 0.03, 0.1, 0.3, 1.0):
        preds = np.zeros_like(y)
        for tr, te in StratifiedKFold(5, shuffle=True, random_state=seed).split(X, y):
            sc = StandardScaler().fit(X[tr])
            lr = LogisticRegression(C=C, max_iter=3000, multi_class="multinomial")
            lr.fit(sc.transform(X[tr]), y[tr])
            preds[te] = lr.predict(sc.transform(X[te]))
        f1 = f1_score(y, preds, average="macro")
        if f1 > best[0]:
            best = (f1, C)
    return best


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--taxonomy", default="cue6", choices=("cue6", "cue9"))
    ap.add_argument("--limit", type=int, default=0)
    ap.add_argument("--device", default="cuda:0")
    ap.add_argument("--clip", default="openai/clip-vit-base-patch32")
    ap.add_argument("--challenger", default="google/siglip2-so400m-patch14-384")
    ap.add_argument("--pca-dims", default="384,256,128",
                    help="common dimensions BOTH encoders are reduced to")
    ap.add_argument("--cache", default="",
                    help="directory to cache embeddings, so reruns are instant")
    ap.add_argument("--out", default="")
    a = ap.parse_args()

    classes, rows, skip = load_gold(a.taxonomy, a.limit)
    paths = [p for p, _ in rows]; y = np.array([t for _, t in rows])
    print(f"crops: {len(rows)} human-gold  (skipped {dict(skip)})")
    print(f"dist : {Counter(classes[i] for i in y).most_common()}\n")

    import hashlib
    cache = Path(a.cache) if a.cache else None
    def _emb(mid):
        if cache:
            f = cache / (hashlib.sha1((mid + str(len(paths))).encode()).hexdigest() + ".npy")
            if f.exists():
                v = np.load(f); print(f"  {mid}: {v.shape} (cached)"); return v
        v = embed(mid, paths, a.device)
        if cache:
            cache.mkdir(parents=True, exist_ok=True); np.save(f, v)
        return v

    E = {"clip": _emb(a.clip), "challenger": _emb(a.challenger)}

    # Symmetric controls. Reducing ONLY the larger encoder to the smaller one's
    # width compresses one side and leaves the other untouched, which is not a
    # dimension control at all -- it is a handicap, and the first run of this
    # reported a FAIL on exactly that basis. Both are reduced to the same
    # dimensions instead, so whatever compression costs, it costs both equally.
    dims = [d for d in (int(x) for x in a.pca_dims.split(","))
            if d < min(v.shape[1] for v in E.values())]
    cols = ["raw"] + [f"PCA{d}" for d in dims]
    res = {}
    print(f"\n{'encoder':>14} {'dim':>5}  " + "  ".join(f"{c:>9}" for c in cols))
    for k, X in E.items():
        row = {"dim": int(X.shape[1])}
        row["raw"] = probe(X, y, classes)[0]
        for d in dims:
            row[f"PCA{d}"] = probe(X, y, classes, pca_dim=d)[0]
        res[k] = row
        name = (a.clip if k == "clip" else a.challenger).split("/")[-1][:14]
        print(f"{name:>14} {X.shape[1]:>5}  "
              + "  ".join(f"{row[c]:>9.4f}" for c in cols))

    print(f"\n{'':>20}" + "  ".join(f"{c:>9}" for c in cols))
    deltas = {c: res["challenger"][c] - res["clip"][c] for c in cols}
    print(f"{'challenger - CLIP':>20}"
          + "  ".join(f"{deltas[c]:>+9.4f}" for c in cols))
    # The gate is the SYMMETRIC comparison: it must win at equal dimension, not
    # only where it is allowed more of them.
    key = cols[-1] if dims else "raw"
    verdict = "PASS" if deltas[key] > 0.02 and deltas["raw"] > 0.02 else "FAIL"
    print(f"\n  GATE: raw {deltas['raw']:+.4f} AND {key} {deltas[key]:+.4f} "
          f"both > +0.02  ->  {verdict}")
    d_raw, d_pca = deltas["raw"], deltas[key]
    if a.out:
        Path(a.out).write_text(json.dumps(
            {"n": len(rows), "taxonomy": a.taxonomy, "results": res,
             "delta_raw": d_raw, "delta_pca": d_pca, "gate": verdict}, indent=1))
    return 0 if verdict == "PASS" else 3


if __name__ == "__main__":
    sys.exit(main())
