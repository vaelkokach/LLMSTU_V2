"""DIPSER external validation: do our visible-cue events track expert engagement?

THE POINT (THESIS_PLAN 6, use 1): our taxonomy deliberately claims only visible
behaviour -- "the student's head is down", never "the student is disengaged".
That is epistemically clean but leaves the thesis asserting, without evidence,
that the cues it detects have anything to do with attention loss. DIPSER carries
independent EXPERT engagement ratings (1 = low ... 5 = maximum, 4 labellers plus
self-report). Correlating our off-task cue rate against those ratings is the
bridge from "we detect head-down episodes" to "these episodes indicate
disengagement".

We never TRAIN on DIPSER's ratings -- that would import inferred mental state
into a visible-cue taxonomy. We only correlate against them.

Design:
  * frame = the DIPSER image; person box = metadata person/body/bounding_box,
    so geometry features are frame-relative exactly as in LLMSTU. (Feeding a
    pre-cropped image with a whole-image box makes 16 of 556 dims constant --
    the bug that produced a false 0/24 in P0.3.)
  * head pose = DIPSER's own metadata (MediaPipe-derived, same estimator as
    ours), normalised identically, so no re-running the model.
  * for each expert label at time t, take the fraction of OFF-TASK cue
    predictions in a +/-window and rank-correlate that against the rating.

Ordinal ratings -> Spearman, never Pearson.

Usage:
    python -m attention.dipser_correlation \
        --dipser-root ../grounding_data/external/DIPSER \
        --config configs/attention_temporal_hp.yaml \
        --ckpt work_dirs/attention_temporal_hp/checkpoints/best.pth \
        --max-subjects 10 --window 15 \
        --out work_dirs/attention_temporal_hp/dipser_correlation.json
"""
import argparse
import json
import re
import shutil
import tempfile
import zipfile
from collections import defaultdict
from pathlib import Path

import numpy as np
import torch
import yaml

from attention.taxonomy import CUE_CLASSES
from attention.temporal_model import AttentionTransformer

# Cues that mean "not oriented to the task". screen_oriented is on-task;
# uncertain is excluded from the numerator AND denominator (unverifiable).
OFF_TASK = {"looking_away", "head_down", "turned_to_peer", "phone_use"}
ON_TASK = {"screen_oriented"}


def parse_ts(s: str) -> float:
    """'10_40_43_347718' or '10:40:43:347718' -> seconds since midnight."""
    p = re.split(r"[:_]", s)
    if len(p) < 3:
        return float("nan")
    h, m, sec = int(p[0]), int(p[1]), int(p[2])
    micro = int(p[3]) if len(p) > 3 else 0
    return h * 3600 + m * 60 + sec + micro / 1e6


def load_subject(zpath: Path, tmp: Path):
    """Extract one subject; return (frames, labels).

    frames: [(t, image_path, bbox, headpose4)]
    labels: [(t, rating)] pooled over all labellers (each is an independent
             observation; they annotate different timestamps).
    """
    with zipfile.ZipFile(zpath) as z:
        names = z.namelist()
        z.extractall(tmp, members=[n for n in names
                                   if n.startswith(("metadata/", "labels/"))])
        img_names = [n for n in names if n.startswith("images/")
                     and n.endswith(".png")]
        z.extractall(tmp, members=img_names)

    labels = []
    for lf in (tmp / "labels").glob("*.json"):
        for rec in json.load(lf.open()):
            if "attention" in rec and "datetime" in rec:
                t = parse_ts(rec["datetime"])
                if not np.isnan(t):
                    labels.append((t, int(rec["attention"]), lf.stem))

    frames = []
    for mf in sorted((tmp / "metadata").glob("*.json")):
        stem = mf.stem
        img = tmp / "images" / f"{stem}.png"
        if not img.exists():
            continue
        try:
            md = json.load(mf.open())
            person = (md or {}).get("person") or {}
            bb = ((person.get("body") or {}).get("bounding_box"))
            if not bb:
                continue
            bbox = [float(bb["x0"]), float(bb["y0"]),
                    float(bb["x1"]), float(bb["y1"])]
            hp = ((person.get("face") or {}).get("headpose") or {}).get("pose")
            if hp:
                v = np.array([hp["yaw"], hp["pitch"], hp["roll"]],
                             dtype=np.float32) / 90.0
                pose = np.concatenate([np.clip(v, -1, 1), [1.0]]).astype(np.float32)
            else:
                pose = np.zeros(4, dtype=np.float32)
        except (KeyError, TypeError, AttributeError, json.JSONDecodeError):
            continue
        t = parse_ts(stem)
        if not np.isnan(t):
            frames.append((t, str(img), bbox, pose))
    frames.sort(key=lambda r: r[0])
    return frames, labels


def pose_off_task(pose: np.ndarray) -> bool:
    """Off-task proxy from head pose alone, using DIPSER's own metadata.

    Deliberately crude and fixed a priori — no tuning against the ratings,
    which would manufacture the correlation it is meant to test:
      * face not found            -> head down / fully turned away
      * |yaw|   > 30 deg (0.33)   -> turned away from the task
      * pitch   > 30 deg (0.33)   -> head down
    """
    yaw, pitch, roll, found = pose
    if found < 0.5:
        return True
    return abs(yaw) > 0.33 or pitch > 0.33


def run_metadata_only(args):
    """Correlate a pose-derived off-task proxy against expert ratings.

    Needs only metadata + labels, so it is fast and — crucially — free of the
    domain shift that makes our model degenerate to the majority class on
    DIPSER's close-up cameras.
    """
    zips = sorted(Path(args.dipser_root).glob("*.zip"))[:args.max_subjects]
    pairs, per_subject = [], {}
    for zp in zips:
        tmp = Path(tempfile.mkdtemp(prefix="dipser_md_"))
        try:
            # A truncated download yields a BadZipFile; skip loudly rather than
            # aborting the whole sweep over one bad archive.
            try:
                zipfile.ZipFile(zp).close()
            except zipfile.BadZipFile:
                print(f"  {zp.stem}: CORRUPT ARCHIVE, skipped")
                continue
            with zipfile.ZipFile(zp) as z:
                z.extractall(tmp, members=[n for n in z.namelist()
                                           if n.startswith(("metadata/", "labels/"))])
            labels = []
            for lf in (tmp / "labels").glob("*.json"):
                for rec in json.load(lf.open()):
                    if "attention" in rec and "datetime" in rec:
                        t = parse_ts(rec["datetime"])
                        if not np.isnan(t):
                            labels.append((t, int(rec["attention"]), lf.stem))
            ts, offs = [], []
            for mf in sorted((tmp / "metadata").glob("*.json")):
                t = parse_ts(mf.stem)
                if np.isnan(t):
                    continue
                try:
                    md = json.load(mf.open())
                    # Any of person / face / headpose can be null when the
                    # upstream detector found nothing — .get() on None raises.
                    face = ((md or {}).get("person") or {}).get("face") or {}
                    hp = (face.get("headpose") or {}).get("pose")
                except (KeyError, TypeError, AttributeError, json.JSONDecodeError):
                    continue
                if hp:
                    v = np.clip(np.array([hp["yaw"], hp["pitch"], hp["roll"]],
                                         dtype=np.float32) / 90.0, -1, 1)
                    pose = np.concatenate([v, [1.0]])
                else:
                    pose = np.zeros(4, dtype=np.float32)
                ts.append(t); offs.append(pose_off_task(pose))
            if not ts or not labels:
                continue
            ts = np.array(ts); offs = np.array(offs)
            got = 0
            for lt, rating, who in labels:
                sel = np.abs(ts - lt) <= args.window
                if sel.sum() < 3:
                    continue
                pairs.append((float(offs[sel].mean()), rating, zp.stem, who))
                got += 1
            per_subject[zp.stem] = {"frames": len(ts), "paired": got,
                                    "off_task_rate": float(offs.mean())}
            print(f"  {zp.stem}: {len(ts)} frames, {got} paired, "
                  f"off-task {offs.mean():.1%}")
        finally:
            shutil.rmtree(tmp, ignore_errors=True)
    _report(pairs, per_subject, args, mode="metadata_only_headpose_proxy")


def _spearman(a, b):
    ra = np.argsort(np.argsort(a)).astype(float)
    rb = np.argsort(np.argsort(b)).astype(float)
    ra -= ra.mean(); rb -= rb.mean()
    return float((ra @ rb) / (np.sqrt((ra**2).sum() * (rb**2).sum()) + 1e-12))


def _report(pairs, per_subject, args, mode):
    if len(pairs) < 10:
        raise SystemExit(f"only {len(pairs)} paired observations — too few")
    off = np.array([p[0] for p in pairs])
    rat = np.array([p[1] for p in pairs], dtype=float)
    rho = _spearman(off, rat)
    rng = np.random.default_rng(0)
    null = np.array([_spearman(off, rng.permutation(rat)) for _ in range(5000)])
    p = float((np.abs(null) >= abs(rho)).mean())
    by = {int(r): float(off[rat == r].mean()) for r in sorted(set(rat.tolist()))}
    cnt = {int(r): int((rat == r).sum()) for r in sorted(set(rat.tolist()))}
    out = {"mode": mode, "n_pairs": len(pairs), "n_subjects": len(per_subject),
           "window_s": args.window, "spearman_rho": rho, "perm_p": p,
           "mean_off_task_rate_by_rating": by, "n_by_rating": cnt,
           "per_subject": per_subject,
           "note": ("Off-task rate vs INDEPENDENT expert engagement ratings "
                    "(1=low..5=max); never trained on. Hypothesis: rho < 0.")}
    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    Path(args.out).write_text(json.dumps(out, indent=2))
    print(f"\n=== DIPSER external validation [{mode}] ===")
    print(f"  pairs {len(pairs)} over {len(per_subject)} subjects, "
          f"window +/-{args.window:.0f}s")
    print(f"  Spearman rho = {rho:+.3f}   permutation p = {p:.4f}")
    print(f"  {'rating':>7} {'n':>5} {'mean off-task rate':>20}")
    for r in sorted(by):
        print(f"  {r:>7} {cnt[r]:>5} {by[r]:>19.1%}")
    print(f"\n  (hypothesis: rho < 0 — more off-task cues, lower engagement)")
    print(f"written: {args.out}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dipser-root", required=True)
    ap.add_argument("--config", required=True)
    ap.add_argument("--ckpt", required=True)
    ap.add_argument("--max-subjects", type=int, default=10)
    ap.add_argument("--window", type=float, default=15.0,
                    help="+/- seconds around each expert label")
    ap.add_argument("--device", default="cuda:0")
    ap.add_argument("--max-frames", type=int, default=1200,
                    help="cap per subject to bound runtime")
    ap.add_argument("--metadata-only", action="store_true",
                    help="skip our model; derive off-task from DIPSER's OWN "
                         "head pose. Tests the thesis PREMISE (do visible "
                         "orientation cues track engagement?) without "
                         "depending on our model transferring off-domain.")
    ap.add_argument("--out", required=True)
    args = ap.parse_args()

    import cv2
    from attention.features import StudentFeatureExtractor

    cfg = yaml.safe_load(open(args.config))
    if args.metadata_only:
        return run_metadata_only(args)
    device = torch.device(args.device if torch.cuda.is_available() else "cpu")
    want = int(cfg["model"]["input_dim"])

    feat = StudentFeatureExtractor(
        clip_model_name=cfg["features"].get("clip_model_name",
                                            "openai/clip-vit-base-patch32"),
        device=str(device))
    model = AttentionTransformer(
        input_dim=want,
        hidden_dim=int(cfg["model"]["hidden_dim"]),
        num_layers=int(cfg["model"]["num_layers"]),
        num_heads=int(cfg["model"]["num_heads"]),
        dropout=float(cfg["model"]["dropout"]),
        num_classes=int(cfg["model"]["num_classes"]),
        max_seq_len=int(cfg["model"]["max_seq_len"]),
        per_frame=True).to(device)
    sd = torch.load(args.ckpt, map_location="cpu")
    model.load_state_dict(sd["model"] if "model" in sd else sd)
    model.eval()

    zips = sorted(Path(args.dipser_root).glob("*.zip"))[:args.max_subjects]
    pairs, per_subject = [], {}
    off_ids = {CUE_CLASSES.index(c) for c in OFF_TASK}

    for zp in zips:
        tmp = Path(tempfile.mkdtemp(prefix="dipser_"))
        try:
            frames, labels = load_subject(zp, tmp)
            if not frames or not labels:
                print(f"  {zp.stem}: no usable frames/labels, skipped")
                continue
            if len(frames) > args.max_frames:
                idx = np.linspace(0, len(frames) - 1, args.max_frames).astype(int)
                frames = [frames[i] for i in idx]

            feats = []
            for _, ip, bbox, pose in frames:
                im = cv2.imread(ip)
                if im is None:
                    feats.append(np.zeros(want, dtype=np.float32))
                    continue
                fv = feat.extract_batch(im, [bbox])[0]
                if feat.output_dim() + 4 == want:
                    fv = np.concatenate([fv, pose]).astype(np.float32)
                feats.append(fv)
            # Chunk to max_seq_len: the positional encoding is sized for it
            # (128, matching max_track_len at build time), and a longer input
            # raises a shape error rather than silently truncating.
            L = int(cfg["model"]["max_seq_len"])
            arr = np.stack(feats)
            preds = []
            with torch.inference_mode():
                for i in range(0, len(arr), L):
                    chunk = arr[i:i + L]
                    xb = torch.from_numpy(chunk[None]).float().to(device)
                    preds.append(model(xb)[0].argmax(-1).cpu().numpy())
            pred = np.concatenate(preds)
            ts = np.array([f[0] for f in frames])
            is_off = np.isin(pred, list(off_ids))

            got = 0
            for lt, rating, who in labels:
                sel = np.abs(ts - lt) <= args.window
                if sel.sum() < 3:
                    continue
                pairs.append((float(is_off[sel].mean()), rating, zp.stem, who))
                got += 1
            per_subject[zp.stem] = {"frames": len(frames), "labels": len(labels),
                                    "paired": got,
                                    "off_task_rate": float(is_off.mean())}
            print(f"  {zp.stem}: {len(frames)} frames, {got} paired, "
                  f"off-task {is_off.mean():.1%}")
        finally:
            shutil.rmtree(tmp, ignore_errors=True)

    if len(pairs) < 10:
        raise SystemExit(f"only {len(pairs)} paired observations — too few")

    off = np.array([p[0] for p in pairs])
    rat = np.array([p[1] for p in pairs], dtype=float)

    def spearman(a, b):
        ra = np.argsort(np.argsort(a)).astype(float)
        rb = np.argsort(np.argsort(b)).astype(float)
        ra -= ra.mean(); rb -= rb.mean()
        return float((ra @ rb) / (np.sqrt((ra**2).sum() * (rb**2).sum()) + 1e-12))

    rho = spearman(off, rat)
    # Permutation test: ordinal data, tiny n, ties — an analytic p-value would
    # be unreliable here.
    rng = np.random.default_rng(0)
    null = np.array([spearman(off, rng.permutation(rat)) for _ in range(5000)])
    p = float((np.abs(null) >= abs(rho)).mean())

    by_rating = {int(r): float(off[rat == r].mean())
                 for r in sorted(set(rat.tolist()))}
    counts = {int(r): int((rat == r).sum()) for r in sorted(set(rat.tolist()))}

    out = {"n_pairs": len(pairs), "n_subjects": len(per_subject),
           "window_s": args.window, "spearman_rho": rho, "perm_p": p,
           "mean_off_task_rate_by_rating": by_rating,
           "n_by_rating": counts, "per_subject": per_subject,
           "off_task_cues": sorted(OFF_TASK),
           "note": ("Our off-task cue rate vs INDEPENDENT expert engagement "
                    "ratings (1=low..5=max). Never trained on these ratings. "
                    "Negative rho is the hypothesis: more off-task -> lower "
                    "engagement.")}
    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    Path(args.out).write_text(json.dumps(out, indent=2))

    print(f"\n=== DIPSER external validation ===")
    print(f"  pairs {len(pairs)} over {len(per_subject)} subjects, "
          f"window +/-{args.window:.0f}s")
    print(f"  Spearman rho = {rho:+.3f}   permutation p = {p:.4f}")
    print(f"  {'rating':>7} {'n':>5} {'mean off-task rate':>20}")
    for r in sorted(by_rating):
        print(f"  {r:>7} {counts[r]:>5} {by_rating[r]:>19.1%}")
    print(f"\n  (hypothesis: rho < 0 — more off-task cues, lower engagement)")
    print(f"written: {args.out}")


if __name__ == "__main__":
    main()
