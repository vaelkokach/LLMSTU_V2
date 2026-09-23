"""P0.3 — score the MODEL's events against the HUMAN event gold.

THE NUMBER THIS PRODUCES IS THE THESIS'S HEADLINE BRANCH-B RESULT.

Background [internal notes, not included]: the previously reported 89% episode-miss rate
compared model events against *teacher-derived* (Qwen3.5) events, so a miss was
uninterpretable — it could mean the event layer is wrong or the reference is.
The dense human gold (984 crops -> 754 accepted frames -> 24 episodes over 8
(video, seat) tracks) settles it. Separately measured: the teacher reproduces
those human episodes at 75%, so the reference was reasonable and the 89% is our
model's gap.

Scope: this evaluates the CUE + EVENT layers in isolation. It consumes the
dense crops directly, so the detector and tracker contribute no error — track
identity comes from the manifest's (video_id, seat_id). That is deliberate:
mixing detector misses into this number would make it uninterpretable in the
same way the pseudo-reference did. Detector quality is reported separately
(Table 1 / threshold sweep).

Three timelines are scored against the same human gold so the comparison is
like-for-like:
  1. model      - temporal transformer over extracted features
  2. teacher    - Qwen3.5 manifest labels (the ceiling any student can inherit)
  3. majority   - always-predict-dominant-class control

Usage (from LLMDet/, GPU: ~1 h for 984 crops incl. CLIP):
    python -m attention.eval_events_vs_human \
        --config configs/attention_temporal.yaml \
        --ckpt work_dirs/attention_temporal_v2/checkpoints/best.pth \
        --manifest ../grounding_data/llmstu_tools/outputs/dense_event_manifest.jsonl \
        --annotations ../gold_annotation_bundle/../event_gold_bundle/gold_annotations_Admin.jsonl \
        --gold-events ../grounding_data/llmstu_tools/outputs/gold_events.jsonl \
        --frames-root ../grounding_data/stu_img/frames \
        --out work_dirs/attention_temporal_v2/events_vs_human.json
"""
import argparse
import json
from collections import Counter, defaultdict
from pathlib import Path

import numpy as np
import torch
import yaml

from attention.taxonomy import CUE_CLASSES, map_record
from attention.temporal_model import AttentionTransformer
from attention.events import Episode, EventConfig, segment_events
from attention.event_metrics import evaluate_events

FIELDS = ["activity", "gaze_direction", "attention_target", "engagement_level",
          "posture", "hand_state", "phone_visible", "laptop_visible",
          "talking", "occluded"]


def load_tracks(manifest, annotations, frames_root):
    """(video_id, seat_id) -> [(t, frame_path, bbox_person, teacher_cue, human_cue)]."""
    ann = {}
    for line in open(annotations):
        r = json.loads(line)
        ann[r["file_name"]] = r  # append-only log: last state wins

    tracks = defaultdict(list)
    n_hum = n_rej = 0
    for line in open(manifest):
        m = json.loads(line)
        a = ann.get(m["file_name"])
        if a is not None and a.get("status") == "rejected":
            n_rej += 1
            continue  # annotator could not verify -> gap in the timeline
        human = None
        if a is not None and a.get("status") in ("ok", "uncertain"):
            rec = dict(m)
            rec.update({f: a[f] for f in FIELDS if f in a})
            human = map_record(rec)
            n_hum += 1
        # Features MUST be extracted the way sequence_builder.py does it:
        # full frame + bbox_person. Using the pre-cropped image with the whole
        # image as the box makes the 8 geometry + 8 posture dims constant
        # (cx=0.5, area=1.0, ...) instead of frame-relative, which is 16 of 552
        # dims silently wrong and pushes the model off-distribution.
        tracks[(m["video_id"], str(m["seat_id"]))].append(
            (float(m["t"]), str(Path(frames_root) / m["src_frame"]),
             [float(v) for v in m["bbox_person"]], map_record(m), human,
             m["file_name"]))
    for k in tracks:
        tracks[k].sort(key=lambda r: r[0])
    print(f"tracks {len(tracks)}  frames human={n_hum} rejected={n_rej}")
    return tracks


def offset_flatten(per_track, duration):
    """Lay tracks on disjoint slices of one timeline.

    Greedy episode matching is global, so without offsets a prediction on
    student A could match a gold episode belonging to student B.
    """
    flat = []
    for i, (_, eps) in enumerate(sorted(per_track.items())):
        off = i * (duration + 1000.0)
        flat += [Episode(e.channel, e.t_start + off, e.t_end + off) for e in eps]
    return flat


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", required=True)
    ap.add_argument("--ckpt", required=True)
    ap.add_argument("--manifest", required=True)
    ap.add_argument("--annotations", required=True)
    ap.add_argument("--gold-events", required=True)
    ap.add_argument("--frames-root", required=True,
                    help="full frames dir (stu_img/frames) — NOT the crops dir")
    ap.add_argument("--affect-cache", default=None,
                    help="npz from precompute_affect.py (4 pose + 7 expression). "
                         "With --dynamic it supplies the 18 extra dims a 570-dim "
                         "config expects.")
    ap.add_argument("--head-pose-cache", default=None,
                    help="npz from precompute_head_pose.py. REQUIRED when the "
                         "config is 556-dim; the feature width must match the "
                         "checkpoint or F.linear raises a shape error.")
    ap.add_argument("--device", default="cuda:0")
    ap.add_argument("--iou-thr", type=float, default=0.3)
    ap.add_argument("--smooth", type=int, default=0,
                    help="median-filter window (frames) over predicted cues; "
                         "0 = off. See P0.5: 31/82 teacher phone misses were "
                         "single-frame flicker.")
    ap.add_argument("--out", required=True)
    args = ap.parse_args()

    import cv2
    from attention.features import StudentFeatureExtractor

    cfg = yaml.safe_load(open(args.config))
    device = torch.device(args.device if torch.cuda.is_available() else "cpu")
    tracks = load_tracks(args.manifest, args.annotations, args.frames_root)

    feat = StudentFeatureExtractor(
        clip_model_name=cfg["features"].get("clip_model_name",
                                            "openai/clip-vit-base-patch32"),
        device=str(device))
    # per_frame MUST be passed. Its default is False, which makes forward()
    # return [B, C] (one label for the whole sequence) instead of [B, T, C].
    # The checkpoint loads cleanly either way — the head shape is identical —
    # so getting this wrong fails silently and produces a flat timeline with
    # no events. Event evaluation needs per-frame cues.
    per_frame = bool(cfg["model"].get("per_frame", False))
    if not per_frame:
        raise SystemExit(
            "config has per_frame=false; event evaluation requires per-frame "
            "cue predictions. Set model.per_frame: true.")
    model = AttentionTransformer(
        input_dim=int(cfg["model"]["input_dim"]),
        hidden_dim=int(cfg["model"]["hidden_dim"]),
        num_layers=int(cfg["model"]["num_layers"]),
        num_heads=int(cfg["model"]["num_heads"]),
        dropout=float(cfg["model"]["dropout"]),
        num_classes=int(cfg["model"]["num_classes"]),
        max_seq_len=int(cfg["model"]["max_seq_len"]),
        per_frame=per_frame).to(device)
    sd = torch.load(args.ckpt, map_location="cpu")
    model.load_state_dict(sd["model"] if "model" in sd else sd)
    model.eval()

    # Feature width must match the checkpoint. A 556-dim config needs the
    # head-pose block; building 552-dim features against it fails inside
    # F.linear with a shape error rather than silently, which is the right
    # behaviour but the cache must be supplied.
    want_dim = int(cfg["model"]["input_dim"])
    base = feat.output_dim()
    cached_hp = affect_map = affect_vecs = None
    use_dynamic = False
    if want_dim != base:
        if args.affect_cache:
            d = np.load(args.affect_cache, allow_pickle=False)
            affect_map = {str(n): i for i, n in enumerate(d["names"])}
            affect_vecs = d["vecs"].astype(np.float32)
            from attention.dynamic_features import DYNAMIC_DIM
            if base + affect_vecs.shape[1] + DYNAMIC_DIM == want_dim:
                use_dynamic = True
            elif base + affect_vecs.shape[1] != want_dim:
                raise SystemExit(
                    f"affect adds {affect_vecs.shape[1]} (+{DYNAMIC_DIM} dynamic) "
                    f"to {base}; config wants {want_dim}")
        elif args.head_pose_cache:
            from attention.head_pose import HeadPoseEstimator
            cached_hp = HeadPoseEstimator(backend="cached",
                                          cache_path=args.head_pose_cache)
            if base + 4 != want_dim:
                raise SystemExit(
                    f"cache adds 4 dims -> {base+4}, config wants {want_dim}")
        else:
            raise SystemExit(
                f"config wants input_dim={want_dim} but features are {base}; "
                f"pass --affect-cache (570-dim) or --head-pose-cache (556-dim)")
    print(f"feature dim: {want_dim} (affect={'on' if affect_vecs is not None else 'off'}, "
          f"dynamic={'on' if use_dynamic else 'off'}, "
          f"head_pose_cache={'on' if cached_hp else 'off'})")

    ecfg = EventConfig()
    eps_model, eps_teacher, eps_major = {}, {}, {}
    frame_model, frame_teacher, frame_human = [], [], []

    # Majority class over the teacher timeline = the control's constant output.
    major = Counter(r[3] for rows in tracks.values() for r in rows).most_common(1)[0][0]

    for key, rows in sorted(tracks.items()):
        ts = [r[0] for r in rows]
        feats, tboxes = [], []
        for _, path, bbox, _, _, fname in rows:
            tboxes.append(list(bbox))
            im = cv2.imread(path)
            if im is None:
                feats.append(np.zeros(want_dim, dtype=np.float32))
                continue
            fv = feat.extract_batch(im, [bbox])[0]
            if cached_hp is not None:
                fv = np.concatenate(
                    [fv, cached_hp.estimate_by_name(fname)]).astype(np.float32)
            if affect_vecs is not None:
                j = affect_map.get(fname)
                av = affect_vecs[j] if j is not None else np.zeros(
                    affect_vecs.shape[1], dtype=np.float32)
                fv = np.concatenate([fv, av]).astype(np.float32)
            feats.append(fv)
        arr = np.stack(feats)
        if use_dynamic:
            # Same construction as sequence_builder: pose is the first 4 of the
            # affect block, and the features are track-level (fidget variance,
            # personalised gaze baseline) so the whole track is required.
            from attention.dynamic_features import compute_dynamic
            a0 = arr.shape[1] - affect_vecs.shape[1]
            arr = np.concatenate(
                [arr, compute_dynamic(tboxes, arr[:, a0:a0 + 4])],
                axis=1).astype(np.float32)
        if arr.shape[1] != want_dim:
            raise SystemExit(
                f"built {arr.shape[1]}-dim features but the config/checkpoint "
                f"wants {want_dim}; check --affect-cache / dynamic settings")
        x = torch.from_numpy(arr[None]).float().to(device)
        with torch.inference_mode():
            logits = model(x)
        pred = np.atleast_1d(logits[0].argmax(-1).cpu().numpy())
        if pred.shape[0] != len(rows):
            # Should not happen with per_frame=True; guard rather than emit a
            # silently wrong timeline. atleast_1d first — a 0-dim result made
            # the previous shape check itself raise IndexError.
            raise RuntimeError(
                f"model returned {pred.shape[0]} predictions for {len(rows)} "
                f"frames on track {key}; expected one per frame")

        if args.smooth and args.smooth > 1:
            k = args.smooth
            padded = np.pad(pred, k // 2, mode="edge")
            pred = np.array([np.bincount(padded[i:i + k]).argmax()
                             for i in range(len(pred))], dtype=np.int64)

        teacher = np.array([r[3] for r in rows], dtype=np.int64)
        for i, r in enumerate(rows):
            if r[4] is not None:  # human-verified frames only
                frame_model.append(pred[i]); frame_teacher.append(teacher[i])
                frame_human.append(r[4])

        eps_model[key] = segment_events(ts, pred.tolist(), ecfg)
        eps_teacher[key] = segment_events(ts, teacher.tolist(), ecfg)
        eps_major[key] = segment_events(ts, [major] * len(ts), ecfg)

    gold_by_track = defaultdict(list)
    for line in open(args.gold_events):
        g = json.loads(line)
        gold_by_track[(g["video_id"], str(g["seat_id"]))].append(
            Episode(g["channel"], g["t_start"], g["t_end"]))

    all_t = [t for rows in tracks.values() for t, *_ in rows]
    duration = max(all_t) - min(all_t) if all_t else 1.0
    keys = sorted(set(gold_by_track) | set(eps_model))
    pad = lambda d: {k: d.get(k, []) for k in keys}
    total = max(duration * len(keys), 1.0)
    gold_flat = offset_flatten(pad(gold_by_track), duration)

    results = {}
    for name, eps in (("model", eps_model), ("teacher", eps_teacher),
                      ("majority", eps_major)):
        results[name] = evaluate_events(offset_flatten(pad(eps), duration),
                                        gold_flat, observed_duration_s=total,
                                        iou_thr=args.iou_thr)

    fm = np.array(frame_model); ft = np.array(frame_teacher); fh = np.array(frame_human)
    frame_acc = {"model_vs_human": float((fm == fh).mean()) if fm.size else None,
                 "teacher_vs_human": float((ft == fh).mean()) if ft.size else None,
                 "n_frames": int(fm.size)}

    out = {"frame_level": frame_acc, "n_gold_events": len(gold_flat),
           "n_tracks": len(keys), "iou_thr": args.iou_thr,
           "smooth_window": args.smooth, "events": results,
           "note": ("Model/teacher/majority events scored against HUMAN gold on "
                    "the same frames. Detector and tracker excluded by design — "
                    "track identity comes from the manifest.")}
    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    Path(args.out).write_text(json.dumps(out, indent=2))

    print(f"\nframe-level (human-verified frames, n={frame_acc['n_frames']}):")
    print(f"  model   vs human : {frame_acc['model_vs_human']}")
    print(f"  teacher vs human : {frame_acc['teacher_vs_human']}")
    print(f"\nevent-level vs HUMAN gold ({len(gold_flat)} episodes, {len(keys)} tracks):")
    print(f"{'system':10s} {'matched':>9} {'missed':>8} {'onset_s':>8} {'dur_s':>7} {'FA/h':>7}")
    for name in ("model", "teacher", "majority"):
        o = results[name]["overall"]
        print(f"{name:10s} {o['num_matched']:5.0f}/{o['num_gt']:<3.0f} "
              f"{o['missed_event_rate']:8.3f} {o['onset_error_s']:8.1f} "
              f"{o['duration_error_s']:7.1f} {o['false_alerts_per_hour']:7.1f}")
    print(f"\nwritten: {args.out}")


if __name__ == "__main__":
    main()
