import argparse
from collections import defaultdict, deque
from pathlib import Path
from typing import Deque, Dict, List, Tuple

import cv2
import numpy as np
import torch
import yaml

from attention.detector_adapter import FrozenLLMDetAdapter, MMDetDetectorAdapter, iou_xyxy
from attention.features import StudentFeatureExtractor
from attention.taxonomy import CUE_CLASSES, CUE_TASK_SCORE
from attention.temporal_model import AttentionTransformer, logits_to_pred
from attention.tracking import IoUTracker


CLASS_NAMES = CUE_CLASSES
CLASS_SCORE = CUE_TASK_SCORE
_ONTASK_LABELS = ("screen_oriented",)


def parse_args():
    p = argparse.ArgumentParser(description="Real-time student attention inference.")
    p.add_argument("--config", type=str, required=True, help="YAML config path.")
    p.add_argument("--source", type=str, default="0", help="Webcam index or video path.")
    p.add_argument("--show", dest="show", action="store_true", help="Display live OpenCV window.")
    p.add_argument("--no-show", dest="show", action="store_false", help="Run headless without GUI display.")
    p.set_defaults(show=True)
    p.add_argument("--out-video", type=str, default="", help="Optional output video path for annotated frames.")
    return p.parse_args()


def load_yaml(path: Path) -> Dict:
    with path.open("r", encoding="utf-8") as f:
        return yaml.safe_load(f)


def _blur_faces(frame: np.ndarray, tracks) -> np.ndarray:
    out = frame.copy()
    for t in tracks:
        x1, y1, x2, y2 = [int(v) for v in t.bbox_xyxy]
        h = max(0, y2 - y1)
        y2f = y1 + int(0.35 * h)
        if x2 <= x1 or y2f <= y1:
            continue
        roi = out[max(0, y1):max(0, y2f), max(0, x1):max(0, x2)]
        if roi.size:
            out[max(0, y1):max(0, y2f), max(0, x1):max(0, x2)] = cv2.GaussianBlur(roi, (31, 31), 0)
    return out


def _det_appearance_feature(frame_bgr: np.ndarray, bbox_xyxy: List[float]) -> np.ndarray:
    x1, y1, x2, y2 = [int(v) for v in bbox_xyxy]
    h, w = frame_bgr.shape[:2]
    x1 = max(0, min(x1, w - 1))
    x2 = max(0, min(x2, w - 1))
    y1 = max(0, min(y1, h - 1))
    y2 = max(0, min(y2, h - 1))
    if x2 <= x1 or y2 <= y1:
        return np.zeros((24,), dtype=np.float32)
    crop = frame_bgr[y1:y2, x1:x2]
    hsv = cv2.cvtColor(crop, cv2.COLOR_BGR2HSV)
    hist_h = cv2.calcHist([hsv], [0], None, [8], [0, 180]).flatten()
    hist_s = cv2.calcHist([hsv], [1], None, [8], [0, 256]).flatten()
    hist_v = cv2.calcHist([hsv], [2], None, [8], [0, 256]).flatten()
    vec = np.concatenate([hist_h, hist_s, hist_v], axis=0).astype(np.float32)
    vec /= float(np.linalg.norm(vec) + 1e-6)
    return vec


def main():
    args = parse_args()
    cfg = load_yaml(Path(args.config))
    hybrid_cfg = cfg.get("hybrid", {})
    hybrid_enabled = bool(hybrid_cfg.get("enabled", False))

    source = int(args.source) if args.source.isdigit() else args.source
    cap = cv2.VideoCapture(source)
    if not cap.isOpened():
        raise RuntimeError(f"Cannot open source: {source}")
    writer = None
    if args.out_video:
        fps = cap.get(cv2.CAP_PROP_FPS)
        if fps is None or fps <= 0 or np.isnan(fps):
            fps = 25.0
        w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        if w <= 0 or h <= 0:
            w, h = 1280, 720
        fourcc = cv2.VideoWriter_fourcc(*"mp4v")
        writer = cv2.VideoWriter(args.out_video, fourcc, fps, (w, h))
        print(f"[info] writing annotated video to: {args.out_video}")

    det = FrozenLLMDetAdapter(
        config_path=cfg["detector"]["config_path"],
        checkpoint_path=cfg["detector"]["checkpoint_path"],
        text_prompt=cfg["detector"].get("text_prompt", "student"),
        score_thr=float(cfg["detector"].get("score_thr", 0.35)),
        device=cfg["detector"].get("device", "cuda:0"),
        max_det=int(cfg["detector"].get("max_det", 100)),
        min_rel_area=float(cfg["detector"].get("min_rel_area", 0.01)),
        max_rel_area=float(cfg["detector"].get("max_rel_area", 0.60)),
        min_aspect_ratio=float(cfg["detector"].get("min_aspect_ratio", 0.22)),
        max_aspect_ratio=float(cfg["detector"].get("max_aspect_ratio", 1.25)),
        nms_iou_thr=float(cfg["detector"].get("nms_iou_thr", 0.5)),
    )
    primary_det = None
    verifier = None
    if hybrid_enabled:
        primary_det = MMDetDetectorAdapter(
            config_path=hybrid_cfg["primary_detector"]["config_path"],
            checkpoint_path=hybrid_cfg["primary_detector"]["checkpoint_path"],
            score_thr=float(hybrid_cfg["primary_detector"].get("score_thr", 0.3)),
            device=hybrid_cfg["primary_detector"].get("device", cfg["detector"].get("device", "cuda:0")),
            max_det=int(hybrid_cfg["primary_detector"].get("max_det", 120)),
            text_prompt=hybrid_cfg["primary_detector"].get("text_prompt", None),
            class_whitelist=hybrid_cfg["primary_detector"].get("class_whitelist", None),
        )
        verifier = det
    tracker = IoUTracker(
        iou_match_thr=float(cfg["tracking"].get("iou_match_thr", 0.35)),
        max_age=int(cfg["tracking"].get("max_age", 30)),
        min_hits=int(cfg["tracking"].get("min_hits", 3)),
        appearance_weight=float(cfg["tracking"].get("appearance_weight", 0.35)),
        min_match_score=float(cfg["tracking"].get("min_match_score", 0.25)),
    )
    feat = StudentFeatureExtractor(
        clip_model_name=cfg["features"].get("clip_model_name", "openai/clip-vit-base-patch32"),
        device=cfg["features"].get("device", "cuda:0"),
    )

    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    model = AttentionTransformer(
        input_dim=int(cfg["model"]["input_dim"]),
        hidden_dim=int(cfg["model"]["hidden_dim"]),
        num_layers=int(cfg["model"]["num_layers"]),
        num_heads=int(cfg["model"]["num_heads"]),
        dropout=float(cfg["model"]["dropout"]),
        num_classes=int(cfg["model"]["num_classes"]),
        max_seq_len=int(cfg["model"]["max_seq_len"]),
        per_frame=bool(cfg["model"].get("per_frame", True)),
    ).to(device)
    ckpt = torch.load(cfg["temporal_checkpoint"], map_location="cpu")
    model.load_state_dict(ckpt["model"], strict=False)
    model.eval()

    win = int(cfg["inference"]["window_size"])
    feats: Dict[int, Deque[np.ndarray]] = defaultdict(lambda: deque(maxlen=win))
    latest_score: Dict[int, float] = {}
    label_hist: Dict[int, Deque[int]] = defaultdict(lambda: deque(maxlen=int(cfg["inference"].get("label_smooth_window", 9))))
    conf_hist: Dict[int, Deque[float]] = defaultdict(lambda: deque(maxlen=int(cfg["inference"].get("label_smooth_window", 9))))
    stable_label: Dict[int, int] = {}
    frame_idx = 0

    while True:
        ok, frame = cap.read()
        if not ok:
            break

        if hybrid_enabled and primary_det is not None:
            dets = primary_det.detect(frame)
            verify_every = int(hybrid_cfg.get("verify_every_n_frames", 3))
            verify_iou_thr = float(hybrid_cfg.get("verify_iou_thr", 0.25))
            if frame_idx % max(1, verify_every) == 0 and verifier is not None:
                vdets = verifier.detect(frame)
                kept = []
                for d in dets:
                    max_iou = 0.0
                    for v in vdets:
                        max_iou = max(max_iou, iou_xyxy(d.bbox_xyxy, v.bbox_xyxy))
                    if max_iou >= verify_iou_thr:
                        kept.append(d)
                dets = kept
        else:
            dets = det.detect(frame)
        det_feats = [_det_appearance_feature(frame, d.bbox_xyxy) for d in dets]
        tracks = tracker.update(dets, det_feats)
        preds: Dict[int, Tuple[str, float]] = {}

        active_ids = set()
        track_feats = (
            feat.extract_batch(frame, [t.bbox_xyxy for t in tracks]) if tracks else []
        )
        for t, f in zip(tracks, track_feats):
            active_ids.add(t.track_id)
            feats[t.track_id].append(f)
            if len(feats[t.track_id]) < int(cfg["inference"].get("min_frames_for_pred", 4)):
                continue

            x = np.stack(list(feats[t.track_id]), axis=0).astype(np.float32)
            x = torch.from_numpy(x).unsqueeze(0).to(device)
            with torch.inference_mode(), torch.cuda.amp.autocast(enabled=device.type == "cuda"):
                logits = model(x)
                if logits.dim() == 3:  # per-frame model: use the newest frame
                    logits = logits[:, -1, :]
                pred, conf = logits_to_pred(logits)
            raw_label_idx = int(pred.item())
            raw_conf = float(conf.item())
            label_hist[t.track_id].append(raw_label_idx)
            conf_hist[t.track_id].append(raw_conf)
            # Majority vote smoothing per track.
            vals = list(label_hist[t.track_id])
            binc = np.bincount(np.array(vals, dtype=np.int64), minlength=len(CLASS_NAMES))
            smooth_idx = int(np.argmax(binc))
            # Hysteresis to suppress rapid flicker.
            prev_idx = stable_label.get(t.track_id, smooth_idx)
            if smooth_idx != prev_idx:
                prev_votes = int(binc[prev_idx]) if prev_idx < len(binc) else 0
                new_votes = int(binc[smooth_idx])
                if new_votes - prev_votes < int(cfg["inference"].get("label_switch_margin", 2)):
                    smooth_idx = prev_idx
            stable_label[t.track_id] = smooth_idx
            label = CLASS_NAMES[smooth_idx]
            score = float(np.mean(conf_hist[t.track_id])) if conf_hist[t.track_id] else raw_conf
            preds[t.track_id] = (label, score)
            latest_score[t.track_id] = CLASS_SCORE.get(label, score)

        for tid in list(feats.keys()):
            if tid not in active_ids:
                feats.pop(tid, None)
                latest_score.pop(tid, None)
                label_hist.pop(tid, None)
                conf_hist.pop(tid, None)
                stable_label.pop(tid, None)

        classroom = float(np.mean(list(latest_score.values()))) if latest_score else 0.0
        vis = _blur_faces(frame, tracks) if bool(cfg["privacy"].get("anonymize_faces", True)) else frame.copy()

        for t in tracks:
            x1, y1, x2, y2 = [int(v) for v in t.bbox_xyxy]
            label, conf = preds.get(t.track_id, ("pending", 0.0))
            col = (0, 255, 0) if label in _ONTASK_LABELS else (0, 165, 255)
            cv2.rectangle(vis, (x1, y1), (x2, y2), col, 2)
            cv2.putText(vis, f"id={t.track_id} {label}:{conf:.2f}", (x1, max(20, y1 - 8)), cv2.FONT_HERSHEY_SIMPLEX, 0.5, col, 2)

        cv2.putText(vis, f"classroom_task_orientation={classroom:.2f}", (12, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.85, (255, 255, 255), 2)
        if writer is not None:
            writer.write(vis)
        if args.show:
            try:
                cv2.imshow("LLMDet Attention Realtime", vis)
                if cv2.waitKey(1) & 0xFF == ord("q"):
                    break
            except cv2.error as e:
                print(f"[warn] display is unavailable, switching to headless mode: {e}")
                args.show = False
        frame_idx += 1

    cap.release()
    if writer is not None:
        writer.release()
    if args.show:
        cv2.destroyAllWindows()


if __name__ == "__main__":
    main()

