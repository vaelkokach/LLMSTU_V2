"""Per-stage latency/FPS/GPU profiling of the real-time attention pipeline.

Measures wall-time per stage (detector, tracker, features, temporal model,
overlay), overall FPS, peak GPU memory, and optionally per-stage scaling with
synthetic student counts (5/10/20/30) so the thesis can report real-time
figures rather than a "real-time-oriented prototype" claim.

Run (GPU job — coordinate before launching on a shared node):
  cd LLMDet
  python -m profiling.profile_pipeline --config configs/attention_temporal.yaml \
      --video 0325.mp4 --max-frames 300 --out work_dirs/profiling/report.json
  # scaling mode:
  python -m profiling.profile_pipeline --config configs/attention_temporal.yaml \
      --video 0325.mp4 --max-frames 200 --scaling 5,10,20,30 \
      --out work_dirs/profiling/report_scaling.json
"""
import argparse
import copy
import json
import time
from collections import defaultdict, deque
from pathlib import Path
from typing import Deque, Dict, List

import numpy as np


class StageTimer:
    """Collects wall-times per named stage; use as `with timer("detector"): ...`."""

    def __init__(self):
        self.samples: Dict[str, List[float]] = defaultdict(list)
        self._stage = None
        self._t0 = 0.0
        self._sync = None  # set to torch.cuda.synchronize for accurate GPU timing

    def __call__(self, stage: str) -> "StageTimer":
        self._stage = stage
        return self

    def __enter__(self):
        if self._sync is not None:
            self._sync()
        self._t0 = time.perf_counter()
        return self

    def __exit__(self, *exc):
        if self._sync is not None:
            self._sync()
        self.samples[self._stage].append(time.perf_counter() - self._t0)
        return False

    def summary(self) -> Dict[str, Dict[str, float]]:
        out = {}
        for stage, xs in self.samples.items():
            a = np.asarray(xs, dtype=np.float64) * 1000.0  # ms
            out[stage] = {
                "n": int(a.size),
                "mean_ms": float(a.mean()),
                "p50_ms": float(np.percentile(a, 50)),
                "p90_ms": float(np.percentile(a, 90)),
                "p95_ms": float(np.percentile(a, 95)),
                "p99_ms": float(np.percentile(a, 99)),
                "max_ms": float(a.max()),
            }
        return out


def parse_args():
    p = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    p.add_argument("--config", type=str, required=True, help="attention_temporal.yaml path")
    p.add_argument("--video", type=str, required=True, help="Video path or webcam index")
    p.add_argument("--max-frames", type=int, default=300)
    p.add_argument("--device", type=str, default="cuda:0")
    p.add_argument("--scaling", type=str, default="",
                   help="Comma-separated synthetic student counts, e.g. 5,10,20,30. "
                        "Detections are tiled/truncated to each count so tracker/feature/"
                        "model scaling is isolated from detector scaling.")
    p.add_argument("--warmup", type=int, default=10, help="Frames excluded from stats")
    p.add_argument("--out", type=str, default="work_dirs/profiling/report.json")
    return p.parse_args()


def _synthesize_dets(dets, target_n, frame_shape):
    """Tile or truncate real detections to exactly target_n boxes with jitter."""
    if not dets:
        return dets
    out = []
    h, w = frame_shape[:2]
    i = 0
    rng = np.random.default_rng(0)
    while len(out) < target_n:
        d = copy.deepcopy(dets[i % len(dets)])
        if len(out) >= len(dets):  # jitter clones so NMS/tracker sees distinct boxes
            dx, dy = rng.uniform(-0.03, 0.03, size=2) * (w, h)
            x1, y1, x2, y2 = d.bbox_xyxy
            bw, bh = x2 - x1, y2 - y1
            x1 = float(np.clip(x1 + dx, 0, w - bw))
            y1 = float(np.clip(y1 + dy, 0, h - bh))
            d.bbox_xyxy = [x1, y1, x1 + bw, y1 + bh]
        out.append(d)
        i += 1
    return out[:target_n]


def run_pass(cfg, args, student_count=None):
    """One profiling pass over the video. student_count=None = real detections."""
    import cv2
    import torch

    from attention.detector_adapter import FrozenLLMDetAdapter
    from attention.features import StudentFeatureExtractor
    from attention.head_pose import HeadPoseEstimator
    from attention.temporal_model import AttentionTransformer, logits_to_pred
    from attention.tracking import IoUTracker
    from attention.realtime_infer import CLASS_NAMES, _det_appearance_feature

    device = torch.device(args.device if torch.cuda.is_available() else "cpu")
    timer = StageTimer()
    if device.type == "cuda":
        torch.cuda.init()  # context must exist before peak-memory reset
        torch.cuda.reset_peak_memory_stats(device)
        timer._sync = lambda: torch.cuda.synchronize(device)

    det = FrozenLLMDetAdapter(
        config_path=cfg["detector"]["config_path"],
        checkpoint_path=cfg["detector"]["checkpoint_path"],
        text_prompt=cfg["detector"].get("text_prompt", "student"),
        score_thr=float(cfg["detector"].get("score_thr", 0.35)),
        device=str(device),
        max_det=int(cfg["detector"].get("max_det", 100)),
        min_rel_area=float(cfg["detector"].get("min_rel_area", 0.01)),
        max_rel_area=float(cfg["detector"].get("max_rel_area", 0.60)),
        min_aspect_ratio=float(cfg["detector"].get("min_aspect_ratio", 0.22)),
        max_aspect_ratio=float(cfg["detector"].get("max_aspect_ratio", 1.25)),
        nms_iou_thr=float(cfg["detector"].get("nms_iou_thr", 0.5)),
    )
    from attention.thesis_eval.runtime import StrideController as _SC
    _stride_cfg = _SC(int(cfg.get("inference", {}).get("detector_stride", 1)),
                      int(cfg.get("inference", {}).get("temporal_stride", 1)))
    tracker = IoUTracker(
        iou_match_thr=float(cfg["tracking"].get("iou_match_thr", 0.35)),
        max_age=int(cfg["tracking"].get("max_age", 30)),
        min_hits=_stride_cfg.adjusted_min_hits(int(cfg["tracking"].get("min_hits", 3))),
        appearance_weight=float(cfg["tracking"].get("appearance_weight", 0.35)),
        min_match_score=float(cfg["tracking"].get("min_match_score", 0.25)),
    )
    # Head pose must be timed if the deployed model uses it. Profiling a
    # 552-dim extractor and then deploying a 556-dim model would understate the
    # frame budget by the entire MediaPipe cost.
    backend = cfg.get("features", {}).get("head_pose_backend")
    hp = HeadPoseEstimator(backend=backend) if backend else None
    if hp is not None and not hp.available():
        raise SystemExit(f"head-pose backend {backend!r} unavailable")
    feat = StudentFeatureExtractor(
        clip_model_name=cfg["features"].get("clip_model_name", "openai/clip-vit-base-patch32"),
        device=str(device), head_pose=hp,
    )
    # Build from the checkpoint's own spec where one exists, so the profiled
    # model is the deployed model. The legacy branch keeps old configs runnable.
    bundle = None
    ckpt_path = cfg.get("temporal_checkpoint")
    try:
        from attention.thesis_eval.runtime import load_runtime_model
        bundle = load_runtime_model(ckpt_path, device=str(device))
        model = bundle.model
        want_dim = bundle.input_dim
        print(f"[profiling] temporal model: {bundle.describe()}")
    except SystemExit:
        bundle = None
    if bundle is None:
        model = AttentionTransformer(
            input_dim=int(cfg["model"]["input_dim"]),
            hidden_dim=int(cfg["model"]["hidden_dim"]),
            num_layers=int(cfg["model"]["num_layers"]),
            num_heads=int(cfg["model"]["num_heads"]),
            dropout=float(cfg["model"]["dropout"]),
            num_classes=int(cfg["model"]["num_classes"]),
            max_seq_len=int(cfg["model"]["max_seq_len"]),
        ).to(device)
        want_dim = int(cfg["model"]["input_dim"])
        # Timing does not require trained weights, but a mismatch must be LOUD:
        # loading nothing and profiling a random network is how the dashboard
        # came to "verify" itself against untrained weights.
        try:
            ckpt = torch.load(ckpt_path, map_location="cpu")
            model.load_state_dict(ckpt["model"], strict=True)
        except (FileNotFoundError, RuntimeError, KeyError) as e:
            print(f"[profiling] WARNING: temporal checkpoint NOT loaded ({e}); "
                  "timing is still valid but these are RANDOM weights")
    model.eval()
    live_cols = None
    if bundle is not None and bundle.live_columns is not None:
        live_cols = bundle.live_columns          # model selects a subset
        want_dim = bundle.live_input_width
    if feat.output_dim() != want_dim:
        raise SystemExit(
            f"extractor yields {feat.output_dim()} dims, model wants {want_dim}; "
            "set features.head_pose_backend or use a matching checkpoint")

    source = int(args.video) if str(args.video).isdigit() else args.video
    cap = cv2.VideoCapture(source)
    if not cap.isOpened():
        raise RuntimeError(f"Cannot open source: {source}")

    win = int(cfg["inference"]["window_size"])
    feats: Dict[int, Deque[np.ndarray]] = defaultdict(lambda: deque(maxlen=win))
    min_frames = int(cfg["inference"].get("min_frames_for_pred", 4))
    from attention.thesis_eval.runtime import StrideController
    stride = StrideController(
        detector_stride=int(cfg["inference"].get("detector_stride", 1)),
        temporal_stride=int(cfg["inference"].get("temporal_stride", 1)))
    if stride.detector_stride > 1 or stride.temporal_stride > 1:
        print(f"[profiling] detector_stride={stride.detector_stride} "
              f"temporal_stride={stride.temporal_stride}")

    frame_times = []
    track_counts = []
    n_done = 0
    while n_done < args.max_frames:
        ok, frame = cap.read()
        if not ok:
            break
        f0 = time.perf_counter()

        run_detector = stride.should_detect(n_done)
        with timer("detector"):
            dets = det.detect(frame) if run_detector else []
        if student_count is not None and run_detector:
            dets = _synthesize_dets(dets, student_count, frame.shape)

        with timer("tracker"):
            if run_detector:
                det_feats = [_det_appearance_feature(frame, d.bbox_xyxy) for d in dets]
                tracks = tracker.update(dets, det_feats)
            else:
                tracks = tracker.coast()
            stride.drop_missing(t.track_id for t in tracks)

        with timer("features"):
            if tracks:
                batch = feat.extract_batch(frame, [t.bbox_xyxy for t in tracks])
                for t, f in zip(tracks, batch):
                    feats[t.track_id].append(f)

        with timer("temporal_model"):
            for t in tracks:
                if len(feats[t.track_id]) < min_frames:
                    continue
                if not stride.should_predict(t.track_id, n_done):
                    continue          # hold the cached cue
                x = np.stack(list(feats[t.track_id]), axis=0).astype(np.float32)
                if live_cols is not None:
                    x = np.ascontiguousarray(x[:, live_cols])
                x = torch.from_numpy(x).unsqueeze(0).to(device)
                with torch.inference_mode():
                    out = model(x)
                    logits = out["logits"] if isinstance(out, dict) else out
                    pred, conf = logits_to_pred(logits)
                stride.store(t.track_id, n_done, {"pred": int(pred.flatten()[-1])})

        with timer("overlay"):
            vis = frame.copy()
            for t in tracks:
                x1, y1, x2, y2 = [int(v) for v in t.bbox_xyxy]
                cv2.rectangle(vis, (x1, y1), (x2, y2), (0, 255, 0), 2)
                cv2.putText(vis, f"id={t.track_id}", (x1, max(20, y1 - 8)),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)

        if n_done >= args.warmup:
            frame_times.append(time.perf_counter() - f0)
            track_counts.append(len(tracks))
        else:  # warmup frames pollute per-stage stats too
            timer.samples = defaultdict(list) if n_done == args.warmup - 1 else timer.samples
        n_done += 1
    cap.release()

    ft = np.asarray(frame_times, dtype=np.float64)
    report = {
        "student_count": student_count if student_count is not None else "real",
        "frames_measured": int(ft.size),
        "fps_mean": float(1.0 / ft.mean()) if ft.size else None,
        "fps_p50": float(1.0 / np.percentile(ft, 50)) if ft.size else None,
        "frame_ms_mean": float(ft.mean() * 1000) if ft.size else None,
        "frame_ms_p50": float(np.percentile(ft * 1000, 50)) if ft.size else None,
        "frame_ms_p90": float(np.percentile(ft * 1000, 90)) if ft.size else None,
        "frame_ms_p95": float(np.percentile(ft * 1000, 95)) if ft.size else None,
        "frame_ms_p99": float(np.percentile(ft * 1000, 99)) if ft.size else None,
        "frame_ms_max": float(ft.max() * 1000) if ft.size else None,
        # Fraction of frames a live 25 fps camera would have had to drop. FPS
        # alone hides tail latency, and the tail is what an instructor notices.
        "dropped_frame_rate_at_25fps": float((ft > 1.0 / 25.0).mean()) if ft.size else None,
        "dropped_frame_rate_at_10fps": float((ft > 1.0 / 10.0).mean()) if ft.size else None,
        "n_tracks_mean": float(np.mean(track_counts)) if track_counts else None,
        "stages": timer.summary(),
    }
    if device.type == "cuda":
        import torch
        report["gpu_peak_mem_mb"] = float(torch.cuda.max_memory_allocated(device)) / 1e6
        report["gpu_reserved_mem_mb"] = float(torch.cuda.max_memory_reserved(device)) / 1e6
        report["gpu_name"] = torch.cuda.get_device_name(device)
        report["n_gpus_used"] = 1
    try:
        import resource
        report["host_peak_rss_mb"] = (
            resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1024.0)
    except Exception:
        pass
    return report


def main():
    import yaml

    args = parse_args()
    with open(args.config, "r", encoding="utf-8") as f:
        cfg = yaml.safe_load(f)

    reports = [run_pass(cfg, args)]
    if args.scaling:
        for n in [int(x) for x in args.scaling.split(",") if x.strip()]:
            reports.append(run_pass(cfg, args, student_count=n))

    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    with out.open("w", encoding="utf-8") as f:
        json.dump({"config": args.config, "video": str(args.video),
                   "reports": reports}, f, indent=2)
    print(json.dumps(reports, indent=2))
    print(f"Report written to: {out}")


if __name__ == "__main__":
    main()
