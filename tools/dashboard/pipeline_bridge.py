"""Bridge the real-time pipeline into the dashboard.

Runs detector -> tracker -> features -> temporal model over a video and pushes
per-frame state to the dashboard. Also records a replay log so the dashboard can
be demonstrated later without a GPU (useful for a thesis defence on a laptop).

Kept separate from server.py so the dashboard has no torch/mmdet dependency in
replay mode — `python server.py --replay session.jsonl` needs stdlib only.
"""
import json
import sys
import time
from collections import defaultdict, deque
from pathlib import Path

LLMDET_ROOT = Path(__file__).resolve().parents[2] / "LLMDet"
sys.path.insert(0, str(LLMDET_ROOT))

#: URL schemes OpenCV can open directly (FFMPEG/GStreamer backends).
_STREAM_SCHEMES = ("rtsp://", "rtsps://", "http://", "https://", "udp://",
                   "tcp://", "rtmp://")


def classify_source(video):
    """-> ("camera" | "stream" | "file", opencv_argument)

    Three source kinds need three different treatments and only the file case
    is a path. ``Path(url).resolve()`` silently turns ``rtsp://cam/stream`` into
    ``<cwd>/rtsp:/cam/stream``, so an IP camera used to fail with "cannot open"
    naming a path the user never typed.

    Camera and stream are *live*: they produce frames whether or not anything is
    consuming them, which changes both how frames must be read and what a
    timestamp means. A file does neither.
    """
    s = str(video)
    if s.isdigit():
        return "camera", int(s)
    if s.lower().startswith(_STREAM_SCHEMES):
        return "stream", s
    return "file", str(Path(s).resolve())


class LatestFrame:
    """Reader thread that keeps only the newest frame from a live source.

    A live camera produces frames on its own clock. The pipeline consumes them
    at a few per second, so reading sequentially means consuming a *queue*: the
    displayed frame falls further behind the room the longer the dashboard runs,
    with no upper bound and nothing on screen to say so. An instructor would be
    alerted about a student who put their phone away two minutes ago.

    ``cv2.CAP_PROP_BUFFERSIZE`` is honoured by some backends and ignored by the
    FFMPEG one that handles RTSP, so it cannot be relied on. This drains the
    source in its own thread instead and hands out the most recent frame,
    turning the lag into dropped frames — which is the right trade for a
    dashboard, and is counted so the drop rate can be reported rather than
    guessed at.
    """

    def __init__(self, cap):
        import threading
        self.cap = cap
        self.lock = threading.Lock()
        self.frame = None
        self.seq = 0            # frames read from the source
        self.taken = 0          # frames the pipeline actually processed
        self.alive = True
        self.thread = threading.Thread(target=self._loop, daemon=True)
        self.thread.start()

    def _loop(self):
        while self.alive:
            ok, f = self.cap.read()
            if not ok:
                self.alive = False
                break
            with self.lock:
                self.frame = f
                self.seq += 1

    def read(self, timeout=5.0):
        """Newest unseen frame, or (False, None) if the source stopped."""
        deadline = time.time() + timeout
        last = -1
        while time.time() < deadline:
            with self.lock:
                if self.frame is not None and self.seq != last:
                    self.taken += 1
                    return True, self.frame
                last = self.seq
            if not self.alive:
                return False, None
            time.sleep(0.005)
        return self.alive, None

    @property
    def dropped(self):
        with self.lock:
            return max(0, self.seq - self.taken)

    def release(self):
        self.alive = False
        self.thread.join(timeout=2.0)
        self.cap.release()


class PushedFrames:
    """Frames arriving from somewhere that is not a `cv2.VideoCapture`.

    The dashboard runs on the GPU box; the camera is in front of whoever opened
    the page. `cv2.VideoCapture(0)` would open the *server's* camera, which on a
    Hugging Face Space does not exist, and on a shared machine belongs to
    somebody else. So the browser captures with `getUserMedia` and POSTs JPEGs,
    and this stands in for the reader.

    It deliberately implements the same contract as :class:`LatestFrame` —
    `read()`, `dropped`, `release()` — including the part that matters: keep only
    the NEWEST frame. The browser pushes on its own clock and the pipeline
    consumes at a few per second, so buffering would put the overlay further
    behind the room the longer it ran, with nothing on screen to say so.

    `alive` is driven by a timeout rather than by end-of-stream: a browser tab
    that is closed, backgrounded or loses its permission simply stops posting,
    and there is no event to observe. Without the timeout the worker thread
    would block on an empty queue forever, holding the GPU and leaving the page
    reporting "running".
    """

    def __init__(self, idle_timeout=10.0, fps=15.0):
        import threading
        self.lock = threading.Lock()
        self.frame = None
        self.seq = 0
        self.taken = 0
        #: seq of the frame last handed out. Instance state, not a local, so a
        #: frame is analysed ONCE -- see read().
        self.last_seq = -1
        self.fps = float(fps)
        self.idle_timeout = float(idle_timeout)
        self.last_push = time.time()
        self.closed = False

    def put(self, frame_bgr):
        """Called from the HTTP thread with a decoded frame."""
        with self.lock:
            self.frame = frame_bgr
            self.seq += 1
            self.last_push = time.time()

    @property
    def alive(self):
        with self.lock:
            return not self.closed and \
                (time.time() - self.last_push) < self.idle_timeout

    def read(self, timeout=5.0):
        """The newest UNSEEN frame, or (False, None) once the pusher is gone.

        Two deliberate differences from :class:`LatestFrame`, both because this
        source can stall in a way a camera cannot:

        **A frame is handed out once.** `LatestFrame` keeps a per-call `last`, so
        if nothing new has arrived it returns the current frame again. For an
        RTSP camera that barely matters — frames keep coming. Here the browser
        can stop for seconds while the tab is backgrounded, and re-analysing one
        stale frame would advance each student's dwell on evidence that is no
        longer true, which is how an alert fires on a frozen image. `last_seq` is
        instance state so that cannot happen.

        **Death is checked first.** A buffer whose last push was longer ago than
        `idle_timeout` has no live frame to offer, even if it still holds one.
        """
        deadline = time.time() + timeout
        while True:
            if not self.alive:
                return False, None
            with self.lock:
                if self.frame is not None and self.seq != self.last_seq:
                    self.last_seq = self.seq
                    self.taken += 1
                    return True, self.frame
            if time.time() >= deadline:
                break
            time.sleep(0.005)
        # Not a failure: the browser may just be posting slowly. Returning
        # (alive, None) matches LatestFrame and lets run_live keep waiting.
        return self.alive, None

    @property
    def dropped(self):
        with self.lock:
            return max(0, self.seq - self.taken)

    def release(self):
        with self.lock:
            self.closed = True
            self.frame = None

    def reopen(self):
        """Ready this buffer for a new capture session."""
        with self.lock:
            self.closed = False
            self.frame = None
            self.seq = self.taken = 0
            self.last_seq = -1
            self.last_push = time.time()


def run_live(config_path, video, push_fn, blur_faces=False, max_frames=0,
             record=None, device=None, entry=None, should_stop=None,
             stats_fn=None, frame_source=None, on_bundle=None):
    """Detector -> tracker -> features -> temporal model over a live video.

    ``entry`` is an optional ``model_registry.ModelEntry``. When given, its
    checkpoint, calibration and head-pose backend override the config's, which
    is what lets the dashboard's model selector drive a live run as well as a
    cached one. The head-pose backend has to come from the entry rather than the
    config because it decides what the extractor *produces*: a model trained on
    the BlazeFace flag and fed the FaceLandmarker block still runs and still
    prints cues.

    ``stats_fn(dict)`` is called once a second for a live source with the
    processed frame rate and the drop rate. A live dashboard that cannot keep up
    with the camera is still useful, but only if it says so — an overlay that
    looks current and is thirty seconds stale is worse than one labelled stale.

    ``frame_source`` replaces the capture entirely: a :class:`PushedFrames` fed
    by the browser's own webcam over HTTP. ``video`` is then only a label. This
    is the only way a Space can analyse "the camera", since the process runs on
    a GPU host with no camera attached to it.
    """
    import cv2
    import numpy as np
    import torch
    import yaml

    from attention.detector_adapter import FrozenLLMDetAdapter
    from attention.features import StudentFeatureExtractor
    from attention.head_pose import HeadPoseEstimator
    from attention.thesis_eval.runtime import load_runtime_model, predict_window
    from attention.tracking import IoUTracker
    from attention.realtime_infer import _det_appearance_feature
    from attention.display_smoothing import BoxSmoother, LabelSmoother
    from session_replay import (WARMUP_COLOUR, WARMUP_LABEL, calibration_path,
                                colour_for, warming_record)

    cfg = yaml.safe_load(open(config_path))
    # Config paths (detector config/checkpoint, CLIP dirs) are written relative
    # to LLMDet/. Resolve from there so the dashboard can be launched from
    # anywhere rather than only from that directory.
    import os
    cwd0 = os.getcwd()
    # Resolve caller-relative paths BEFORE chdir, or outputs land in LLMDet/
    # instead of where the caller asked for them.
    if record:
        record = str(Path(record).resolve())
    kind, source = classify_source(video)
    os.chdir(LLMDET_ROOT)
    # The caller decides, then the config, then CPU. Grabbing cuda:0 whenever a
    # GPU exists is wrong on a shared box: the GPUs may belong to someone else's
    # job, and the dashboard is expected to be demonstrable without one.
    dev = device or cfg.get("detector", {}).get("device") or "cpu"
    if str(dev).startswith("cuda") and not torch.cuda.is_available():
        dev = "cpu"

    det = FrozenLLMDetAdapter(
        config_path=cfg["detector"]["config_path"],
        checkpoint_path=cfg["detector"]["checkpoint_path"],
        text_prompt=cfg["detector"].get("text_prompt", "a student sitting"),
        score_thr=float(cfg["detector"].get("score_thr", 0.10)),
        device=dev,
        max_det=int(cfg["detector"].get("max_det", 80)),
        min_rel_area=float(cfg["detector"].get("min_rel_area", 0.01)),
        max_rel_area=float(cfg["detector"].get("max_rel_area", 0.60)),
        min_aspect_ratio=float(cfg["detector"].get("min_aspect_ratio", 0.22)),
        max_aspect_ratio=float(cfg["detector"].get("max_aspect_ratio", 1.25)),
        nms_iou_thr=float(cfg["detector"].get("nms_iou_thr", 0.5)))
    from attention.thesis_eval.runtime import StrideController as _SC
    _stride_cfg = _SC(int(cfg.get("inference", {}).get("detector_stride", 1)),
                      int(cfg.get("inference", {}).get("temporal_stride", 1)))
    tracker = IoUTracker(
        iou_match_thr=float(cfg["tracking"].get("iou_match_thr", 0.35)),
        max_age=int(cfg["tracking"].get("max_age", 30)),
        min_hits=_stride_cfg.adjusted_min_hits(int(cfg["tracking"].get("min_hits", 3))),
        appearance_weight=float(cfg["tracking"].get("appearance_weight", 0.35)),
        min_match_score=float(cfg["tracking"].get("min_match_score", 0.25)))

    # Head pose is REQUIRED, not best-effort: the deployed model is 556-dim and
    # 4 of those dims are head pose. Swallowing a failure here used to leave the
    # extractor at 552 dims, which the old zero-padding path then hid.
    backend = (entry.head_pose_backend if entry is not None
               else cfg.get("features", {}).get("head_pose_backend", "mediapipe"))
    hp = HeadPoseEstimator(backend=backend) if backend else None
    if hp is not None and not hp.available():
        raise SystemExit(
            f"head-pose backend {backend!r} is unavailable, but the deployed "
            "model needs its 4 dims. Install it or point --config at a "
            "checkpoint trained without head pose.")
    # Build from the CHECKPOINT's own spec. A YAML/checkpoint disagreement used
    # to raise inside a bare except and run the dashboard on random weights.
    #
    # Loaded BEFORE the extractor on purpose: whether the 518-dim head stream is
    # switched on is a property of the checkpoint (``1074_hp_head`` reads it),
    # and it changes what the extractor produces. Building the extractor first
    # and asking afterwards would leave the best validation model in the
    # registry permanently unservable — the width assert below would reject it
    # with a message about padding.
    if entry is not None:
        ckpt = str(Path(__file__).resolve().parents[2] / entry.checkpoint)
        cal = str(calibration_path(entry.variant_id))
    else:
        ckpt, cal = cfg["temporal_checkpoint"], cfg.get("calibration")
    bundle = load_runtime_model(ckpt, device=dev, calibration=cal)
    # The bundle is built here, so without this nothing outside the run could
    # reach the thresholds of the model that is actually running.
    if on_bundle is not None:
        on_bundle(bundle)

    feat = StudentFeatureExtractor(
        clip_model_name=cfg["features"].get("clip_model_name",
                                            "openai/clip-vit-base-patch32"),
        device=dev, head_pose=hp, head_stream=bundle.needs_head_stream)
    if bundle.needs_head_stream:
        print(f"[dashboard] head stream ON — a second CLIP pass over the head "
              f"crop, {feat.head_stream_dim} extra dims per student")
    # The extractor always emits base + the 4 head-pose columns; a checkpoint
    # trained on a subset (e.g. 553_facefound) selects its columns inside
    # predict_window. Assert the EXTRACTOR width, not the model width.
    # A second detector, only for a checkpoint that reads the object block. Its
    # (measured value removed from the handover copy)
    # and they cost a full extra open-vocabulary pass per frame.
    from attention.object_features import OBJECT_DIM, ObjectDetector
    objdet = None
    if bundle.needs_objects:
        oc = cfg.get("object_detector") or {}
        if not oc.get("config_path") or not oc.get("checkpoint_path"):
            raise SystemExit(
                f"{bundle.experiment_id} reads the object block, but the config "
                "has no `object_detector` section. Its six columns cannot be "
                "faked: a zero block means 'no phone present', which is a "
                "measurement, not a missing value.")
        ock = Path(oc["checkpoint_path"])
        if not ock.is_absolute():
            ock = (LLMDET_ROOT / ock).resolve()
        if not ock.exists():
            raise SystemExit(
                f"{bundle.experiment_id} needs the pretrained open-vocabulary "
                f"detector at {ock}, which is not on this host. It is a "
                "separate 1.1 GB checkpoint from the student detector.")
        objdet = ObjectDetector(
            str((LLMDET_ROOT / oc["config_path"]).resolve()), str(ock),
            device=str(dev), prompts=tuple(oc.get("prompts") or ()),
            min_score=float(oc.get("min_score", 0.05)))
        print(f"[dashboard] object detector ON — a second open-vocabulary pass "
              f"per frame for {list(objdet.prompts)}, {OBJECT_DIM} extra dims")

    want = bundle.live_input_width
    # The extractor produces everything EXCEPT the object block, which is
    # appended below from a different model; assert against what each side
    # actually contributes rather than against the total.
    produced = feat.output_dim() + (OBJECT_DIM if objdet is not None else 0)
    if produced != want:
        raise SystemExit(
            f"live features are {produced}-dim but "
            f"{bundle.experiment_id} needs an extractor producing {want} "
            f"({bundle.feature_config}). Refusing to pad — a zero block is "
            "indistinguishable from a real measurement.")
    print(f"[dashboard] temporal model: {bundle.describe()}")
    #: The names this checkpoint predicts. Not CUE_CLASSES: a coarse-taxonomy
    #: model has two or three of them, and the UI renders whatever it is sent.
    class_names = list(bundle.class_names)

    from attention.thesis_eval.runtime import HistorySampler, StrideController
    from attention.track_handoff import SeatRegistry
    # Feeds each student's history at the rate the model was trained on, however
    # fast the detector runs. Without it the 32-frame window spans 1/fps * 32
    # seconds instead of the ~31 s it spanned in training -- see HistorySampler.
    sampler = HistorySampler(float(cfg["inference"].get("temporal_input_fps", 1.0)))
    stride = StrideController(
        detector_stride=int(cfg["inference"].get("detector_stride", 1)),
        temporal_stride=int(cfg["inference"].get("temporal_stride", 1)))
    win = int(cfg["inference"]["window_size"])
    minf = int(cfg["inference"].get("min_frames_for_pred", 4))
    # The config has declared these since it was written and this path never
    # read them, so the live overlay ran with NO smoothing while the file said
    # otherwise. They steady what is DRAWN; the raw prediction, its confidence
    # and the alert gate are passed through untouched below.
    smooth_box = BoxSmoother()
    smooth_label = LabelSmoother(
        window=int(cfg["inference"].get("label_smooth_window", 3)),
        margin=int(cfg["inference"].get("label_switch_margin", 0)))
    hist = defaultdict(lambda: deque(maxlen=win))
    dwell = {}
    # Per-student state is keyed by SEAT, not the tracker's id, so an id switch
    # neither changes the number on screen nor throws away the 1 Hz history.
    # See SeatRegistry.
    handoff = SeatRegistry()
    rec = open(record, "w") if record else None

    if frame_source is not None:
        # Frames are being POSTed to us; there is nothing to open.
        cap = None
        kind, source = "camera", "browser"
        live = True
        reader = frame_source
        fps = float(getattr(frame_source, "fps", 15.0))
    else:
        cap = cv2.VideoCapture(source)
        if not cap.isOpened():
            raise RuntimeError(
                f"cannot open {kind} source {source!r}. For a webcam pass the "
                f"device index (--video 0); for an IP camera pass the full URL "
                f"(--video rtsp://user:pass@host/stream). To use the camera of "
                f"the machine viewing the page, pick the browser camera in the "
                f"UI instead — this process cannot reach it.")
        live = kind in ("camera", "stream")
        # A live source is drained by a reader thread so the pipeline always gets
        # the newest frame; a file is read sequentially, since every frame matters
        # and none of them are going stale.
        reader = LatestFrame(cap) if live else cap
        fps = cap.get(cv2.CAP_PROP_FPS) or 25.0
    n = 0
    t_prev = time.time()
    t_start = time.time()
    t_stats = t_start
    stop_reason = ""
    print(f"[dashboard] source: {kind} {source!r}"
          + (" — frames are dropped to stay current" if live else ""))
    # max_frames <= 0 means no cap. A live camera has no natural end, so the run
    # stops on should_stop(), on the source dying, or on the frame buffer going
    # idle -- not on a frame count nobody chose.
    while max_frames <= 0 or n < max_frames:
        # Lets the dashboard abandon a run mid-video when the user picks a
        # different model, instead of leaving two pipelines pushing frames.
        if should_stop is not None and should_stop():
            break
        ok, frame = reader.read()
        if not ok:
            # The source is gone. For a file that is simply the end; for a live
            # source it is a diagnosis the page needs, because the run otherwise
            # ends looking exactly like a clean finish and the caller records no
            # error at all.
            if live and n == 0:
                stop_reason = ("the frame source ended before a single frame "
                               "was analysed"
                               + (" (the buffer was already closed)"
                                  if getattr(reader, "closed", False) else
                                  " (no frames arrived within the idle timeout)"))
            elif live:
                stop_reason = (f"the frame source stopped after {n} frames "
                               "(the browser stopped posting, or the tab was "
                               "closed)")
            break
        if frame is None:       # live source stalled; keep waiting
            continue
        # Timestamps drive the dwell thresholds that raise alerts ("head down
        # for 30 s"), so they have to be in the same seconds the instructor is
        # living in. For a file, frame index over fps IS that clock. For a live
        # source it is not: frames are dropped, so n/fps would run slow and a
        # 30 s episode would be announced minutes late.
        t = (time.time() - t_start) if live else (n / fps)
        if stride.should_detect(n):
            dets = det.detect(frame)
            tracks = tracker.update(dets, [_det_appearance_feature(frame, d.bbox_xyxy)
                                           for d in dets])
        else:
            tracks = tracker.coast()
        key_of = handoff.resolve(t, {tr.track_id: tr.bbox_xyxy for tr in tracks})
        for gone in handoff.pop_cleared():
            hist.pop(gone, None)
            dwell.pop(gone, None)
        for was, now in handoff.pop_moved():
            if was in hist:
                hist[now] = hist.pop(was)
            if was in dwell:
                dwell[now] = dwell.pop(was)
        live_ids = [key_of[tr.track_id] for tr in tracks]
        stride.drop_missing(live_ids)
        sampler.drop(live_ids)
        smooth_box.drop(live_ids)
        smooth_label.drop(live_ids)
        students = {}
        if tracks:
            # Decide BEFORE paying. `t` is the same clock the dwell thresholds
            # use -- elapsed wall time for a live source, frame index / fps for a
            # file -- so a file sampled here yields the same frames however fast
            # it is replayed.
            due = [tr for tr in tracks if sampler.due(key_of[tr.track_id], t)]
            if due:
                boxes = [tr.bbox_xyxy for tr in due]
                fv = feat.extract_batch(frame, boxes)
                if objdet is not None:
                    # One detector pass for the frame, then per-student
                    # containment. Appended in the order the sequence builder
                    # used, which is why the column layout is named: reading
                    # these six as anything else would train and serve happily
                    # on nonsense. Skipped entirely when nobody is due -- it is
                    # the most expensive thing in the loop after the detector.
                    fv = np.hstack([fv, objdet.features(frame, boxes)])
                for tr, f in zip(due, fv):
                    hist[key_of[tr.track_id]].append(f)
                    sampler.mark(key_of[tr.track_id], t)

            for tr in tracks:
                sid = key_of[tr.track_id]
                h = hist[sid]
                if len(h) < minf:
                    # Reported, not skipped: the table and the counts see a
                    # student on screen who has no prediction yet.
                    students[str(sid)] = warming_record(
                        smooth_box(sid, tr.bbox_xyxy))
                    continue
                # predict_window applies the validation-fitted temperature and
                # both abstention thresholds, and asserts the feature width.
                if stride.should_predict(sid, n):
                    r = stride.store(sid, n,
                                     predict_window(bundle, np.stack(list(h))))
                else:
                    r = stride.cached(sid)
                # Smoothed for display; `raw_cue` below keeps the model's own
                # answer, so nothing downstream is fooled about what it said.
                cue = smooth_label(sid, r["displayed_cue"])
                conf = r["confidence"]
                # Dwell accumulates on the DISPLAYED cue, so an abstention
                # interrupts an episode rather than silently extending it.
                prev = dwell.get(sid)
                if prev and prev["cue"] == cue:
                    prev["dwell"] = t - prev["since"]
                else:
                    dwell[sid] = {"cue": cue, "since": t, "dwell": 0.0,
                                  "alerted": False}
                st = dwell[sid]
                students[str(sid)] = {
                    "cue": cue, "conf": round(conf, 2),
                    # Track ids are assigned in detection order and are NOT
                    # comparable between runs; the box is, and it is what lets
                    # tools/verify_stride_equivalence.py match students across
                    # configurations the way a human comparing two overlays would.
                    "bbox": [round(float(v), 1)
                             for v in smooth_box(sid, tr.bbox_xyxy)],
                    "dwell": st["dwell"], "alerted": st["alerted"],
                    # raw prediction preserved even when abstaining: the point
                    # of abstention is to withhold an alert, not evidence
                    "raw_cue": r["cue"], "abstained": r["abstained"],
                    "alert_allowed": r["alert_allowed"],
                    "rescued": bool(r.get("rescued", False))}

        vis = frame.copy()
        for tr in tracks:
            s = students.get(str(key_of[tr.track_id]))
            # Draw the SAME box that was reported, or the raw one for a track
            # too new to have a prediction yet.
            x1, y1, x2, y2 = [int(v) for v in
                              (s["bbox"] if s else tr.bbox_xyxy)]
            warming = s is None or s.get("warming", False)
            cue = s["cue"] if s else WARMUP_LABEL
            col = WARMUP_COLOUR if warming else colour_for(cue)
            if blur_faces:
                hh = max(1, int(0.35 * (y2 - y1)))
                roi = vis[y1:y1 + hh, x1:x2]
                if roi.size:
                    vis[y1:y1 + hh, x1:x2] = cv2.GaussianBlur(roi, (31, 31), 0)
            cv2.rectangle(vis, (x1, y1), (x2, y2), col, 1 if warming else 2)
            cv2.putText(vis, f"{key_of[tr.track_id]} {cue.replace('_',' ')}",
                        (x1, max(18, y1 - 6)), cv2.FONT_HERSHEY_SIMPLEX, 0.5, col, 2)

        small = cv2.resize(vis, (960, int(960 * vis.shape[0] / vis.shape[1])))
        ok2, buf = cv2.imencode(".jpg", small, [cv2.IMWRITE_JPEG_QUALITY, 70])
        push_fn(t, buf.tobytes() if ok2 else None, students, class_names)
        if rec:
            rec.write(json.dumps({"t": t, "students": students,
                                  "dt": time.time() - t_prev,
                                  "cue_names": class_names}) + "\n")
        t_prev = time.time()
        n += 1

        if live and stats_fn is not None and time.time() - t_stats >= 1.0:
            t_stats = time.time()
            dropped = reader.dropped
            stats_fn({
                "live": True,
                "processed_fps": round(n / max(t_stats - t_start, 1e-9), 2),
                "frames_processed": n,
                "frames_dropped": dropped,
                "drop_pct": round(100 * dropped / max(dropped + n, 1)),
                "source_fps": round(fps, 1),
            })

    if live:
        # The drop rate is the honest measure of how far behind real time the
        # pipeline is running: 9 dropped for every 1 processed means the model
        # sees the room at a tenth of its actual frame rate.
        dropped = reader.dropped
        total = dropped + n
        print(f"[dashboard] live source: processed {n}, dropped {dropped} "
              f"({100 * dropped / max(total, 1):.0f}% of {total}), "
              f"{n / max(time.time() - t_start, 1e-9):.2f} processed fps")
    # Both live readers own their own teardown (a reader thread, or the frame
    # buffer); only the sequential file path hands back a bare capture.
    reader.release() if live else cap.release()
    os.chdir(cwd0)
    if rec:
        rec.close()
    if stop_reason:
        print(f"[dashboard] live source ended: {stop_reason}")
        if stats_fn is not None:
            stats_fn({"processed": n, "dropped": int(getattr(reader, "dropped", 0)),
                      "stopped_because": stop_reason})
    print(f"[dashboard] finished after {n} frames")
    return stop_reason
