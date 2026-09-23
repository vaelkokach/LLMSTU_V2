"""Replay a cached session through any one temporal model.

``precompute_session`` cached everything that does not depend on the model. This
runs the part that does: assemble the live feature vector, slide a window per
track, and ask one checkpoint what each student's visible cue is.

Two things this must get right, because both are silent when wrong.

**Which head-pose block.** The cache holds the FaceLandmarker block and the
BlazeFace block side by side. Each model gets the one its training features were
built from (``ModelEntry.head_pose_backend``, derived from the checkpoint's own
``run_record.json``). Handing a model the other block would still produce a
556-wide vector, still run, and still print cues — just worse ones, for a reason
nothing in the output would reveal.

**Which thresholds.** Confidence is comparable across models only after each has
its own temperature. ``calibrate_registry`` fits one per model on validation;
this refuses to invent a fallback if that file is missing, because a model
running at ``threshold = 0`` never abstains and would look *more* decisive than
a well-calibrated one — the wrong way round.

Everything else mirrors ``pipeline_bridge.run_live`` frame for frame, so a
cached replay and a live run of the same model produce the same cue stream.
"""

from __future__ import annotations

import json
import sys
import time
from collections import defaultdict, deque
from pathlib import Path
from typing import Callable, Dict, Optional

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO / "LLMDet"))
sys.path.insert(0, str(Path(__file__).parent))

import numpy as np                                              # noqa: E402

CALIBRATION_DIR = (REPO / "LLMDet" / "work_dirs" / "thesis" / "runtime"
                   / "dashboard")

#: BGR overlay colours, matching the chip colours in index.html.
#:
#: Keyed by class NAME rather than by id, so a model predicting a regrouped
#: taxonomy draws in the right colour without a second table: `on_task` is the
#: green `screen_oriented` is, `off_task` and `down_or_hidden` are the red the
#: cues they merge are. Any name not listed falls back to grey, which is the
#: honest default — an unrecognised class is one this overlay cannot interpret.
#: The `inference` block of the runtime config, when the dashboard was started
#: with one. Replay-only servers leave it empty and fall back to whatever the
#: cache recorded. Set by server.py at startup rather than threaded through
#: every call, because `replay` is also used directly from scripts.
CONTEXT_INFERENCE: dict = {}

CUE_COLOUR = {
    # cue6
    "screen_oriented": (61, 220, 132),      # green
    "head_down": (86, 95, 255),             # red
    "phone_use": (86, 95, 255),
    "looking_away": (84, 180, 255),         # amber
    "turned_to_peer": (84, 180, 255),
    "uncertain": (160, 160, 160),           # grey
    # onoff / onoff_reliable
    "on_task": (61, 220, 132),
    "off_task": (86, 95, 255),
    # coarse3_reliable
    "down_or_hidden": (86, 95, 255),
    # cue9 / cue7 -- the four classes screen_oriented was split into.
    #
    # These were missing, and the consequence was not cosmetic: they are ~74% of
    # students, so nearly every box drew in the fallback grey, and grey is ALSO
    # what an abstention draws as. "writing notes" and "the model is not sure"
    # were the same colour on screen.
    #
    # All four are on-task, so they stay in the cool half of the palette, well
    # away from the amber and red the off-task cues own. They are distinct from
    # each other because the whole point of the split is that these are
    # different behaviours.
    "using_laptop": (61, 220, 132),         # green -- screen_oriented's heir
    "reading": (180, 210, 90),              # teal
    "listening": (220, 170, 70),            # blue
    "writing_notes": (120, 235, 200),       # lime
    # cue8 -- `reading` and `listening` merged. Sits between the two it replaces
    # (180,210,90) and (220,170,70), and stays well clear of the amber and red
    # the off-task cues own: it is an ON-TASK class and the palette carries that
    # meaning, so the wrong side would be worse than no colour at all.
    "engaged": (200, 190, 80),
}

#: Grey. Also what an abstention draws as, since `displayed_cue` is the
#: ABSTAIN_LABEL ("uncertain") whenever the model was not confident enough.
UNKNOWN_COLOUR = (160, 160, 160)


def colour_for(cue: str):
    """BGR for one class name, grey for anything this table does not know."""
    return CUE_COLOUR.get(cue, UNKNOWN_COLOUR)


#: What a tracked student reads before the model has enough history to predict.
#: At the trained 1 Hz and ``min_frames_for_pred: 10`` that is the first ~10 s
#: of every track [internal notes, not included], which used to mean no box at all. It is not a
#: class and not an abstention: it never counts as on- or off-task, never
#: alerts, and draws thin in its own light grey so it cannot be mistaken for the
#: abstention grey.
WARMUP_LABEL = "warming_up"
WARMUP_COLOUR = (215, 215, 215)


def warming_record(bbox) -> Dict:
    """The student entry for a track that has a box but no prediction yet."""
    return {"cue": WARMUP_LABEL, "conf": None,
            "bbox": [round(float(v), 1) for v in bbox],
            "dwell": 0.0, "alerted": False, "raw_cue": None,
            "abstained": False, "alert_allowed": False, "warming": True}


def calibration_path(variant_id: str) -> Path:
    """Where this variant's fitted thresholds live.

    A `+vlm` entry is virtual — the temporal half is the base checkpoint and
    runs identically, so it uses the base model's thresholds. Fitting separate
    ones would be fitting the same predictions twice under a different name.
    (The FUSION has its own decision rule, in fusion.Policy; it is not a
    threshold on this distribution.)
    """
    base = variant_id[:-4] if variant_id.endswith("+vlm") else variant_id
    return CALIBRATION_DIR / f"{base.replace('/', '__')}.json"


#: One loaded VLM at a time, keyed by (model_id, classes). A Space has a system
#: RAM limit (16 GB on a10g-small) and a VLM is ~4 GB of it, so loading a second
#: without releasing the first is what killed the app with "Memory limit
#: exceeded" on the first switch between two `+vlm` entries. Switching is the
#: dashboard's main interaction, so this is not an edge case.
_GROUNDER: "Optional[tuple]" = None          # (key, AsyncGrounder)


def release_grounder() -> None:
    """Drop the cached VLM and give the memory back.

    Called before loading a different one and whenever a non-VLM model is
    selected: the weights are useless then and they are the largest single
    allocation in the process.
    """
    global _GROUNDER
    if _GROUNDER is None:
        return
    key, g = _GROUNDER
    _GROUNDER = None
    try:
        g.close()
    except Exception:                                        # noqa: BLE001
        pass
    del g
    import gc
    gc.collect()
    try:
        import torch
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
    except Exception:                                        # noqa: BLE001
        pass
    print(f"[replay] released the VLM for {key[0]}", flush=True)


def load_grounder(entry, device: str = "cpu"):
    """The VLM for a `+vlm` entry, or None. Loaded lazily, and at most one."""
    if not getattr(entry, "vlm", False):
        # A plain model has no use for a loaded VLM, and holding it costs ~4 GB
        # of a 16 GB budget for nothing.
        release_grounder()
        return None
    from attention.vlm_grounder import (AsyncGrounder, LlavaOneVisionGrounder,
                                        QwenGrounder)
    ok, why = QwenGrounder.available()
    if not ok:
        # Fail loudly at selection, not silently at the first frame, and do NOT
        # fall back to the stub: a deterministic fake presented as a second
        # opinion would put an agreement rate on screen that measures nothing.
        raise SystemExit(
            f"{entry.variant_id} needs a VLM this environment cannot load.\n"
            f"  {why}\n"
            f"  Pick the base model {entry.vlm_base!r} instead — it is the same "
            f"checkpoint without the second opinion.")
    print(f"[replay] VLM second opinion: {entry.vlm_model_id} on {device} "
          f"(policy {entry.vlm_policy}). Expect seconds per frame.")
    # Wrapped so the caller never blocks on it. A synchronous VLM cannot be
    # live: ~0.2-0.5 s per student is 1.5-3 s for six, against a pipeline that
    # manages 1-4 fps.
    # The grounder must score the SAME label space the temporal model predicts.
    # Both vectors are fused positionally, so handing a cue9 model a six-wide
    # opinion would not raise -- it would pair every cue with the wrong name.
    from attention.taxonomy import taxonomy_classes
    classes = taxonomy_classes(getattr(entry, "taxonomy", "cue6"))

    global _GROUNDER
    key = (entry.vlm_model_id, tuple(classes), device)
    if _GROUNDER is not None and _GROUNDER[0] == key:
        print("[replay] reusing the loaded VLM", flush=True)
        return _GROUNDER[1]
    release_grounder()          # different model or class space: free first
    g = AsyncGrounder(QwenGrounder(model_id=entry.vlm_model_id,
                                   device=device, classes=classes))
    _GROUNDER = (key, g)
    return g


class SessionCache:
    """The precomputed front end for one video."""

    def __init__(self, cache_dir: str):
        self.dir = Path(cache_dir).resolve()
        meta_path = self.dir / "meta.json"
        if not meta_path.exists():
            raise SystemExit(
                f"{self.dir} is not a session cache (no meta.json). Build one "
                f"with tools/dashboard/precompute_session.py.")
        self.meta = json.loads(meta_path.read_text())
        z = np.load(self.dir / "features.npz")
        self.frame_idx = z["frame_idx"]
        self.track_id = z["track_id"]
        self.bbox = z["bbox"]
        self.base = z["base"]
        self.head = {"mediapipe": z["head_landmarker"],
                     "mediapipe_detector": z["head_detector"]}
        self.frame_index = z["frame_index"]

        # How many JPEGs the cache actually has. Replay works without them --
        # it just pushes cue data and no image -- so a cache whose frame writes
        # failed looks exactly like a cue log at the UI: an empty video panel
        # and no error. Counting them here lets the caller say so.
        fdir = self.dir / "frames"
        self.n_cached_frames = (sum(1 for _ in fdir.glob("*.jpg"))
                                if fdir.is_dir() else 0)

        # rows grouped by frame, in frame order
        self._by_frame: Dict[int, list] = defaultdict(list)
        for r, f in enumerate(self.frame_idx):
            self._by_frame[int(f)].append(r)

    @property
    def fps(self) -> float:
        return float(self.meta.get("fps", 25.0))

    def rows_for(self, frame: int):
        return self._by_frame.get(int(frame), [])

    def jpeg(self, frame: int) -> Optional[bytes]:
        p = self.dir / "frames" / f"{int(frame):06d}.jpg"
        return p.read_bytes() if p.exists() else None

    def live_vector(self, rows, backend: str) -> np.ndarray:
        """[n, 556] = base | the head-pose block this model was trained on."""
        if backend not in self.head:
            raise SystemExit(f"cache has no {backend!r} head-pose block")
        return np.concatenate([self.base[rows], self.head[backend][rows]], axis=1)


def load_model(entry, device: str = "cpu"):
    """Build the runtime bundle for a registry entry, with its own thresholds."""
    from attention.thesis_eval.runtime import load_runtime_model

    cal = calibration_path(entry.variant_id)
    if not cal.exists():
        raise SystemExit(
            f"no calibration for {entry.variant_id} at {cal}. Run "
            f"tools/dashboard/calibrate_registry.py. Refusing to fall back to "
            f"threshold 0: an uncalibrated model never abstains, so it would "
            f"appear more confident than a calibrated one purely because "
            f"nobody fitted it.")
    bundle = load_runtime_model(str(REPO / entry.checkpoint), device=device,
                                calibration=str(cal))
    return bundle, json.loads(cal.read_text())


def replay(cache: SessionCache, entry, bundle, push_fn: Callable,
           should_stop: Callable[[], bool] = lambda: False,
           blur_faces: bool = False, realtime: bool = True,
           speed: float = 1.0, overlay: bool = True,
           grounder=None) -> int:
    """Stream one model's cues over the cached session.

    ``push_fn(t, jpeg_bytes, students, cue_names)`` matches
    ``pipeline_bridge.run_live`` so the server does not care which produced it.
    """
    import cv2
    from attention.display_smoothing import BoxSmoother, LabelSmoother
    from attention.track_handoff import SeatRegistry
    from attention.thesis_eval.runtime import (HistorySampler, StrideController,
                                                predict_window)

    # The names the MODEL predicts, from its own checkpoint spec. Pushing
    # CUE_CLASSES here would have told the UI to render six cue chips for a
    # two-class model, so the legend and the model would disagree.
    class_names = list(bundle.class_names)

    # The VLM's second opinion, when this entry asked for one. It needs the
    # frame, which the cache has; `fuse_frame` combines the two distributions
    # per student. Both are strictly optional -- if the VLM is absent the run is
    # exactly the run without it, which is the property that lets the slow path
    # be opt-in rather than a fork of the pipeline.
    submit = fuse_latest = None
    if grounder is not None:
        import cv2 as _cv2
        from attention.fusion import Policy, fuse_frame
        policy = Policy(getattr(entry, "vlm_policy", "agreement"))
        from attention.taxonomy import taxonomy_classes as _tc
        fuse_classes = _tc(getattr(entry, "taxonomy", "cue6"))

        def submit(jpeg_bytes, rows, t):
            """Offer the newest frame to the VLM. Never blocks."""
            if jpeg_bytes is None or not rows:
                return
            img = _cv2.imdecode(np.frombuffer(jpeg_bytes, np.uint8),
                                _cv2.IMREAD_COLOR)
            if img is None:
                return
            # Boxes are cached at full resolution; the JPEG is downscaled.
            sw = cache.meta.get("source_width")
            sc = img.shape[1] / float(sw) if sw else 1.0
            # Student keys, not tracker ids: fuse_latest matches the VLM's
            # opinions against probs_by_track, which is keyed by student.
            grounder.submit(img,
                            [[v * sc for v in cache.bbox[r]] for r in rows],
                            [key_of.get(int(cache.track_id[r]),
                                        int(cache.track_id[r])) for r in rows], t)

        def fuse_latest(probs_by_track, t):
            """Fuse against the most recent VLM opinion, whatever its age.

            Returns ({track_id: FusedStudent}, age_seconds). A student the VLM
            has not seen -- new track, or scored before it appeared -- is fused
            temporal-only by fuse_frame rather than dropped, which is what lets
            a slow opinion be useful instead of disruptive.
            """
            got = grounder.latest(t)
            if got is None:
                return {}, None
            scores, tids, age = got
            vlm = {tid: scores[i] for i, tid in enumerate(tids)
                   if i < len(scores) and tid in probs_by_track}
            return fuse_frame(probs_by_track, vlm, policy=policy,
                              classes=fuse_classes), age

    # The cache records the inference settings it was BUILT with, which is the
    # right provenance for the features it stores -- but window_size and
    # min_frames_for_pred are deployment decisions, not properties of a feature
    # cache. Pinning them here means a measured improvement (window 32 -> 48,
    # (measured value removed from the handover copy)
    # would quietly serve two different configurations depending on which
    # recording you picked. The live config wins where it says something.
    inf = dict(cache.meta.get("inference", {}))
    live_inf = (CONTEXT_INFERENCE or {})
    for k in ("window_size", "min_frames_for_pred", "temporal_input_fps",
              "label_smooth_window", "label_switch_margin"):
        if k in live_inf:
            inf[k] = live_inf[k]
    win = int(inf.get("window_size", 32))
    minf = int(inf.get("min_frames_for_pred", 4))
    # The detector stride is already baked into the cache (frames it skipped
    # have coasted boxes). Only the temporal stride is still ours to apply.
    stride = StrideController(1, int(inf.get("temporal_stride", 1)))
    # Same smoothing as the live path, so a cached replay and a live run of the
    # same model look alike as well as decide alike.
    smooth_box = BoxSmoother()
    smooth_label = LabelSmoother(window=int(inf.get("label_smooth_window", 3)),
                                 margin=int(inf.get("label_switch_margin", 0)))

    hist = defaultdict(lambda: deque(maxlen=win))
    # The cache holds EVERY frame of the source video -- 30 fps for session 0325
    # -- and this used to append all of them, so the 32-frame window spanned 1.0 s
    # (measured value removed from the handover copy)
    # the dashboard's default view. See HistorySampler.
    sampler = HistorySampler(float(inf.get("temporal_input_fps", 1.0)))
    dwell: Dict[int, Dict] = {}
    # Everything per-student below is keyed by SEAT, not the tracker's id: the
    # tracker re-issues ids, which changed the number on screen and discarded the
    # 1 Hz history the model needs. See SeatRegistry.
    handoff = SeatRegistry()
    key_of: Dict[int, int] = {}
    #: Last frame the VLM was asked, and how often to ask. 25 frames is about
    #: one opinion per second of source video.
    vlm_every = int(cache.meta.get("fps", 25.0))
    last_vlm_frame = -10 ** 9
    vlm_reported = False
    fps = cache.fps
    n_pushed = 0
    wall0 = time.time()

    for frame in cache.frame_index:
        if should_stop():
            break
        frame = int(frame)
        t = frame / fps
        rows = cache.rows_for(frame)
        students: Dict[str, Dict] = {}
        probs_by_track: Dict[int, list] = {}

        # Forget tracks the tracker dropped, on empty frames too — otherwise a
        # reused track id would inherit a stale cached prediction.
        key_of = handoff.resolve(t, {int(cache.track_id[r]): cache.bbox[r]
                                     for r in rows})
        for gone in handoff.pop_cleared():
            hist.pop(gone, None)
            dwell.pop(gone, None)
        for was, now in handoff.pop_moved():
            if was in hist:
                hist[now] = hist.pop(was)
            if was in dwell:
                dwell[now] = dwell.pop(was)
        live_ids = [key_of[int(cache.track_id[r])] for r in rows]
        stride.drop_missing(live_ids)
        sampler.drop(live_ids)
        smooth_box.drop(live_ids)
        smooth_label.drop(live_ids)

        if rows:
            vec = cache.live_vector(rows, entry.head_pose_backend)
            for k, r in enumerate(rows):
                # `t` is frame / fps, i.e. source time, so the frames admitted do
                # not change when the page replays faster or slower.
                sid = key_of[int(cache.track_id[r])]
                if sampler.should_append(sid, t):
                    hist[sid].append(vec[k])

            for k, r in enumerate(rows):
                tid = key_of[int(cache.track_id[r])]
                h = hist[tid]
                if len(h) < minf:
                    # A box from the first confirmed frame, labelled as not yet
                    # decided, instead of no box for the ~10 s the history takes.
                    students[str(tid)] = warming_record(
                        smooth_box(tid, cache.bbox[r]))
                    continue
                if stride.should_predict(tid, frame):
                    res = stride.store(tid, frame,
                                       predict_window(bundle, np.stack(list(h))))
                else:
                    res = stride.cached(tid)
                cue = smooth_label(tid, res["displayed_cue"])

                prev = dwell.get(tid)
                if prev and prev["cue"] == cue:
                    prev["dwell"] = t - prev["since"]
                else:
                    dwell[tid] = {"cue": cue, "since": t, "dwell": 0.0,
                                  "alerted": False}
                st = dwell[tid]
                probs_by_track[tid] = res["probs"]
                students[str(tid)] = {
                    "cue": cue,
                    "conf": round(float(res["confidence"]), 2),
                    "bbox": [round(float(v), 1)
                             for v in smooth_box(tid, cache.bbox[r])],
                    "dwell": st["dwell"],
                    "alerted": st["alerted"],
                    "raw_cue": res["cue"],
                    "abstained": bool(res["abstained"]),
                    "alert_allowed": bool(res["alert_allowed"]),
                    "rescued": bool(res.get("rescued", False)),
                }

        jpg = cache.jpeg(frame)

        # The VLM runs on a stride of its own. It costs seconds per frame, so
        # asking it every frame would make the replay unwatchable; between its
        # opinions the previous one is carried, which is the same thing the
        # temporal stride already does for the model itself. A student the VLM
        # has never seen is fused temporal-only rather than dropped.
        if submit is not None and students:
            # Offer work on a stride, and read whatever is ready EVERY frame.
            # The two rates are independent on purpose: the VLM answers when it
            # answers, and the display never waits for it.
            if frame - last_vlm_frame >= vlm_every:
                last_vlm_frame = frame
                submit(jpg, rows, t)
            fused, age = fuse_latest(probs_by_track, t)
            err = grounder.error
            if err and not vlm_reported:
                # Degrade to the temporal model and say so once, rather than
                # per frame: the fused entry is the same checkpoint plus an
                # opinion, and the checkpoint is still right without it.
                print(f"[replay] VLM failed, continuing temporal-only: {err}",
                      flush=True)
                vlm_reported = True
            # Distinguish "the VLM has not loaded yet" from "the VLM looked and
            # had nothing to add". Both show no fusion, and only one is a
            # problem.
            if not fused and not getattr(grounder, "ready", True):
                for srec in students.values():
                    srec["vlm_loading"] = True
            for tid, fs in fused.items():
                srec = students.get(str(tid))
                if srec is None:
                    continue
                srec["fusion"] = fs.to_json()
                # Staleness is REPORTED, never hidden. An opinion seconds old
                # about a student who has since moved is worth showing and worth
                # labelling; it is not worth passing off as current.
                srec["fusion"]["age_s"] = None if age is None else round(age, 1)
                if fs.cue:                  # the policy picked a label
                    srec["cue"] = fs.cue
                srec["contested"] = bool(fs.contested)
        if jpg is not None and overlay and students:
            jpg = _draw(cv2, jpg, students, cache.meta, blur_faces)

        push_fn(t, jpg, students, class_names)
        n_pushed += 1

        if realtime and speed > 0:
            target = wall0 + (n_pushed / (fps * speed))
            time.sleep(max(0.0, target - time.time()))

    return n_pushed


def _draw(cv2, jpeg_bytes: bytes, students: Dict[str, Dict], meta: Dict,
          blur_faces: bool) -> bytes:
    """Draw boxes and cue labels onto the cached (clean, downscaled) frame.

    Boxes are cached in full-resolution coordinates and the frame is stored
    downscaled, so every coordinate is scaled by the same factor the JPEG was.
    """
    import numpy as np
    img = cv2.imdecode(np.frombuffer(jpeg_bytes, np.uint8), cv2.IMREAD_COLOR)
    if img is None:
        return jpeg_bytes
    src_w = meta.get("source_width")
    if not src_w:
        raise SystemExit(
            "this session cache predates `source_width` in meta.json, so the "
            "cached boxes cannot be scaled onto the cached frames. Rebuild it "
            "with precompute_session.py — drawing at the wrong scale would put "
            "every box in the wrong place while still looking like a result.")
    scale = img.shape[1] / float(src_w)
    for seat, s in students.items():
        x1, y1, x2, y2 = [int(v * scale) for v in s["bbox"]]
        warming = s.get("warming", False)
        col = WARMUP_COLOUR if warming else colour_for(s["cue"])
        if blur_faces:
            hh = max(1, int(0.35 * (y2 - y1)))
            roi = img[max(0, y1):max(0, y1) + hh, max(0, x1):max(0, x2)]
            if roi.size:
                img[max(0, y1):max(0, y1) + hh, max(0, x1):max(0, x2)] = \
                    cv2.GaussianBlur(roi, (31, 31), 0)
        cv2.rectangle(img, (x1, y1), (x2, y2), col, 1 if warming else 2)
        cv2.putText(img, f"{seat} {s['cue'].replace('_', ' ')}",
                    (x1, max(14, y1 - 5)), cv2.FONT_HERSHEY_SIMPLEX, 0.45, col, 1,
                    cv2.LINE_AA)
    ok, buf = cv2.imencode(".jpg", img, [cv2.IMWRITE_JPEG_QUALITY, 70])
    return buf.tobytes() if ok else jpeg_bytes


def main():
    """Headless replay: useful to diff two models over the same cache."""
    import argparse
    import model_registry as MR

    ap = argparse.ArgumentParser(description=main.__doc__)
    ap.add_argument("--cache", required=True)
    ap.add_argument("--model", default=None,
                    help="variant id, e.g. arch/asrf_556_hp (default: registry default)")
    ap.add_argument("--device", default="cpu")
    ap.add_argument("--out", default=None, help="write per-frame cues as JSONL")
    args = ap.parse_args()

    entries = MR.scan()
    entry = MR.find(entries, args.model) if args.model else MR.default_entry(entries)
    if entry is None:
        raise SystemExit(f"unknown model {args.model!r}")
    if not entry.deployable:
        raise SystemExit(f"{entry.variant_id} is not deployable: {entry.blocked_reason}")
    # A cache holds base + both head-pose blocks and nothing else. Without this
    # the run reaches predict_window's width assert, which correctly reports
    # 556 against 1074 but reads as a broken extractor rather than as a model
    # this cache cannot serve.
    if not entry.replay_capable:
        raise SystemExit(
            f"{entry.variant_id} cannot replay a session cache: "
            f"{entry.blocked_reason}")

    cache = SessionCache(args.cache)
    bundle, cal = load_model(entry, args.device)
    print(f"[replay] {entry.variant_id} seed {entry.seed} — {bundle.describe()}")
    print(f"[replay] head-pose block: {entry.head_pose_backend}")

    fh = open(args.out, "w") if args.out else None
    counts: Dict[str, int] = defaultdict(int)

    def push(t, jpg, students, cue_names):
        for s in students.values():
            if s.get("warming"):
                continue
            counts[s["cue"]] += 1
        if fh:
            fh.write(json.dumps({"t": t, "students": students}) + "\n")

    t0 = time.time()
    n = replay(cache, entry, bundle, push, realtime=False, overlay=False)
    el = time.time() - t0
    print(f"[replay] {n} frames in {el:.1f}s ({n / max(el, 1e-9):.1f} fps, "
          f"temporal head only)")
    total = sum(counts.values()) or 1
    for cue, c in sorted(counts.items(), key=lambda kv: -kv[1]):
        print(f"  {cue:16} {c:6d}  {100 * c / total:5.1f}%")
    if fh:
        fh.close()
        print(f"written: {args.out}")


if __name__ == "__main__":
    main()
