#!/usr/bin/env python3
"""Instructor dashboard — real-time analytics, alerts, and a model selector.

Implements the `Thesis_Topic.md` deliverable:

    "The project will give teachers/professors a dynamic dashboard showing
     real-time analytics and alerts so enabling quick interventions..."

Serves a live view over the pipeline: annotated video frame, per-student cue
state, a rolling alert log for sustained off-task episodes, and class-level
analytics over time — plus two controls the thesis needs and a product would
not: **which recording is being analysed**, and **which trained model is
producing the cues**.

Sources
-------
The page can be pointed at three kinds of thing, and the difference between them
is cost, not features:

``session``   a cache built by ``precompute_session`` from one video. Everything
              except the temporal head is already computed, so replay needs no
              detector, no GPU, and switching models is instant. This is what an
              uploaded recording becomes once analysed.
``video``     a file, webcam index or camera URL run through the whole pipeline
              live. Expensive, and a model switch restarts it.
``cue log``   a recorded JSONL of decisions. Stdlib only — no torch, no mmdet —
              and no model to switch, because a cue log stores conclusions
              rather than features.

A recording uploaded through the page lands as a ``video``, is analysed once
into a ``session``, and is used from the session thereafter.

Why a model selector is part of the deliverable
-----------------------------------------------
The thesis is a comparison of architectures and feature blocks, and the tables
in `work_dirs/thesis/tables/` are frame-level macro-F1 on held-out splits. A
number like 0.50 does not tell a reader what the difference between two models
*looks like* to an instructor. Running the same classroom minute through each
checkpoint does.

Selection is on **validation** only, in the dropdown ordering and in the
default. Test numbers are shown but never rank anything: `[internal notes, not included]`
spent the test split once, and a UI that sorted by it would spend it again on
every page load.

DESIGN CONSTRAINTS honoured from the rest of the project:
  * Language is VISIBLE-CUE only. The UI says "head down 45 s", never
    "not paying attention" — the taxonomy's founding principle
    (attention/taxonomy.py) and the supervisor's reframing.
  * Alerts fire on sustained EPISODES (attention/events.py), not single frames,
    and only when the model's own validation-fitted alert threshold allows it.
    A model that cannot reach 85% selective accuracy raises no alerts at all,
    and the UI says which.
  * Privacy: no identity, no demographics. Students are seat numbers. Faces can
    be blurred with --blur-faces, and the served frame is downscaled.

    python server.py --session tools/dashboard/sessions/0325   # cheapest
    python server.py --config LLMDet/configs/attention_runtime.yaml --video 0
    python server.py --replay live_session.jsonl
    python server.py --config LLMDet/configs/attention_runtime.yaml
        # no source: upload one from the page

UPLOADS AND EXPOSURE. The page can write video files to ``uploads/``. The server
binds 0.0.0.0 by default, which on a shared machine means anyone who can reach
the port can upload. There is no authentication — this is a thesis demo, not a
deployed service. Use ``--host 127.0.0.1`` when the browser is on the same
machine, or ``--no-upload`` to serve read-only.
"""
import argparse
import base64
import json
import sys
import threading
import time
import traceback
from collections import defaultdict, deque
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from urllib.parse import parse_qs, urlparse

sys.path.insert(0, str(Path(__file__).resolve().parent))

import sources as SRC                                           # noqa: E402

STATE = {
    "frame_jpeg_b64": None,
    "t": 0.0,
    "students": {},        # seat -> {cue, since, conf}
    "alerts": deque(maxlen=100),
    "history": deque(maxlen=600),   # (t, off_task_fraction)
    "class_summary": {},
    "running": False,
    "source": "",
    "source_id": None,
    "model": {},           # the active entry, as shown in the header
    "notice": "",          # e.g. "alerts disabled for this model"
    "capture": {},         # live-source throughput and drop rate
    "job": {},             # background analyse job
    # Why the last run's failure is state and not just a log line: the worker
    # runs in a daemon thread, so an exception there kills the thread and
    # leaves the page showing "idle" with an empty video panel and no reason.
    # That is indistinguishable from a source that simply has not been
    # started, and it is what made a failed session replay look like a
    # rendering bug for an afternoon.
    "error": "",           # why the last run stopped, if it stopped badly
    "thresholds": {},      # the running model's thresholds; see set_thresholds
}
LOCK = threading.Lock()

#: Frames POSTed by the page's own webcam. One buffer for the process: the
#: dashboard runs one pipeline at a time, and a second camera would be a second
#: pipeline pushing into the same STATE.
CAMERA = None


def camera_buffer(reset=False):
    """The browser-camera frame buffer, created on first use.

    Imported lazily because pipeline_bridge pulls in torch, and replay mode is
    deliberately stdlib-only.

    ``reset`` replaces it with a fresh one, and starting a camera run MUST pass
    it. `run_live` ends by releasing its reader, and for the browser camera the
    reader is this process-wide buffer -- so after the first camera run the
    buffer is permanently `closed`. A second run then attaches to a dead buffer:
    `read()` reports the pusher gone on its first call, `run_live` returns
    normally, and because nothing raised, STATE["error"] stays empty. Meanwhile
    `put()` still accepts frames and bumps `seq`. The page shows `sent` climbing,
    `analysed` frozen and "the camera pipeline is not running", with no error
    anywhere to explain it. That cost a day; it is the second run that is broken,
    which is why it looked intermittent.
    """
    global CAMERA
    if CAMERA is None or reset:
        from pipeline_bridge import PushedFrames
        CAMERA = PushedFrames()
    return CAMERA

# Alert only after a cue has persisted this long — matches EventConfig
# min_duration_s. Single-frame flicker must never page an instructor.
# Keyed by the SIX cue classes; `active_policy` projects it onto whichever
# taxonomy the running model predicts.
ALERT_AFTER_S = {"phone_use": 15.0, "head_down": 30.0,
                 "turned_to_peer": 30.0, "looking_away": 20.0}

#: The cue6 policy, used in replay mode and as the fallback. A recorded cue log
#: stores decisions with no model attached, and every recorded log predates the
#: coarse taxonomies, so cue6 is the correct reading of one.
CUE6_POLICY = {
    "taxonomy": "cue6",
    "off_task_classes": ["looking_away", "head_down", "turned_to_peer",
                         "phone_use"],
    "alert_dwell": dict(ALERT_AFTER_S),
    "off_task_impure": {},
}


def _vlm_status():
    """The registry's verdict on the VLM backend, or why it could not ask."""
    try:
        import model_registry as MR
        return dict(MR.VLM_STATUS)
    except Exception as e:                                   # noqa: BLE001
        return {"available": False, "reason": f"{type(e).__name__}: {e}"}


def active_policy(taxonomy="cue6"):
    """Off-task classes and alert dwells for the taxonomy now running.

    A model trained on `onoff_reliable` predicts `on_task`/`off_task`, neither
    of which appears in ALERT_AFTER_S. Without this projection the off-task
    share read 0% for every frame and no alert could ever fire — the page would
    have shown a calm room full of phones. The rules, and why the two questions
    get different ones, are in ``attention.taxonomy``.

    Falls back to cue6 if ``attention`` is not importable: replay mode is
    deliberately stdlib-only, and a cue log is cue6 anyway.
    """
    if taxonomy == "cue6":
        return CUE6_POLICY
    try:
        sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "LLMDet"))
        from attention.taxonomy import (taxonomy_alert_dwell,
                                        taxonomy_off_task_classes,
                                        taxonomy_off_task_is_impure)
        return {
            "taxonomy": taxonomy,
            "off_task_classes": taxonomy_off_task_classes(taxonomy),
            "alert_dwell": taxonomy_alert_dwell(taxonomy, ALERT_AFTER_S),
            "off_task_impure": taxonomy_off_task_is_impure(taxonomy),
        }
    except Exception as e:
        print(f"[dashboard] cannot project the alert policy onto {taxonomy!r} "
              f"({type(e).__name__}: {e}); using the cue6 policy", flush=True)
        return CUE6_POLICY


# ---------------------------------------------------------------------------
# runner: owns the one thread that is currently producing frames
# ---------------------------------------------------------------------------

class Runner:
    """Starts, stops and replaces one background thread.

    Switching model or source must not leave the old thread running: two
    pipelines pushing into one STATE would interleave output from two
    configurations under one configuration's name, which is the sort of
    plausible-looking result this project keeps having to hunt down.
    """

    def __init__(self, on_start=None):
        self.thread = None
        self._stop = threading.Event()
        self.lock = threading.Lock()
        self.on_start = on_start

    def should_stop(self):
        return self._stop.is_set()

    def alive(self):
        return self.thread is not None and self.thread.is_alive()

    def request_stop(self):
        """Ask, without waiting. A cancel button should return immediately;
        the worker notices at its next frame, which on CPU can be seconds."""
        self._stop.set()

    # Generous, because the live source checks the stop flag once per frame and
    # a single detector frame on CPU is seconds. A tight timeout would turn a
    # slow switch into a spurious 409.
    #
    # 210s, not 60s, and the reason is measured: a cold live start on the Space
    # spends ~139s loading the detector, CLIP and the head-pose backend BEFORE
    # it reaches the loop that checks the stop flag [internal notes, not included]. A switch
    # requested during that window could never be honoured in 60s, so it always
    # raised.
    def stop(self, join_timeout=210.0):
        self._stop.set()
        t = self.thread
        if t and t.is_alive():
            t.join(timeout=join_timeout)
            if t.is_alive():
                # Refusing to start another must leave the current one ALONE.
                # Clearing the flag is the whole point: it was set two lines up,
                # and without this the run we just declined to replace sees it
                # at its next frame and exits -- so a refused switch killed the
                # camera, reported no error (run_live returned normally), and
                # the page showed "sent" climbing against "analysed 0" with
                # "the camera pipeline is not running". That is the SAME symptom
                # as the released-buffer bug, reached by a completely different
                # route, which is why fixing that one did not make it go away.
                self._stop.clear()
                raise RuntimeError(
                    "the previous run did not stop within "
                    f"{join_timeout:.0f}s; refusing to start another")
        self.thread = None

    def start(self, target, *args, **kwargs):
        with self.lock:
            self.stop()
            self._stop = threading.Event()
            if self.on_start:
                self.on_start()
            self.thread = threading.Thread(target=self._guard, args=(target,) + args,
                                           kwargs=kwargs, daemon=True)
            self.thread.start()

    @staticmethod
    def _guard(target, *args, **kwargs):
        """Run ``target``, and make a crash visible on the page.

        Without this the thread dies, ``running`` stays False and the UI shows
        an idle dashboard with no video and no explanation -- the same thing it
        shows before anything has been started. The traceback still goes to the
        log; this puts the one-line reason where the operator is looking.
        """
        try:
            target(*args, **kwargs)
        except BaseException as e:                     # noqa: BLE001
            traceback.print_exc()
            with LOCK:
                STATE["running"] = False
                STATE["error"] = f"{type(e).__name__}: {e}"


def reset_state():
    with LOCK:
        STATE["frame_jpeg_b64"] = None
        STATE["t"] = 0.0
        STATE["students"] = {}
        STATE["alerts"].clear()
        STATE["history"].clear()
        STATE["class_summary"] = {}
        STATE["capture"] = {}
        STATE["running"] = False
        STATE["error"] = ""
        # A threshold override belongs to one run of one model. Detaching here,
        # on every start, is what stops it outliving the model it was set on.
        STATE["thresholds"] = {}
        CONTEXT.pop("bundle", None)


RUNNER = Runner(on_start=reset_state)    # produces frames
JOBS = Runner()                          # builds session caches
CONTEXT = {}


# ---------------------------------------------------------------------------
# HTTP
# ---------------------------------------------------------------------------

class Handler(BaseHTTPRequestHandler):
    def log_message(self, *a):
        pass

    def _send(self, code, body, ctype="application/json"):
        b = body.encode() if isinstance(body, str) else body
        self.send_response(code)
        self.send_header("Content-Type", ctype)
        self.send_header("Content-Length", str(len(b)))
        self.send_header("Cache-Control", "no-store")
        self.end_headers()
        self.wfile.write(b)

    def _err(self, code, msg):
        self._send(code, json.dumps({"error": str(msg)}))

    def do_GET(self):
        path = urlparse(self.path).path
        if path in ("/", "/index.html"):
            p = Path(__file__).with_name("index.html")
            return self._send(200, p.read_bytes(), "text/html; charset=utf-8")
        if path == "/api/state":
            with LOCK:
                return self._send(200, json.dumps({
                    "t": STATE["t"],
                    "running": STATE["running"],
                    "source": STATE["source"],
                    "source_id": STATE["source_id"],
                    "frame": STATE["frame_jpeg_b64"],
                    "students": STATE["students"],
                    "alerts": list(STATE["alerts"])[-25:],
                    "history": list(STATE["history"]),
                    "class_summary": STATE["class_summary"],
                    "model": STATE["model"],
                    "notice": STATE["notice"],
                    "capture": STATE["capture"],
                    "job": STATE["job"],
                    "error": STATE["error"],
                    "thresholds": STATE["thresholds"],
                }))
        if path == "/api/models":
            return self._send(200, json.dumps({
                "models": [e.to_json() for e in CONTEXT.get("entries", [])],
                "active": STATE["model"].get("variant_id"),
                "switchable": bool(CONTEXT.get("entries")),
                "switch_cost": CONTEXT.get("switch_cost", ""),
                # A session cache cannot serve a head-stream model, so the page
                # needs to know which kind of source is running to disable the
                # right options instead of offering one that will be refused.
                "source_kind": (CONTEXT.get("source") or {}).get("kind", ""),
                # Why the VLM-assisted entry is or is not on the list. A model
                # that is simply absent looks the same as one nobody tried to
                # offer; this says which, and on what version.
                "vlm_status": _vlm_status(),
            }))
        if path == "/api/sources":
            return self._send(200, json.dumps({
                "sources": SRC.list_sources(CONTEXT.get("cli_video"),
                                            CONTEXT.get("cli_session")),
                "active": STATE["source_id"],
                "upload_enabled": CONTEXT.get("upload_enabled", False),
                "max_upload_mb": round(SRC.MAX_UPLOAD_BYTES / 1e6),
                "accepts": sorted(SRC.VIDEO_EXTS),
                "can_analyse": bool(CONTEXT.get("config")),
                "stream_enabled": bool(CONTEXT.get("config")),
                "stream_schemes": [x.rstrip(":/") for x in SRC.STREAM_SCHEMES],
                # The page captures with getUserMedia and POSTs frames, so this
                # needs the detector (--config) but nothing on this machine.
                "camera_enabled": bool(CONTEXT.get("config")),
                "camera_id": SRC.BROWSER_CAMERA_ID,
                "persist_enabled": SRC.ARCHIVE_DIR is not None,
            }))
        self._err(404, "not found")

    def do_POST(self):
        u = urlparse(self.path)
        q = parse_qs(u.query)
        try:
            if u.path == "/api/model":
                e = switch_model(self._json().get("variant_id", ""))
                return self._send(200, json.dumps({"active": e.variant_id}))
            if u.path == "/api/thresholds":
                return self._send(200, json.dumps(set_thresholds(self._json())))
            if u.path == "/api/source":
                b = self._json()
                sid = select_source(b.get("source_id", ""), b.get("mode"))
                return self._send(200, json.dumps({"active": sid}))
            if u.path == "/api/upload":
                return self._upload(q.get("name", [""])[0])
            if u.path == "/api/analyse":
                b = self._json()
                return self._send(200, json.dumps(
                    start_analyse(b.get("source_id", ""),
                                  b.get("max_frames"))))
            if u.path == "/api/camera/frame":
                return self._camera_frame()
            if u.path == "/api/cancel":
                JOBS.request_stop()
                with LOCK:
                    if STATE["job"].get("status") == "running":
                        STATE["job"] = {**STATE["job"],
                                        "message": "cancelling…"}
                return self._send(200, json.dumps({"cancelling": True}))
        except ValueError as e:
            return self._err(400, e)
        except FileNotFoundError as e:
            return self._err(404, e)
        except SystemExit as e:            # raised by the model loaders
            return self._err(400, e)
        except RuntimeError as e:
            return self._err(409, e)
        except Exception as e:             # noqa: BLE001 - report, don't hang
            traceback.print_exc()
            return self._err(500, f"{type(e).__name__}: {e}")
        self._err(404, "not found")

    def _json(self):
        n = int(self.headers.get("Content-Length") or 0)
        try:
            return json.loads(self.rfile.read(n) or b"{}")
        except json.JSONDecodeError:
            raise ValueError("malformed JSON")

    def _upload(self, name):
        if not CONTEXT.get("upload_enabled"):
            raise RuntimeError("uploads are disabled (--no-upload)")
        if not name:
            raise ValueError("missing ?name=")
        n = int(self.headers.get("Content-Length") or 0)
        dest = SRC.save_upload(self.rfile, n, name)
        print(f"[dashboard] uploaded {dest.name} ({n / 1e6:.1f} MB)")
        return self._send(200, json.dumps({
            "id": f"video:{dest}", "name": dest.name,
            "size_mb": round(n / 1e6, 1)}))

    def _camera_frame(self):
        """One JPEG from the page's webcam, straight into the frame buffer.

        Decoded here rather than on the worker thread so a corrupt frame is
        rejected with a 400 at the source instead of killing the pipeline, and
        so the buffer only ever holds something the pipeline can use.

        Returns the buffer's own counters, which is what lets the page throttle
        itself: if it is pushing far faster than the pipeline consumes, the drop
        count is the honest signal to slow down.
        """
        if not CONTEXT.get("config"):
            raise RuntimeError(
                "this dashboard was started without --config, so it cannot run "
                "the detector on camera frames")
        n = int(self.headers.get("Content-Length") or 0)
        if n <= 0:
            raise ValueError("empty camera frame")
        if n > MAX_CAMERA_FRAME_BYTES:
            raise ValueError(
                f"camera frame is {n / 1e6:.1f} MB, over the "
                f"{MAX_CAMERA_FRAME_BYTES / 1e6:.0f} MB limit")
        blob = self.rfile.read(n)
        import numpy as np
        import cv2
        img = cv2.imdecode(np.frombuffer(blob, np.uint8), cv2.IMREAD_COLOR)
        if img is None:
            raise ValueError("camera frame is not a decodable image")
        buf = camera_buffer()
        buf.put(img)

        # Whether anything is actually READING this buffer, and if not, why.
        #
        # Without this the endpoint returns 200 and a rising `dropped` count in
        # three completely different situations — the pipeline is keeping up,
        # the detector is still loading (~40 s on first use), and the camera is
        # not the active source at all — and the page renders all three as
        # "skipped". That made a working pipeline and a pipeline that was never
        # started look identical from the browser.
        src = CONTEXT.get("source") or {}
        is_cam = src.get("kind") == "camera"
        alive = RUNNER.alive()
        if not is_cam:
            status = ("not consuming: the active source is "
                      f"{src.get('kind') or 'none'}, not the camera. Press "
                      f"Start camera again to switch.")
        elif not alive:
            status = "not consuming: the camera pipeline is not running"
        elif buf.taken == 0:
            # Measured on the deployed t4-medium Space from a cold start:
            # 125 s from selecting the camera to the first analysed frame. The
            # docstring in app.py quotes ~40 s for the detector weights alone;
            # end to end it is GroundingDINO + CLIP + the head-pose backend, and
            # on a Space that has just woken it is slower still. Understating it
            # is what makes a warming pipeline look like a dead one.
            status = ("warming up: loading the detector, CLIP and the head-pose "
                      "backend. Cold start measured at ~2 minutes on this "
                      "hardware. Frames pushed until then are skipped; watch "
                      "'analysed', not 'skipped'.")
        else:
            status = "consuming"
        return self._send(200, json.dumps({
            "received": int(buf.seq), "processed": int(buf.taken),
            "dropped": int(buf.dropped),
            "consuming": bool(is_cam and alive),
            "status": status,
            "error": STATE.get("error", "")}))


#: Largest single webcam frame accepted, before decoding. A 1280x720 JPEG at
#: quality 0.7 is ~120 KB; this is generous and still bounds a malicious post.
MAX_CAMERA_FRAME_BYTES = 4 << 20


def push_frame(t, jpeg_bytes, students, cue_names):
    """Update dashboard state from one processed frame."""
    with LOCK:
        STATE["t"] = float(t)
        if jpeg_bytes is not None:
            STATE["frame_jpeg_b64"] = base64.b64encode(jpeg_bytes).decode()
        STATE["students"] = students

        pol = CONTEXT.get("policy") or CUE6_POLICY
        off_classes = pol["off_task_classes"]
        dwell_for = pol["alert_dwell"]
        # The DISPLAYED cue, not the raw one: a frame the model abstained on
        # reads `uncertain`, and an abstention must not be counted as off-task
        # evidence. `cue` is already the displayed value everywhere it is set.
        # A track still warming up has a box and no prediction: a student on
        # screen, but no evidence either way, so it stays out of the off-task
        # share AND its denominator. Counting it would dilute the percentage
        # for the first ~10 s of every new track.
        decided = [s for s in students.values() if not s.get("warming")]
        off = [s for s in decided if s["cue"] in off_classes]
        frac = len(off) / max(len(decided), 1)
        STATE["history"].append([round(float(t), 1), round(frac, 3)])

        counts = defaultdict(int)
        for s in decided:
            counts[s["cue"]] += 1
        STATE["class_summary"] = {
            "n_students": len(students),
            "warming_up": len(students) - len(decided),
            "off_task": len(off),
            "off_task_pct": round(100 * frac),
            "by_cue": dict(counts),
            "taxonomy": pol["taxonomy"],
            # Named so the UI can footnote a percentage that is NOT comparable
            # to cue6's: `onoff_reliable`'s off_task merges `uncertain`, so it
            # counts students who merely could not be seen.
            "off_task_impure": pol["off_task_impure"],
        }

        for seat, s in students.items():
            thr = dwell_for.get(s["cue"])
            # Selective prediction: an alert also needs the model to be confident
            # enough, at the threshold fitted on ITS OWN validation predictions
            # (tools/dashboard/calibrate_registry.py). Replay logs recorded
            # before abstention existed carry no `alert_allowed` key, so the
            # default admits them and old sessions still replay.
            if not s.get("alert_allowed", True):
                continue
            if thr and s.get("dwell", 0) >= thr and not s.get("alerted"):
                STATE["alerts"].append({
                    "t": round(float(t), 1),
                    "seat": seat,
                    "cue": s["cue"],
                    "dwell": round(s["dwell"]),
                    # Visible-cue phrasing only. Never "inattentive".
                    "text": f"Seat {seat}: {s['cue'].replace('_', ' ')} "
                            f"for {round(s['dwell'])}s",
                })
                s["alerted"] = True


# ---------------------------------------------------------------------------
# sources
# ---------------------------------------------------------------------------

def run_replay(path):
    """Replay a recorded cue log — no model, no torch, no GPU."""
    with LOCK:
        STATE["running"] = True
        STATE["source"] = f"cue log: {Path(path).name}"
        STATE["model"] = {}
    rows = [json.loads(l) for l in open(path)]
    for r in rows:
        if RUNNER.should_stop():
            break
        push_frame(r["t"], None, r["students"], r.get("cue_names", []))
        time.sleep(max(0.0, min(1.0, r.get("dt", 1.0))))
    with LOCK:
        STATE["running"] = False


def run_session(entry, cache_dir):
    """Replay a precomputed session cache through one model."""
    import session_replay as SR
    cache = SR.SessionCache(cache_dir)
    bundle, cal = SR.load_model(entry, CONTEXT["device"])
    attach_bundle(bundle)
    with LOCK:
        STATE["running"] = True
        STATE["source"] = (f"session: {Path(cache_dir).name} "
                           f"({cache.meta['n_frames']} frames @ "
                           f"{cache.fps:.0f} fps)")
        card = model_card(entry, cal)
        STATE["model"] = card
        STATE["notice"] = "" if card["alerts_enabled"] else \
            card["alerts_disabled_reason"]
        # A frameless cache replays as cue data over a blank panel, which reads
        # as a broken player rather than as a cache that was built without
        # frames. Name it instead of leaving the operator to guess.
        if cache.n_cached_frames == 0:
            STATE["error"] = (
                f"this session has no cached frames (no .jpg under "
                f"{Path(cache_dir).name}/frames), so cues replay over an "
                f"empty video panel. Delete the session and analyse the "
                f"recording again.")
        elif cache.n_cached_frames < int(cache.meta.get("n_frames", 0)):
            STATE["notice"] = (
                f"{cache.n_cached_frames} of {cache.meta.get('n_frames')} "
                f"frames are cached; the rest of the replay shows cues with "
                f"no image")
    print(f"[dashboard] {cache.n_cached_frames} cached frames")
    print(f"[dashboard] {entry.variant_id} — {bundle.describe()}")
    # A `+vlm` entry loads an 8 GB VLM on top of the temporal model. Built here,
    # after the cache and the checkpoint, so a failure to load it is reported
    # against a running dashboard rather than at import time.
    grounder = SR.load_grounder(entry, CONTEXT["device"])
    SR.replay(cache, entry, bundle, push_frame,
              should_stop=RUNNER.should_stop,
              blur_faces=CONTEXT["blur_faces"], realtime=True,
              speed=CONTEXT["speed"], grounder=grounder)
    with LOCK:
        STATE["running"] = False


def run_live_source(entry, video):
    """Full pipeline over a file, a webcam or an IP camera.

    Expensive: the detector dominates, and on CPU it dominates completely. For a
    live source that shows up as a drop rate rather than as a slowdown — the
    reader thread keeps the newest frame and the pipeline skips the rest — so
    the capture stats are surfaced to the page instead of only the log.
    """
    import session_replay as SR
    from pipeline_bridge import classify_source, run_live
    cal_path = SR.calibration_path(entry.variant_id)
    cal = json.loads(cal_path.read_text()) if cal_path.exists() else {}
    browser_cam = str(video) == SRC.BROWSER_CAMERA_ID
    # reset=True: a fresh buffer per run. See camera_buffer().
    frame_source = camera_buffer(reset=True) if browser_cam else None
    kind = "camera" if browser_cam else classify_source(video)[0]
    with LOCK:
        STATE["running"] = True
        STATE["source"] = (
            "camera: your browser's webcam" if browser_cam else
            f"{kind}: {Path(video).name if kind == 'file' else video}"
        ) + f" on {CONTEXT['device']}"
        card = model_card(entry, cal)
        STATE["model"] = card
        STATE["notice"] = "" if card["alerts_enabled"] else \
            card["alerts_disabled_reason"]

    def on_stats(s):
        with LOCK:
            STATE["capture"] = s

    stopped_because = run_live(
        CONTEXT["config"], video, push_frame,
        blur_faces=CONTEXT["blur_faces"], max_frames=CONTEXT["max_frames"],
        device=CONTEXT["device"], entry=entry,
        should_stop=RUNNER.should_stop, stats_fn=on_stats,
        frame_source=frame_source, on_bundle=attach_bundle)
    with LOCK:
        STATE["running"] = False
        # A live run that analysed nothing ended for a reason the page cannot
        # infer: it looks identical to a clean finish. Anything else -- the
        # browser closing the tab, Stop -- is an ordinary end, not an error.
        if stopped_because and not (STATE.get("capture") or {}).get("processed"):
            STATE["error"] = stopped_because


def model_card(entry, cal):
    """What the header shows about the running model.

    Both the deployed single-seed figure and the seed mean are carried, because
    they are different claims: the dashboard runs one checkpoint chosen as the
    best of three on validation, while the thesis tables report the mean over
    those three. Showing only the first would quietly overstate the model.
    """
    return {
        "variant_id": entry.variant_id,
        "label": entry.label,
        "experiment_id": entry.experiment_id,
        "seed": entry.seed,
        "n_seeds": entry.n_seeds,
        "sweep_label": entry.sweep_label,
        "head_pose_backend": entry.head_pose_backend,
        "val_macro_f1": entry.val.get("macro_f1"),
        "val_seed_mean": entry.val_seed_mean,
        "test_macro_f1": entry.test.get("macro_f1"),
        "test_seed_mean": entry.test_seed_mean,
        "display_threshold": cal.get("display_threshold"),
        "alert_threshold": cal.get("alert_threshold"),
        # Two independent reasons alerts can be off, and the card must not
        # claim they are on when either holds:
        #   calibration — no threshold reaches 85% selective accuracy;
        #   taxonomy    — every off-task class merges `uncertain`, so none is
        #                 alertable (attention.taxonomy.taxonomy_alert_dwell).
        # onoff_reliable hits the second: its calibration is fine, and it may
        # still page nobody.
        "alerts_enabled": bool(cal.get("alerts_enabled", True)) and bool(
            (CONTEXT.get("policy") or CUE6_POLICY)["alert_dwell"]),
        "alerts_disabled_reason": cal.get("alerts_disabled_reason", "") or (
            "" if (CONTEXT.get("policy") or CUE6_POLICY)["alert_dwell"] else
            f"no class of the {getattr(entry, 'taxonomy', 'cue6')} taxonomy may "
            f"raise an alert: each of its off-task classes merges "
            f"\u201cuncertain\u201d, and a student who cannot be seen is not "
            f"evidence of being off task. Cues are still displayed."),
        # Alert coverage is the instructor-facing consequence of model quality
        # and is NOT ordered like macro-F1. Every model is held to the same 85%
        # alert precision, so what a weaker one gives up is the *share of the
        # (measured value removed from the handover copy)
        # (measured value removed from the handover copy)
        "alert_coverage": cal.get("alert_coverage"),
        "alert_selective_accuracy": cal.get("alert_selective_accuracy"),
        "display_coverage": cal.get("display_coverage"),
        "checkpoint": entry.checkpoint,
        # What this model predicts. A 2-class abstaining model's macro-F1 is
        # not comparable to a 6-class one's, so the class count and the
        # taxonomy's own validation coverage travel with the number.
        "taxonomy": getattr(entry, "taxonomy", "cue6"),
        "class_names": list(getattr(entry, "class_names", []) or []),
        "n_classes": getattr(entry, "n_classes", 0),
        "taxonomy_coverage": getattr(entry, "coverage", 1.0),
        "abstains_on": list(getattr(entry, "abstains_on", []) or []),
        "comparable_group": getattr(entry, "comparable_group", ""),
        "is_canonical": getattr(entry, "is_canonical", True),
        "vlm": bool(getattr(entry, "vlm", False)),
        "vlm_model_id": getattr(entry, "vlm_model_id", ""),
        "vlm_policy": getattr(entry, "vlm_policy", ""),
        "vlm_base": getattr(entry, "vlm_base", ""),
        "alert_dwell": (CONTEXT.get("policy") or CUE6_POLICY)["alert_dwell"],
        "off_task_impure": (CONTEXT.get("policy") or CUE6_POLICY)["off_task_impure"],
    }


def _thresholds_view(bundle):
    return {
        "display": bundle.display_threshold,
        "alert": bundle.alert_threshold,
        "classes": list(bundle.class_names),
        "cues": bundle.effective_cue_thresholds(),
        "overridden": bool(bundle.cue_thresholds),
    }


def attach_bundle(bundle):
    """Make the running model's per-cue display bars adjustable from the page.

    Called by both run paths once their bundle exists. ``predict_window`` reads
    the bars off the bundle on every call, so changing them here takes effect
    from the next prediction without restarting anything.
    """
    with LOCK:
        CONTEXT["bundle"] = bundle
        STATE["thresholds"] = _thresholds_view(bundle)


def set_thresholds(body):
    """Set per-cue display bars on the running model, or reset them.

    ``{"cues": {"phone_use": 0.25}}`` merges into the bars already set;
    ``{"reset": true}`` returns to the calibrated rule. A demonstration control,
    not a calibration, so the page flags the run for as long as any bar is set.
    """
    with LOCK:
        bundle = CONTEXT.get("bundle")
        if bundle is None:
            raise RuntimeError(
                "no model is running, so there are no thresholds to change")
        if body.get("reset"):
            bundle.reset_cue_thresholds()
        else:
            cues = body.get("cues")
            if not isinstance(cues, dict) or not cues:
                raise ValueError('send "cues": {cue: bar}, or "reset": true')
            bundle.set_cue_thresholds(cues)
        STATE["thresholds"] = _thresholds_view(bundle)
        return dict(STATE["thresholds"])


def start(source, entry=None):
    """Point the dashboard at ``source`` (a dict from ``sources.list_sources``).

    ``source["kind"]`` decides the cost: a session replays from the cache, a
    video runs the whole pipeline, a cue log runs neither.
    """
    entry = entry or CONTEXT.get("entry")
    CONTEXT["source"] = source
    with LOCK:
        STATE["source_id"] = source.get("id")

    if source["kind"] == "cuelog":
        CONTEXT["switch_cost"] = ""
        CONTEXT["policy"] = CUE6_POLICY
        return RUNNER.start(run_replay, source["path"])

    if entry is None:
        raise RuntimeError("no model selected")
    CONTEXT["entry"] = entry
    # Before the first frame is pushed: push_frame reads this to decide what
    # counts as off-task and what may alert.
    CONTEXT["policy"] = active_policy(getattr(entry, "taxonomy", "cue6"))

    if source["kind"] == "camera" and source["path"] == "browser":
        if not CONTEXT.get("config"):
            raise RuntimeError(
                "no detector config; start the server with --config to analyse "
                "camera frames")
        CONTEXT["switch_cost"] = (
            "restarts the capture; the browser keeps streaming and the new "
            "model picks up from the next frame")
        # No reopen() here. There used to be one, and it was undone by the
        # very next line: RUNNER.start() begins by joining the PREVIOUS run,
        # whose last act is reader.release() -- on this same buffer. The revive
        # therefore always lost to the teardown that followed it, and the camera
        # worked exactly once per process. run_live_source takes a fresh buffer
        # instead, after the join, where nothing can close it behind us.
        return RUNNER.start(run_live_source, entry, SRC.BROWSER_CAMERA_ID)

    if source["kind"] == "session":
        # A session cache stores base + both head-pose blocks and nothing else.
        # A head-stream model would reach the width assert in predict_window and
        # fail there with a message about padding; say the real reason instead.
        if not getattr(entry, "replay_capable", True):
            raise ValueError(
                f"{entry.variant_id} cannot re-decide a cached session: "
                f"{entry.blocked_reason} Analyse the video or a camera with "
                f"this model instead, or pick a model that reads only base + "
                f"head pose.")
        CONTEXT["switch_cost"] = (
            "instant — the detector, tracker and features are cached, so only "
            "the temporal head re-runs")
        return RUNNER.start(run_session, entry, source["path"])

    if source["kind"] in ("video", "stream", "camera"):
        if not CONTEXT.get("config"):
            raise RuntimeError(
                "no detector config; start the server with --config to run "
                "the full pipeline")
        CONTEXT["switch_cost"] = (
            "reconnects to the camera and re-runs the whole pipeline; a live "
            "source has no cache to fall back on"
            if source["kind"] in ("stream", "camera") else
            "restarts the source and re-runs the whole pipeline; analyse it "
            "into a session to make switching instant")
        return RUNNER.start(run_live_source, entry, source["path"])

    raise ValueError(f"unknown source kind {source['kind']!r}")


def switch_model(variant_id):
    import model_registry as MR
    entry = MR.find(CONTEXT.get("entries", []), variant_id)
    if entry is None:
        raise ValueError(f"unknown model {variant_id!r}")
    if not entry.deployable:
        raise ValueError(f"{variant_id} cannot run live: {entry.blocked_reason}")
    src = CONTEXT.get("source")
    if src is None or src["kind"] == "cuelog":
        raise ValueError("this source has no model to switch "
                         "(a recorded cue log holds cues, not features)")
    start(src, entry)
    return entry


def select_source(source_id, mode=None):
    """Point the dashboard at a different recording.

    ``mode`` picks what to do with a video that already has a session:
    ``"session"`` (default when one exists) replays the cache, ``"video"``
    forces the full pipeline. It is not a preference — running a video live when
    a cache exists is a genuine choice between fidelity to the current config
    and speed.
    """
    kind, path = SRC.parse_id(source_id)
    if kind == "camera":
        # Not discovered on disk either: the page names it, and for "browser"
        # the frames come from the viewer rather than from this machine.
        src = {"id": f"camera:{path}", "kind": "camera",
               "name": ("your browser's webcam" if path == "browser"
                        else f"capture device {path}"),
               "path": path, "ready": True}
        start(src)
        return src["id"]
    if kind == "stream":
        # Typed by the user rather than discovered on disk, so there is nothing
        # in list_sources to look up. parse_id has already validated the URL.
        src = {"id": f"stream:{path}", "kind": "stream",
               "name": path, "path": path, "ready": True}
        start(src)
        return src["id"]

    listed = {s["id"]: s for s in SRC.list_sources(CONTEXT.get("cli_video"),
                                                   CONTEXT.get("cli_session"))}
    src = listed.get(source_id)
    if src is None:
        raise FileNotFoundError(f"no such source {source_id!r}")

    if kind == "video" and src.get("session") and mode != "video":
        src = listed.get(f"session:{Path(src['session']).resolve()}") or src
    start(src)
    return src["id"]


# ---------------------------------------------------------------------------
# analyse: build a session cache in the background
# ---------------------------------------------------------------------------

def start_analyse(source_id, max_frames=None):
    """Kick off a precompute pass over an uploaded/recorded video."""
    kind, video = SRC.parse_id(source_id)
    if kind == "stream":
        raise ValueError(
            "a live stream cannot be analysed into a session: precompute needs "
            "a finite source, and a camera has no end. Select it as a source "
            "to run the pipeline live instead.")
    if kind != "video":
        raise ValueError("only a video can be analysed; a session already is")
    if not video.exists():
        raise FileNotFoundError(f"{video} is gone")
    if not CONTEXT.get("config"):
        raise RuntimeError("no detector config; start the server with --config")
    if JOBS.alive():
        raise RuntimeError("an analysis is already running")

    out = SRC.session_dir_for(video)
    if SRC.session_meta(out) is not None:
        raise RuntimeError(f"{out.name} already has a session; delete it first")

    # Analysis is the detector-bound part of the system. Leaving a live pipeline
    # running alongside it would have both fighting for the same cores and make
    # the progress estimate meaningless.
    RUNNER.stop()
    with LOCK:
        STATE["running"] = False
        STATE["job"] = {"status": "running", "name": video.name,
                        "source_id": source_id, "pct": 0,
                        "message": "starting…"}
    JOBS.start(_analyse_thread, video, out,
               int(max_frames or CONTEXT["analyse_frames"]))
    return {"status": "running", "name": video.name}


def _analyse_thread(video, out, max_frames):
    from precompute_session import precompute

    def on_progress(p):
        with LOCK:
            STATE["job"] = {
                "status": "running", "name": video.name, **p,
                "message": (f"{p['frames_done']}/{p['frames_target']} frames · "
                            f"{p['fps']:.2f} fps · ~{p['eta_s']}s left"),
            }

    t0 = time.time()
    try:
        precompute(CONTEXT["config"], str(video), str(out),
                   device=CONTEXT["device"], max_frames=max_frames,
                   progress_fn=on_progress, should_stop=JOBS.should_stop)
    except KeyboardInterrupt as e:                    # cancelled
        with LOCK:
            STATE["job"] = {"status": "cancelled", "name": video.name,
                            "message": str(e)}
        return
    except BaseException as e:                        # noqa: BLE001
        traceback.print_exc()
        with LOCK:
            STATE["job"] = {"status": "failed", "name": video.name,
                            "message": f"{type(e).__name__}: {e}"}
        return

    meta = SRC.session_meta(out) or {}
    summary = (f"{meta.get('n_frames', '?')} frames, "
               f"{meta.get('n_tracks', '?')} students, "
               f"{round(time.time() - t0)}s")
    saving = SRC.ARCHIVE_DIR is not None
    with LOCK:
        STATE["job"] = {
            "status": "saving" if saving else "done", "name": video.name,
            "pct": 100, "session_id": f"session:{out.resolve()}",
            "message": summary + (" · saving so it is still here after a "
                                  "restart…" if saving else ""),
        }
    # Show the result rather than leaving the operator to find the new entry.
    # Before the save, which copies the whole recording to the bucket: playback
    # needs only the local cache, so there is no reason to make anyone wait.
    try:
        select_source(f"session:{out.resolve()}")
    except Exception:                                  # noqa: BLE001
        traceback.print_exc()
    if not saving:
        return

    try:
        SRC.archive_session(out, video)
        note = "saved — it will still be here after a restart"
    except BaseException as e:                         # noqa: BLE001
        traceback.print_exc()
        note = (f"NOT saved ({type(e).__name__}: {e}); it plays now but will "
                f"be gone after a restart")
    with LOCK:
        STATE["job"] = {**STATE["job"], "status": "done",
                        "message": f"{summary} · {note}"}


def restore_saved_recordings():
    """Bring back what earlier runs analysed. Runs in the background at boot.

    Not before serving: each saved session is a ~150 MB read from the bucket,
    and a page that does not answer for a minute looks like a Space that failed
    to start. Recordings appear in the list as each one lands.
    """
    try:
        got = SRC.restore_archive()
    except Exception:                                  # noqa: BLE001
        traceback.print_exc()
        return
    print(f"[dashboard] restored {len(got['sessions'])} saved session(s) and "
          f"{len(got['videos'])} upload(s) from {SRC.ARCHIVE_DIR}", flush=True)
    for name, why in got["skipped"]:
        print(f"[dashboard]   not restored: {name} ({why})", flush=True)


# ---------------------------------------------------------------------------

def main():
    ap = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--host", default="0.0.0.0",
                    help="bind address. Use 127.0.0.1 when the browser is on "
                         "this machine — uploads are unauthenticated.")
    ap.add_argument("--port", type=int, default=8080)
    ap.add_argument("--session", default=None,
                    help="a session cache from precompute_session.py")
    ap.add_argument("--replay", default=None, help="a recorded cue log (JSONL)")
    ap.add_argument("--config", default="LLMDet/configs/attention_runtime.yaml",
                    help="detector config, needed to analyse or stream a video")
    ap.add_argument("--video", default=None,
                    help="a file, a webcam index (0), or a camera URL")
    ap.add_argument("--model", default=None,
                    help="variant id, e.g. arch/asrf_556_hp. "
                         "Default: best deployable variant on validation.")
    ap.add_argument("--device", default="cpu", help="cpu (default) or cuda:N")
    ap.add_argument("--speed", type=float, default=1.0,
                    help="session replay speed multiplier (1.0 = source fps)")
    ap.add_argument("--analyse-frames", type=int, default=0,
                    help="frame cap when analysing an uploaded video")
    ap.add_argument("--blur-faces", action="store_true")
    ap.add_argument("--max-frames", type=int, default=0,
                    help="cap on a LIVE run; 0 (default) = no cap")
    ap.add_argument("--no-upload", action="store_true",
                    help="serve read-only: no file uploads accepted")
    args = ap.parse_args()

    cfg = Path(args.config)
    CONTEXT.update({
        "config": str(cfg) if cfg.exists() else None,
        "device": args.device,
        "blur_faces": args.blur_faces,
        "speed": args.speed,
        "max_frames": args.max_frames,
        "analyse_frames": args.analyse_frames,
        "cli_video": args.video,
        "cli_session": args.session,
        "upload_enabled": not args.no_upload,
        "entries": [],
    })
    if SRC.ARCHIVE_DIR is not None:
        print(f"[dashboard] analysed recordings are saved to {SRC.ARCHIVE_DIR}")
        threading.Thread(target=restore_saved_recordings, daemon=True,
                         name="restore-saved").start()
    if cfg.exists():
        SRC.UPLOAD_DIR.mkdir(parents=True, exist_ok=True)
        SRC.SESSION_DIR.mkdir(parents=True, exist_ok=True)
        # Hand the live inference settings to replay. A session cache records
        # the settings it was BUILT with, which is correct provenance for its
        # features but wrong for decisions: window_size and the history rate are
        # deployment choices, and pinning them to the cache means a measured
        # improvement never reaches a recording made before it.
        try:
            import yaml
            import session_replay as _SR
            _SR.CONTEXT_INFERENCE = dict(
                (yaml.safe_load(cfg.read_text()) or {}).get("inference") or {})
        except Exception as e:                                   # noqa: BLE001
            print(f"[dashboard] could not read inference config ({e}); "
                  f"replay will use whatever each cache recorded")
    else:
        print(f"[dashboard] no detector config at {args.config}: sessions and "
              f"cue logs will work, analysing and streaming a video will not")

    entry = None
    if not args.replay:
        import model_registry as MR
        entries = MR.scan()
        entry = (MR.find(entries, args.model) if args.model
                 else MR.default_entry(entries))
        if entry is None:
            raise SystemExit(f"unknown model {args.model!r}. Known: "
                             + ", ".join(e.variant_id for e in entries))
        if not entry.deployable:
            raise SystemExit(f"{entry.variant_id}: {entry.blocked_reason}")
        # A seed-pinned id (variant@sNN) is not one of the ranked rows, which
        # carry each variant's best validation seed. Show it anyway, first, or
        # the dropdown reports no active model while one is plainly running.
        if all(e.variant_id != entry.variant_id for e in entries):
            entries = [entry] + entries
        CONTEXT["entries"] = entries
        CONTEXT["entry"] = entry

    if args.replay:
        start({"kind": "cuelog", "id": f"cuelog:{args.replay}",
               "path": args.replay})
    elif args.session:
        start({"kind": "session", "id": f"session:{Path(args.session).resolve()}",
               "path": str(Path(args.session).resolve())}, entry)
    elif args.video:
        start({"kind": "video", "id": f"video:{args.video}",
               "path": args.video}, entry)
    else:
        print("[dashboard] no source yet — upload a recording from the page, "
              "or pick one already in tools/dashboard/{uploads,sessions}/")

    if CONTEXT["upload_enabled"] and args.host == "0.0.0.0":
        print("[dashboard] WARNING: bound to 0.0.0.0 with uploads enabled and "
              "no authentication. Use --host 127.0.0.1 or --no-upload on a "
              "shared machine.")

    # Threading, not the plain HTTPServer: a model switch stops the producer
    # thread and can take seconds, and on a single-threaded server that would
    # freeze the polling GETs too — the page would look crashed while working.
    srv = ThreadingHTTPServer((args.host, args.port), Handler)
    print(f"dashboard: http://{'localhost' if args.host == '0.0.0.0' else args.host}"
          f":{args.port}")
    srv.serve_forever()


if __name__ == "__main__":
    main()
