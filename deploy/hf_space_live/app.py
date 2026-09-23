#!/usr/bin/env python
"""Hugging Face Space entrypoint.

Fetches the artifacts the dashboard needs from a private model repo, then starts
`tools/dashboard/server.py` on the port Spaces routes to.

Why weights are fetched at runtime rather than baked into the image:

* `artifacts.lock.json` marks the detector and the LLMSTU corpus **restricted**,
  gated on an unresolved ethics/consent question. Baking them into image layers
  would put them in every copy of the Space and in its build cache, where they
  cannot be revoked. Fetching under a token keeps the access decision in one
  place — the model repo's own permissions.
* Temporal checkpoints are ~50 MB each and there are 13 deployable ones. Layers
  that large make every rebuild slow for no benefit, since they change far less
  often than the code.

Environment (set these as Space secrets/variables, not in the Dockerfile):

    HF_TOKEN         read token for the private artifact repo   (secret)
    ARTIFACT_REPO    e.g. "CHANGE_ME/your-dashboard-artifacts"    (variable)
    ARTIFACT_TYPE    "model" (default) or "dataset"             (variable)
    DASHBOARD_MODEL  registry id, default ff_det/mstcn_553_facefound@s42
                     -- the checkpoint attention_runtime.yaml deploys  (variable)
    SESSION          session dir name under sessions/, default 0325_full
    RUNTIME_CONFIG   detector+features config for live analysis  (variable)
    DETECTOR_CONFIG  mmdet config to use instead of the one named
                     inside RUNTIME_CONFIG, e.g.
                     configs/student_llmstu_exact_deploy.py       (variable)
    DEVICE           cuda:0 (default) or cpu                     (variable)
    PERSIST_DIR      durable mount path, if not /data            (variable)
    DASHBOARD_ARCHIVE_DIR  where analysed uploads are saved so they survive
                     a restart; default <durable mount>/saved_recordings
"""

from __future__ import annotations

import os
import sys
import threading
from pathlib import Path

HOME = Path(__file__).resolve().parent
PORT = int(os.environ.get("PORT", "7860"))


def _cache_root() -> Path:
    """Prefer Spaces persistent storage, fall back to the ephemeral image.

    With a 1-hour sleep timer the Space cold-starts often, and the artifacts are
    ~5 GB (the detector alone is 4.29 GB). On ephemeral disk that is re-fetched
    on every wake, which dominates time-to-first-frame. /data survives sleeps, so
    the download happens once.

    PERSIST_DIR overrides the location, because Spaces offers more than one kind
    of durable storage and they do not all mount at the same path — a mounted
    bucket in particular may appear somewhere else entirely. Set PERSIST_DIR to
    whatever the Space actually mounts and this follows it.

    Each candidate is probed by _writable(), which both writes and gives up.
    """
    candidates = []
    override = os.environ.get("PERSIST_DIR", "").strip()
    if override:
        candidates.append(Path(override))
    candidates += [Path("/data"), Path("/mnt/data")]

    for cand in candidates:
        if _writable(cand):
            return cand
        print(f"[app] {cand} not usable for the cache — skipping", flush=True)
    return HOME


def _writable(cand: Path, timeout_s: float = 20.0) -> bool:
    """Can we actually write here — answered within a bounded time?

    The probe writes a file rather than calling exists(), because /data is
    present but read-only when persistent storage is off, and an exists() check
    would route the cache somewhere that fails on first write.

    The bound matters just as much. A mounted bucket can wedge: the FUSE call
    blocks and never returns, and because this runs before the first print, the
    Space dies silently after its startup banner with no clue why. That happened.
    A daemon thread lets a wedged mount be abandoned instead of waited on, so the
    app falls back to ephemeral disk and boots.
    """
    ok = []

    def probe():
        try:
            cand.mkdir(parents=True, exist_ok=True)
            p = cand / ".write_test"
            p.write_text("ok")
            p.unlink()
            ok.append(True)
        except Exception:
            pass

    t = threading.Thread(target=probe, daemon=True)
    t.start()
    t.join(timeout_s)
    return bool(ok)


CACHE_ROOT = _cache_root()
# Set before huggingface_hub is imported anywhere, or it reads the default.
os.environ.setdefault("HF_HOME", str(CACHE_ROOT / ".cache" / "huggingface"))
print(f"[app] cache root: {CACHE_ROOT} "
      f"({'persistent' if CACHE_ROOT != HOME else 'EPHEMERAL — artifacts re-download on every wake'})",
      flush=True)

# Where server.py expects things, relative to the repo root it is run from.
SESSIONS_DIR = HOME / "tools" / "dashboard" / "sessions"
WORKDIRS = HOME / "LLMDet" / "work_dirs"
LLMDET_ROOT = HOME / "LLMDet"
#: Where '../huggingface/...' in the detector config actually lands.
#: It is resolved while cwd is LLMDet/, not HOME — precompute_session.py:110
#: chdirs there so the config's own "configs/..." and "work_dirs/..." paths
#: work. Derived from that rule rather than written out, because writing out
#: the guess (HOME.parent) put 2.1 GB of models one directory away from where
#: the detector looked, and the run still died on the GPU with the startup
#: check reporting everything present.
HF_MODELS = (LLMDET_ROOT / ".." / "huggingface").resolve()


def fetch_artifacts() -> bool:
    """Pull checkpoints + session cache from the private repo. True if usable."""
    repo = os.environ.get("ARTIFACT_REPO", "").strip()
    if not repo:
        print("[app] ARTIFACT_REPO not set — starting in replay-only mode "
              "(cue logs render; no model switching).", flush=True)
        return False

    token = os.environ.get("HF_TOKEN") or os.environ.get("HUGGING_FACE_HUB_TOKEN")
    if not token:
        print("[app] ARTIFACT_REPO is set but HF_TOKEN is not. A private repo "
              "cannot be read without it; add it as a Space secret.", flush=True)
        return False

    from huggingface_hub import snapshot_download

    def pull(allow, dest, label):
        """Fetch one slice of the artifact repo, surviving a poisoned sidecar.

        With local_dir=, huggingface_hub keeps bookkeeping in
        <dest>/.cache/huggingface/download/*.metadata. On a bucket mount an
        interrupted fetch leaves those half-written, and the next run does not
        recover: read_download_metadata() raises

            UnboundLocalError: cannot access local variable 'metadata'

        because its parse failure path leaves the name unbound. The sidecars
        are pure bookkeeping — the payload is content-addressed on the hub — so
        deleting them and retrying costs one re-download and fixes it, whereas
        leaving them wedges the Space on every boot.
        """
        def go():
            return Path(snapshot_download(
                repo_id=repo, repo_type=os.environ.get("ARTIFACT_TYPE", "model"),
                token=token, allow_patterns=allow, local_dir=str(dest)))

        print(f"[app] fetching {label} -> {dest}", flush=True)
        try:
            return go()
        except Exception as e:
            sidecar = dest / ".cache" / "huggingface"
            print(f"[app] {label}: {type(e).__name__}: {str(e)[:160]}", flush=True)
            print(f"[app] clearing {sidecar} and retrying once", flush=True)
            import shutil
            shutil.rmtree(sidecar, ignore_errors=True)
            return go()

    # Two destinations, because the artifacts split cleanly by shape and the
    # durable mount may be a BUCKET, which is good at few large objects and bad
    # at many small ones:
    #
    #   LLMDet/work_dirs   273 files, 4.94 GB, ~18 MB each  -> durable
    #   sessions/*/frames  900 files, 0.07 GB, ~78 KB each  -> ephemeral
    #
    # (measured value removed from the handover copy)
    # a mounted bucket writes 900 cache-metadata sidecars, which is what made a
    # boot stall at 25% with repeated "[Errno 5] Input/output error". They cost
    # seconds to re-fetch onto local disk, so persisting them buys nothing and
    # costs a hang. The 4.94 GB of weights is the part worth keeping across a
    # sleep, and it is exactly the shape a bucket handles well.
    weights = pull(["LLMDet/work_dirs/**"], CACHE_ROOT / "artifacts", "weights (durable)")
    session = pull(["tools/dashboard/sessions/**"], HOME / "_artifacts_session",
                   "session cache (ephemeral)")
    # The detector config addresses its text encoder and LMM by RELATIVE path
    # (grounding_dino_swin_t.py: lang_model_name = '../huggingface/bert-base-uncased/',
    # lmm = '../huggingface/my_llava-onevision-qwen2-0.5b-ov-2/'), resolved from
    # inside LLMDet/ — see HF_MODELS. ~2.1 GB in a handful of large files, so
    # this goes to durable storage with the weights.
    models = pull(["huggingface/**"], CACHE_ROOT / "hf_models",
                  "detector models (durable)")

    for src, dest, what in ((session / "tools" / "dashboard" / "sessions", SESSIONS_DIR,
                             "session cache"),
                            (weights / "LLMDet" / "work_dirs", WORKDIRS, "work_dirs"),
                            (models / "huggingface", HF_MODELS, "detector models")):
        if not src.exists():
            print(f"[app] note: {what} not present in the artifact repo", flush=True)
            continue
        dest.parent.mkdir(parents=True, exist_ok=True)
        if dest.exists() or dest.is_symlink():
            continue
        try:
            dest.symlink_to(src, target_is_directory=True)
        except OSError:
            import shutil
            shutil.copytree(src, dest)
        print(f"[app] {what} -> {dest}", flush=True)
    return True


#: Files the detector config names directly. Checked at startup because a
#: missing one otherwise surfaces ~40 s into an analysis job as an opaque
#: transformers OSError ("Incorrect path_or_model_id"), on the GPU, in a
#: background thread — about the least useful place to learn it.
DETECTOR_ASSETS = (
    "bert-base-uncased/config.json",                    # lang_model_name
    "my_llava-onevision-qwen2-0.5b-ov-2/config.json",   # lmm=
    # The LMM's own config.json carries mm_vision_tower =
    # '../huggingface/siglip-so400m-patch14-384', so the model directory we
    # were handed pulls in a third one. Public (google/...), unlike the LMM.
    "siglip-so400m-patch14-384/config.json",
    "mediapipe/face_detection_full_range.tflite",       # head_pose_backend
)


def check_detector_assets() -> None:
    missing = [a for a in DETECTOR_ASSETS if not (HF_MODELS / a).exists()]
    if missing:
        print(f"[app] WARNING: {len(missing)} detector asset(s) missing under "
              f"{HF_MODELS} — replay works, analysing video will not:", flush=True)
        for a in missing:
            print(f"[app]   missing: {a}", flush=True)
    else:
        print(f"[app] detector assets present under {HF_MODELS}", flush=True)


def apply_detector_override(config: str) -> str:
    """Swap the detector config, without editing the one the thesis cites.

    DETECTOR_CONFIG names an mmdet config (relative to LLMDet/) to use instead
    of the one in the runtime YAML. It exists so the Space can run
    student_llmstu_exact_deploy.py — the same detector with `lmm=None`, whose
    predictions are identical because predict() never reads the LMM — while
    LLMDet/configs/attention_runtime.yaml stays byte-identical to the recorded
    deployment configuration.

    The rewritten YAML is a derived runtime artifact written next to the app;
    the canonical file on disk is never touched.
    """
    override = os.environ.get("DETECTOR_CONFIG", "").strip()
    # Knobs a Space variable can turn without a code push or a rebuild (a
    # variable change only restarts the Space). Each is validated; a bad value
    # is reported and ignored rather than crashing the boot.
    knobs = {name: os.environ[name].strip() for name in KNOBS
             if os.environ.get(name, "").strip()}
    if not override and not knobs:
        return config

    import yaml
    src = HOME / config
    try:
        cfg = yaml.safe_load(src.read_text(encoding="utf-8"))
        if override:
            was = cfg["detector"]["config_path"]
            cfg["detector"]["config_path"] = override
            print(f"[app] detector config override: {was} -> {override}", flush=True)
    except Exception as e:
        print(f"[app] could not rewrite {src} ({type(e).__name__}: {e}); "
              f"using it unchanged", flush=True)
        return config

    for name, raw in knobs.items():
        section, key, cast = KNOBS[name]
        try:
            value = cast(raw)
        except ValueError:
            print(f"[app] {name}={raw!r} is not a valid {cast.__name__}; ignored",
                  flush=True)
            continue
        was = cfg.setdefault(section, {}).get(key)
        cfg[section][key] = value
        print(f"[app] {name}: {section}.{key} {was} -> {value}", flush=True)

    out = HOME / "_runtime_override.yaml"
    out.write_text(yaml.safe_dump(cfg, sort_keys=False), encoding="utf-8")
    print(f"[app] (derived from {config}; that file is unmodified)", flush=True)
    return str(out.relative_to(HOME))


#: Space variable -> (runtime YAML section, key, type). Detector filters apply
#: to live camera and NEW analyses; a session already analysed keeps the boxes
#: it was built with. MIN_FRAMES_FOR_PRED applies to replay as well.
KNOBS = {
    "DETECTOR_SCORE_THR": ("detector", "score_thr", float),
    "DETECTOR_MIN_REL_AREA": ("detector", "min_rel_area", float),
    "DETECTOR_MAX_ASPECT": ("detector", "max_aspect_ratio", float),
    "DETECTOR_NMS_IOU": ("detector", "nms_iou_thr", float),
    "TRACK_MIN_HITS": ("tracking", "min_hits", int),
    "MIN_FRAMES_FOR_PRED": ("inference", "min_frames_for_pred", int),
}


def main() -> int:
    have_artifacts = fetch_artifacts()
    if have_artifacts:
        check_detector_assets()

    # 0325_full (64.4 s), not 0325 (30.0 s). The temporal window is 32
    # frames and frames now enter the history at 1 fps [internal notes, not included], so a
    # 30 s clip yields 30 samples and never fills a single window. 64 s
    # fills two.
    session = os.environ.get("SESSION", "0325_full")
    session_dir = SESSIONS_DIR / session
    # The checkpoint attention_runtime.yaml names as deployed, seed-pinned.
    #
    # This used to default to arch/mstcn_556_hp, inherited from the phase-1
    # CPU-only Space, where that choice is argued for explicitly
    # (README_DEPLOY.md: best test macro-F1 among deployable variants AND ~4.6x
    # faster to replay on CPU). Neither half of that argument applies here: this
    # Space has a T4, and the thesis cites ff_det/mstcn_553_ff_s42 as the
    # deployed system. With the old default, a Space whose DASHBOARD_MODEL
    # variable was unset served a *different* model than the one written up --
    # trained on the FaceLandmarker mesh rather than the BlazeFace detector, so
    # live analysis also gave up the +30% FPS that backend change bought
    # [internal notes, not included] -- while the page still presented itself as the
    # deployed system.
    #
    # Pinned to @s42 deliberately: the unpinned variant id resolves to the best
    # VALIDATION seed, which is s43, a different checkpoint with its own fitted
    # calibration. See model_registry.find().
    model = os.environ.get("DASHBOARD_MODEL",
                           "ff_det/mstcn_553_facefound@s42")

    # Live analysis needs the detector config and a device. Without --config the
    # Analyse button can only replay; with it, an uploaded video runs the full
    # detector -> tracker -> features -> temporal chain.
    config = os.environ.get("RUNTIME_CONFIG", "LLMDet/configs/attention_runtime.yaml")
    config = apply_detector_override(config)
    device = os.environ.get("DEVICE", "cuda:0")

    try:
        import torch
        if device.startswith("cuda") and not torch.cuda.is_available():
            print(f"[app] {device} requested but torch reports no CUDA device — "
                  f"falling back to cpu. Analysis will run at roughly 1 fps.",
                  flush=True)
            device = "cpu"
        elif device.startswith("cuda"):
            print(f"[app] GPU: {torch.cuda.get_device_name(0)}", flush=True)
    except Exception as e:
        print(f"[app] could not query torch for CUDA ({e}); using {device}", flush=True)

    argv = [sys.argv[0], "--host", "0.0.0.0", "--port", str(PORT),
            "--config", config, "--device", device]
    if have_artifacts and session_dir.exists():
        argv += ["--session", str(session_dir), "--model", model]
        print(f"[app] session {session_dir} | model {model}", flush=True)
    else:
        # Falls back to a tracked cue log so the Space always renders something
        # rather than failing to boot. Model switching is unavailable here: a
        # recorded cue log stores decisions, not features.
        #
        # Name what was asked for and what is actually present. Landing in replay
        # mode is visible; landing here because SESSION names a directory the
        # artifact repo does not have is not, and the two look identical on the
        # page -- no model selector, no explanation.
        have = sorted(d.name for d in SESSIONS_DIR.glob("*") if d.is_dir()) \
            if SESSIONS_DIR.exists() else []
        print(f"[app] SESSION={session!r} unavailable (artifacts fetched: "
              f"{have_artifacts}; sessions present: {have or 'none'})", flush=True)
        replay = HOME / "tools" / "dashboard" / "demo_session.jsonl"
        argv += ["--replay", str(replay)]
        print(f"[app] replay mode: {replay}", flush=True)

    # Uploads and the sessions built from them are written next to the app, on
    # ephemeral disk, so a restart or a sleep used to lose every recording
    # analysed since boot. Keep a copy on the durable mount instead: one tar per
    # session, because the mount is the bucket that stalled on 900 small JPEGs
    # (see fetch_artifacts). Nowhere to keep it means nothing is kept, loudly.
    if CACHE_ROOT != HOME:
        os.environ.setdefault("DASHBOARD_ARCHIVE_DIR",
                              str(CACHE_ROOT / "saved_recordings"))
    else:
        print("[app] no durable storage: uploaded recordings will not survive "
              "a restart", flush=True)

    sys.path.insert(0, str(HOME / "tools" / "dashboard"))
    sys.argv = argv
    os.chdir(HOME)

    import runpy
    runpy.run_path(str(HOME / "tools" / "dashboard" / "server.py"),
                   run_name="__main__")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
