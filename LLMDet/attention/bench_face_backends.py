"""Benchmark cheaper sources of the ``face_found`` bit.

Motivation. ~80% of the head-pose block's contribution is the binary "was a face
detected" flag, not the metric angles [internal notes, not included] — yet the pipeline pays
~100 ms per frame at ~6 students to run a full FaceLandmarker mesh, more than
the detector itself [internal notes, not included]. If a plain face *detector* preserves the
flag, most of that time is recoverable.

"Preserves the flag" is not the same as "agrees with FaceLandmarker". What the
model actually consumes is the flag's ability to separate cues:
``face_found`` fires on 67% of ``screen_oriented`` crops and 14% of
``head_down`` ones, and that **spread** is the signal [internal notes, not included] measured
71.0 percentage points for the current cache; a variant with a smaller spread
trained to a worse model even though it looked reasonable). So the decisive
column below is the spread, not the agreement.

Three strategies:

``landmarker_crop``  FaceLandmarker per student crop            (current)
``detector_crop``    BlazeFace FaceDetector per student crop
``detector_frame``   BlazeFace once on the whole frame, faces assigned to
                     students by centre containment — one call per frame
                     instead of N, the only variant that changes the scaling
                     behaviour rather than the constant

    python -m attention.bench_face_backends --n 2000
"""

import argparse
import json
import time
from collections import defaultdict
from pathlib import Path

import numpy as np

MODELS = Path("../huggingface/mediapipe")


def _landmarker():
    from mediapipe.tasks.python import vision, BaseOptions
    return vision.FaceLandmarker.create_from_options(
        vision.FaceLandmarkerOptions(
            base_options=BaseOptions(model_asset_path=str(MODELS / "face_landmarker.task")),
            running_mode=vision.RunningMode.IMAGE, num_faces=1,
            output_facial_transformation_matrixes=True))


def _detector(model: str, conf: float, max_faces: int = 50):
    from mediapipe.tasks.python import vision, BaseOptions
    return vision.FaceDetector.create_from_options(
        vision.FaceDetectorOptions(
            base_options=BaseOptions(model_asset_path=str(MODELS / model)),
            running_mode=vision.RunningMode.IMAGE,
            min_detection_confidence=conf))


def _mp_image(bgr):
    import cv2
    import mediapipe as mp
    return mp.Image(image_format=mp.ImageFormat.SRGB,
                    data=cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB))


def main():
    import cv2
    from attention.taxonomy import CUE_CLASSES, map_record

    ap = argparse.ArgumentParser()
    ap.add_argument("--labels",
                    default="../grounding_data/llmstu_tools/outputs/labels_tracked.jsonl")
    ap.add_argument("--frames-root", default="../grounding_data/stu_img/frames")
    ap.add_argument("--n", type=int, default=2000, help="records to sample")
    ap.add_argument("--conf", type=float, default=0.5)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--out", default="work_dirs/thesis/face_backend_bench.json")
    args = ap.parse_args()

    recs = [json.loads(l) for l in open(args.labels)]
    rng = np.random.default_rng(args.seed)
    # sample whole FRAMES, so detector_frame pays a realistic per-frame cost and
    # every student of a sampled frame is scored
    by_frame = defaultdict(list)
    for r in recs:
        by_frame[r["src_frame"]].append(r)
    frames = sorted(by_frame)
    pick = rng.choice(len(frames), size=min(args.n, len(frames)), replace=False)
    frames = [frames[i] for i in pick]
    sample = [(f, by_frame[f]) for f in frames]
    n_students = sum(len(v) for _, v in sample)
    print(f"{len(sample)} frames, {n_students} students "
          f"({n_students / len(sample):.2f}/frame)")

    cues = {}
    for _, rs in sample:
        for r in rs:
            cues[r["file_name"]] = map_record(r)

    results = {}
    lm = _landmarker()
    det_s = _detector("blaze_face_short_range.tflite", args.conf)
    det_f = _detector("face_detection_full_range.tflite", args.conf)

    def run(name, fn):
        found, t0 = {}, time.perf_counter()
        for src, rs in sample:
            im = cv2.imread(str(Path(args.frames_root) / src))
            if im is None:
                continue
            fn(im, rs, found)
        dt = time.perf_counter() - t0
        results[name] = {"ms_per_frame": dt / len(sample) * 1000,
                         "ms_per_student": dt / max(n_students, 1) * 1000,
                         "found": found}
        print(f"  {name:18s} {results[name]['ms_per_frame']:7.1f} ms/frame "
              f"{results[name]['ms_per_student']:6.2f} ms/student", flush=True)

    def crop_of(im, r):
        h, w = im.shape[:2]
        x1, y1, x2, y2 = [int(v) for v in r["bbox_person"]]
        x1 = max(0, min(x1, w - 1)); x2 = max(0, min(x2, w))
        y1 = max(0, min(y1, h - 1)); y2 = max(0, min(y2, h))
        return im[y1:y2, x1:x2] if (x2 > x1 and y2 > y1) else None

    def lm_crop(im, rs, found):
        for r in rs:
            c = crop_of(im, r)
            ok = 0.0
            if c is not None and c.size and min(c.shape[:2]) >= 16:
                ok = 1.0 if lm.detect(_mp_image(c)).facial_transformation_matrixes else 0.0
            found[r["file_name"]] = ok

    def det_crop(det):
        def inner(im, rs, found):
            for r in rs:
                c = crop_of(im, r)
                ok = 0.0
                if c is not None and c.size and min(c.shape[:2]) >= 16:
                    ok = 1.0 if det.detect(_mp_image(c)).detections else 0.0
                found[r["file_name"]] = ok
        return inner

    def det_frame(det):
        def inner(im, rs, found):
            res = det.detect(_mp_image(im))
            centres = []
            for d in res.detections:
                b = d.bounding_box
                centres.append((b.origin_x + b.width / 2.0, b.origin_y + b.height / 2.0))
            for r in rs:
                x1, y1, x2, y2 = r["bbox_person"]
                found[r["file_name"]] = float(any(x1 <= cx <= x2 and y1 <= cy <= y2
                                                  for cx, cy in centres))
        return inner

    print("timing:")
    run("landmarker_crop", lm_crop)
    run("detector_crop_short", det_crop(det_s))
    run("detector_crop_full", det_crop(det_f))
    run("detector_frame_full", det_frame(det_f))

    def mutual_information(f, c, n_cls):
        """I(face_found ; cue) in bits — how much the flag tells you about the cue.

        max-min spread is a poor summary: a backend can match it while collapsing
        the one contrast the feature exists for. detector_crop_full scores the
        same 71.0% spread as the landmarker yet separates head_down from
        screen_oriented by 22 points instead of 49.
        """
        n = len(f)
        H = lambda p: float(-sum(q * np.log2(q) for q in p if q > 0))
        pf = np.array([(f == 0).mean(), (f == 1).mean()])
        h = H(pf)
        for i in range(n_cls):
            m = c == i
            if not m.any():
                continue
            pi = m.mean()
            pfi = np.array([(f[m] == 0).mean(), (f[m] == 1).mean()])
            h -= pi * H(pfi)
        return h

    base = results["landmarker_crop"]["found"]
    keys = sorted(base)
    print(f"\n{'backend':22s} {'overall':>8} {'agree':>7} {'spread':>8} {'MI bits':>8} "
          f"{'screen-headdown':>16}   per-cue rate")
    summary = {}
    for name, res in results.items():
        f = np.array([res["found"].get(k, 0.0) for k in keys])
        b = np.array([base[k] for k in keys])
        c = np.array([cues[k] for k in keys])
        rates = {CUE_CLASSES[i]: float(f[c == i].mean()) if (c == i).any() else float("nan")
                 for i in range(len(CUE_CLASSES))}
        vals = [v for v in rates.values() if np.isfinite(v)]
        spread = max(vals) - min(vals)
        mi = mutual_information(f, c, len(CUE_CLASSES))
        contrast = rates["screen_oriented"] - rates["head_down"]
        summary[name] = {"overall": float(f.mean()),
                         "agreement_with_landmarker": float((f == b).mean()),
                         "spread": spread, "mutual_information_bits": mi,
                         "screen_minus_headdown": contrast, "per_cue": rates,
                         "ms_per_frame": res["ms_per_frame"],
                         "ms_per_student": res["ms_per_student"]}
        print(f"{name:22s} {f.mean():8.1%} {(f == b).mean():7.1%} {spread:8.1%} {mi:8.4f} "
              f"{contrast:15.1%}   " + "  ".join(f"{k[:9]} {v:.0%}" for k, v in rates.items()))

    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    Path(args.out).write_text(json.dumps(
        {"n_frames": len(sample), "n_students": n_students,
         "min_detection_confidence": args.conf, "backends": summary}, indent=2))
    print(f"\nwritten: {args.out}")


if __name__ == "__main__":
    main()
