"""Head-pose backends for the attention pipeline.

WHY THIS MATTERS HERE: the two weakest cue classes are `turned_to_peer`
(F1 0.18) and `looking_away` (F1 0.29). Both are *orientation* judgements,
which CLIP embeddings + bbox geometry + colour statistics cannot represent.
The dense gold audit [internal notes, not included] also showed gaze carries the phone-use
signal when the phone itself is hidden behind a monitor. Head pose is the
single highest-value missing feature.

Backends, in order of preference:

``opencv``    Zero new dependencies. Haar frontal + left/right profile
              cascades give a coarse but genuinely informative YAW signal —
              which is precisely the axis that separates screen-oriented from
              turned-to-peer. Pitch is proxied from the face's vertical
              position inside the head crop, roll is not observable this way
              and is returned as 0.
``mediapipe`` PREFERRED and installed (mediapipe 1.0.0, Tasks API). Metric
              yaw/pitch/roll from the facial transformation matrix, plus the
              face_found dimension. Detection rate on LLMSTU crops: 60%
              overall, but see the face_found note below -- the misses are
              informative, not lost data.
``6drepnet``  Direct yaw/pitch/roll regression. Needs external weights.

All backends return ``np.ndarray([yaw, pitch, roll, face_found], float32)``.
Angles are normalised to roughly [-1, 1] (degrees/90) so the block matches the
scale of the other feature groups.

WHY face_found IS A DIMENSION, NOT A FAILURE FLAG (measured 2026-07-31, 40
crops per class): detection rate is itself the strongest single cue signal we
have --

    screen_oriented 92%   looking_away 90%   phone_use 62%
    turned_to_peer  52%   uncertain    40%   head_down     8%

head_down vs screen_oriented is 8% vs 92%. Treating a missed face as "zeros"
would throw that away and make it indistinguishable from "facing forward" --
the exact ambiguity that made the OpenCV backend useless. Exposing it as an
explicit dimension turns the failure mode into the most discriminative feature
in the block.

Contract: ``available()`` must be False unless the backend can actually run,
because StudentFeatureExtractor.output_dim() branches on it — a backend that
claims availability then fails would change the feature width mid-run.
"""
import os
from typing import Optional

from pathlib import Path

import numpy as np

try:
    import cv2
except ImportError:  # pragma: no cover
    cv2 = None

OUTPUT_DIM = 4  # yaw, pitch, roll, face_found


class OpenCVHeadPose:
    """Coarse yaw/pitch from Haar frontal + profile cascades.

    Yaw is derived from which cascade fires and where the face sits
    horizontally within the head crop:

      * frontal face found        -> yaw from horizontal face-centre offset
      * left-profile only         -> strong negative yaw
      * right-profile only        -> strong positive yaw (mirrored detection)
      * nothing found             -> yaw 0, confidence 0 (see note below)

    SUPERSEDED by the mediapipe backend (25% vs 60% detection, and coarse
    rather than metric angles). Kept as a zero-dependency fallback only.
    The 4th dimension (face_found) resolves the old ambiguity where a missed
    face and a forward-facing face both returned zeros.
    """

    def __init__(self):
        if cv2 is None:
            raise RuntimeError("opencv is required for the opencv head-pose backend")
        base = cv2.data.haarcascades
        self._frontal = cv2.CascadeClassifier(base + "haarcascade_frontalface_alt2.xml")
        self._profile = cv2.CascadeClassifier(base + "haarcascade_profileface.xml")
        if self._frontal.empty() or self._profile.empty():
            raise RuntimeError("Haar cascade XML failed to load")

    def _detect(self, gray: np.ndarray):
        f = self._frontal.detectMultiScale(gray, 1.15, 4, minSize=(24, 24))
        if len(f):
            return "frontal", max(f, key=lambda r: r[2] * r[3])
        p = self._profile.detectMultiScale(gray, 1.15, 4, minSize=(24, 24))
        if len(p):
            return "right", max(p, key=lambda r: r[2] * r[3])
        # The profile cascade is trained on one side only; mirror to find the other.
        pm = self._profile.detectMultiScale(cv2.flip(gray, 1), 1.15, 4, minSize=(24, 24))
        if len(pm):
            x, y, w, h = max(pm, key=lambda r: r[2] * r[3])
            return "left", (gray.shape[1] - x - w, y, w, h)
        return None, None

    def estimate(self, head_crop_bgr: np.ndarray) -> np.ndarray:
        if head_crop_bgr is None or head_crop_bgr.size == 0:
            return np.zeros(OUTPUT_DIM, dtype=np.float32)
        h, w = head_crop_bgr.shape[:2]
        if h < 8 or w < 8:
            return np.zeros(OUTPUT_DIM, dtype=np.float32)
        gray = cv2.cvtColor(head_crop_bgr, cv2.COLOR_BGR2GRAY)
        gray = cv2.equalizeHist(gray)
        kind, box = self._detect(gray)
        if kind is None:
            return np.zeros(OUTPUT_DIM, dtype=np.float32)

        x, y, fw, fh = box
        cx = (x + fw / 2.0) / max(1.0, w)   # 0..1 across the crop
        cy = (y + fh / 2.0) / max(1.0, h)
        if kind == "frontal":
            yaw = float(np.clip((cx - 0.5) * 2.0, -1.0, 1.0)) * 0.45
        elif kind == "left":
            yaw = -0.75 + float(np.clip((cx - 0.5) * 0.5, -0.25, 0.25))
        else:
            yaw = 0.75 + float(np.clip((cx - 0.5) * 0.5, -0.25, 0.25))
        # Face high in the crop => head up; low => head tilted down.
        pitch = float(np.clip((cy - 0.45) * 2.0, -1.0, 1.0))
        return np.array([yaw, pitch, 0.0, 1.0], dtype=np.float32)


class MediaPipeHeadPose:
    """MediaPipe FaceLandmarker (Tasks API) -> facial transformation matrix.

    mediapipe >= 1.0 removed the legacy ``mp.solutions`` API, so this uses
    ``mediapipe.tasks.python.vision.FaceLandmarker`` with
    ``output_facial_transformation_matrixes=True``. That yields head pose
    directly from the 4x4 transform, which is both simpler and better
    conditioned than landmark + solvePnP.

    Needs the model bundle (3.7 MB):
        curl -o face_landmarker.task https://storage.googleapis.com/\
mediapipe-models/face_landmarker/face_landmarker/float16/1/face_landmarker.task

    Input: pass the FULL student crop, not a top-fraction slice. Measured
    detection rate on LLMSTU crops -- top 35%: 38%, top 50%: 55%,
    full crop: 60%. FaceLandmarker runs its own face detector, so pre-cropping
    to a guessed head region only removes context it uses.
    """

    DEFAULT_MODEL = "../huggingface/mediapipe/face_landmarker.task"

    def __init__(self, model_path: Optional[str] = None):
        import mediapipe as mp
        from mediapipe.tasks.python import vision, BaseOptions
        self._mp = mp
        path = model_path or self.DEFAULT_MODEL
        if not os.path.exists(path):
            raise RuntimeError(
                f"FaceLandmarker bundle not found at {path}; download it (see "
                f"class docstring)")
        self._lm = vision.FaceLandmarker.create_from_options(
            vision.FaceLandmarkerOptions(
                base_options=BaseOptions(model_asset_path=path),
                running_mode=vision.RunningMode.IMAGE,
                num_faces=1,
                output_facial_transformation_matrixes=True))

    def estimate(self, crop_bgr: np.ndarray) -> np.ndarray:
        if crop_bgr is None or crop_bgr.size == 0:
            return np.zeros(OUTPUT_DIM, dtype=np.float32)
        h, w = crop_bgr.shape[:2]
        if h < 16 or w < 16:
            return np.zeros(OUTPUT_DIM, dtype=np.float32)
        img = self._mp.Image(image_format=self._mp.ImageFormat.SRGB,
                             data=cv2.cvtColor(crop_bgr, cv2.COLOR_BGR2RGB))
        res = self._lm.detect(img)
        if not res.facial_transformation_matrixes:
            # face_found = 0. NOT the same as "facing forward" -- see module docstring.
            return np.zeros(OUTPUT_DIM, dtype=np.float32)
        M = np.asarray(res.facial_transformation_matrixes[0])[:3, :3]
        sy = float(np.sqrt(M[0, 0] ** 2 + M[1, 0] ** 2))
        pitch = np.degrees(np.arctan2(-M[2, 0], sy))
        yaw = np.degrees(np.arctan2(M[1, 0], M[0, 0]))
        roll = np.degrees(np.arctan2(M[2, 1], M[2, 2]))
        v = np.array([yaw, pitch, roll], dtype=np.float32) / 90.0
        return np.concatenate([np.clip(v, -1, 1), [1.0]]).astype(np.float32)


class CachedHeadPose:
    """Look up precomputed pose by crop file_name. No model, no per-crop cost.

    Running FaceLandmarker inline costs ~140 ms/crop; over 283,913 dense crops
    that is ~11 h and dominates a sequence rebuild. `precompute_head_pose.py`
    computes the same vectors across all cores in ~3 min and caches them, so
    the build reverts to being CLIP-bound.

    Values are IDENTICAL to the mediapipe backend — same model, same matrix
    decomposition — so a cached build and an inline build are interchangeable.
    """

    def __init__(self, cache_path: str):
        if not os.path.exists(cache_path):
            raise RuntimeError(f"head-pose cache not found: {cache_path}")
        d = np.load(cache_path, allow_pickle=False)
        self._map = {str(n): i for i, n in enumerate(d["names"])}
        self._vecs = d["vecs"].astype(np.float32)
        self.n_missing = 0

    def estimate_by_name(self, file_name: str) -> np.ndarray:
        i = self._map.get(file_name)
        if i is None:
            self.n_missing += 1
            return np.zeros(OUTPUT_DIM, dtype=np.float32)
        return self._vecs[i]

    def estimate(self, crop_bgr: np.ndarray) -> np.ndarray:
        raise RuntimeError(
            "CachedHeadPose is keyed by file_name; call estimate_by_name(). "
            "Callers with only pixels must use the mediapipe backend.")


class MediaPipeFaceDetected:
    """``face_found`` only, from BlazeFace — no pose.

    Rationale [internal notes, not included]: ~80% of the head-pose block's downstream
    value is the binary flag, and the metric angles add nothing measurable on
    top of it. A full FaceLandmarker mesh is an expensive way to obtain one bit:
    measured over 1,200 frames, 74.9 ms/frame against this backend's 47.2 ms.

    Returns ``[0, 0, 0, found]``, so a model built on the ``553_facefound``
    feature config — base + column 555 only — sees exactly the same semantics
    as it would from the landmarker. Do NOT pair this with a 556-dim
    checkpoint: the three zeroed angle columns would be silently wrong, which is
    the failure mode `attention/thesis_eval/runtime.py` exists to make
    impossible.
    """

    DEFAULT_MODEL = "../huggingface/mediapipe/face_detection_full_range.tflite"

    def __init__(self, model_path: str = None, min_confidence: float = 0.5):
        model_path = model_path or self.DEFAULT_MODEL
        if not Path(model_path).exists():
            raise RuntimeError(
                f"face detector model not found at {model_path}; fetch it with\n"
                "  curl -o face_detection_full_range.tflite https://storage."
                "googleapis.com/mediapipe-assets/face_detection_full_range.tflite")
        from mediapipe.tasks.python import vision, BaseOptions
        self._det = vision.FaceDetector.create_from_options(
            vision.FaceDetectorOptions(
                base_options=BaseOptions(model_asset_path=str(model_path)),
                running_mode=vision.RunningMode.IMAGE,
                min_detection_confidence=min_confidence))

    def estimate(self, crop_bgr: np.ndarray) -> np.ndarray:
        import cv2
        import mediapipe as mp
        if crop_bgr is None or crop_bgr.size == 0 or min(crop_bgr.shape[:2]) < 16:
            return np.zeros(OUTPUT_DIM, dtype=np.float32)
        res = self._det.detect(mp.Image(
            image_format=mp.ImageFormat.SRGB,
            data=cv2.cvtColor(crop_bgr, cv2.COLOR_BGR2RGB)))
        out = np.zeros(OUTPUT_DIM, dtype=np.float32)
        out[3] = 1.0 if res.detections else 0.0
        return out


_BACKENDS = {"opencv": OpenCVHeadPose, "mediapipe": MediaPipeHeadPose,
             "mediapipe_detector": MediaPipeFaceDetected}


class HeadPoseEstimator:
    """Drop-in replacement for the stub in features.py.

    ``backend=None`` keeps the previous behaviour (unavailable, feature block
    omitted) so existing 552-dim checkpoints stay loadable. Pass
    ``backend="opencv"`` to enable the 3 extra dims -> 555-dim features, which
    requires rebuilding sequences and retraining the temporal model.
    """

    OUTPUT_DIM = OUTPUT_DIM

    def __init__(self, backend: Optional[str] = None, device: str = "cuda:0",
                 cache_path: Optional[str] = None):
        self.backend = backend
        self.device = device
        self._model = None
        if backend == "cached":
            if not cache_path:
                raise ValueError("backend='cached' requires cache_path")
            self._model = CachedHeadPose(cache_path)
            return
        if backend is None:
            return
        if backend not in _BACKENDS:
            raise ValueError(f"unknown head-pose backend {backend!r}; "
                             f"choose from {sorted(_BACKENDS)}")
        try:
            self._model = _BACKENDS[backend]()
        except Exception as e:
            # Fail loudly: a silently-disabled backend would change the feature
            # width and train a model that cannot see orientation at all.
            raise RuntimeError(
                f"head-pose backend {backend!r} could not be initialised: {e}"
            ) from e

    def available(self) -> bool:
        return self._model is not None

    def estimate(self, head_crop_bgr: np.ndarray) -> np.ndarray:
        if not self.available():
            raise RuntimeError("No head-pose backend configured.")
        return self._model.estimate(head_crop_bgr)

    def estimate_by_name(self, file_name: str) -> np.ndarray:
        """Cache-backed lookup. Only valid for backend='cached'."""
        if not self.available():
            raise RuntimeError("No head-pose backend configured.")
        if not hasattr(self._model, "estimate_by_name"):
            raise RuntimeError(f"backend {self.backend!r} has no cache lookup")
        return self._model.estimate_by_name(file_name)
