from typing import List, Optional

import cv2
import numpy as np
import torch
from PIL import Image

try:
    from transformers import CLIPModel, CLIPProcessor
except ImportError:
    CLIPModel = None
    CLIPProcessor = None


# Real implementation lives in attention/head_pose.py. Re-exported here so
# existing imports (`from attention.features import HeadPoseEstimator`) keep
# working and there is exactly ONE implementation rather than a stub plus a
# real class that can drift apart.
#
# backend=None  -> unavailable, feature block omitted, 552-dim (unchanged;
#                  existing checkpoints stay loadable)
# backend="opencv"    -> +4 dims = 556-dim; coarse, 25% face-detection rate
# backend="mediapipe" -> +4 dims = 556-dim, metric angles, 60% detection rate
#                  (PREFERRED). The 4th dim is face_found, which is itself the
#                  strongest single cue signal: head_down detects at 8% vs
#                  screen_oriented at 92%. Enabling either requires rebuilding
#                  sequences and retraining the temporal model.
from .head_pose import HeadPoseEstimator  # noqa: F401  (re-export)


def crop_boxes(frame_bgr: np.ndarray, bboxes_xyxy: List[List[float]]):
    """Clip boxes to the frame and cut the crops the feature blocks see.

    Factored out of ``extract_batch`` so that anything needing *the same* crops
    — the dashboard's session precompute, which runs two head-pose backends over
    one CLIP pass — cuts them identically instead of reimplementing the
    rounding and clipping and drifting away from it. A head-pose block computed
    on a crop one pixel different from the one the model was trained on is the
    kind of mismatch that shows up as an unexplained accuracy drop.

    Returns ``(crops, clipped_xyxy, valid_idx)``; degenerate boxes are dropped,
    so ``valid_idx`` maps each crop back to its position in ``bboxes_xyxy``.
    """
    h, w = frame_bgr.shape[:2]
    crops, clipped, valid_idx = [], [], []
    for i, bbox in enumerate(bboxes_xyxy):
        x1, y1, x2, y2 = [int(v) for v in bbox]
        x1 = max(0, min(x1, w - 1))
        x2 = max(0, min(x2, w - 1))
        y1 = max(0, min(y1, h - 1))
        y2 = max(0, min(y2, h - 1))
        if x2 <= x1 or y2 <= y1:
            continue
        crops.append(frame_bgr[y1:y2, x1:x2])
        clipped.append((x1, y1, x2, y2))
        valid_idx.append(i)
    return crops, clipped, valid_idx


class StudentFeatureExtractor:
    """Per-student feature vector: CLIP embedding + bbox geometry + color
    statistics + posture-geometry proxies.

    CLIP failures are fatal by default: silently zeroing 512 of the feature
    dims previously let a broken environment train a garbage model. Pass
    ``allow_clip_fallback=True`` to opt into zeros explicitly.
    """

    def __init__(
        self,
        clip_model_name: str = "openai/clip-vit-base-patch32",
        head_stream: bool = False,
        device: str = "cuda:0",
        allow_clip_fallback: bool = False,
        head_pose: Optional[HeadPoseEstimator] = None,
    ):
        self.device = device if torch.cuda.is_available() else "cpu"
        #: Width of the visual embedding. NOT a constant: `siglip2-so400m
        #: -patch14-384` is 1152 against CLIP ViT-B/32's 512, and the named
        #: LAYOUTS are built from this, so it is read from the loaded model
        #: rather than assumed. A wrong value here would place every block after
        #: the embedding at the wrong column -- the exact failure the layouts
        #: exist to make impossible.
        self.clip_dim = 512
        #: True when the encoder is a SigLIP tower, which has no
        #: `get_image_features` and is pooled from `pooler_output` instead.
        self._siglip = False
        self.clip_model: Optional["CLIPModel"] = None
        self.clip_processor: Optional["CLIPProcessor"] = None
        self.head_pose = head_pose
        #: Encode the head region as its own CLIP image (see _head_box).
        self.head_stream = bool(head_stream)
        self.clip_enabled = False

        if CLIPModel is None or CLIPProcessor is None:
            if not allow_clip_fallback:
                raise RuntimeError(
                    "transformers CLIP is not importable. Install transformers, or pass "
                    "allow_clip_fallback=True to train without CLIP features (zeros)."
                )
        else:
            try:
                import transformers
                mt = transformers.AutoConfig.from_pretrained(clip_model_name).model_type
                if mt in ("siglip", "siglip2"):
                    # The fixed-resolution siglip2 checkpoints declare model_type
                    # "siglip" and load with the v1 classes; only `-naflex` uses
                    # the Siglip2* ones, whose patch embedding is a Linear.
                    vm = getattr(transformers, "Siglip2VisionModel") \
                        if mt == "siglip2" else getattr(transformers, "SiglipVisionModel")
                    # AutoImageProcessor, not AutoProcessor: the latter also
                    # builds the TEXT tokenizer, which for SigLIP needs
                    # sentencepiece and fails the whole load over a component
                    # nothing here uses. Only images are encoded.
                    self.clip_processor = transformers.AutoImageProcessor.from_pretrained(
                        clip_model_name)
                    self.clip_model = vm.from_pretrained(clip_model_name).to(self.device)
                    self._siglip = True
                    self.clip_dim = int(self.clip_model.config.hidden_size)
                else:
                    self.clip_processor = CLIPProcessor.from_pretrained(clip_model_name)
                    self.clip_model = CLIPModel.from_pretrained(clip_model_name).to(self.device)
                    self.clip_dim = int(self.clip_model.config.projection_dim)
                self.clip_model.eval()
                self.clip_enabled = True
            except Exception as e:
                if not allow_clip_fallback:
                    raise RuntimeError(
                        f"CLIP weights failed to load ({e}). Fix the environment, or pass "
                        "allow_clip_fallback=True to continue with zeroed CLIP features."
                    ) from e
                print(f"[warn] CLIP disabled by explicit fallback; using zeros: {e}")

    #: 512 CLIP over the head crop + 6 head-box geometry dims. Kept as a class
    #: attribute because callers and tests read it off the class; instances with
    #: a wider encoder override it in __init__ (see head_stream_dim).
    HEAD_STREAM_DIM = 512 + 6

    @property
    def head_stream_dim(self) -> int:
        """Head-stream width for THIS encoder: embedding + 6 geometry dims."""
        return self.clip_dim + 6

    def output_dim(self) -> int:
        # 512 clip + 8 bbox geom + 24 color stats + 8 posture geom
        # (+4 head pose: yaw, pitch, roll, face_found — if a backend is configured)
        # (+518 head stream — if enabled)
        d = self.clip_dim + 8 + 24 + 8
        if self.head_pose is not None and self.head_pose.available():
            d += HeadPoseEstimator.OUTPUT_DIM
        if self.head_stream:
            d += self.head_stream_dim
        return d

    @staticmethod
    def _head_box(x1: int, y1: int, x2: int, y2: int, w: int, h: int,
                  frac: float = 0.30):
        """Where the head is, as absolute frame coordinates.

        Geometric rather than detected, deliberately. MediaPipe finds a face in
        60% of these crops overall and in only **8% of head_down** frames
        [internal notes, not included] — precisely the class where the head region matters
        most. A detector-gated crop would therefore be absent exactly where it
        is needed, and its presence/absence would leak the label. A fixed
        geometric prior is available on every frame and leaks nothing; the
        binary detection flag already lives in the head-pose block, where it
        belongs.

        Square, spanning the top ``frac`` of the person box and centred
        horizontally. Square because CLIP resizes to a square and a tall thin
        crop would be squashed; centred because a seated student's head is
        near the horizontal centre of a torso box.
        """
        bw, bh = max(1, x2 - x1), max(1, y2 - y1)
        side = max(8.0, frac * bh)
        cx = x1 + bw / 2.0
        cy = y1 + side / 2.0
        hx1 = int(round(max(0, cx - side / 2.0)))
        hy1 = int(round(max(0, cy - side / 2.0)))
        hx2 = int(round(min(w, cx + side / 2.0)))
        hy2 = int(round(min(h, cy + side / 2.0)))
        clipped = float(hx1 == 0 or hy1 == 0 or hx2 == w or hy2 == h)
        return hx1, hy1, hx2, hy2, clipped

    def _head_geom(self, hb, x1, y1, x2, y2, w, h) -> np.ndarray:
        """6 dims describing the head box, so the encoder is not the only cue."""
        hx1, hy1, hx2, hy2, clipped = hb
        bw, bh = max(1.0, float(x2 - x1)), max(1.0, float(y2 - y1))
        hw, hh = max(1.0, float(hx2 - hx1)), max(1.0, float(hy2 - hy1))
        return np.array([
            ((hx1 + hx2) / 2.0 - x1) / bw,     # head centre x within the person box
            ((hy1 + hy2) / 2.0 - y1) / bh,     # head centre y within the person box
            hw / bw, hh / bh,                  # head size relative to the person
            (hw * hh) / float(max(1, w * h)),  # head size relative to the frame
            clipped,                           # touched a frame edge
        ], dtype=np.float32)

    def extract(self, frame_bgr: np.ndarray, bbox_xyxy: List[float]) -> np.ndarray:
        return self.extract_batch(frame_bgr, [bbox_xyxy])[0]

    def extract_many(self, pairs, batch: int = 32) -> np.ndarray:
        """Features for (frame, bbox) pairs from DIFFERENT frames, batched.

        `extract_batch` batches the students of ONE frame, which is the right
        shape for live analysis. The sequence builder has the opposite shape: one
        student across many frames, so it was calling `extract` per crop and
        paying a batch-of-one forward pass each time. That is tolerable for
        CLIP ViT-B/32 at 2.97 ms but not for a so400m tower at 384px, where a
        batch of one runs ~3.5x the per-crop cost of a batch of eight and turns a
        3-hour corpus build into an 11-hour one.

        Only the encoder is batched; the geometry, colour and posture parts are
        per-crop arithmetic and stay exactly where they were, so this is the same
        function computed in a different order.
        """
        pairs = list(pairs)
        out = np.zeros((len(pairs), self.output_dim()), dtype=np.float32)
        for s in range(0, len(pairs), batch):
            grp = pairs[s:s + batch]
            crops, metas = [], []
            for frame, box in grp:
                h, w = frame.shape[:2]
                c, clipped, valid = crop_boxes(frame, [box])
                if not c:
                    metas.append(None)
                    continue
                crops.append(c[0])
                metas.append((frame, c[0], clipped[0], w, h))
            feats = self._clip_batch(crops) if crops else []
            j = 0
            for r, m in enumerate(metas):
                if m is None:
                    continue
                frame, crop, (x1, y1, x2, y2), w, h = m
                parts = [feats[j], self._geom(x1, y1, x2, y2, w, h),
                         self._color_stats(crop),
                         self._posture_geom(crop, x1, y1, x2, y2, w, h)]
                j += 1
                if self.head_pose is not None and self.head_pose.available():
                    parts.append(self.head_pose.estimate(crop))
                if self.head_stream:
                    hb = self._head_box(x1, y1, x2, y2, w, h)
                    hc = frame[hb[1]:hb[3], hb[0]:hb[2]]
                    hf = (self._clip_batch([hc])[0] if hc is not None and hc.size
                          else np.zeros(self.clip_dim, dtype=np.float32))
                    parts.append(hf)
                    parts.append(self._head_geom(hb, x1, y1, x2, y2, w, h)
                                 if hasattr(self, "_head_geom")
                                 else np.zeros(6, dtype=np.float32))
                v = np.concatenate(parts).astype(np.float32)
                out[s + r, :len(v)] = v
        return out

    def extract_batch(self, frame_bgr: np.ndarray, bboxes_xyxy: List[List[float]]) -> np.ndarray:
        """Features for every box of one frame with a single CLIP forward pass.

        Per-crop CLIP calls cost ~20 ms/student and dominated the real-time
        budget at high student counts; batching flattens that to one call per
        frame. Returns [N, output_dim()]; degenerate boxes yield zero rows.
        """
        n = len(bboxes_xyxy)
        out = np.zeros((n, self.output_dim()), dtype=np.float32)
        if n == 0:
            return out

        h, w = frame_bgr.shape[:2]
        crops, clipped, valid_idx = crop_boxes(frame_bgr, bboxes_xyxy)

        if not crops:
            return out

        clip_feats = self._clip_batch(crops)

        # Second CLIP pass over head crops taken from the FULL FRAME at native
        # resolution. This is the point of the stream: a person crop resized to
        # 224x224 puts the head on ~1 of CLIP-B/32's 49 patches, so gaze is
        # gone before the temporal model sees anything. Cropping the head and
        # letting it fill the same 224x224 gives it the whole 7x7 grid.
        head_feats = head_boxes = None
        if self.head_stream:
            head_boxes = [self._head_box(*cl, w, h) for cl in clipped]
            head_crops = [frame_bgr[b[1]:b[3], b[0]:b[2]] for b in head_boxes]
            ok = [c is not None and c.size > 0 for c in head_crops]
            feats = self._clip_batch([c for c, k in zip(head_crops, ok) if k])
            head_feats = np.zeros((len(head_crops), self.clip_dim), dtype=np.float32)
            j = 0
            for r, k in enumerate(ok):
                if k:
                    head_feats[r] = feats[j]
                    j += 1

        for row, (crop, (x1, y1, x2, y2), i) in enumerate(zip(crops, clipped, valid_idx)):
            parts = [
                clip_feats[row],
                self._geom(x1, y1, x2, y2, w, h),
                self._color_stats(crop),
                self._posture_geom(crop, x1, y1, x2, y2, w, h),
            ]
            if self.head_pose is not None and self.head_pose.available():
                # FULL crop, not a top-fraction slice. MediaPipe FaceLandmarker
                # runs its own face detector; pre-cropping to a guessed head
                # region only removes context. Measured detection rate on LLMSTU
                # crops: top 35% -> 38%, top 50% -> 55%, full crop -> 60%.
                parts.append(self.head_pose.estimate(crop).astype(np.float32))
            if self.head_stream:
                parts.append(head_feats[row])
                parts.append(self._head_geom(head_boxes[row], x1, y1, x2, y2, w, h))
            out[i] = np.concatenate(parts, axis=0).astype(np.float32)
        return out

    def _clip_batch(self, crops_bgr: List[np.ndarray]) -> np.ndarray:
        if not self.clip_enabled or self.clip_model is None or self.clip_processor is None:
            return np.zeros((len(crops_bgr), self.clip_dim), dtype=np.float32)
        images = [Image.fromarray(cv2.cvtColor(c, cv2.COLOR_BGR2RGB)) for c in crops_bgr]
        inp = self.clip_processor(images=images, return_tensors="pt").to(self.device)
        with torch.inference_mode():
            f = (self.clip_model(**inp).pooler_output if self._siglip
                 else self.clip_model.get_image_features(**inp))
            f = f / (f.norm(dim=-1, keepdim=True) + 1e-6)
        return f.detach().float().cpu().numpy()

    def _geom(self, x1: int, y1: int, x2: int, y2: int, w: int, h: int) -> np.ndarray:
        bw = max(1.0, float(x2 - x1))
        bh = max(1.0, float(y2 - y1))
        cx = (x1 + x2) * 0.5 / max(1.0, float(w))
        cy = (y1 + y2) * 0.5 / max(1.0, float(h))
        area = (bw * bh) / max(1.0, float(w * h))
        ar = bw / bh
        left = x1 / max(1.0, float(w))
        right = x2 / max(1.0, float(w))
        top = y1 / max(1.0, float(h))
        bottom = y2 / max(1.0, float(h))
        return np.array([cx, cy, area, ar, left, right, top, bottom], dtype=np.float32)

    def _color_stats(self, crop_bgr: np.ndarray) -> np.ndarray:
        chans = cv2.split(crop_bgr)
        feats = []
        for ch in chans:
            chf = ch.astype(np.float32)
            # one partition-based call for all three percentiles: identical
            # values to separate np.percentile calls at ~1/3 the sort cost
            p25, p50, p75 = np.percentile(chf, (25, 50, 75))
            feats.extend(
                [
                    float(chf.mean()),
                    float(chf.std()),
                    float(p25),
                    float(p50),
                    float(p75),
                    float(chf.min()),
                    float(chf.max()),
                    float((chf > 200).mean()),
                ]
            )
        return np.array(feats, dtype=np.float32)

    def _posture_geom(self, crop_bgr: np.ndarray, x1: int, y1: int, x2: int, y2: int, w: int, h: int) -> np.ndarray:
        """Cheap posture proxies from box shape and crop intensity layout.

        The head occupies the top of a seated person's box; its horizontal
        offset and the vertical mass distribution shift with leaning,
        head-down and turned postures. All values are in [0, 1]-ish ranges so
        they can sit next to the normalized geometry block.
        """
        bw = max(1.0, float(x2 - x1))
        bh = max(1.0, float(y2 - y1))
        elongation = bh / bw  # tall = upright, squat = slumped/leaning
        gray = cv2.cvtColor(crop_bgr, cv2.COLOR_BGR2GRAY).astype(np.float32)

        # vertical mass distribution: center of intensity mass, top-third share
        col_profile = gray.mean(axis=1)
        total = float(col_profile.sum()) + 1e-6
        ys = np.arange(col_profile.shape[0], dtype=np.float32)
        v_center = float((col_profile * ys).sum() / total / max(1.0, gray.shape[0]))
        third = max(1, gray.shape[0] // 3)
        top_share = float(col_profile[:third].sum() / total)
        bottom_share = float(col_profile[-third:].sum() / total)

        # head-region estimate: brightest-motion proxy is unavailable per-frame,
        # so use the horizontal intensity centroid of the top 35% as a head
        # x-offset proxy (turned head/torso shifts it off center).
        head_h = max(1, int(0.35 * gray.shape[0]))
        head_strip = gray[:head_h]
        row_profile = head_strip.mean(axis=0)
        xs = np.arange(row_profile.shape[0], dtype=np.float32)
        head_cx = float((row_profile * xs).sum() / (float(row_profile.sum()) + 1e-6) / max(1.0, gray.shape[1]))

        # contrast between head strip and torso strip (head-down lowers it)
        torso = gray[head_h:] if gray.shape[0] > head_h else gray
        head_torso_contrast = float(
            (head_strip.mean() - torso.mean()) / (head_strip.mean() + torso.mean() + 1e-6)
        )
        # normalized top-edge position of box (standing/leaning forward changes it)
        top_rel = y1 / max(1.0, float(h))

        return np.array(
            [
                elongation,
                v_center,
                top_share,
                bottom_share,
                head_cx,
                head_torso_contrast,
                top_rel,
                min(1.0, bw / max(1.0, float(w))),
            ],
            dtype=np.float32,
        )
