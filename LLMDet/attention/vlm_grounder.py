"""A VLM's second opinion on each tracked student.

Interface
---------
A grounder answers one question: given a frame and the students' boxes, how
does each student score against the six cue phrases? Everything else -- the
tracker, the temporal model, the fusion policy -- is unchanged.

    scores = grounder.score_students(frame_bgr, boxes)   # [n_boxes, 6]

Rows are probability distributions over ``taxonomy.CUE_CLASSES``, in that
order. A backend that cannot judge a student returns a uniform row rather than
guessing, so the fusion layer sees "no opinion" instead of a confident wrong
one.

Why option-likelihood scoring instead of generating JSON
--------------------------------------------------------
The labelling pipeline had the VLM emit the full 10-field schema and then
collapsed it through ``map_record``. Faithful, but it costs up to 512 generated
tokens per student -- seconds each on a T4, against the pipeline's 3.48 fps.

This scores the six options in a single forward pass and reads the logits at
the answer position, giving a distribution over cues at roughly the cost of one
prefill. The trade is real and worth stating: option-letter scoring is
sensitive to option ORDER and to wording, in a way that free generation is not.
The order here is fixed to CUE_CLASSES and recorded in the output so a later
audit can re-run with a permuted order and measure the sensitivity rather than
assume it away.

Independence caveat
-------------------
The temporal model was trained on labels from Qwen3.5-27B. Querying a
Qwen-family VLM is therefore not a fully independent second opinion. Using a
smaller base model that was never adapted to these labels weakens the coupling;
it does not remove it. See fusion.py.
"""

from __future__ import annotations

from typing import List, Optional, Protocol, Sequence

import numpy as np

from .cue_phrases import phrases_in_class_order
from .taxonomy import CUE_CLASSES

NUM_CUES = len(CUE_CLASSES)
#: Letters presented to the model, one per cue, in class order. The six-class
#: list is the default; :func:`option_letters` builds one for any width.
OPTION_LETTERS = [chr(ord("A") + i) for i in range(NUM_CUES)]


def option_letters(n: int) -> List[str]:
    """``n`` single-character option labels.

    Capped at 26 deliberately: past Z the labels would need two characters, and
    option-likelihood scoring reads ONE token at the answer position. A
    two-character label would be measuring a prefix, and every score after it
    would be quietly wrong rather than absent.
    """
    if not 1 <= n <= 26:
        raise ValueError(
            f"{n} options: option-likelihood scoring needs one single-character "
            f"label per class, so it supports 1-26.")
    return [chr(ord("A") + i) for i in range(n)]


class Grounder(Protocol):
    """Anything that can score students against the cue phrases."""

    def score_students(self, frame_bgr: np.ndarray,
                       boxes: Sequence[Sequence[float]]) -> np.ndarray:
        ...


def uniform_rows(n: int, k: int = NUM_CUES) -> np.ndarray:
    """`n` rows of "no opinion" over `k` classes -- the honest output when
    scoring fails."""
    return np.full((n, k), 1.0 / k, dtype=np.float32)


def build_prompt(classes: Optional[Sequence[str]] = None) -> str:
    """The multiple-choice question, built from the phrase mapping.

    Mirrors the labelling pipeline's guard clause: judge the student in the
    centre of the crop, describe visible behaviour, never infer identity.
    """
    cl = list(CUE_CLASSES if classes is None else classes)
    letters = option_letters(len(cl))
    opts = "\n".join(f"{L}. {p}" for L, p in
                     zip(letters, phrases_in_class_order(cl)))
    return (
        "Look only at the student in the centre of this crop. Describe what is "
        "visible; do not guess identity, age, gender or ethnicity.\n\n"
        "Which ONE option best describes this student?\n"
        f"{opts}\n\n"
        f"Answer with a single letter ({letters[0]}-{letters[-1]})."
    )


def crop_students(frame_bgr: np.ndarray,
                  boxes: Sequence[Sequence[float]],
                  pad: float = 0.10) -> List[Optional[np.ndarray]]:
    """Crops for each box, padded slightly, or None where degenerate.

    Padding gives the model a little context around the student -- a phone or a
    neighbour can sit just outside a tight detector box, and both matter to the
    cue being judged.
    """
    h, w = frame_bgr.shape[:2]
    out: List[Optional[np.ndarray]] = []
    for b in boxes:
        x1, y1, x2, y2 = (float(v) for v in b[:4])
        px, py = (x2 - x1) * pad, (y2 - y1) * pad
        xa, ya = int(max(0, x1 - px)), int(max(0, y1 - py))
        xb, yb = int(min(w, x2 + px)), int(min(h, y2 + py))
        out.append(frame_bgr[ya:yb, xa:xb] if xb > xa and yb > ya else None)
    return out


class StubGrounder:
    """Deterministic fake, for tests and for running the UI without weights.

    Not random: a stub that returned noise would make a disagreement rate look
    like a measurement. This returns a fixed distribution derived from the box
    index, so tests can assert exact fusion outcomes.
    """

    def __init__(self, cue_ids: Optional[Sequence[int]] = None, conf: float = 0.7):
        self.cue_ids = cue_ids
        self.conf = float(conf)

    def score_students(self, frame_bgr: np.ndarray,
                       boxes: Sequence[Sequence[float]]) -> np.ndarray:
        n = len(boxes)
        rows = uniform_rows(n).copy()
        rest = (1.0 - self.conf) / (NUM_CUES - 1)
        for i in range(n):
            # Cycle rather than index: a caller that wants "everyone is
            # screen_oriented" passes [0], and indexing would raise IndexError
            # from inside the grounder on the second box -- a confusing failure
            # for a deterministic fake whose whole job is to be predictable.
            cid = (self.cue_ids[i % len(self.cue_ids)]
                   if self.cue_ids else i % NUM_CUES)
            rows[i, :] = rest
            rows[i, cid] = self.conf
        return rows


class AsyncGrounder:
    """Runs any grounder on its own thread, and never makes a frame wait.

    A VLM cannot be synchronous in a live path. Measured arithmetic: one
    option-likelihood forward pass per student is ~0.2-0.5 s on an L4, so six
    students is 1.5-3 s, against a pipeline that manages 1-4 fps. Calling
    `score_students` inline would drop the frame rate by an order of magnitude
    and the overlay would fall behind the room -- the exact failure `LatestFrame`
    exists to prevent, reintroduced one layer up.

    So the caller never blocks. `submit()` hands over the newest frame and
    returns immediately; `latest()` returns the most recent completed scores, or
    None before the first one lands. `fuse_frame` already treats a student with
    no VLM row as temporal-only, so a stale or absent opinion degrades exactly
    the way the data model was built for.

    The staleness is REPORTED rather than hidden: `latest()` also returns the
    age of the opinion, so a UI can say "VLM opinion 2.4 s old" instead of
    implying it is current.
    """

    def __init__(self, grounder, max_age_s: float = 10.0):
        import threading
        self._g = grounder
        self._lock = threading.Lock()
        self._pending = None          # (frame, boxes, tids, t)
        self._result = None           # (scores, tids, t)
        self._max_age = float(max_age_s)
        self._stop = False
        self._err = None
        #: False until the backend's weights are resident. Surfaced so the UI
        #: can say "loading" instead of showing an absent opinion as if the VLM
        #: had considered the frame and declined to answer.
        self._ready = False
        self._thread = threading.Thread(target=self._loop, daemon=True)
        self._thread.start()

    def _loop(self):
        import time
        # Load the weights BEFORE the first job rather than on it. The load is
        # ~60 s and the demo session is 64 s of video, so a lazy load means the
        # replay ends before the first opinion exists and the feature looks
        # broken on every first selection. Warming here overlaps the load with
        # the replay instead of serialising after it.
        try:
            ensure = getattr(self._g, "_ensure", None)
            if callable(ensure):
                ensure()
            self._ready = True
        except Exception as e:                               # noqa: BLE001
            self._err = f"{type(e).__name__}: {e}"
        while not self._stop:
            with self._lock:
                job, self._pending = self._pending, None
            if job is None:
                time.sleep(0.02)
                continue
            frame, boxes, tids, t = job
            try:
                sc = self._g.score_students(frame, boxes)
                with self._lock:
                    self._result = (sc, tids, t)
            except Exception as e:                           # noqa: BLE001
                with self._lock:
                    self._err = f"{type(e).__name__}: {e}"

    def submit(self, frame, boxes, tids, t):
        """Offer work. Replaces any job not yet started -- newest wins."""
        if not len(boxes):
            return
        with self._lock:
            self._pending = (frame, list(boxes), list(tids), t)

    def latest(self, now: float):
        """(scores, track_ids, age_seconds) or None. Stale results expire."""
        with self._lock:
            if self._result is None:
                return None
            sc, tids, t = self._result
            age = now - t
            if age > self._max_age:
                return None
            return sc, tids, age

    @property
    def ready(self) -> bool:
        """Have the weights finished loading?"""
        return self._ready

    @property
    def error(self):
        with self._lock:
            return self._err

    def close(self):
        """Stop the thread AND drop the model.

        Setting the flag alone leaves `self._g` holding several GB of weights
        that nothing will collect while this object is referenced. On a Space
        with a 16 GB system-RAM budget that is the difference between switching
        models and being killed for it.
        """
        self._stop = True
        try:
            self._thread.join(timeout=2.0)
        except Exception:                                    # noqa: BLE001
            pass
        g, self._g = self._g, None
        release = getattr(g, "release", None)
        if callable(release):
            release()
        with self._lock:
            self._pending = self._result = None


class LlavaOneVisionGrounder:
    """The 0.5B LLaVA-OneVision that already ships with the detector.

    Chosen for one reason: it loads on THIS stack. The detector config names
    `../huggingface/my_llava-onevision-qwen2-0.5b-ov-2` as its LMM, so the
    weights are already on disk and already in the artifact repo -- and
    `predict()` never reads them, so they are free to repurpose. At 0.5B it is
    roughly 8x faster than a 4B, which is what makes an async second opinion
    keep up with a live camera at all.

    It is loaded through the VENDORED `llava` package rather than transformers,
    which is why it works where `QwenGrounder` does not: it never touches
    `AutoModelForImageTextToText`.

    Independence caveat, unchanged and if anything stronger: this is a Qwen2
    backbone, and the labels came from Qwen3.5-27B. It is a cheaper opinion, not
    an independent one. Report it as an ensemble member.
    """

    MODEL_DIR = "../huggingface/my_llava-onevision-qwen2-0.5b-ov-2"

    def __init__(self, model_dir: Optional[str] = None,
                 device: str = "cuda:0", dtype: str = "float16",
                 max_students: int = 12):
        self.model_dir = model_dir or self.MODEL_DIR
        self.device = device
        self.dtype = dtype
        self.max_students = int(max_students)
        self._model = None
        self._tok = None
        self._letter_ids: Optional[List[int]] = None
        self._warned = False

    @staticmethod
    def available(model_dir: Optional[str] = None) -> "tuple[bool, str]":
        import os
        d = model_dir or LlavaOneVisionGrounder.MODEL_DIR
        if not os.path.isdir(d):
            return False, f"{d} is not on disk"
        try:
            import llava            # noqa: F401
        except Exception as e:      # noqa: BLE001
            return False, f"vendored llava package not importable: {e}"
        return True, ""

    def _ensure(self):
        if self._model is not None:
            return
        import torch
        import transformers
        from llava.model.language_model.llava_qwen import LlavaQwenForCausalLM

        ok, why = self.available(self.model_dir)
        if not ok:
            raise RuntimeError(f"{self.model_dir} is not loadable: {why}")

        td = getattr(torch, self.dtype)
        self._model = LlavaQwenForCausalLM.from_pretrained(
            self.model_dir, torch_dtype=td).to(self.device).eval()
        self._tok = transformers.AutoTokenizer.from_pretrained(self.model_dir)

        # One token per option letter, checked. If a letter tokenises to more
        # than one token the logit read would be measuring a prefix and every
        # score after it would be wrong -- silently.
        ids = []
        for L in OPTION_LETTERS:
            enc = self._tok.encode(L, add_special_tokens=False)
            if len(enc) != 1:
                raise RuntimeError(
                    f"option letter {L!r} tokenises to {len(enc)} tokens under "
                    f"{self.model_dir}; option-likelihood scoring needs one "
                    f"token per letter.")
            ids.append(enc[0])
        self._letter_ids = ids

    def score_students(self, frame_bgr: np.ndarray,
                       boxes: Sequence[Sequence[float]]) -> np.ndarray:
        """[n_boxes, NUM_CUES], rows summing to 1, in CUE_CLASSES order.

        Text-only scoring over the crop is NOT what this does: the crop is
        encoded by the SigLip tower and projected into the LM's embedding space
        exactly as the model was trained to consume it, then the six option
        letters are read off the logits at the answer position -- one forward
        pass per student, no generation.

        A student that cannot be scored gets a uniform row, so the fusion layer
        sees "no opinion" rather than a confident wrong one.
        """
        import torch
        n = len(boxes)
        out = uniform_rows(n)
        if n == 0:
            return out
        self._ensure()

        crops = crop_students(frame_bgr, boxes)
        # The prompt MUST carry the <image> token, and it must be tokenised by
        # llava's own helper, which replaces it with IMAGE_TOKEN_INDEX (-200).
        # A plain tokenizer call produces ids with no image slot, the `images`
        # argument is then never spliced in, and the model answers from the text
        # alone -- which returns the SAME distribution for every student. That
        # identical-rows symptom is the only thing that distinguishes it from
        # working, since the output is otherwise perfectly well-formed.
        from llava.constants import DEFAULT_IMAGE_TOKEN
        from llava.mm_utils import tokenizer_image_token
        prompt = DEFAULT_IMAGE_TOKEN + "\n" + build_prompt(self.classes)
        ids = tokenizer_image_token(prompt, self._tok,
                                    return_tensors="pt").unsqueeze(0).to(self.device)

        with torch.no_grad():
            for i, crop in enumerate(crops[:self.max_students]):
                if crop is None or crop.size == 0:
                    continue
                try:
                    px = self._pixels(crop)
                    logits = self._model(input_ids=ids, images=px).logits
                    last = logits[0, -1].float()
                    sel = last[self._letter_ids]
                    out[i] = torch.softmax(sel, dim=-1).cpu().numpy()
                except Exception as e:                   # noqa: BLE001
                    # One unscorable student must not cost the whole frame its
                    # opinion; the row stays uniform. But a failure that hits
                    # EVERY student is a broken backend, not a hard crop, and
                    # swallowing it silently returns six uniform rows that look
                    # like considered "no opinion" -- so the first one is
                    # reported, once.
                    if not self._warned:
                        self._warned = True
                        print(f"[vlm] scoring failed, rows stay uniform: "
                              f"{type(e).__name__}: {e}", flush=True)
                    continue
        return out

    def _pixels(self, crop_bgr: np.ndarray):
        """Crop -> the vision tower's expected tensor."""
        import torch
        from PIL import Image
        vt = self._model.get_model().get_vision_tower()
        proc = vt.image_processor
        img = Image.fromarray(crop_bgr[:, :, ::-1])
        # `.preprocess`, not `__call__`: the vendored SigLipImageProcessor is
        # not a transformers ImageProcessingMixin and defines only the former.
        # Calling it raises "'SigLipImageProcessor' object is not callable".
        px = (proc.preprocess(images=img, return_tensors="pt")["pixel_values"]
              if hasattr(proc, "preprocess")
              else proc(images=img, return_tensors="pt")["pixel_values"])
        return px.to(self.device, dtype=getattr(torch, self.dtype))


class QwenGrounder:
    """A Qwen VLM scoring the six options in one forward pass per student.

    Defaults to **Qwen2-VL-2B-Instruct**, not the 4B Qwen3-VL it was written
    for. Qwen3-VL needs transformers >= 4.57; Qwen2-VL needs >= 4.45, and 4.45
    is a far smaller step from the pinned 4.44.2 than 4.57 is. The pin exists to
    protect mmcv's compiled `_ext` against torch 2.2.2, so the smallest bump
    that unblocks the feature is the right one.

    2B rather than 4B is also the right size here: at ~4 GB in fp16 it leaves
    room for the detector on a 24 GB card, and the cost of this call is
    dominated by the vision tower and the prefill rather than by LM width.

    Independence caveat, unchanged: the labels came from Qwen3.5-27B, so a Qwen
    backbone is a cheaper opinion rather than an independent one. Report it as
    an ensemble member.
    """

    def __init__(self,
                 model_id: str = "Qwen/Qwen2-VL-2B-Instruct",
                 device: str = "cuda:0",
                 dtype: str = "float16",
                 max_students: int = 12,
                 classes: Optional[Sequence[str]] = None):
        #: The label space this grounder scores. Defaults to the six cue
        #: classes; a cue9 model needs its own nine, with their own phrases and
        #: their own nine option letters.
        self.classes = list(CUE_CLASSES if classes is None else classes)
        self.model_id = model_id
        self.device = device
        self.dtype = dtype
        self.max_students = int(max_students)
        self._model = None
        self._proc = None
        self._letter_ids: Optional[List[int]] = None

    # -- lazy load: the dashboard must start without paying for the VLM -----
    @staticmethod
    def available() -> "tuple[bool, str]":
        """(usable here, why not). Checked BEFORE a run commits to this backend.

        Qwen3-VL needs ``AutoModelForImageTextToText``, which arrived in
        transformers 4.45. This stack pins **4.44.2**, and that pin is
        load-bearing rather than lazy: the Space's requirements.txt documents
        that torch 2.2.2 and mmcv's compiled ``_ext`` are built against each
        other, and a dependency bump that drags torch forward produces

            mmcv/_ext...so: undefined symbol: _ZN3c104cuda9SetDeviceEi

        i.e. it breaks the detector, which is the part that works. So this
        reports unavailable rather than being made to work by upgrading.
        """
        try:
            import transformers
        except Exception as e:                               # noqa: BLE001
            return False, f"transformers not importable: {e}"
        v = transformers.__version__
        # Two things are needed and they arrived in DIFFERENT releases, which is
        # the trap: a loader class, and the Qwen2-VL architecture itself.
        #
        # `AutoModelForVision2Seq` has existed since long before 4.44, so its
        # presence proves nothing. `AutoModelForImageTextToText` is NOT the test
        # either -- it landed well after 4.45, and checking for it reported
        # "unavailable" on a 4.45.2 install that could in fact load Qwen2-VL.
        # The architecture class is the thing that actually gates this.
        if not hasattr(transformers, "Qwen2VLForConditionalGeneration"):
            return False, (
                f"transformers {v} has no Qwen2VLForConditionalGeneration "
                f"(added in 4.45); this backend cannot load. Bumping the pin "
                f"risks mmcv's compiled _ext against torch 2.2.2, so this "
                f"reports unavailable rather than upgrading in place.")
        if not (hasattr(transformers, "AutoModelForImageTextToText")
                or hasattr(transformers, "AutoModelForVision2Seq")):
            return False, f"transformers {v} has no vision-to-text auto class"
        return True, ""

    def _ensure(self):
        if self._model is not None:
            return
        ok, why = self.available()
        if not ok:
            raise RuntimeError(f"{self.model_id} is not loadable here: {why}")
        import torch
        import transformers
        from transformers import AutoProcessor

        # Prefer the newer auto class where it exists, fall back to the one that
        # has been there for years. Both resolve Qwen2-VL from its config.
        Loader = getattr(transformers, "AutoModelForImageTextToText", None) \
            or transformers.AutoModelForVision2Seq

        td = getattr(torch, self.dtype)
        self._proc = AutoProcessor.from_pretrained(self.model_id)
        # low_cpu_mem_usage streams the weights straight to the target device
        # instead of materialising a full CPU copy first. On a 16 GB Space that
        # copy is the peak, not the resident size.
        self._model = Loader.from_pretrained(
            self.model_id, torch_dtype=td,
            low_cpu_mem_usage=True).to(self.device).eval()

        # Token id of each option letter. Resolved once, and checked: if a
        # letter does not map to a single token the logit read would be
        # measuring a prefix, and every score after it would be wrong.
        tok = self._proc.tokenizer
        ids = []
        for L in option_letters(len(self.classes)):
            enc = tok.encode(L, add_special_tokens=False)
            if len(enc) != 1:
                raise RuntimeError(
                    f"option letter {L!r} tokenises to {len(enc)} tokens under "
                    f"{self.model_id}; option-likelihood scoring needs one "
                    f"token per letter.")
            ids.append(enc[0])
        self._letter_ids = ids

    def prompt(self) -> str:
        """The question this grounder asks, over ITS OWN classes.

        Separate from `build_prompt()` because calling that with no argument is
        exactly the bug this replaced: it returns the SIX cue6 options, while
        `_letter_ids` is built from `self.classes`. A cue9 grounder then scored
        nine letters against a question that offered six, and G/H/I -- three of
        the four classes cue9 exists to separate -- were read off logits for
        options the model was never shown. Well-formed, plausible, and noise.
        """
        return build_prompt(self.classes)

    def release(self) -> None:
        """Drop the weights so the process gets the memory back."""
        self._model = None
        self._proc = None
        self._letter_ids = None

    def score_students(self, frame_bgr: np.ndarray,
                       boxes: Sequence[Sequence[float]]) -> np.ndarray:
        import cv2
        import torch
        from PIL import Image

        n = len(boxes)
        scores = uniform_rows(n, len(self.classes))
        if n == 0:
            return scores
        self._ensure()

        crops = crop_students(frame_bgr, boxes)
        idx = [i for i, c in enumerate(crops) if c is not None][:self.max_students]
        if not idx:
            return scores

        prompt = self.prompt()
        images = [Image.fromarray(cv2.cvtColor(crops[i], cv2.COLOR_BGR2RGB))
                  for i in idx]
        msgs = [[{"role": "user", "content": [{"type": "image"},
                                              {"type": "text", "text": prompt}]}]
                for _ in idx]
        texts = [self._proc.apply_chat_template(
            m, tokenize=False, add_generation_prompt=True) for m in msgs]
        inputs = self._proc(text=texts, images=images, return_tensors="pt",
                            padding=True).to(self.device)

        with torch.inference_mode():
            logits = self._model(**inputs).logits[:, -1, :].float()

        sel = logits[:, self._letter_ids]              # [k, 6]
        probs = torch.softmax(sel, dim=-1).cpu().numpy()
        for row, i in enumerate(idx):
            scores[i] = probs[row]
        return scores
