"""The VLM must never make a frame wait.

A synchronous VLM cannot be live. One option-likelihood forward pass per student
is ~0.2-0.5 s on an L4, so six students is 1.5-3 s per frame against a pipeline
that manages 1-4 fps -- an order of magnitude slower, with the overlay falling
further behind the room the longer it runs. That is the same failure
`LatestFrame` exists to prevent, reintroduced one layer up.

`AsyncGrounder` decouples the two rates: `submit()` returns immediately and
`latest()` hands back the most recent completed opinion, or nothing. These tests
pin the properties that make that safe.
"""
import time

import numpy as np
import pytest

from attention.vlm_grounder import AsyncGrounder, StubGrounder

BOXES = [[0, 0, 10, 10], [10, 10, 20, 20]]
TIDS = [7, 9]
FRAME = np.zeros((32, 32, 3), dtype=np.uint8)


def _wait(g, timeout=5.0):
    t0 = time.time()
    while time.time() - t0 < timeout:
        r = g.latest(time.time())
        if r is not None:
            return r
        time.sleep(0.01)
    return None


def test_submit_does_not_block():
    """The whole point. A 0.3 s scorer must not cost the caller 0.3 s."""
    class Slow(StubGrounder):
        def score_students(self, frame, boxes):
            time.sleep(0.3)
            return super().score_students(frame, boxes)

    g = AsyncGrounder(Slow())
    try:
        t0 = time.time()
        for _ in range(5):
            g.submit(FRAME, BOXES, TIDS, time.time())
        assert time.time() - t0 < 0.1, "submit() blocked on the scorer"
    finally:
        g.close()


def test_nothing_before_the_first_result():
    g = AsyncGrounder(StubGrounder())
    try:
        assert g.latest(time.time()) is None
    finally:
        g.close()


def test_a_result_arrives_and_carries_its_age():
    g = AsyncGrounder(StubGrounder(cue_ids=[4], conf=0.8))
    try:
        t = time.time()
        g.submit(FRAME, BOXES, TIDS, t)
        got = _wait(g)
        assert got is not None
        scores, tids, age = got
        assert scores.shape == (2, 6)
        assert tids == TIDS
        assert age >= 0.0, "age must be reported so staleness can be shown"
    finally:
        g.close()


def test_newest_submission_wins():
    """A queue would make the overlay lag without bound. Unstarted work is
    replaced, so the VLM always answers about the most recent frame offered."""
    seen = []

    class Recording(StubGrounder):
        def score_students(self, frame, boxes):
            time.sleep(0.25)
            seen.append(len(boxes))
            return super().score_students(frame, boxes)

    g = AsyncGrounder(Recording())
    try:
        g.submit(FRAME, BOXES, TIDS, time.time())          # this one starts
        time.sleep(0.05)
        for n in (3, 4, 5):                                 # these queue up...
            g.submit(FRAME, BOXES[:1] * n, list(range(n)), time.time())
        time.sleep(1.0)
        # ...and only the LAST of them should ever have been scored.
        assert 4 not in seen and 3 not in seen, f"stale work was scored: {seen}"
    finally:
        g.close()


def test_a_stale_opinion_expires():
    """An opinion older than max_age is withheld rather than shown as current."""
    g = AsyncGrounder(StubGrounder(), max_age_s=0.2)
    try:
        g.submit(FRAME, BOXES, TIDS, time.time())
        assert _wait(g) is not None
        time.sleep(0.35)
        assert g.latest(time.time()) is None, "expired opinion was still served"
    finally:
        g.close()


def test_a_failing_vlm_is_reported_not_raised():
    """The fused entry is the base checkpoint plus an opinion. Without the
    opinion the checkpoint is still right, so a VLM failure must degrade the
    run rather than end it."""
    class Broken:
        def score_students(self, frame, boxes):
            raise RuntimeError("no weights")

    g = AsyncGrounder(Broken())
    try:
        g.submit(FRAME, BOXES, TIDS, time.time())
        t0 = time.time()
        while time.time() - t0 < 3.0 and g.error is None:
            time.sleep(0.02)
        assert g.error is not None and "no weights" in g.error
        assert g.latest(time.time()) is None
    finally:
        g.close()


def test_empty_boxes_are_not_submitted():
    g = AsyncGrounder(StubGrounder())
    try:
        g.submit(FRAME, [], [], time.time())
        time.sleep(0.2)
        assert g.latest(time.time()) is None
    finally:
        g.close()


def test_fuse_frame_tolerates_a_student_the_vlm_never_saw():
    """Tracks appear between VLM opinions; they must be fused temporal-only
    rather than dropped from the display."""
    from attention.fusion import Policy, fuse_frame
    temporal = {1: [0.7, 0.1, 0.05, 0.05, 0.05, 0.05],
                2: [0.1, 0.7, 0.05, 0.05, 0.05, 0.05]}
    vlm = {1: [0.8, 0.05, 0.05, 0.04, 0.03, 0.03]}      # nothing for track 2
    out = fuse_frame(temporal, vlm, policy=Policy.AGREEMENT)
    assert set(out) == {1, 2}
    assert out[2].vlm_cue is None
    assert out[1].vlm_cue is not None


# --------------------------------------------------------------------------
# the question must match the letters being scored
#
# `_letter_ids` comes from `self.classes`; the prompt used to come from
# `build_prompt()` with no argument, i.e. always the six cue6 options. A cue9
# grounder therefore read nine logits off a six-option question, and three of
# the four classes cue9 exists to separate were scored from options the model
# never saw. The output was well-formed the whole time, which is why it took
# reading the call to find.
# --------------------------------------------------------------------------

def test_the_prompt_offers_exactly_the_classes_being_scored():
    from attention.taxonomy import CUE9_CLASSES
    from attention.vlm_grounder import QwenGrounder, option_letters

    for classes in (None, list(CUE9_CLASSES)):
        g = QwenGrounder(classes=classes)
        text = g.prompt()
        letters = option_letters(len(g.classes))
        offered = [L for L in letters if f"\n{L}. " in "\n" + text]
        assert offered == list(letters), (
            f"{len(g.classes)} classes scored, {len(offered)} offered")
        assert f"({letters[0]}-{letters[-1]})" in text


def test_a_nine_class_grounder_does_not_ask_a_six_option_question():
    """The regression itself, stated as the difference it makes."""
    from attention.taxonomy import CUE9_CLASSES
    from attention.vlm_grounder import QwenGrounder

    six = QwenGrounder().prompt()
    nine = QwenGrounder(classes=list(CUE9_CLASSES)).prompt()
    assert six != nine
    assert "\nG. " in nine and "\nG. " not in six
