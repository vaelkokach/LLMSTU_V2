"""Fusing the temporal model with a VLM's opinion.

The VLM call itself is not tested here -- it needs weights and a GPU. Everything
around it is: the phrase mapping, the cropping, the decision policies and the
agreement accounting. That split is deliberate, so the part that can be wrong
silently is the part under test.
"""
from __future__ import annotations

import numpy as np
import pytest

from attention.cue_phrases import CUE_PHRASES, phrases_in_class_order
from attention.fusion import (NUM_CUES, FusedStudent, Policy, agreement_summary,
                              fuse_frame, fuse_student)
from attention.taxonomy import CUE_CLASSES, CUE_TO_ID
from attention.vlm_grounder import (OPTION_LETTERS, StubGrounder, build_prompt,
                                    crop_students, uniform_rows)


def peaked(cue: str, conf: float = 0.8) -> np.ndarray:
    """A distribution peaked on one cue."""
    v = np.full(NUM_CUES, (1.0 - conf) / (NUM_CUES - 1))
    v[CUE_TO_ID[cue]] = conf
    return v


# --------------------------------------------------------------------------
# phrases
# --------------------------------------------------------------------------

def test_every_cue_has_a_phrase_and_order_follows_the_taxonomy():
    """Index i of the score vector must be class i, or every cue is paired
    with the wrong phrase and nothing downstream notices.

    The invariant now spans BOTH label spaces: cue9 splits `screen_oriented`
    into four classes that each need their own phrase, and a phrase with no
    class is dead weight that will eventually be pressed into service for the
    wrong one.
    """
    from attention.taxonomy import LABEL_SPACES, TAXONOMIES
    every_class = {c for cl in LABEL_SPACES.values() for c in cl}
    # Every class of a base SPACE needs a phrase: those are what the VLM scores.
    assert every_class <= set(CUE_PHRASES), (
        f"classes with no phrase: {sorted(every_class - set(CUE_PHRASES))}")
    # No ORPHAN phrases -- dead weight eventually gets pressed into service for
    # the wrong cue. But a regrouping may introduce a class name of its own
    # (cue8's `engaged` merges reading + listening) and that is not an orphan,
    # so the allowed set is every class of every taxonomy, not only the spaces.
    # Merges that are not observable behaviours (`on_task`, `down_or_hidden`)
    # still have no phrase, and `_has_phrases` skips those taxonomies.
    owned = every_class | {c for t in TAXONOMIES.values() for c in t["classes"]}
    assert set(CUE_PHRASES) <= owned, (
        f"phrases with no class: {sorted(set(CUE_PHRASES) - owned)}")

    for space, classes in LABEL_SPACES.items():
        got = phrases_in_class_order(classes)
        assert len(got) == len(classes), space
        for i, c in enumerate(classes):
            assert got[i] == CUE_PHRASES[c], f"{space}[{i}] is not {c}"


def test_a_class_without_a_phrase_raises_rather_than_shortening_the_list():
    """A gap would shift every option letter after it onto the wrong cue, and
    the scores would be well-formed and wrong."""
    import pytest as _pytest
    with _pytest.raises(KeyError, match="no cue phrase"):
        phrases_in_class_order(list(CUE_CLASSES) + ["not_a_cue"])


def test_phrases_describe_behaviour_not_mental_state():
    """The project's standing rule: claim what is visible, never infer
    attention. A VLM asked about attention answers a different question."""
    banned = ("paying attention", "not paying attention", "engaged",
              "disengaged", "bored", "interested", "focused")
    for cue, phrase in CUE_PHRASES.items():
        low = phrase.lower()
        for b in banned:
            assert b not in low, f"{cue} phrase asserts mental state: {phrase!r}"


def test_prompt_lists_every_option_once():
    p = build_prompt()
    for letter, phrase in zip(OPTION_LETTERS, phrases_in_class_order()):
        assert f"{letter}. {phrase}" in p
    assert len(OPTION_LETTERS) == NUM_CUES
    assert len(set(OPTION_LETTERS)) == NUM_CUES


# --------------------------------------------------------------------------
# cropping
# --------------------------------------------------------------------------

def test_crop_students_pads_and_clamps():
    frame = np.zeros((1080, 1918, 3), np.uint8)
    crops = crop_students(frame, [[800, 400, 958, 630]], pad=0.10)
    assert crops[0] is not None
    h, w = crops[0].shape[:2]
    assert h > 230 and w > 158, "padding must enlarge the box"
    edge = crop_students(frame, [[0, 0, 60, 80]], pad=0.5)[0]
    assert edge is not None and edge.size > 0, "must clamp, not go negative"


def test_crop_students_returns_none_for_degenerate_boxes():
    frame = np.zeros((100, 100, 3), np.uint8)
    assert crop_students(frame, [[10, 10, 10, 10]])[0] is None


# --------------------------------------------------------------------------
# fusion policies
# --------------------------------------------------------------------------

def test_agreement_shows_the_cue_when_both_agree():
    f = fuse_student(peaked("phone_use"), peaked("phone_use"))
    assert f.agree is True and f.contested is False
    assert f.cue == "phone_use"


def test_agreement_refuses_to_pick_a_winner_when_they_disagree():
    """Choosing the more confident one would compare a calibrated probability
    against an uncalibrated one."""
    f = fuse_student(peaked("head_down", 0.9), peaked("looking_away", 0.4))
    assert f.agree is False and f.contested is True
    assert f.cue is None
    assert "head_down" in f.note and "looking_away" in f.note


def test_missing_vlm_opinion_falls_back_to_temporal():
    """The VLM is slower than the tracker; a gap must not blank a student."""
    f = fuse_student(peaked("screen_oriented"), None)
    assert f.cue == "screen_oriented"
    assert f.agree is None and f.contested is False
    assert f.vlm_cue is None


def test_pool_produces_a_distribution_and_can_overturn():
    f = fuse_student(peaked("head_down", 0.45), peaked("phone_use", 0.95),
                     policy=Policy.POOL, temporal_weight=0.5)
    assert f.fused is not None
    assert pytest.approx(sum(f.fused), abs=1e-9) == 1.0
    assert f.cue == "phone_use"
    assert f.contested is True, "disagreement is still reported, not hidden"


def test_pool_weight_one_is_the_temporal_model():
    f = fuse_student(peaked("head_down", 0.5), peaked("phone_use", 0.99),
                     policy=Policy.POOL, temporal_weight=1.0)
    assert f.cue == "head_down"


def test_product_of_experts_agrees_with_itself():
    f = fuse_student(peaked("uncertain"), peaked("uncertain"),
                     policy=Policy.PRODUCT)
    assert f.cue == "uncertain" and f.agree is True


def test_product_handles_disjoint_support():
    a = np.zeros(NUM_CUES); a[0] = 1.0
    b = np.zeros(NUM_CUES); b[1] = 1.0
    f = fuse_student(a, b, policy=Policy.PRODUCT)
    assert f.cue is None and f.contested is True


# --------------------------------------------------------------------------
# validation
# --------------------------------------------------------------------------

@pytest.mark.parametrize("bad", [
    [0.5] * (NUM_CUES - 1),                 # wrong length
    [0.0] * NUM_CUES,                       # sums to zero
    [-1.0] + [1.0] * (NUM_CUES - 1),        # negative
])
def test_malformed_scores_are_rejected(bad):
    with pytest.raises(ValueError):
        fuse_student(bad, peaked("uncertain"))


def test_unnormalised_scores_are_normalised_not_rejected():
    f = fuse_student(peaked("phone_use") * 7.0, None)
    assert f.cue == "phone_use"
    assert 0.0 <= f.temporal_conf <= 1.0


# --------------------------------------------------------------------------
# frame-level accounting
# --------------------------------------------------------------------------

def test_agreement_summary_excludes_students_the_vlm_never_saw():
    """Counting them as agreeing would inflate the rate, which is the number
    a reader would quote."""
    fused = fuse_frame(
        {1: peaked("phone_use"), 2: peaked("head_down"), 3: peaked("uncertain")},
        {1: peaked("phone_use"), 2: peaked("looking_away")},   # no opinion on 3
    )
    s = agreement_summary(fused)
    assert s["n_students"] == 3
    assert s["n_compared"] == 2, "student 3 must not count"
    assert s["n_agree"] == 1 and s["n_contested"] == 1
    assert s["rate"] == pytest.approx(0.5)


def test_agreement_rate_is_none_when_nothing_was_compared():
    s = agreement_summary(fuse_frame({1: peaked("phone_use")}, {}))
    assert s["rate"] is None, "0/0 must not be reported as 0.0 agreement"


def test_to_json_is_serialisable_and_keeps_both_opinions():
    import json
    f = fuse_student(peaked("head_down"), peaked("looking_away"))
    d = f.to_json()
    json.dumps(d)
    assert d["temporal"]["cue"] == "head_down"
    assert d["vlm"]["cue"] == "looking_away"
    assert d["cue"] is None and d["contested"] is True


# --------------------------------------------------------------------------
# stub backend
# --------------------------------------------------------------------------

def test_stub_grounder_is_deterministic_and_well_formed():
    """A stub returning noise would make a disagreement rate look measured."""
    frame = np.zeros((200, 200, 3), np.uint8)
    boxes = [[10, 10, 90, 90], [100, 10, 190, 90]]
    g = StubGrounder(cue_ids=[CUE_TO_ID["phone_use"], CUE_TO_ID["head_down"]])
    a = g.score_students(frame, boxes)
    b = g.score_students(frame, boxes)
    assert np.array_equal(a, b)
    assert a.shape == (2, NUM_CUES)
    assert np.allclose(a.sum(1), 1.0)
    assert CUE_CLASSES[int(a[0].argmax())] == "phone_use"
    assert CUE_CLASSES[int(a[1].argmax())] == "head_down"


def test_uniform_rows_are_no_opinion():
    r = uniform_rows(3)
    assert r.shape == (3, NUM_CUES)
    assert np.allclose(r, 1.0 / NUM_CUES)
