"""Partial-label learning: candidate sets and the PRODEN objective.

The claim being protected is narrow and load-bearing: turning on
``--partial-labels`` must change the loss ONLY on frames whose annotation
genuinely supports more than one cue. If it also moved unambiguous frames, any
measured gain would be uninterpretable -- it could be the partial labels, or it
could be an accidentally different objective everywhere.
"""
from __future__ import annotations

import pytest

torch = pytest.importorskip("torch")
F = pytest.importorskip("torch.nn.functional")

from attention.taxonomy import (CUE_CLASSES, CUE_TO_ID, candidate_set,
                                cue_conditions, map_record)
from attention.thesis_eval.train import proden_loss

K = len(CUE_CLASSES)


# --------------------------------------------------------------------------
# candidate_set / map_record share one definition
# --------------------------------------------------------------------------

RECORDS = [
    # unambiguous: a student listening at the board
    dict(activity="listening", gaze_direction="teacher_or_board",
         attention_target="instruction", engagement_level="engaged",
         posture="upright", hand_state="unknown"),
    # slumped AND distracted -> head_down wins, looking_away is suppressed
    dict(activity="other", gaze_direction="unknown",
         attention_target="distracted", engagement_level="disengaged",
         posture="slumped", hand_state="unknown"),
    # phone beats everything below it
    dict(activity="using_phone", gaze_direction="phone",
         attention_target="distracted", engagement_level="disengaged",
         posture="upright", hand_state="on_phone", phone_visible=True),
    # unverifiable: the gate
    dict(activity="other", gaze_direction="unknown", attention_target="unknown",
         engagement_level="unknown", posture="unknown", hand_state="unknown",
         occluded=True, face_kpts=1),
]


@pytest.mark.parametrize("rec", RECORDS)
def test_precedence_winner_is_always_a_candidate(rec):
    """Otherwise the single-label target would sit outside the candidate set
    and PRODEN would put zero weight on the very class CE trains toward."""
    assert map_record(rec) in candidate_set(rec)


def test_candidate_set_is_never_empty():
    assert candidate_set({}) == [CUE_TO_ID["uncertain"]]


def test_unverifiable_gate_admits_no_other_candidate():
    """A student who cannot be seen supports no positive cue."""
    rec = dict(RECORDS[3])
    assert candidate_set(rec) == [CUE_TO_ID["uncertain"]]


def test_suppressed_class_is_recovered_as_a_candidate():
    """The measured failure mode: slumped + distracted is labelled head_down,
    and looking_away -- true by its own rule -- is discarded."""
    rec = RECORDS[1]
    assert map_record(rec) == CUE_TO_ID["head_down"]
    assert CUE_TO_ID["looking_away"] in candidate_set(rec)


def test_map_record_is_the_first_firing_condition():
    for rec in RECORDS:
        fired = [n for n, hit in cue_conditions(rec) if hit]
        assert CUE_CLASSES[map_record(rec)] == (fired[0] if fired else "uncertain")


# --------------------------------------------------------------------------
# PRODEN
# --------------------------------------------------------------------------

def test_singleton_candidates_reduce_to_cross_entropy():
    """The whole design rests on this: unambiguous frames must be untouched."""
    torch.manual_seed(0)
    logits = torch.randn(256, K) * 2
    y = torch.randint(0, K, (256,))
    valid = torch.ones(256, dtype=torch.bool)
    got = proden_loss(logits, F.one_hot(y, K).float(), valid)
    assert torch.allclose(got, F.cross_entropy(logits, y), atol=1e-6)


def test_singleton_candidates_reduce_to_cross_entropy_with_class_weights():
    """Compared against F.cross_entropy itself, NOT a hand-rolled formula.

    The first version of this test asserted against sum(w*ce)/N -- my own
    arithmetic rather than the function PRODEN stands in for. Torch's weighted
    mean divides by the SUM OF TARGET WEIGHTS, so the loss shipped 2.21x too
    small on the real class histogram and the partial-label runs trained with
    a proportionally weaker gradient. Verifying against the reference
    implementation is the only version of this test worth having.
    """
    torch.manual_seed(1)
    logits = torch.randn(512, K) * 2
    # A realistic, heavily imbalanced target distribution: the bug was
    # invisible at uniform weights and only bit under real class weights.
    counts = torch.tensor([0.762, 0.068, 0.069, 0.022, 0.031, 0.048])
    y = torch.multinomial(counts, 512, replacement=True)
    hist = torch.bincount(y, minlength=K).float()
    inv = 1.0 / hist.clamp(min=1).sqrt()
    w = inv / inv.sum() * K                       # as data.py builds them
    got = proden_loss(logits, F.one_hot(y, K).float(),
                      torch.ones(512, dtype=torch.bool), w)
    want = F.cross_entropy(logits, y, weight=w)
    assert torch.allclose(got, want, atol=1e-5), (got.item(), want.item())


def test_ambiguous_frame_costs_less_than_forcing_the_suppressed_member():
    logits = torch.tensor([[3.0, 2.5, -1.0, -1.0, -1.0, -1.0]])
    cand = torch.tensor([[1.0, 1.0, 0.0, 0.0, 0.0, 0.0]])
    valid = torch.ones(1, dtype=torch.bool)
    assert proden_loss(logits, cand, valid) < F.cross_entropy(
        logits, torch.tensor([1]))


def test_loss_is_differentiable():
    """It is decorated by nothing: an accidental @torch.no_grad() above this
    function would train the model on a constant. That happened once."""
    logits = torch.randn(8, K, requires_grad=True)
    cand = (torch.rand(8, K) > 0.5).float()
    cand[:, 0] = 1.0
    proden_loss(logits, cand, torch.ones(8, dtype=torch.bool)).backward()
    assert torch.isfinite(logits.grad).all()
    assert (logits.grad != 0).any()


def test_invalid_frames_do_not_contribute():
    torch.manual_seed(2)
    logits = torch.randn(8, K)
    cand = (torch.rand(8, K) > 0.3).float()
    cand[:, 0] = 1.0
    v = torch.zeros(8, dtype=torch.bool)
    v[:3] = True
    # Same first three frames, different tail -> identical loss.
    other = logits.clone()
    other[3:] = torch.randn(5, K) * 10
    assert torch.allclose(proden_loss(logits, cand, v),
                          proden_loss(other, cand, v), atol=1e-6)


# --------------------------------------------------------------------------
# abstained frames must never reach the metrics
# --------------------------------------------------------------------------

def test_metrics_reject_unfiltered_ignore_labels():
    """The evaluator filters IGNORE; if it ever stops, fail loudly here.

    The natural symptom was an IndexError from brier_score's one-hot indexing
    by -100, which points at the wrong place. Worse, a metric that did not
    index by label would have silently averaged over frames the model was
    never asked to predict.
    """
    import numpy as np
    from attention.thesis_eval import metrics as M
    from attention.thesis_eval.data import IGNORE_INDEX

    rng = np.random.default_rng(0)
    y = rng.integers(0, 3, 64)
    y[:5] = IGNORE_INDEX
    p = rng.random((64, 3))
    p /= p.sum(1, keepdims=True)
    with pytest.raises(ValueError, match="IGNORE_INDEX"):
        M.frame_metrics(p, y, num_classes=3, class_names=["a", "b", "c"])


def test_metrics_work_once_ignore_is_filtered():
    import numpy as np
    from attention.thesis_eval import metrics as M
    from attention.thesis_eval.data import IGNORE_INDEX

    rng = np.random.default_rng(1)
    y = rng.integers(0, 3, 64)
    y[:5] = IGNORE_INDEX
    p = rng.random((64, 3))
    p /= p.sum(1, keepdims=True)
    keep = y != IGNORE_INDEX
    res = M.frame_metrics(p[keep], y[keep], num_classes=3,
                          class_names=["a", "b", "c"])
    assert res["n_frames"] == int(keep.sum()) == 59
    assert list(res["per_class"]) == ["a", "b", "c"]
