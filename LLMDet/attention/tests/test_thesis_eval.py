"""Unit tests for the unified evaluation stack.

Every assertion uses a small synthetic case whose answer is derivable by hand.
The project's history is a sequence of evaluation bugs that produced plausible
numbers (per-shard macro-F1, mean-of-batch-means accuracy, per_frame defaulting
to False, crop-vs-frame feature extraction), so the tests target exactly the
properties those bugs violated: class ordering, padding invariance, batch-size
invariance, and metrics computed over the whole set rather than per chunk.

Run from LLMDet/:
    python -m pytest attention/tests/test_thesis_eval.py -q
"""

import numpy as np
import pytest
import torch

from attention.events import Episode
from attention.taxonomy import CUE_CLASSES
from attention.thesis_eval import bootstrap as B
from attention.thesis_eval import data as D
from attention.thesis_eval import metrics as M
from attention.thesis_eval import segmentation as S
from attention.thesis_eval.models import boundary_targets_from_labels, build_model


# ---------------------------------------------------------------- feature slicing

def test_feature_block_layout_is_contiguous_and_complete():
    edges = sorted(D.FEATURE_BLOCKS.values())
    assert edges[0][0] == 0
    for (a_lo, a_hi), (b_lo, b_hi) in zip(edges, edges[1:]):
        assert a_hi == b_lo, "feature blocks must tile [0, 570) without gaps"
    assert edges[-1][1] == 570


@pytest.mark.parametrize("name,dim", [
    ("552_base", 552), ("556_hp", 556), ("563_expr", 563),
    ("563_dyn", 563), ("570_full", 570)])
def test_config_dims(name, dim):
    assert D.config_dim(name) == dim
    assert len(D.column_index(name)) == dim


def test_563_variants_select_different_columns():
    e, d = D.column_index("563_expr"), D.column_index("563_dyn")
    assert len(e) == len(d) == 563
    assert not np.array_equal(e, d), "the two 563 rungs must not be the same slice"
    # both keep base+headpose, and differ exactly by express vs dynamic
    assert np.array_equal(e[:556], d[:556])
    assert set(e[556:]) == set(range(556, 563))
    assert set(d[556:]) == set(range(563, 570))


def test_headpose_decomposition_partitions_the_headpose_block():
    """553_facefound and 555_angles must split [552,556) with no overlap."""
    ff = D.column_index("553_facefound")
    an = D.column_index("555_angles")
    base = D.column_index("552_base")
    assert D.config_dim("553_facefound") == 553 and D.config_dim("555_angles") == 555
    assert set(ff) - set(base) == {555}
    assert set(an) - set(base) == {552, 553, 554}
    assert (set(ff) | set(an)) - set(base) == set(range(552, 556))
    assert (set(ff) & set(an)) == set(base), "the two halves must not overlap"


def test_legacy_remap_folds_idle_other_and_uncertain():
    assert D.LEGACY_REMAP_LUT[5] == 5 and D.LEGACY_REMAP_LUT[6] == 5
    assert list(D.LEGACY_REMAP_LUT[:5]) == [0, 1, 2, 3, 4]


# ---------------------------------------------------------------- frame metrics

def test_confusion_matrix_orientation():
    y = np.array([0, 0, 1]); p = np.array([0, 1, 1])
    cm = M.confusion_matrix(y, p, 2)
    assert cm[0, 0] == 1 and cm[0, 1] == 1 and cm[1, 1] == 1 and cm[1, 0] == 0


def test_per_class_prf_hand_computed():
    # class 0: tp=2 fp=1 fn=1 -> p=2/3 r=2/3 f=2/3
    cm = np.array([[2, 1], [1, 3]])
    prf = M.per_class_prf(cm)
    assert prf["precision"][0] == pytest.approx(2 / 3)
    assert prf["recall"][0] == pytest.approx(2 / 3)
    assert prf["f1"][0] == pytest.approx(2 / 3)
    assert list(prf["support"]) == [3, 4]


def test_macro_f1_ignores_absent_classes_by_default():
    cm = np.zeros((6, 6), dtype=np.int64)
    cm[0, 0] = 10
    cm[1, 1] = 10
    assert M.macro_f1(cm, present_only=True) == pytest.approx(1.0)
    assert M.macro_f1(cm, present_only=False) == pytest.approx(2 / 6)


def test_balanced_accuracy_is_mean_recall_not_accuracy():
    # 90 of class 0 all correct, 10 of class 1 all wrong
    cm = np.array([[90, 0], [10, 0]])
    assert M.accuracy(cm) == pytest.approx(0.9)
    assert M.balanced_accuracy(cm) == pytest.approx(0.5)


def test_accuracy_is_not_a_mean_of_batch_means():
    """The March bug: averaging per-batch accuracy over unequal batches.

    One wrong frame in 100. Computed over the whole set that is 0.99; averaged
    over a 2-frame batch and a 98-frame batch it is not.
    """
    y = np.array([0] * 99 + [1])
    p = np.zeros(100, dtype=int)
    assert M.accuracy(M.confusion_matrix(y, p, 2)) == pytest.approx(0.99)
    uneven = np.mean([(p[:2] == y[:2]).mean(), (p[2:] == y[2:]).mean()])
    assert uneven != pytest.approx(0.99), \
        "a mean of unequal batch means must differ from the true accuracy"


def test_auprc_and_auroc_on_perfect_ranker():
    probs = np.array([[0.1, 0.9], [0.2, 0.8], [0.9, 0.1], [0.8, 0.2]])
    y = np.array([1, 1, 0, 0])
    r = M.ranking_metrics(probs, y, 2)
    assert r["auprc"][1] == pytest.approx(1.0)
    assert r["auroc"][1] == pytest.approx(1.0)
    assert r["auroc"][0] == pytest.approx(1.0)


def test_auroc_of_random_constant_scores_is_half():
    probs = np.full((100, 2), 0.5)
    y = np.array([0, 1] * 50)
    assert M.ranking_metrics(probs, y, 2)["auroc"][1] == pytest.approx(0.5)


def test_perfectly_calibrated_predictions_have_zero_ece():
    rng = np.random.default_rng(0)
    p1 = rng.uniform(0.5, 1.0, 20000)
    y = (rng.uniform(size=20000) < p1).astype(int)
    probs = np.stack([1 - p1, p1], 1)
    assert M.expected_calibration_error(probs, y, n_bins=15) < 0.02


def test_overconfident_predictions_have_large_ece():
    n = 10000
    probs = np.stack([np.full(n, 0.01), np.full(n, 0.99)], 1)
    y = np.array([1] * (n // 2) + [0] * (n // 2))
    assert M.expected_calibration_error(probs, y) == pytest.approx(0.49, abs=0.01)


def test_brier_and_nll_bounds():
    probs = np.array([[0.0, 1.0], [1.0, 0.0]])
    y = np.array([1, 0])
    assert M.brier_score(probs, y) == pytest.approx(0.0)
    assert M.negative_log_likelihood(probs, y) == pytest.approx(0.0, abs=1e-9)
    probs_bad = np.array([[1.0, 0.0], [0.0, 1.0]])
    assert M.brier_score(probs_bad, y) == pytest.approx(2.0)


def test_reliability_bins_cover_all_frames():
    rng = np.random.default_rng(1)
    conf = rng.uniform(0.2, 1.0, 500)
    correct = (rng.uniform(size=500) < conf).astype(float)
    rows = M.reliability_bins(conf, correct, 15)
    assert sum(r["count"] for r in rows) == 500


def test_frame_metrics_uses_taxonomy_class_order():
    y = np.arange(6); probs = np.eye(6)[y]
    res = M.frame_metrics(probs, y)
    assert res["class_order"] == list(CUE_CLASSES)
    assert list(res["per_class"].keys()) == list(CUE_CLASSES)
    assert res["macro_f1"] == pytest.approx(1.0)


# ---------------------------------------------------------------- segmentation

def test_label_segments_run_length_encoding():
    assert S.label_segments([0, 0, 1, 1, 1, 0]) == [(0, 0, 2), (1, 2, 5), (0, 5, 6)]
    assert S.label_segments([]) == []
    assert S.label_segments([3]) == [(3, 0, 1)]


def test_segment_iou_hand_computed():
    assert S.segment_iou((1, 0, 10), (1, 5, 15)) == pytest.approx(5 / 15)
    assert S.segment_iou((1, 0, 10), (1, 0, 10)) == pytest.approx(1.0)
    assert S.segment_iou((1, 0, 5), (1, 5, 10)) == pytest.approx(0.0)


def test_segmental_f1_perfect_and_fragmented():
    gt = [0] * 10 + [1] * 10
    assert S.segmental_f1(gt, gt, 0.5)["f1"] == pytest.approx(1.0)
    # same frames, but the second segment is chopped in two by one wrong frame
    frag = [0] * 10 + [1] * 4 + [0] + [1] * 5
    r = S.segmental_f1(gt, frag, 0.5)
    assert r["n_pred_segments"] == 4 and r["f1"] < 1.0


def test_segmental_f1_threshold_sensitivity():
    gt = [0] * 10 + [1] * 10
    pred = [0] * 10 + [1] * 3 + [0] * 7        # predicted seg covers 3/10 of gt
    assert S.segmental_f1(gt, pred, 0.1)["tp"] == 2   # both classes match loosely
    assert S.segmental_f1(gt, pred, 0.5)["tp"] < 2    # class-1 segment fails at 0.5


def test_edit_score_perfect_is_100_and_fragmentation_penalised():
    gt = [0] * 10 + [1] * 10
    assert S.edit_score(gt, gt) == pytest.approx(100.0)
    frag = [0] * 10 + [1] * 4 + [0] + [1] * 5
    assert 0 <= S.edit_score(gt, frag) < 100.0


def test_edit_score_ignores_segment_length():
    """Edit score is over the label *sequence*, so lengths must not matter."""
    assert S.edit_score([0] * 5 + [1] * 5, [0] * 2 + [1] * 8) == pytest.approx(100.0)


def test_temporal_iou_and_zero_length_markers():
    a = Episode("x", 0.0, 10.0); b = Episode("x", 5.0, 15.0)
    assert S.temporal_iou(a, b) == pytest.approx(5 / 15)
    m1 = Episode("r", 4.0, 4.0); m2 = Episode("r", 4.0, 4.0)
    assert S.temporal_iou(m1, m2) == pytest.approx(1.0)
    assert S.temporal_iou(m1, Episode("r", 5.0, 5.0)) == pytest.approx(0.0)


def test_match_episodes_is_one_to_one_and_channel_aware():
    gt = [Episode("a", 0, 10), Episode("b", 0, 10)]
    pred = [Episode("a", 0, 10), Episode("a", 1, 11)]
    m = S.match_episodes(pred, gt, 0.3)
    assert len(m) == 1, "two predictions cannot both take one gold event"
    assert m[0][1] == 0, "channel b must never match a channel-a prediction"


def test_dedup_removes_inactivity_alias_and_exact_duplicates():
    eps = [Episode("head_down", 0, 5), Episode("inactivity", 0, 5),
           Episode("return_to_task", 9, 9), Episode("return_to_task", 9, 9)]
    out = S.dedup_episodes(eps)
    assert [e.channel for e in out] == ["head_down", "return_to_task"]


def test_duplicate_zero_length_gold_marker_is_unmatchable_without_dedup():
    """Documents the defect: greedy one-to-one matching cannot cover a gold
    event that appears twice, so one is a guaranteed miss."""
    gt = [Episode("r", 9, 9), Episode("r", 9, 9)]
    pred = [Episode("r", 9, 9)]
    assert len(S.match_episodes(pred, gt, 0.3)) == 1
    assert len(S.match_episodes(pred, S.dedup_episodes(gt), 0.3)) == 1
    assert len(S.dedup_episodes(gt)) == 1


def test_event_metrics_precision_recall_and_false_alerts():
    gt = [Episode("a", 0, 10), Episode("a", 100, 110)]
    pred = [Episode("a", 0, 10), Episode("a", 200, 210)]
    r = S.event_metrics(pred, gt, observed_duration_s=3600.0, primary_iou=0.3)
    row = r["by_tiou"]["0.30"]
    assert row["matched"] == 1 and row["missed"] == 1 and row["false_alerts"] == 1
    assert row["precision"] == pytest.approx(0.5)
    assert row["recall"] == pytest.approx(0.5)
    assert row["false_alerts_per_hour"] == pytest.approx(1.0)


def test_boundary_errors_and_detection_delay_signs():
    gt = [Episode("a", 10.0, 20.0)]
    pred = [Episode("a", 12.0, 21.0)]           # 2 s late onset, 1 s late offset
    r = S.event_metrics(pred, gt, 3600.0, primary_iou=0.3)["by_tiou"]["0.30"]
    assert r["onset_mae_s"] == pytest.approx(2.0)
    assert r["offset_mae_s"] == pytest.approx(1.0)
    assert r["duration_mae_s"] == pytest.approx(1.0)
    assert r["detection_delay_s"] == pytest.approx(2.0)
    early = S.event_metrics([Episode("a", 8.0, 20.0)], gt, 3600.0)["by_tiou"]["0.30"]
    assert early["detection_delay_s"] == pytest.approx(0.0), "early alerts are not delays"


def test_common_subset_boundaries_restricts_to_shared_matches():
    gt = [Episode("a", 0, 10), Episode("a", 100, 110)]
    lo = [Episode("a", 0, 10)]                       # finds the easy one only
    hi = [Episode("a", 0, 10), Episode("a", 90, 115)]  # also finds a sloppy second
    out = S.common_subset_boundaries({"lo": lo, "hi": hi}, gt, 0.3)
    assert out["n_common_gold_events"] == 1
    assert out["per_system"]["lo"]["onset_mae_s"] == pytest.approx(0.0)
    assert out["per_system"]["hi"]["onset_mae_s"] == pytest.approx(0.0), \
        "the extra hard event must not pollute the like-for-like comparison"
    full = S.event_metrics(hi, gt, 3600.0)["by_tiou"]["0.30"]
    assert full["onset_mae_s"] > 0.0, "unrestricted boundary error does include it"


def test_segmentation_metrics_over_tracks_rejects_length_mismatch():
    with pytest.raises(ValueError):
        S.segmentation_metrics_over_tracks({"a": [0, 0]}, {"a": [0]})


# ---------------------------------------------------------------- bootstrap

def test_cluster_bootstrap_resamples_clusters_not_rows():
    # 4 clusters, cluster 3 is entirely wrong -> the statistic must vary a lot
    cid = np.repeat(np.arange(4), 250)
    y = np.zeros(1000, dtype=int)
    pred = y.copy(); pred[750:] = 1
    r = B.cluster_bootstrap(cid, lambda idx: float((y[idx] == pred[idx]).mean()),
                            n_boot=500, seed=0)
    assert r["point"] == pytest.approx(0.75)
    assert r["ci_low"] < 0.75 < r["ci_high"]
    assert r["n_clusters"] == 4
    # frame-level resampling would give a far tighter interval; assert ours is not
    assert r["ci_high"] - r["ci_low"] > 0.1


def test_paired_bootstrap_detects_no_difference_between_identical_systems():
    cid = np.repeat(np.arange(20), 50)
    y = np.random.default_rng(0).integers(0, 2, 1000)
    f = lambda idx: float((y[idx] == y[idx]).mean())
    r = B.paired_cluster_bootstrap(cid, f, f, n_boot=300, seed=1)
    assert r["difference"] == pytest.approx(0.0)
    assert not r["significant_at_alpha"]


def test_paired_bootstrap_detects_a_real_difference():
    cid = np.repeat(np.arange(20), 50)
    y = np.zeros(1000, dtype=int)
    good, bad = y.copy(), y.copy()
    bad[::2] = 1                                  # bad is wrong half the time
    a = lambda idx: float((y[idx] == good[idx]).mean())
    b = lambda idx: float((y[idx] == bad[idx]).mean())
    r = B.paired_cluster_bootstrap(cid, a, b, n_boot=300, seed=1)
    assert r["difference"] == pytest.approx(0.5)
    assert r["significant_at_alpha"]


def test_seed_summary():
    r = B.seed_summary([0.40, 0.42, 0.44])
    assert r["mean"] == pytest.approx(0.42)
    assert r["n"] == 3 and r["std"] > 0


# ---------------------------------------------------------------- models

@pytest.mark.parametrize("name", ["transformer", "mstcn", "asrf"])
def test_model_output_is_per_frame(name):
    m = build_model(name, input_dim=16, num_classes=6).eval()
    x = torch.randn(2, 7, 16)
    out = m(x, pad_mask=torch.zeros(2, 7, dtype=torch.bool))
    assert out["logits"].shape == (2, 7, 6), "one label per FRAME, not per sequence"


@pytest.mark.parametrize("name", ["transformer", "mstcn", "asrf"])
def test_padding_does_not_change_real_frame_predictions(name):
    """The property a padding-mask bug breaks: appending padding to a batch
    must not change the predictions for the real frames."""
    torch.manual_seed(0)
    m = build_model(name, input_dim=16, num_classes=6).eval()
    x = torch.randn(1, 10, 16)
    with torch.no_grad():
        solo = m(x, pad_mask=torch.zeros(1, 10, dtype=torch.bool))["logits"]
        padded_x = torch.cat([x, torch.randn(1, 6, 16)], dim=1)
        mask = torch.zeros(1, 16, dtype=torch.bool); mask[:, 10:] = True
        both = m(padded_x, pad_mask=mask)["logits"][:, :10]
    assert torch.allclose(solo, both, atol=1e-4), f"{name} leaks padding into real frames"


@pytest.mark.parametrize("name", ["transformer", "mstcn", "asrf"])
def test_batching_is_equivalent_to_single_sequence(name):
    """Batch size must not change any reported metric."""
    torch.manual_seed(0)
    m = build_model(name, input_dim=16, num_classes=6).eval()
    xs = [torch.randn(t, 16) for t in (5, 11, 8)]
    with torch.no_grad():
        solo = [m(x[None], pad_mask=torch.zeros(1, x.shape[0], dtype=torch.bool))["logits"][0]
                for x in xs]
        bx, _, bmask = D.collate([(x.numpy(), np.zeros(x.shape[0], dtype=np.int64)) for x in xs])
        batched = m(bx, pad_mask=bmask)["logits"]
    for i, s in enumerate(solo):
        assert torch.allclose(s, batched[i, : s.shape[0]], atol=1e-4), \
            f"{name}: batch size changes the result for sequence {i}"


def test_boundary_targets_mark_transitions_only():
    y = torch.tensor([[0, 0, 1, 1, 2]])
    b = boundary_targets_from_labels(y)
    assert b.tolist() == [[0.0, 0.0, 1.0, 0.0, 1.0]], "frame 0 is not a transition"


def test_boundary_targets_ignore_padding():
    y = torch.tensor([[0, 1, -100, -100]])
    assert boundary_targets_from_labels(y).tolist() == [[0.0, 1.0, 0.0, 0.0]]


def test_asrf_refine_removes_single_frame_flicker():
    m = build_model("asrf", input_dim=8, num_classes=2).eval()
    # class-0 segment with one flickered frame, and no predicted boundaries
    logits = torch.tensor([[[3.0, 0.0], [3.0, 0.0], [0.0, 3.0],
                            [3.0, 0.0], [3.0, 0.0]]])
    bnd = torch.full((1, 5), -10.0)          # sigmoid ~ 0 -> one segment
    refined = m.refine(logits, bnd, torch.tensor([5]))
    assert refined.tolist() == [[0, 0, 0, 0, 0]]


def test_asrf_refine_respects_predicted_boundaries():
    m = build_model("asrf", input_dim=8, num_classes=2).eval()
    logits = torch.tensor([[[3.0, 0.0], [3.0, 0.0], [0.0, 3.0], [0.0, 3.0]]])
    bnd = torch.tensor([[-10.0, -10.0, 10.0, -10.0]])   # boundary at index 2
    assert m.refine(logits, bnd, torch.tensor([4])).tolist() == [[0, 0, 1, 1]]


# ---------------------------------------------------------------- collate

def test_collate_masks_and_pads_consistently():
    a = (np.ones((3, 4), np.float32), np.array([1, 1, 1]))
    b = (np.ones((5, 4), np.float32) * 2, np.array([2, 2, 2, 2, 2]))
    x, y, mask = D.collate([a, b])
    assert x.shape == (2, 5, 4) and y.shape == (2, 5)
    assert mask[0, 3:].all() and not mask[0, :3].any() and not mask[1].any()
    assert (y[0, 3:] == D.IGNORE_INDEX).all()
    assert (x[0, 3:] == 0).all(), "padded features must be zero, not stale"


# ---------------------------------------------------------------- calibration

from attention.thesis_eval import calibrate as C


def test_temperature_scaling_never_changes_predictions():
    rng = np.random.default_rng(0)
    probs = rng.dirichlet(np.ones(6), size=5000)
    for T in (0.3, 1.0, 2.7, 9.0):
        assert np.array_equal(probs.argmax(1), C.apply_temperature(probs, T).argmax(1))


def test_temperature_above_one_reduces_confidence():
    probs = np.array([[0.9, 0.05, 0.05]])
    assert C.apply_temperature(probs, 2.0).max() < probs.max()
    assert C.apply_temperature(probs, 0.5).max() > probs.max()


def test_fit_temperature_recovers_a_known_overconfidence():
    """Sharpen a calibrated model by a known factor; the fit must undo it."""
    rng = np.random.default_rng(1)
    logits = rng.normal(size=(20000, 4)) * 2.0
    p = np.exp(logits - logits.max(1, keepdims=True))
    p /= p.sum(1, keepdims=True)
    y = np.array([rng.choice(4, p=row) for row in p])
    sharpened = C.apply_temperature(p, 0.5)          # 2x overconfident
    assert C.fit_temperature(sharpened, y) == pytest.approx(2.0, rel=0.15)


def test_fit_temperature_on_calibrated_input_is_about_one():
    rng = np.random.default_rng(2)
    logits = rng.normal(size=(20000, 4)) * 1.5
    p = np.exp(logits - logits.max(1, keepdims=True))
    p /= p.sum(1, keepdims=True)
    y = np.array([rng.choice(4, p=row) for row in p])
    assert C.fit_temperature(p, y) == pytest.approx(1.0, rel=0.15)


def test_temperature_scaling_reduces_ece_on_overconfident_predictions():
    rng = np.random.default_rng(3)
    logits = rng.normal(size=(20000, 4)) * 1.5
    p = np.exp(logits - logits.max(1, keepdims=True))
    p /= p.sum(1, keepdims=True)
    y = np.array([rng.choice(4, p=row) for row in p])
    bad = C.apply_temperature(p, 0.4)
    T = C.fit_temperature(bad, y)
    assert (M.expected_calibration_error(C.apply_temperature(bad, T), y)
            < M.expected_calibration_error(bad, y))


def test_coverage_falls_and_selective_error_improves_as_the_threshold_rises():
    """A confidence-ranked model must trade coverage for accuracy."""
    rng = np.random.default_rng(4)
    n = 4000
    conf = rng.uniform(0.5, 1.0, n)
    correct = rng.uniform(size=n) < conf            # confidence is informative
    probs = np.stack([1 - conf, conf], 1)           # class 1 is the prediction
    y = correct.astype(int)                         # correct <=> truth is class 1
    rows = C.coverage_risk_curve(probs, y)
    cov = [r["coverage"] for r in rows]
    assert cov == sorted(cov, reverse=True), "coverage must fall as the threshold rises"
    assert rows[0]["coverage"] == pytest.approx(1.0)
    lo = next(r for r in rows if r["threshold"] == pytest.approx(0.5))
    hi = next(r for r in rows if r["threshold"] == pytest.approx(0.9))
    assert hi["selective_error"] < lo["selective_error"], \
        "abstaining on low-confidence frames must reduce error on the retained ones"


def test_abstention_maps_low_confidence_to_uncertain():
    probs = np.array([[0.9, 0.05, 0.02, 0.01, 0.01, 0.01],
                      [0.30, 0.28, 0.22, 0.10, 0.05, 0.05]])
    y = np.array([0, 1])
    rows = {r["threshold"]: r for r in C.coverage_risk_curve(probs, y, [0.0, 0.5])}
    assert rows[0.0]["coverage"] == pytest.approx(1.0)
    assert rows[0.5]["coverage"] == pytest.approx(0.5), "only the confident frame is kept"
    # the abstained frame becomes `uncertain`, which is wrong here but explicit
    assert rows[0.5]["abstain_accuracy"] == pytest.approx(0.5)


def test_logits_from_probs_roundtrips_through_softmax():
    rng = np.random.default_rng(5)
    p = rng.dirichlet(np.ones(6), size=100)
    assert np.allclose(C.apply_temperature(p, 1.0), p, atol=1e-9)


# ---------------------------------------------------------------- ordinal (CMOSE)

from attention.thesis_eval import cmose as CM


def test_cmose_subject_parsing():
    assert CM.subject_of("video5_146_person5") == "v5_p5"
    assert CM.subject_of("video12_3_person1") == "v12_p1"
    with pytest.raises(ValueError):
        CM.subject_of("not_a_cmose_clip")


def test_subject_disjoint_split_shares_no_subject():
    subj = np.array([f"s{i // 10}" for i in range(500)])
    y = np.arange(500) % 4
    tr, va, te = CM.subject_disjoint_split(subj, y)
    assert tr.sum() + va.sum() + te.sum() == 500
    assert not (set(subj[tr]) & set(subj[te]))
    assert not (set(subj[tr]) & set(subj[va]))
    assert not (set(subj[va]) & set(subj[te]))


def test_quadratic_weighted_kappa_bounds():
    y = np.array([0, 1, 2, 3] * 25)
    assert CM.quadratic_weighted_kappa(y, y) == pytest.approx(1.0)
    # systematic reversal is worse than chance
    assert CM.quadratic_weighted_kappa(y, 3 - y) < 0.0


def test_qwk_penalises_distant_confusions_more():
    """The property that makes QWK the right ordinal statistic."""
    y = np.array([0] * 50 + [3] * 50)
    near = np.array([1] * 50 + [2] * 50)      # off by 1 and 1
    far = np.array([3] * 50 + [0] * 50)       # off by 3 and 3
    assert CM.quadratic_weighted_kappa(y, near) > CM.quadratic_weighted_kappa(y, far)


def test_ordinal_mae_uses_the_level_distance():
    y = np.array([0, 0]); p = np.array([1, 3])
    assert CM.ordinal_metrics(y, p)["mae"] == pytest.approx(2.0)


def test_spearman_monotone_and_reversed():
    a = np.arange(50)
    assert CM.spearman(a, a * 3.0) == pytest.approx(1.0)
    assert CM.spearman(a, -a * 1.0) == pytest.approx(-1.0)


def test_average_accuracy_is_not_accuracy_under_imbalance():
    """CMOSE is 69% one level, so the two must be reported separately."""
    y = np.array([2] * 90 + [0] * 10)
    p = np.full(100, 2)
    m = CM.ordinal_metrics(y, p)
    assert m["accuracy"] == pytest.approx(0.9)
    assert m["average_accuracy"] == pytest.approx(0.5)


def test_ordinal_metrics_preserve_level_order():
    y = np.arange(4); p = np.arange(4)
    m = CM.ordinal_metrics(y, p)
    assert m["level_order"] == CM.LEVELS
    assert list(m["per_class_f1"].keys()) == CM.LEVELS
    assert m["macro_f1"] == pytest.approx(1.0)


# ---------------------------------------------------------------- striding

from attention.thesis_eval.runtime import StrideController
from attention.tracking import IoUTracker
from attention.detector_adapter import DetectionResult


def test_stride_controller_detect_schedule():
    s = StrideController(detector_stride=5, temporal_stride=3)
    assert [n for n in range(12) if s.should_detect(n)] == [0, 5, 10]
    assert all(StrideController(1, 1).should_detect(n) for n in range(10))


def test_stride_controller_predicts_immediately_for_a_new_track():
    """A newly confirmed track must not wait for the next global tick."""
    s = StrideController(1, 10)
    assert s.should_predict(track_id=7, frame_idx=3)
    s.store(7, 3, {"cue": "head_down"})
    assert not s.should_predict(7, 4)
    assert s.should_predict(7, 13)
    assert s.should_predict(track_id=99, frame_idx=4), "a different track is new"


def test_stride_controller_caches_and_returns_the_held_value():
    s = StrideController(1, 5)
    s.store(1, 0, {"cue": "phone_use"})
    assert s.cached(1) == {"cue": "phone_use"}
    assert s.cached(2) is None


def test_stride_controller_drops_dead_tracks_so_ids_cannot_go_stale():
    s = StrideController(1, 5)
    s.store(1, 0, {"cue": "a"}); s.store(2, 0, {"cue": "b"})
    s.drop_missing([2])
    assert s.cached(1) is None and s.cached(2) == {"cue": "b"}
    # a recycled id must look new, not inherit the old cue
    assert s.should_predict(1, 1)


def test_min_hits_scales_so_confirmation_time_is_constant():
    for stride in (1, 3, 5, 10):
        adj = StrideController(stride, 1).adjusted_min_hits(8)
        assert adj >= 1
        assert abs(stride * adj - 8) <= 4, \
            f"stride {stride}: {stride * adj} real frames to confirm, want ~8"


def test_stride_controller_rejects_zero_and_negative():
    for bad in (0, -1):
        with pytest.raises(ValueError):
            StrideController(bad, 1)
        with pytest.raises(ValueError):
            StrideController(1, bad)


def _det(x1, y1, x2, y2):
    return DetectionResult(bbox_xyxy=[x1, y1, x2, y2], score=0.9, label="student")


def test_coast_returns_confirmed_tracks_without_ageing_them():
    """max_age must count frames we looked at, not frames we skipped."""
    tr = IoUTracker(iou_match_thr=0.3, max_age=2, min_hits=1)
    tr.update([_det(0, 0, 10, 10)])
    tid = next(iter(tr.tracks))
    age_before = tr.ages[tid]      # note: update() ages a track on creation too
    for _ in range(10):
        out = tr.coast()
        assert [t.track_id for t in out] == [tid]
    assert tr.ages[tid] == age_before, "coasting must not age a track"
    # whereas update([]) — looking and finding nothing — does age it away
    for _ in range(4):
        tr.update([])
    assert tid not in tr.tracks


def test_coast_holds_the_last_box_and_hides_unconfirmed_tracks():
    tr = IoUTracker(iou_match_thr=0.3, max_age=5, min_hits=3)
    tr.update([_det(0, 0, 10, 10)])
    assert tr.coast() == [], "a track below min_hits is not published"
    tr.update([_det(0, 0, 10, 10)]); tr.update([_det(1, 1, 11, 11)])
    out = tr.coast()
    assert len(out) == 1 and out[0].bbox_xyxy == [1, 1, 11, 11], "holds the LAST box"
