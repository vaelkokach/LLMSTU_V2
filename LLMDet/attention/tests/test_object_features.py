"""Object-presence features, and the layout that carries them.

Every number these tests assert against comes from the matched-pair validation
in [internal notes, not included] -- 250 pairs of students drawn from the SAME source frame, one
with `phone_visible` and one without. Two earlier versions of that measurement
were wrong (a padded-centre test that fired on 90.8% of students with no phone,
and an unmatched sample whose `y_frac` AUROC of [value removed] was a default-value
artefact), so what is pinned here is specifically the version that survived.
"""
import numpy as np
import pytest

from attention.object_features import (DIMS_PER_OBJECT, OBJECT_DIM,
                                       OBJECT_PROMPTS, student_object_features)
from attention.thesis_eval import data as D

PERSON = [100.0, 100.0, 200.0, 300.0]        # 100 wide, 200 tall
NONE = (np.zeros((0, 4)), np.zeros(0))


def _obj(cx, cy, score, half=5.0):
    return (np.array([[cx - half, cy - half, cx + half, cy + half]]),
            np.array([score]))


def test_dim_matches_the_layout_block():
    lo, hi = D.LAYOUTS["v1080_obj"]["objects"]
    assert hi - lo == OBJECT_DIM == len(OBJECT_PROMPTS) * DIMS_PER_OBJECT


def test_a_low_phone_scores_far_above_a_high_one():
    """The whole signal. A held phone sits at hand/desk height (measured [value removed]
    of the way down the student's box); the detector's false positives on
    students without phones cluster at [value removed] -- head and shoulders."""
    low = student_object_features([_obj(150, 240, 0.30), NONE], PERSON)
    high = student_object_features([_obj(150, 115, 0.30), NONE], PERSON)
    assert low[0] == pytest.approx(high[0]), "same detector score"
    assert low[1] > 0.6 and high[1] < 0.2
    assert low[2] > 5 * high[2], "score*y_frac is what separates them"


def test_absence_is_zero_everywhere():
    f = student_object_features([NONE, NONE], PERSON)
    assert f.shape == (OBJECT_DIM,)
    assert not f.any()


def test_absence_is_distinguishable_from_a_phone_at_the_top():
    """y_frac alone cannot tell "nothing found" from "found at the very top" --
    both are 0. The score column is what disambiguates, which is why it is
    first and is never dropped. Getting this wrong is how the [value removed] artefact
    happened."""
    nothing = student_object_features([NONE, NONE], PERSON)
    at_top = student_object_features([_obj(150, 100, 0.30), NONE], PERSON)
    assert nothing[1] == at_top[1] == 0.0
    assert nothing[0] == 0.0 and at_top[0] > 0.0


def test_objects_outside_the_box_are_ignored_with_no_padding():
    """A 15% pad reaches the neighbour in a classroom this dense and made the
    feature fire on 90.8% of students who had no phone [internal notes, not included]."""
    just_outside = student_object_features([_obj(210, 240, 0.9), NONE], PERSON)
    assert not just_outside.any()


def test_the_highest_scoring_contained_object_wins():
    boxes = np.array([[145, 130, 155, 140], [145, 250, 155, 260]])
    scores = np.array([0.20, 0.60])
    f = student_object_features([(boxes, scores), NONE], PERSON)
    assert f[0] == pytest.approx(0.60)
    assert f[1] > 0.6, "the winner's position, not the other one's"


def test_below_threshold_detections_do_not_count():
    from attention.object_features import MIN_SCORE
    f = student_object_features([_obj(150, 240, MIN_SCORE / 2), NONE], PERSON)
    assert not f.any()


def test_the_two_objects_occupy_separate_columns():
    phone = student_object_features([_obj(150, 240, 0.4), NONE], PERSON)
    laptop = student_object_features([NONE, _obj(150, 240, 0.4)], PERSON)
    assert phone[:DIMS_PER_OBJECT].any() and not phone[DIMS_PER_OBJECT:].any()
    assert laptop[DIMS_PER_OBJECT:].any() and not laptop[:DIMS_PER_OBJECT].any()


# --------------------------------------------------------------------------
# the layout
# --------------------------------------------------------------------------

def test_v1080_extends_v1074_without_moving_a_column():
    """An object build must serve every existing config unchanged, exactly as
    v1074_head serves 556_hp. A column that moved would be silent."""
    for blk in ("base", "headpose", "hp_angles", "hp_facefound", "head"):
        assert D.LAYOUTS["v1080_obj"][blk] == D.LAYOUTS["v1074_head"][blk]
    assert D.LAYOUT_WIDTH["v1080_obj"] == 1080


@pytest.mark.parametrize("config", ["556_hp", "553_facefound", "1074_hp_head"])
def test_existing_configs_read_a_v1080_build(config):
    stored_layout = D.config_layout(config)
    assert D.layout_is_compatible(stored_layout, "v1080_obj",
                                  D.FEATURE_CONFIGS[config])


def test_object_configs_are_refused_by_older_builds():
    """The converse, and the one that matters: a config reading `objects` must
    NOT be servable from a build that has no object columns."""
    for older in ("v570", "v1074_head"):
        assert not D.layout_is_compatible(
            "v1080_obj", older, D.FEATURE_CONFIGS["1080_hp_head_obj"])


def test_object_config_dims():
    assert D.config_dim("1080_hp_head_obj") == 1080
    assert D.config_dim("562_obj") == 552 + 4 + OBJECT_DIM


# --------------------------------------------------------------------------
# the deploy config must be the same function as the one that built the cache
#
# The object columns were precomputed over 283,913 crops with the LMM
# constructed; the live path runs `grounding_dino_swin_t_original_deploy.py`,
# which sets `lmm=None` to avoid a 0.5B model and a 3.5 GB SigLIP read that
# `predict()` never touches. If the two disagreed by even one score, a
# checkpoint trained on cached columns would be served from
# differently-produced ones -- train/serve skew INSIDE a named layout, which is
# the failure the layouts exist to prevent and which nothing downstream could
# detect.
#
# Slow and needs the 1.1 GB checkpoint, so it is opt-in:
#     LLMSTU_SLOW_TESTS=1 python -m pytest attention/tests/test_object_features.py -k lmm
# Measured 2026-09-12 on 8 frames x 3 boxes: byte-identical, max abs diff 0.0
# [internal notes, not included].
# --------------------------------------------------------------------------

OBJ_CKPT = ("../huggingface/mm_grounding_dino/grounding_dino_swin-t_pretrain"
            "_obj365_goldg_grit9m_v3det_20231204_095047-b448804b.pth")


@pytest.mark.skipif(
    not __import__("os").environ.get("LLMSTU_SLOW_TESTS"),
    reason="loads two 1.1 GB detectors; set LLMSTU_SLOW_TESTS=1")
def test_lmm_none_gives_identical_object_features():
    import glob
    from pathlib import Path

    import cv2
    pytest.importorskip("mmdet")
    from attention.object_features import ObjectDetector

    root = Path(__file__).resolve().parents[2]          # LLMDet/
    if not (root / OBJ_CKPT).exists():
        pytest.skip(f"{OBJ_CKPT} not on this host")
    frames = sorted(glob.glob(str(root / ".." / "grounding_data" /
                                  "stu_img" / "frames" / "*.jpg")))[:8]
    if not frames:
        pytest.skip("no frames on this host")

    imgs = [cv2.imread(f) for f in frames]

    def boxes_for(img):
        h, w = img.shape[:2]
        return [[w * i / 3, h * 0.2, w * (i + 1) / 3, h * 0.95] for i in range(3)]

    out = {}
    for tag, cfg in (("with_lmm", "configs/grounding_dino_swin_t_original.py"),
                     ("lmm_none",
                      "configs/grounding_dino_swin_t_original_deploy.py")):
        det = ObjectDetector(str(root / cfg), str(root / OBJ_CKPT),
                             device="cuda:0")
        out[tag] = np.concatenate([det.features(im, boxes_for(im))
                                   for im in imgs], axis=0)
        det._model = None

    a, b = out["with_lmm"], out["lmm_none"]
    # Non-trivial: an all-zero comparison would pass whatever the configs did.
    assert (a != 0).any(axis=1).sum() >= len(a) // 2, \
        "the detector found nothing, so this compares two empty answers"
    assert np.array_equal(a, b), f"max abs diff {np.abs(a - b).max():.6g}"
