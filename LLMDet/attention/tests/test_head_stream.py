"""The head stream: a second encoder pass over the head region.

Why it exists, in one measurement. Features are CLIP ViT-B/32 over the PERSON
crop. CLIP resizes any crop to 224x224 and tokenises it in 32x32 patches, a 7x7
grid. A seated student's head is roughly a fifth of the visible crop, so after
the resize it lands on ~1.4 of those 49 patches -- and that is true for every
student regardless of camera distance, because the resize normalises it away.
`looking_away` vs `screen_oriented` is then a decision about where the eyes
point, made from one 32-pixel token. Face detection fires on 92% of
screen_oriented and 90% of looking_away [internal notes, not included]: indistinguishable.

Cropping the head and letting it fill the same 224x224 gives it the whole grid.

These tests run without CLIP (allow_clip_fallback=True zeroes the embedding),
so they check wiring, geometry and column layout -- not embedding quality.
"""
from __future__ import annotations

import numpy as np
import pytest

from attention.features import StudentFeatureExtractor as SFE
from attention.thesis_eval import data as D

W, H = 1918, 1080
#: The median LLMSTU person box: 158x230 px.
BOX = (800, 400, 958, 630)


@pytest.fixture
def stub_clip(monkeypatch):
    """Replace the CLIP forward with a deterministic stub.

    These tests check feature ASSEMBLY -- widths, column offsets, additivity --
    not embedding quality, so running the real encoder only couples them to
    whichever transformers version happens to be installed (4.44.2 returns a
    Tensor from get_image_features, 5.x returns an output object). Stubbing is
    both faster and the honest scope.
    """
    def _fake(self, crops_bgr):
        n = len(crops_bgr)
        out = np.zeros((n, self.clip_dim), dtype=np.float32)
        for i, c in enumerate(crops_bgr):
            if c is not None and c.size:
                out[i] = float(c.shape[0] * c.shape[1] % 97) / 97.0
        return out
    monkeypatch.setattr(SFE, "_clip_batch", _fake)


class _StubHeadPose:
    OUTPUT_DIM = 4

    def available(self):
        return True

    def estimate(self, crop):
        return np.zeros(4, dtype=np.float32)


# --------------------------------------------------------------------------
# geometry
# --------------------------------------------------------------------------

def test_head_box_is_square_and_at_the_top():
    x1, y1, x2, y2 = BOX
    hx1, hy1, hx2, hy2, clipped = SFE._head_box(x1, y1, x2, y2, W, H)
    assert abs((hx2 - hx1) - (hy2 - hy1)) <= 1, "must be square: CLIP resizes to a square"
    assert hy1 == y1, "head sits at the top of the person box"
    assert abs(((hx1 + hx2) / 2) - ((x1 + x2) / 2)) <= 1, "centred horizontally"
    assert not clipped


def test_head_box_is_clamped_to_the_frame_and_flagged():
    hx1, hy1, hx2, hy2, clipped = SFE._head_box(0, 0, 120, 200, W, H)
    assert hx1 >= 0 and hy1 >= 0 and hx2 <= W and hy2 <= H
    assert clipped == 1.0, "a truncated head must be marked, not silently used"


def test_head_box_survives_a_degenerate_person_box():
    hx1, hy1, hx2, hy2, _ = SFE._head_box(10, 10, 11, 11, W, H)
    assert hx2 > hx1 and hy2 > hy1, "never emit an empty crop"


def test_head_box_is_scale_invariant():
    """Same relative geometry for a near and a far student -- the point is that
    the head then fills the encoder input in both cases."""
    near = SFE._head_box(0, 0, 400, 600, W, H)
    far = SFE._head_box(0, 0, 100, 150, W, H)
    rn = (near[3] - near[1]) / 600
    rf = (far[3] - far[1]) / 150
    assert abs(rn - rf) < 0.01


def test_head_crop_gives_the_head_far_more_encoder_tokens():
    """The claim the whole stream rests on, made numerically."""
    x1, y1, x2, y2 = BOX
    bh = y2 - y1
    head_px = 0.2 * bh                       # a head is ~1/5 of a seated crop
    _, hy1, _, hy2, _ = SFE._head_box(x1, y1, x2, y2, W, H)
    patches_person = (head_px * (224.0 / bh)) / 32.0
    patches_head = (head_px * (224.0 / (hy2 - hy1))) / 32.0
    assert patches_person < 2.0, "the head is ~1 patch in a person crop"
    assert patches_head / patches_person > 3.0, "head crop must be a large gain"


# --------------------------------------------------------------------------
# feature assembly
# --------------------------------------------------------------------------

@pytest.mark.parametrize("head_pose,head_stream,want", [
    (False, False, 552),
    (False, True, 552 + 518),
    (True, True, 552 + 4 + 518),
])
def test_output_dim(head_pose, head_stream, want):
    e = SFE(allow_clip_fallback=True,
            head_pose=_StubHeadPose() if head_pose else None,
            head_stream=head_stream)
    assert e.output_dim() == want


def test_head_stream_dim_is_declared_once():
    """output_dim and the layout block must agree, or slicing misaligns."""
    lo, hi = D.LAYOUTS["v1074_head"]["head"]
    assert hi - lo == SFE.HEAD_STREAM_DIM


def test_head_stream_is_strictly_additive(stub_clip):
    """The first 552 columns must be untouched, or every published number that
    used them silently means something else."""
    frame = np.random.default_rng(0).integers(0, 255, (H, W, 3), dtype=np.uint8)
    boxes = [list(BOX), [1200, 380, 1400, 660]]
    a = SFE(allow_clip_fallback=True, head_stream=False).extract_batch(frame, boxes)
    b = SFE(allow_clip_fallback=True, head_stream=True).extract_batch(frame, boxes)
    assert a.shape[1] == 552 and b.shape[1] == 1070
    assert np.array_equal(a, b[:, :552])


def test_degenerate_boxes_still_yield_a_full_width_row(stub_clip):
    frame = np.zeros((H, W, 3), dtype=np.uint8)
    out = SFE(allow_clip_fallback=True, head_stream=True).extract_batch(
        frame, [[10, 10, 10, 10], list(BOX)])
    assert out.shape == (2, 1070)


# --------------------------------------------------------------------------
# column layout
# --------------------------------------------------------------------------

def test_head_configs_use_the_head_layout():
    for name in ("1070_head", "1074_hp_head"):
        assert D.config_layout(name) == "v1074_head"
    for name in ("552_base", "556_hp", "570_full", "553_facefound"):
        assert D.config_layout(name) == "v570"


def test_layout_blocks_do_not_overlap_and_dims_agree():
    for layout, blocks in D.LAYOUTS.items():
        width = D.LAYOUT_WIDTH[layout]
        for lo, hi in blocks.values():
            assert 0 <= lo < hi <= width, f"{layout}: block outside [0,{width})"
    for name in D.FEATURE_CONFIGS:
        cols = D.column_index(name)
        assert len(cols) == D.config_dim(name)
        assert len(set(cols.tolist())) == len(cols), f"{name}: duplicate columns"
        assert cols.max() < D.LAYOUT_WIDTH[D.config_layout(name)]


def test_v570_configs_are_unchanged_by_the_head_layout():
    """Regression guard: adding a layout must not move an existing config's
    columns, or every archived checkpoint reads the wrong features."""
    assert D.column_index("556_hp").tolist() == list(range(556))
    assert D.column_index("552_base").tolist() == list(range(552))
    assert D.column_index("553_facefound").tolist() == list(range(552)) + [555]
    assert D.column_index("570_full").tolist() == list(range(570))


# --------------------------------------------------------------------------
# cross-layout slicing
# --------------------------------------------------------------------------

def test_v570_configs_readable_from_a_head_build():
    """556_hp and friends must slice out of a v1074_head build.

    Both layouts define base and the head-pose blocks at identical columns, so
    the numbers are the same. This is what lets --partial-labels (which needs
    y_cand, hence the new build) run WITHOUT the head stream, so the two
    contributions can be attributed separately.
    """
    for name in ("552_base", "553_facefound", "555_angles", "556_hp"):
        assert D.layout_is_compatible(
            D.config_layout(name), "v1074_head", D.FEATURE_CONFIGS[name]), name


def test_express_and_dynamic_configs_are_refused_by_a_head_build():
    """Those blocks do not exist in v1074_head; the head block occupies 556+.
    Allowing them would read head-CLIP dims as facial expression."""
    for name in ("563_expr", "563_dyn", "570_full"):
        assert not D.layout_is_compatible(
            D.config_layout(name), "v1074_head", D.FEATURE_CONFIGS[name]), name


def test_head_configs_are_refused_by_a_v570_build():
    for name in ("1070_head", "1074_hp_head"):
        assert not D.layout_is_compatible(
            D.config_layout(name), "v570", D.FEATURE_CONFIGS[name]), name


def test_compatibility_compares_columns_not_names():
    """A block present in both layouts but at different columns must fail."""
    saved = D.LAYOUTS["v1074_head"]["base"]
    try:
        D.LAYOUTS["v1074_head"]["base"] = (8, 560)      # same name, moved
        assert not D.layout_is_compatible("v570", "v1074_head", ["base"])
    finally:
        D.LAYOUTS["v1074_head"]["base"] = saved
    assert D.layout_is_compatible("v570", "v1074_head", ["base"])
