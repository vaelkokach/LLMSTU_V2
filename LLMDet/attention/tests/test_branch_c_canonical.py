"""Tests for seat-relative pose canonicalisation.

The invariance test and the causality test are the two that matter. Everything
else guards the numerics that make them meaningful.
"""

from __future__ import annotations

import math

import pytest
import torch

from attention.branch_c import canonical as C


def random_rotations(n: int, seed: int = 0) -> torch.Tensor:
    """Uniform-ish random rotations via QR of a Gaussian, det forced to +1."""
    g = torch.Generator().manual_seed(seed)
    A = torch.randn(n, 3, 3, generator=g, dtype=torch.float64)
    Q, R = torch.linalg.qr(A)
    Q = Q * torch.sign(torch.diagonal(R, dim1=-2, dim2=-1)).unsqueeze(-2)
    flip = torch.det(Q) < 0
    Q[flip, :, 0] = -Q[flip, :, 0]
    return Q


# ---------------------------------------------------------------------------
# The core claim
# ---------------------------------------------------------------------------


def test_relative_rotation_is_invariant_to_global_camera_rotation():
    """R_ref^T R_head is unchanged when both are premultiplied by any Q in SO(3).

    This is the branch's entire mathematical guarantee. 256 random (Q, ref, head)
    triples.
    """
    n = 256
    Q = random_rotations(n, seed=1)
    R_ref = random_rotations(n, seed=2)
    R_head = random_rotations(n, seed=3)

    base = C.relative_rotation(R_ref, R_head)
    rotated = C.relative_rotation(Q @ R_ref, Q @ R_head)

    assert torch.allclose(base, rotated, atol=1e-10), (
        f"max deviation {(base - rotated).abs().max().item():.3e}"
    )


def test_invariance_holds_for_the_derived_features_too():
    """Invariance is worthless if the feature assembler reintroduces camera frame."""
    T = 40
    Q = random_rotations(1, seed=11)[0]
    R_head = random_rotations(T, seed=12)
    R_ref = random_rotations(1, seed=13).expand(T, 3, 3)
    valid = torch.ones(T, dtype=torch.bool)
    conf = torch.ones(T, dtype=torch.float64)

    f1, v1 = C.assemble_pose_features(R_head, R_ref, valid, conf)
    f2, v2 = C.assemble_pose_features(Q @ R_head, Q @ R_ref, valid, conf)

    assert torch.equal(v1, v2)
    assert torch.allclose(f1, f2, atol=1e-8), (
        f"features are not viewpoint-invariant; max dev {(f1 - f2).abs().max().item():.3e}"
    )


def test_rotation_invariance_does_not_imply_homography_invariance():
    """Guards against over-claiming. A projective perturbation MUST change R_rel.

    If this ever starts passing as an invariance, someone has confused a rotation
    with a general camera change, and the thesis claim would be wrong.
    """
    R_ref = random_rotations(1, seed=21)
    R_head = random_rotations(1, seed=22)
    base = C.relative_rotation(R_ref, R_head)

    # A non-orthogonal (shear/perspective-like) transform applied in camera space.
    H = torch.tensor(
        [[1.0, 0.25, 0.0], [0.0, 1.0, 0.10], [0.05, 0.0, 1.0]], dtype=torch.float64
    )
    perturbed = C.relative_rotation(H @ R_ref, H @ R_head)
    assert not torch.allclose(base, perturbed, atol=1e-3), (
        "a projective perturbation left R_rel unchanged; the invariance claim is "
        "being over-read"
    )


# ---------------------------------------------------------------------------
# Causality
# ---------------------------------------------------------------------------


def test_angular_velocity_is_causal():
    """Appending future frames must not change any earlier value."""
    R = random_rotations(60, seed=31)
    full = C.relative_angular_velocity(R)
    prefix = C.relative_angular_velocity(R[:25])
    assert torch.allclose(full[:25], prefix, atol=1e-12), (
        "future frames changed an earlier angular velocity"
    )


def test_angular_acceleration_is_causal():
    R = random_rotations(60, seed=32)
    full = C.relative_angular_acceleration(R)
    prefix = C.relative_angular_acceleration(R[:25])
    assert torch.allclose(full[:25], prefix, atol=1e-12)


def test_causal_seat_reference_never_uses_the_current_or_future_frame():
    """The reference read at frame t must depend only on frames < t.

    Run the estimator over a sequence, recording what it returned at each step;
    then re-run over a strict prefix. The overlapping outputs must be identical.
    """
    R = random_rotations(40, seed=33)

    def run(n):
        est = C.CausalSeatReferenceEstimator(min_observations=4, momentum=0.2)
        out = []
        for t in range(n):
            ref = est.current("seat_0", dtype=torch.float64)
            out.append((ref.R_ref.clone(), float(ref.confidence)))
            est.update("seat_0", R[t], quality=1.0)
        return out

    full, prefix = run(40), run(18)
    for t, (a, b) in enumerate(zip(full[:18], prefix)):
        assert torch.allclose(a[0], b[0], atol=1e-12), f"reference at t={t} depends on the future"
        assert a[1] == b[1]


def test_causal_seat_reference_abstains_before_enough_evidence():
    est = C.CausalSeatReferenceEstimator(min_observations=6)
    R = random_rotations(10, seed=34)
    for t in range(5):
        assert float(est.current("s").confidence) == 0.0, "emitted a reference too early"
        est.update("s", R[t], quality=1.0)
    for t in range(5, 10):
        est.update("s", R[t], quality=1.0)
    assert float(est.current("s").confidence) > 0.0


def test_low_quality_observations_are_ignored():
    est = C.CausalSeatReferenceEstimator(min_observations=2, min_quality=0.5)
    R = random_rotations(5, seed=35)
    for t in range(5):
        est.update("s", R[t], quality=0.1)
    assert est.observations("s") == 0
    assert float(est.current("s").confidence) == 0.0


# ---------------------------------------------------------------------------
# Wraparound - the thing Euler subtraction gets wrong
# ---------------------------------------------------------------------------


def test_yaw_wraparound_is_a_small_angle_not_a_large_one():
    """179 deg -> -179 deg is a 2 degree turn. Naive subtraction says 358."""
    d = math.pi / 180.0
    R = torch.stack(
        [
            C.euler_to_matrix(torch.tensor(179.0 * d), torch.tensor(0.0), torch.tensor(0.0)),
            C.euler_to_matrix(torch.tensor(-179.0 * d), torch.tensor(0.0), torch.tensor(0.0)),
        ]
    )
    omega = C.relative_angular_velocity(R)
    turn = omega[1].norm().item() / d
    assert turn == pytest.approx(2.0, abs=1e-3), f"got {turn} degrees, expected 2"

    naive = abs(179.0 - (-179.0))
    assert naive == pytest.approx(358.0), "sanity: naive subtraction really is wrong"


def test_so2_heading_wraps_correctly():
    d = math.pi / 180.0
    out = C.relative_heading_so2(
        torch.tensor([179.0, -179.0, 10.0]) * d, torch.tensor([-179.0, 179.0, -10.0]) * d
    )
    assert torch.allclose(out / d, torch.tensor([-2.0, 2.0, 20.0]), atol=1e-4)


def test_so2_heading_is_invariant_to_image_plane_rotation():
    d = math.pi / 180.0
    head = torch.tensor([10.0, 100.0, -170.0]) * d
    ref = torch.tensor([-30.0, 20.0, 150.0]) * d
    for shift in (0.0, 37.0, 180.0, -95.0):
        got = C.relative_heading_so2(head + shift * d, ref + shift * d)
        assert torch.allclose(got, C.relative_heading_so2(head, ref), atol=1e-6)


# ---------------------------------------------------------------------------
# Representation numerics
# ---------------------------------------------------------------------------


def test_rot6d_round_trip_and_properness():
    R = random_rotations(128, seed=41)
    back = C.rot6d_to_matrix(C.matrix_to_rot6d(R))
    assert torch.allclose(R, back, atol=1e-10)
    eye = torch.eye(3, dtype=back.dtype).expand_as(back)
    assert torch.allclose(back @ back.transpose(-1, -2), eye, atol=1e-10)
    assert torch.allclose(torch.det(back), torch.ones(128, dtype=back.dtype), atol=1e-10)


def test_rot6d_projects_arbitrary_input_onto_so3():
    """Even for non-rotation input, the output must be a proper rotation."""
    x = torch.randn(64, 6, dtype=torch.float64, generator=torch.Generator().manual_seed(42))
    R = C.rot6d_to_matrix(x)
    eye = torch.eye(3, dtype=R.dtype).expand_as(R)
    assert torch.allclose(R @ R.transpose(-1, -2), eye, atol=1e-10)
    assert torch.allclose(torch.det(R), torch.ones(64, dtype=R.dtype), atol=1e-10)


def test_euler_round_trip():
    g = torch.Generator().manual_seed(43)
    yaw = (torch.rand(200, generator=g, dtype=torch.float64) - 0.5) * 2 * math.pi
    pitch = (torch.rand(200, generator=g, dtype=torch.float64) - 0.5) * math.pi * 0.9
    roll = (torch.rand(200, generator=g, dtype=torch.float64) - 0.5) * 2 * math.pi
    R = C.euler_to_matrix(yaw, pitch, roll)
    y2, p2, r2 = C.matrix_to_euler(R)
    assert torch.allclose(C.euler_to_matrix(y2, p2, r2), R, atol=1e-10)


def test_so3_log_exp_round_trip():
    R = random_rotations(128, seed=44)
    assert torch.allclose(C.so3_exp(C.so3_log(R)), R, atol=1e-8)


def test_so3_log_is_safe_at_identity():
    eye = torch.eye(3, dtype=torch.float64).expand(4, 3, 3)
    out = C.so3_log(eye)
    assert torch.isfinite(out).all()
    assert torch.allclose(out, torch.zeros_like(out), atol=1e-12)


def test_so3_log_is_safe_at_pi():
    """The degenerate case: sin(theta) -> 0 while the rotation is maximal."""
    axes = torch.tensor(
        [[1.0, 0, 0], [0, 1.0, 0], [0, 0, 1.0], [1.0, 1.0, 0]], dtype=torch.float64
    )
    axes = torch.nn.functional.normalize(axes, dim=-1)
    R = C.so3_exp(axes * math.pi)
    out = C.so3_log(R)
    assert torch.isfinite(out).all(), "log map produced NaN/Inf at theta = pi"
    assert torch.allclose(out.norm(dim=-1), torch.full((4,), math.pi, dtype=torch.float64), atol=1e-4)


def test_geodesic_distance_bounds_and_identity():
    R = random_rotations(64, seed=45)
    d_self = C.geodesic_distance(R, R)
    assert torch.allclose(d_self, torch.zeros_like(d_self), atol=1e-7)
    d = C.geodesic_distance(R, random_rotations(64, seed=46))
    assert torch.isfinite(d).all()
    assert (d >= -1e-9).all() and (d <= math.pi + 1e-9).all()


def test_geodesic_distance_is_symmetric():
    A, B = random_rotations(64, seed=47), random_rotations(64, seed=48)
    assert torch.allclose(C.geodesic_distance(A, B), C.geodesic_distance(B, A), atol=1e-9)


# ---------------------------------------------------------------------------
# Masks and the legacy adapter
# ---------------------------------------------------------------------------


def test_missing_evidence_yields_zero_features_and_a_zero_mask():
    """A zero vector must never be emitted without the mask that explains it."""
    T = 12
    R_head = random_rotations(T, seed=51)
    R_ref = random_rotations(1, seed=52).expand(T, 3, 3)
    valid = torch.ones(T, dtype=torch.bool)
    valid[3:6] = False
    conf = torch.ones(T, dtype=torch.float64)
    conf[8] = 0.0

    feats, out_valid = C.assemble_pose_features(R_head, R_ref, valid, conf)
    assert feats.shape == (T, C.POSE_FEATURE_DIM)
    assert not out_valid[3:6].any(), "invalid head frames stayed valid"
    assert not out_valid[8], "a zero-confidence reference stayed valid"
    assert torch.count_nonzero(feats[3:6]) == 0
    assert torch.count_nonzero(feats[8]) == 0
    assert out_valid[0] and torch.count_nonzero(feats[0]) > 0


def test_feature_layout_matches_declared_dim():
    """The schema hash in RUNS.jsonl is meaningless if the layout drifts."""
    assert sum(n for _, n in C.POSE_FEATURE_LAYOUT) == C.POSE_FEATURE_DIM
    R = random_rotations(5, seed=53)
    feats, _ = C.assemble_pose_features(
        R, R, torch.ones(5, dtype=torch.bool), torch.ones(5, dtype=torch.float64)
    )
    assert feats.shape[-1] == C.POSE_FEATURE_DIM


def test_legacy_block_adapter_matches_head_pose_scaling():
    """head_pose.py emits degrees/90; the adapter must invert exactly that."""
    block = torch.tensor([[30.0 / 90, -45.0 / 90, 10.0 / 90, 1.0]], dtype=torch.float64)
    R, found = C.from_head_pose_block(block)
    d = math.pi / 180.0
    expected = C.euler_to_matrix(
        torch.tensor(30.0 * d, dtype=torch.float64),
        torch.tensor(-45.0 * d, dtype=torch.float64),
        torch.tensor(10.0 * d, dtype=torch.float64),
    )
    assert torch.allclose(R[0], expected, atol=1e-12)
    assert bool(found[0])


def test_legacy_block_saturation_is_visible_not_silent():
    """Two students facing away by different amounts land on the SAME legacy value.

    This is the documented limitation that motivates a full-range estimator: the
    556-dim block clips at +/-90 degrees, and this room is full of people past it.
    """
    clipped_a = torch.tensor([[1.0, 0.0, 0.0, 1.0]], dtype=torch.float64)  # 120 deg -> clipped
    clipped_b = torch.tensor([[1.0, 0.0, 0.0, 1.0]], dtype=torch.float64)  # 170 deg -> clipped
    Ra, _ = C.from_head_pose_block(clipped_a)
    Rb, _ = C.from_head_pose_block(clipped_b)
    assert torch.allclose(Ra, Rb), (
        "expected the legacy clip to collapse distinct away-facing angles; if this "
        "fails the clipping behaviour has changed and the docstring is stale"
    )


def test_scene_reference_missing_seat_abstains():
    est = C.SceneReferenceEstimator({})
    ref = est("unknown_seat", T=5)
    assert float(ref.confidence.max()) == 0.0
    assert ref.source.endswith("missing")


def test_torso_reference_gates_on_confidence():
    est = C.TorsoReferenceEstimator(min_confidence=0.4)
    R = random_rotations(4, seed=61).to(torch.float32)
    conf = torch.tensor([0.9, 0.1, 0.5, 0.0])
    ref = est(R, conf)
    assert torch.equal(ref.confidence > 0, torch.tensor([True, False, True, False]))
