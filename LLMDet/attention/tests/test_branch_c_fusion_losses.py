"""Tests for observability-weighted fusion and the Branch-C objective terms.

BRANCH_C_PROTOCOL.md requires every loss term to carry a unit test, a zero-weight
baseline, and finite-value monitoring. The two tests that carry real weight are
:func:`test_reliability_head_cannot_see_content` (the gate must not be a covert
content path) and :func:`test_all_modalities_missing_falls_back_to_base` (the
model must degrade to the appearance-only baseline, not to NaN).
"""

from __future__ import annotations

import pytest
import torch

from attention.branch_c import fusion as FU
from attention.branch_c import losses as LO


def _shapes(M=3, B=2, T=7, Cc=6, Q=5):
    return M, B, T, Cc, Q


def make_fusion(seed=0, M=3, Q=5):
    torch.manual_seed(seed)
    return FU.ReliabilityFusion(num_modalities=M, quality_dim=Q)


def make_inputs(seed=0, M=3, B=2, T=7, Cc=6, Q=5, all_available=True):
    g = torch.Generator().manual_seed(seed)
    z_base = torch.randn(B, T, Cc, generator=g)
    z_experts = torch.randn(M, B, T, Cc, generator=g)
    quality = torch.randn(M, B, T, Q, generator=g)
    mask = torch.ones(M, B, T, dtype=torch.bool)
    if not all_available:
        mask[:] = False
    return z_base, z_experts, quality, mask


# ---------------------------------------------------------------------------
# masked_softmax edge cases
# ---------------------------------------------------------------------------


def test_masked_softmax_normalises_over_available_only():
    logits = torch.tensor([[1.0], [2.0], [3.0]])
    mask = torch.tensor([[True], [False], [True]])
    a = FU.masked_softmax(logits, mask, dim=0)
    assert a[1].item() == 0.0
    assert a.sum().item() == pytest.approx(1.0, abs=1e-6)


def test_masked_softmax_single_modality_is_one():
    logits = torch.tensor([[5.0], [-3.0], [100.0]])
    mask = torch.tensor([[False], [True], [False]])
    a = FU.masked_softmax(logits, mask, dim=0)
    assert a[1].item() == pytest.approx(1.0, abs=1e-6)


def test_masked_softmax_all_missing_is_zero_not_nan():
    logits = torch.randn(4, 3)
    mask = torch.zeros(4, 3, dtype=torch.bool)
    a = FU.masked_softmax(logits, mask, dim=0)
    assert torch.isfinite(a).all(), "all-missing slice produced NaN/Inf"
    assert torch.count_nonzero(a) == 0


def test_masked_softmax_extreme_logits_stay_finite():
    """Guards the max-subtraction: raw exp of these would overflow."""
    logits = torch.tensor([[1e4], [-1e4], [0.0]])
    mask = torch.ones(3, 1, dtype=torch.bool)
    a = FU.masked_softmax(logits, mask, dim=0)
    assert torch.isfinite(a).all()
    assert a.sum().item() == pytest.approx(1.0, abs=1e-6)


# ---------------------------------------------------------------------------
# Fusion behaviour
# ---------------------------------------------------------------------------


def test_all_modalities_missing_falls_back_to_base():
    """The protocol's degradation requirement: no experts -> appearance-only."""
    mdl = make_fusion()
    z_base, z_exp, q, mask = make_inputs(all_available=False)
    z_fused, diag = mdl(z_base, z_exp, q, mask)
    assert torch.allclose(z_fused, z_base, atol=1e-6)
    assert torch.count_nonzero(diag["alpha"]) == 0
    assert torch.isfinite(z_fused).all()


def test_alpha_sums_to_one_where_anything_is_available():
    mdl = make_fusion()
    z_base, z_exp, q, mask = make_inputs()
    mask[0, 0, 0] = False
    _, diag = mdl(z_base, z_exp, q, mask)
    s = diag["alpha"].sum(dim=0)
    assert torch.allclose(s, torch.ones_like(s), atol=1e-5)


def test_reliability_head_cannot_see_content():
    """Change the expert logits, hold quality fixed: routing must not move.

    If this fails, the gate has become a second content pathway and every
    "reliability" claim in the thesis is unsupported.
    """
    mdl = make_fusion()
    z_base, z_exp, q, mask = make_inputs()
    _, d1 = mdl(z_base, z_exp, q, mask)
    _, d2 = mdl(z_base, torch.randn_like(z_exp) * 17.0, q, mask)
    assert torch.allclose(d1["alpha"], d2["alpha"], atol=1e-12)
    assert torch.allclose(d1["r"], d2["r"], atol=1e-12)


def test_routing_does_respond_to_quality():
    """The converse: the gate must not be inert either."""
    mdl = make_fusion()
    z_base, z_exp, q, mask = make_inputs()
    _, d1 = mdl(z_base, z_exp, q, mask)
    _, d2 = mdl(z_base, z_exp, q + 3.0, mask)
    assert not torch.allclose(d1["alpha"], d2["alpha"], atol=1e-6)


def test_masked_expert_garbage_does_not_leak_into_the_fusion():
    """A missing expert's logits may be junk; alpha=0 must not turn that into NaN.

    IEEE 0 * NaN is NaN, so multiplying a zero weight by an unavailable expert's
    placeholder is not sufficient on its own.
    """
    mdl = make_fusion()
    z_base, z_exp, q, mask = make_inputs()
    mask[1] = False
    z_exp[1] = float("nan")
    z_fused, _ = mdl(z_base, z_exp, q, mask)
    assert torch.isfinite(z_fused).all(), (
        "an unavailable expert's placeholder values leaked into the fused logits"
    )


def test_fusion_gradients_are_finite():
    mdl = make_fusion()
    z_base, z_exp, q, mask = make_inputs()
    q.requires_grad_(True)
    z_fused, _ = mdl(z_base, z_exp, q, mask)
    z_fused.sum().backward()
    assert q.grad is not None and torch.isfinite(q.grad).all()


# ---------------------------------------------------------------------------
# Individual loss terms
# ---------------------------------------------------------------------------


def _logits_targets(B=4, T=9, Cc=6, seed=0):
    g = torch.Generator().manual_seed(seed)
    return (
        torch.randn(B, T, Cc, generator=g, requires_grad=True),
        torch.randint(0, Cc, (B, T), generator=g),
    )


@pytest.mark.parametrize("mode", ["focal", "logit_adjusted"])
def test_cue_loss_finite_and_differentiable(mode):
    logits, targets = _logits_targets()
    counts = torch.tensor([1000.0, 40.0, 46.0, 13.0, 53.0, 60.0])  # fold-0 support
    kwargs = {"class_counts": counts}
    loss = LO.cue_loss(logits, targets, mode=mode, **kwargs)
    assert torch.isfinite(loss)
    loss.backward()
    assert torch.isfinite(logits.grad).all()


def test_quality_order_loss_is_zero_when_ordering_holds():
    r_clean = torch.tensor([2.0, 3.0, 5.0])
    r_corrupt = torch.tensor([1.0, 1.5, 2.0])
    assert LO.quality_order_loss(r_clean, r_corrupt, margin=0.5).item() == pytest.approx(0.0)


def test_quality_order_loss_is_positive_when_ordering_is_violated():
    """The corrupted view claiming higher reliability than the clean one."""
    r_clean = torch.tensor([1.0])
    r_corrupt = torch.tensor([3.0])
    loss = LO.quality_order_loss(r_clean, r_corrupt, margin=0.5)
    assert loss.item() == pytest.approx(0.5 - 1.0 + 3.0)


def test_quality_order_loss_gradient_pushes_the_right_way():
    r_clean = torch.tensor([1.0], requires_grad=True)
    r_corrupt = torch.tensor([3.0], requires_grad=True)
    LO.quality_order_loss(r_clean, r_corrupt, margin=0.5).backward()
    assert r_clean.grad.item() < 0, "clean reliability should be pushed up"
    assert r_corrupt.grad.item() > 0, "corrupt reliability should be pushed down"


def test_counterfactual_loss_stops_gradient_on_the_clean_path():
    """The clean posterior is a target, never a thing this term trains."""
    clean = torch.randn(3, 5, 6, requires_grad=True)
    corrupt = torch.randn(3, 5, 6, requires_grad=True)
    loss = LO.counterfactual_loss(
        clean, corrupt, torch.ones(3, 5), torch.ones(3, 5)
    )
    loss.backward()
    assert clean.grad is None or torch.count_nonzero(clean.grad) == 0, (
        "gradient flowed into the clean path; the stop-gradient is not effective"
    )
    assert corrupt.grad is not None and torch.count_nonzero(corrupt.grad) > 0


def test_counterfactual_loss_is_gated_off_when_evidence_is_not_retained():
    """Do not force invariance when the removed modality carried unique signal."""
    clean = torch.randn(2, 4, 6)
    corrupt = torch.randn(2, 4, 6)
    off = LO.counterfactual_loss(clean, corrupt, torch.ones(2, 4), torch.zeros(2, 4))
    on = LO.counterfactual_loss(clean, corrupt, torch.ones(2, 4), torch.ones(2, 4))
    assert off.item() == pytest.approx(0.0, abs=1e-9)
    assert on.item() > 0.0


def test_counterfactual_loss_is_zero_when_paths_agree():
    z = torch.randn(2, 4, 6)
    loss = LO.counterfactual_loss(z, z.clone(), torch.ones(2, 4), torch.ones(2, 4))
    assert loss.item() == pytest.approx(0.0, abs=1e-6)


def test_jsd_is_symmetric_and_bounded():
    g = torch.Generator().manual_seed(3)
    a = torch.randn(16, 6, generator=g)
    b = torch.randn(16, 6, generator=g)
    ab, ba = LO.jsd(a, b), LO.jsd(b, a)
    assert torch.allclose(ab, ba, atol=1e-6), "JSD is not symmetric"
    assert (ab >= -1e-7).all()
    # JSD with natural log is bounded above by ln 2
    assert (ab <= torch.log(torch.tensor(2.0)) + 1e-5).all()


def test_jsd_is_zero_for_identical_distributions():
    a = torch.randn(8, 6)
    assert LO.jsd(a, a.clone()).abs().max().item() == pytest.approx(0.0, abs=1e-6)


def test_view_consistency_loss_zero_when_views_agree():
    z = torch.randn(2, 5, 6)
    assert LO.view_consistency_loss(z, z.clone()).item() == pytest.approx(0.0, abs=1e-6)


def test_smooth_and_boundary_losses_are_finite_and_differentiable():
    logits = torch.randn(2, 8, 6, requires_grad=True)
    valid = torch.ones(2, 8, dtype=torch.bool)
    s = LO.smooth_loss(logits, valid)
    assert torch.isfinite(s)
    s.backward(retain_graph=True)
    assert torch.isfinite(logits.grad).all()

    b_logits = torch.randn(2, 8, requires_grad=True)
    targets = torch.randint(0, 2, (2, 8)).float()
    b = LO.boundary_loss(b_logits, targets)
    assert torch.isfinite(b)
    b.backward()
    assert torch.isfinite(b_logits.grad).all()


@pytest.mark.parametrize("mode", ["brier", "nll"])
def test_calibration_loss_finite_and_differentiable(mode):
    logits, targets = _logits_targets(seed=5)
    loss = LO.calibration_loss(logits, targets, mode=mode)
    assert torch.isfinite(loss)
    loss.backward()
    assert torch.isfinite(logits.grad).all()


def test_distill_loss_finite_and_differentiable():
    teacher = torch.randn(2, 5, 6)
    student = torch.randn(2, 5, 6, requires_grad=True)
    loss = LO.distill_loss(student, teacher)
    assert torch.isfinite(loss)
    loss.backward()
    assert torch.isfinite(student.grad).all()


# ---------------------------------------------------------------------------
# The zero-weight baseline the protocol requires
# ---------------------------------------------------------------------------


def test_total_loss_with_only_cue_equals_cue_loss_exactly():
    logits, targets = _logits_targets(seed=7)
    cue = LO.cue_loss(logits, targets, mode="focal")
    total = LO.total_loss({"cue": cue}, coefficients={"cue": 1.0})
    assert torch.equal(total, cue)


def test_zero_coefficients_reproduce_the_baseline_exactly():
    """Every extra term at coefficient 0 must leave the baseline untouched."""
    logits, targets = _logits_targets(seed=8)
    cue = LO.cue_loss(logits, targets, mode="focal")
    terms = {
        "cue": cue,
        "boundary": torch.tensor(3.0),
        "smooth": torch.tensor(7.0),
        "view": torch.tensor(11.0),
        "quality_order": torch.tensor(13.0),
        "counterfactual": torch.tensor(17.0),
        "calibration": torch.tensor(19.0),
    }
    coeffs = {k: 0.0 for k in terms}
    coeffs["cue"] = 1.0
    total = LO.total_loss(terms, coefficients=coeffs)
    assert torch.equal(total, cue)


def test_a_nan_term_at_zero_weight_cannot_poison_the_total():
    """Skipping, not multiplying by zero: 0 * NaN would still be NaN."""
    logits, targets = _logits_targets(seed=9)
    cue = LO.cue_loss(logits, targets, mode="focal")
    total = LO.total_loss(
        {"cue": cue, "view": torch.tensor(float("nan"))},
        coefficients={"cue": 1.0, "view": 0.0},
    )
    assert torch.isfinite(total), "a zero-weighted NaN term poisoned the total"
    assert torch.equal(total, cue)


def test_total_loss_requires_a_cue_term():
    with pytest.raises(KeyError):
        LO.total_loss({"smooth": torch.tensor(1.0)})


def test_total_loss_applies_nonzero_coefficients():
    cue = torch.tensor(2.0)
    total = LO.total_loss(
        {"cue": cue, "smooth": torch.tensor(5.0)},
        coefficients={"cue": 1.0, "smooth": 0.5},
    )
    assert total.item() == pytest.approx(2.0 + 2.5)


def test_all_terms_together_are_finite_and_differentiable():
    """End-to-end finite-value monitoring across the whole objective."""
    logits, targets = _logits_targets(seed=10)
    corrupt = torch.randn_like(logits, requires_grad=True)
    valid = torch.ones(logits.shape[:2], dtype=torch.bool)
    terms = {
        "cue": LO.cue_loss(logits, targets, mode="focal"),
        "smooth": LO.smooth_loss(logits, valid),
        "view": LO.view_consistency_loss(logits, corrupt),
        "quality_order": LO.quality_order_loss(
            torch.tensor([1.0], requires_grad=True), torch.tensor([2.0])
        ),
        "counterfactual": LO.counterfactual_loss(
            logits, corrupt, torch.ones(logits.shape[:2]), torch.ones(logits.shape[:2])
        ),
        "calibration": LO.calibration_loss(logits, targets),
    }
    coeffs = {k: 0.3 for k in terms}
    coeffs["cue"] = 1.0
    total = LO.total_loss(terms, coefficients=coeffs)
    assert torch.isfinite(total)
    total.backward()
    assert torch.isfinite(logits.grad).all()
    assert torch.isfinite(corrupt.grad).all()
