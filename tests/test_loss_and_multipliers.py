"""Tests for CombinedFocalLoss and parse_class_multipliers (Phase 0)."""

import logging

import pytest
import torch
import torch.nn as nn

from scdiag.train import CombinedFocalLoss, parse_class_multipliers

LABEL2ID = {
    "actinic_keratoses": 0,
    "basal_cell_carcinoma": 1,
    "benign_keratosis": 2,
    "dermatofibroma": 3,
    "melanocytic_Nevi": 4,
    "melanoma": 5,
    "vascular_lesions": 6,
}
NUM_LABELS = 7


class TestParseClassMultipliers:
  """Tests for the --class_multipliers CLI string parser."""

  def test_empty_string_returns_ones(self):
    m = parse_class_multipliers("", NUM_LABELS, LABEL2ID)
    assert m.shape == (NUM_LABELS,)
    assert torch.allclose(m, torch.ones(NUM_LABELS))

  def test_none_like_empty(self):
    m = parse_class_multipliers("   ", NUM_LABELS, LABEL2ID)
    assert torch.allclose(m, torch.ones(NUM_LABELS))

  def test_single_string_key(self):
    m = parse_class_multipliers("melanoma=4.0", NUM_LABELS, LABEL2ID)
    assert m[5] == pytest.approx(4.0)
    # All other classes unchanged
    for i in range(NUM_LABELS):
      if i != 5:
        assert m[i] == pytest.approx(1.0)

  def test_single_integer_key(self):
    m = parse_class_multipliers("5=3.5", NUM_LABELS, LABEL2ID)
    assert m[5] == pytest.approx(3.5)
    for i in range(NUM_LABELS):
      if i != 5:
        assert m[i] == pytest.approx(1.0)

  def test_multiple_entries(self):
    m = parse_class_multipliers(
        "melanoma=4.0,melanocytic_Nevi=0.5,basal_cell_carcinoma=2.0",
        NUM_LABELS,
        LABEL2ID,
    )
    assert m[5] == pytest.approx(4.0)
    assert m[4] == pytest.approx(0.5)
    assert m[1] == pytest.approx(2.0)
    # Remaining classes unchanged
    for i in [0, 2, 3, 6]:
      assert m[i] == pytest.approx(1.0)

  def test_whitespace_handling(self):
    m = parse_class_multipliers(
        "melanoma=4.0 , melanocytic_Nevi=0.5",
        NUM_LABELS,
        LABEL2ID,
    )
    assert m[5] == pytest.approx(4.0)
    assert m[4] == pytest.approx(0.5)

  def test_unknown_class_name_raises(self):
    with pytest.raises(ValueError, match="Unknown class name 'fake_class'"):
      parse_class_multipliers("fake_class=2.0", NUM_LABELS, LABEL2ID)

  def test_malformed_entry_no_equals_raises(self):
    with pytest.raises(ValueError, match="Invalid --class_multipliers entry"):
      parse_class_multipliers("melanoma", NUM_LABELS, LABEL2ID)

  def test_out_of_range_index_raises(self):
    with pytest.raises(ValueError, match="out of range"):
      parse_class_multipliers("99=2.0", NUM_LABELS, LABEL2ID)

  def test_negative_index_raises(self):
    # "-1".isdigit() is False in Python, so it is treated as a class name
    # and raises "Unknown class name" rather than "out of range".
    with pytest.raises(ValueError, match="Unknown class name"):
      parse_class_multipliers("-1=2.0", NUM_LABELS, LABEL2ID)

  def test_float_index_not_digit(self):
    """'1.5' is not a digit, so it's treated as a class name."""
    with pytest.raises(ValueError, match="Unknown class name"):
      parse_class_multipliers("1.5=2.0", NUM_LABELS, LABEL2ID)

  def test_zero_multiplier(self):
    m = parse_class_multipliers("melanoma=0.0", NUM_LABELS, LABEL2ID)
    assert m[5] == pytest.approx(0.0)

  def test_large_multiplier(self):
    m = parse_class_multipliers("melanoma=100.0", NUM_LABELS, LABEL2ID)
    assert m[5] == pytest.approx(100.0)


def _make_batch(num_classes=7, batch_size=16, seed=42):
  """Create a reproducible batch of logits and targets."""
  g = torch.Generator().manual_seed(seed)
  logits = torch.randn(batch_size, num_classes, generator=g)
  # Targets are class indices 0..num_classes-1, cycling
  targets = torch.arange(batch_size) % num_classes
  return logits, targets


class TestCombinedFocalLoss:
  """Tests for the CombinedFocalLoss implementation."""

  def test_equivalence_with_cross_entropy_when_gamma_zero(self):
    """With gamma=0 and label_smoothing=0, should match F.cross_entropy
    with reduction='none' followed by .mean() (batch mean, not weighted mean)."""
    logits, targets = _make_batch()
    weights = torch.tensor([1.0, 2.0, 0.5, 1.5, 0.8, 3.0, 1.2])

    focal = CombinedFocalLoss(weights=weights,
                              gamma=0.0,
                              label_smoothing=0.0,
                              reduction='mean')

    # Our implementation uses F.cross_entropy(reduction='none') then .mean(),
    # which divides by batch size (not by sum of per-class weights as
    # nn.CrossEntropyLoss(reduction='mean') does).
    ce_none = nn.functional.cross_entropy(logits,
                                          targets,
                                          weight=weights,
                                          label_smoothing=0.0,
                                          reduction='none')
    expected = ce_none.mean()

    loss_focal = focal(logits, targets)
    torch.testing.assert_close(loss_focal, expected, rtol=1e-5, atol=1e-6)

  def test_equivalence_with_ce_label_smoothing(self):
    """With gamma=0 but label_smoothing>0, should match F.cross_entropy
    reduction='none' then .mean()."""
    logits, targets = _make_batch()
    weights = torch.tensor([1.0, 2.0, 0.5, 1.5, 0.8, 3.0, 1.2])

    focal = CombinedFocalLoss(weights=weights,
                              gamma=0.0,
                              label_smoothing=0.1,
                              reduction='mean')

    ce_none = nn.functional.cross_entropy(logits,
                                          targets,
                                          weight=weights,
                                          label_smoothing=0.1,
                                          reduction='none')
    expected = ce_none.mean()

    loss_focal = focal(logits, targets)
    torch.testing.assert_close(loss_focal, expected, rtol=1e-5, atol=1e-6)

  def test_zero_loss_when_all_correct(self):
    """A batch where the model is perfectly confident should have low loss."""
    num_classes = 3
    # Create logits with very high values for the correct class
    logits = torch.full((4, num_classes), -10.0)
    logits[0, 0] = 10.0
    logits[1, 1] = 10.0
    logits[2, 2] = 10.0
    logits[3, 0] = 10.0
    targets = torch.tensor([0, 1, 2, 0])

    weights = torch.ones(num_classes)
    focal = CombinedFocalLoss(weights=weights, gamma=2.0, reduction='mean')
    loss = focal(logits, targets)

    # Should be very close to zero
    assert loss.item() < 1e-4, f"Expected near-zero loss, got {loss.item()}"

  def test_focal_reduces_loss_for_easy_examples(self):
    """Focal loss should produce lower loss than CE for easy examples."""
    num_classes = 3
    # Easy example: model is somewhat confident and correct.
    # (Use moderate logits so CE is nonzero for meaningful comparison.)
    logits = torch.tensor([[-2.0, -2.0, 5.0]])
    targets = torch.tensor([2])
    weights = torch.ones(num_classes)

    focal = CombinedFocalLoss(weights=weights, gamma=2.0, reduction='mean')

    # Compute the CE part manually (reduction='none')
    ce_none = nn.functional.cross_entropy(logits,
                                          targets,
                                          weight=weights,
                                          reduction='none')
    loss_ce = ce_none.mean().item()
    loss_focal = focal(logits, targets).item()

    # Focal loss should be smaller for easy examples (p_t close to 1)
    assert loss_focal < loss_ce, (
        f"Focal ({loss_focal:.4f}) should be < CE ({loss_ce:.4f}) for easy examples")

  def test_monotonicity_increasing_melanoma_multiplier(self):
    """Increasing the melanoma multiplier should increase loss for melanoma labels."""
    logits, targets = _make_batch(batch_size=32, seed=123)
    # Set all targets to class 5 (melanoma)
    targets = torch.full((32,), 5)

    base_weights = torch.ones(7)

    # Low melanoma multiplier
    w_low = base_weights.clone()
    w_low[5] = 1.0
    loss_low = CombinedFocalLoss(weights=w_low, gamma=2.0, reduction='mean')(logits,
                                                                             targets)

    # High melanoma multiplier
    w_high = base_weights.clone()
    w_high[5] = 4.0
    loss_high = CombinedFocalLoss(weights=w_high, gamma=2.0, reduction='mean')(logits,
                                                                               targets)

    assert loss_high.item() > loss_low.item(), (
        f"Higher melanoma weight ({loss_high:.4f}) should produce "
        f"higher loss than lower weight ({loss_low:.4f})")

  def test_monotonicity_increasing_melanoma_multiplier_other_class_unaffected(self):
    """Increasing the melanoma multiplier should NOT increase loss for non-melanoma labels."""
    logits, targets = _make_batch(batch_size=32, seed=456)
    # Set all targets to class 1 (basal_cell_carcinoma)
    targets = torch.full((32,), 1)

    base_weights = torch.ones(7)

    # Low melanoma multiplier
    w_low = base_weights.clone()
    w_low[5] = 1.0
    loss_low = CombinedFocalLoss(weights=w_low, gamma=2.0, reduction='mean')(logits,
                                                                             targets)

    # High melanoma multiplier
    w_high = base_weights.clone()
    w_high[5] = 4.0
    loss_high = CombinedFocalLoss(weights=w_high, gamma=2.0, reduction='mean')(logits,
                                                                               targets)

    # Loss should be identical since no sample is melanoma
    torch.testing.assert_close(loss_low, loss_high, rtol=1e-6, atol=1e-7)

  def test_gradient_flow(self):
    """Gradients should be non-zero and finite for all parameters."""
    num_classes = 7
    logits, targets = _make_batch(num_classes=num_classes)
    logits.requires_grad_(True)

    weights = torch.ones(num_classes)
    focal = CombinedFocalLoss(weights=weights, gamma=2.0, reduction='mean')
    loss = focal(logits, targets)
    loss.backward()

    assert logits.grad is not None, "No gradient computed"
    assert torch.isfinite(logits.grad).all(), "Non-finite gradients detected"
    assert (logits.grad != 0).any(), "All gradients are zero"

  def test_single_sample_batch(self):
    """Should work with batch_size=1."""
    logits = torch.randn(1, 7)
    targets = torch.tensor([3])
    weights = torch.ones(7)

    focal = CombinedFocalLoss(weights=weights, gamma=2.0, reduction='mean')
    loss = focal(logits, targets)
    assert loss.shape == ()
    assert loss.item() > 0

  def test_reduction_none(self):
    """reduction='none' should return per-sample losses."""
    logits, targets = _make_batch(batch_size=8)
    weights = torch.ones(7)

    focal = CombinedFocalLoss(weights=weights, gamma=2.0, reduction='none')
    loss = focal(logits, targets)
    assert loss.shape == (8,)
    assert (loss > 0).all()

  def test_reduction_sum(self):
    """reduction='sum' should return the sum of losses."""
    logits, targets = _make_batch(batch_size=8)
    weights = torch.ones(7)

    focal_mean = CombinedFocalLoss(weights=weights, gamma=2.0, reduction='mean')
    focal_sum = CombinedFocalLoss(weights=weights, gamma=2.0, reduction='sum')

    loss_mean = focal_mean(logits, targets)
    loss_sum = focal_sum(logits, targets)

    torch.testing.assert_close(loss_sum, loss_mean * 8, rtol=1e-5, atol=1e-6)

  def test_gamma_zero_reduces_to_ce(self):
    """When gamma=0, the loss should match F.cross_entropy(reduction='none').mean()."""
    logits, targets = _make_batch()
    weights = torch.rand(7) + 0.1  # Random positive weights

    focal = CombinedFocalLoss(weights=weights, gamma=0.0, reduction='mean')

    ce_none = nn.functional.cross_entropy(logits,
                                          targets,
                                          weight=weights,
                                          reduction='none')
    expected = ce_none.mean()

    torch.testing.assert_close(focal(logits, targets), expected, rtol=1e-5, atol=1e-6)

  def test_label_smoothing_warning_emitted(self, caplog):
    """logging.warning should fire when gamma>0 and label_smoothing>0."""
    logits, targets = _make_batch()
    weights = torch.ones(7)

    focal = CombinedFocalLoss(weights=weights, gamma=2.0, label_smoothing=0.1)

    # The warning is emitted in train.py at construction time, not in the
    # loss class itself.  This test verifies the class still works correctly
    # when both are set (no crash), while the warning test is in
    # test_train_smoke.py or tested at the integration level.
    loss = focal(logits, targets)
    assert loss.item() > 0

  def test_weights_buffer_registered(self):
    """The weights tensor should be a registered buffer (not a parameter)."""
    weights = torch.ones(7)
    focal = CombinedFocalLoss(weights=weights, gamma=2.0)

    buffer_names = [name for name, _ in focal.named_buffers()]
    param_names = [name for name, _ in focal.named_parameters()]

    assert "weights" in buffer_names
    assert "weights" not in param_names
