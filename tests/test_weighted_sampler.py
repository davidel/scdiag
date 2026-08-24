"""Tests for scdiag.datasets.weighted_sampler."""

import torch
from torch.utils.data import WeightedRandomSampler

from scdiag.datasets.weighted_sampler import build_weighted_sampler


class _FakeDataset:
  """Minimal dataset backed by a list of label dicts."""

  def __init__(self, labels):
    self._data = [{"label": int(l)} for l in labels]

  def __len__(self):
    return len(self._data)

  def __getitem__(self, idx):
    return self._data[idx]


class TestBuildWeightedSampler:

  def test_frequency_weights_inversely_proportional(self):
    """Rare classes should get higher weights than common classes."""
    # 90 samples of class 0, 10 samples of class 1.
    labels = [0] * 90 + [1] * 10
    ds = _FakeDataset(labels)
    sampler = build_weighted_sampler(ds, num_labels=2, weight_mode="frequency")
    assert isinstance(sampler, WeightedRandomSampler)
    weights = list(sampler.weights)
    # Class 1 weight should be ~9x class 0 weight.
    assert weights[90] > weights[0] * 5

  def test_frequency_weights_num_samples(self):
    labels = [0, 0, 1, 1, 2]
    ds = _FakeDataset(labels)
    sampler = build_weighted_sampler(ds, num_labels=3, weight_mode="frequency")
    assert sampler.num_samples == 5

  def test_combined_weights_match_loss_weights(self):
    """Combined mode should multiply inv_freq by multipliers."""
    labels = [0] * 80 + [1] * 20
    ds = _FakeDataset(labels)
    multipliers = torch.tensor([2.0, 1.0])
    sampler = build_weighted_sampler(ds,
                                     num_labels=2,
                                     weight_mode="combined",
                                     multipliers=multipliers)
    weights = list(sampler.weights)
    # Class 0: inv_freq=2/80=0.025, weight=0.025*2.0=0.05
    # Class 1: inv_freq=2/20=0.1,   weight=0.1*1.0=0.1
    # So class 1 should still be higher (freq dominates).
    assert weights[80] > weights[0]

  def test_multipliers_mode(self):
    """Multipliers mode should use the multiplier values directly."""
    labels = [0] * 50 + [1] * 50
    ds = _FakeDataset(labels)
    multipliers = torch.tensor([3.0, 1.0])
    sampler = build_weighted_sampler(ds,
                                     num_labels=2,
                                     weight_mode="multipliers",
                                     multipliers=multipliers)
    weights = list(sampler.weights)
    # Equal counts, so weights = multipliers.
    assert abs(weights[0] - 3.0) < 1e-5
    assert abs(weights[50] - 1.0) < 1e-5

  def test_replacement_true_allows_duplicates(self):
    labels = [0, 0, 0, 1]
    ds = _FakeDataset(labels)
    sampler = build_weighted_sampler(ds,
                                     num_labels=2,
                                     weight_mode="frequency",
                                     replacement=True)
    # With replacement, same index can appear multiple times.
    indices = list(sampler)
    assert len(indices) == 4

  def test_replacement_false_no_duplicates(self):
    labels = list(range(10))
    ds = _FakeDataset(labels)
    sampler = build_weighted_sampler(ds,
                                     num_labels=10,
                                     weight_mode="frequency",
                                     replacement=False)
    indices = list(sampler)
    # Without replacement, all indices should appear exactly once.
    assert sorted(indices) == list(range(10))

  def test_missing_multipliers_raises(self):
    ds = _FakeDataset([0, 1])
    try:
      build_weighted_sampler(ds, num_labels=2, weight_mode="combined")
      assert False, "Expected ValueError"
    except ValueError:
      pass

  def test_invalid_weight_mode_raises(self):
    ds = _FakeDataset([0, 1])
    try:
      build_weighted_sampler(ds, num_labels=2, weight_mode="bogus")
      assert False, "Expected ValueError"
    except ValueError:
      pass

  def test_single_class(self):
    labels = [0] * 100
    ds = _FakeDataset(labels)
    sampler = build_weighted_sampler(ds, num_labels=1, weight_mode="frequency")
    indices = list(sampler)
    assert len(indices) == 100
    # All weights are equal (single class), so every index appears once.
    assert sorted(indices) == list(range(100))
