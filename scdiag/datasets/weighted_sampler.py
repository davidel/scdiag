"""Build a WeightedRandomSampler for class-imbalanced training."""

import logging

import torch
from torch.utils.data import WeightedRandomSampler


def build_weighted_sampler(dataset,
                           num_labels,
                           weight_mode,
                           multipliers=None,
                           replacement=False):
  """Compute per-sample weights and return a WeightedRandomSampler.

    Args:
      dataset: Training proxy dataset with ``"label"`` key accessible by index.
      num_labels: Number of classes.
      weight_mode: ``"frequency"`` (inverse-freq), ``"multipliers"``
        (clinical severity), or ``"combined"`` (freq × multipliers).
      multipliers: Per-class multiplier tensor. Required when *weight_mode*
        is ``"multipliers"`` or ``"combined"``.
      replacement: Sample with replacement.

    Returns:
      A ``torch.utils.data.WeightedRandomSampler`` that yields indices with
      probabilities proportional to the computed per-sample weights.
    """
  labels = [dataset[i]["label"] for i in range(len(dataset))]

  class_counts = torch.zeros(num_labels)
  for label in labels:
    class_counts[label] += 1

  # Inverse-frequency weights (normalized so sum == num_labels).
  inv_freq = num_labels / class_counts.clamp(min=1)

  if weight_mode == "frequency":
    class_weights = inv_freq
  elif weight_mode == "multipliers":
    if multipliers is None:
      raise ValueError("multipliers must be provided when weight_mode='multipliers'")
    class_weights = multipliers.float()
  elif weight_mode == "combined":
    if multipliers is None:
      raise ValueError("multipliers must be provided when weight_mode='combined'")
    class_weights = inv_freq * multipliers.float()
  else:
    raise ValueError(f"Unknown weight_mode: {weight_mode!r}")

  sample_weights = [class_weights[label].item() for label in labels]

  sampler = WeightedRandomSampler(
      weights=sample_weights,
      num_samples=len(sample_weights),
      replacement=replacement,
  )

  logging.info(f"WeightedRandomSampler: {len(sample_weights):,} samples, "
               f"weight_mode={weight_mode!r}, replacement={replacement}")
  for i, w in enumerate(class_weights):
    logging.info(f"  class {i}: count={int(class_counts[i])}, "
                 f"weight={w:.4f}")
  return sampler
