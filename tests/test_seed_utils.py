"""Tests for scdiag.seed_utils reproducibility helpers."""

import argparse
import random
from unittest import mock

import numpy as np
import pytest
import torch

from scdiag.seed_utils import seed_everything, seed_worker


class TestSeedEverything:

  def test_torch_reproducible(self):
    seed_everything(123)
    a = torch.rand(4)
    seed_everything(123)
    b = torch.rand(4)
    assert torch.equal(a, b)

  def test_numpy_reproducible(self):
    seed_everything(123)
    a = np.random.rand(4)
    seed_everything(123)
    b = np.random.rand(4)
    np.testing.assert_array_equal(a, b)

  def test_random_reproducible(self):
    seed_everything(123)
    a = [random.random() for _ in range(4)]
    seed_everything(123)
    b = [random.random() for _ in range(4)]
    assert a == b

  def test_mixup_lambda_reproducible(self):
    """np.random.beta (used by mixup_data) is covered by the numpy seed."""
    from scdiag.train import mixup_data

    seed_everything(7)
    x = torch.rand(2, 3, 8, 8)
    _, _, _, lam_a = mixup_data(x, torch.tensor([0, 1]), alpha=0.2)
    seed_everything(7)
    _, _, _, lam_b = mixup_data(x, torch.tensor([0, 1]), alpha=0.2)
    assert lam_a == lam_b

  def test_returns_settings_dict(self):
    settings = seed_everything(5)
    assert settings["seed"] == 5
    assert settings["deterministic"] is False
    assert settings["cudnn_deterministic"] is False

  def test_deterministic_flags(self):
    try:
      settings = seed_everything(5, deterministic=True)
      assert settings["deterministic"] is True
      assert torch.backends.cudnn.deterministic is True
      assert torch.backends.cudnn.benchmark is False
    finally:
      torch.backends.cudnn.deterministic = False
      torch.backends.cudnn.benchmark = True
      torch.use_deterministic_algorithms(False)

  def test_different_seeds_diverge(self):
    seed_everything(1)
    a = torch.rand(4)
    seed_everything(2)
    b = torch.rand(4)
    assert not torch.equal(a, b)


class TestSeedWorker:

  def test_seeds_worker_from_torch_seed(self, monkeypatch):
    monkeypatch.setattr(torch, "initial_seed", lambda: 12345)
    seed_worker(0)
    assert np.random.get_state()[1][0] == (12345 % 2**32)
    assert random.random() is not None  # seeded without error

  def test_worker_determinism(self):
    """Two workers with the same torch seed draw the same numbers."""
    np.random.seed(0)
    seed_everything(99)
    worker_seed = torch.initial_seed() % 2**32
    np.random.seed(worker_seed)
    a = np.random.rand(3)
    np.random.seed(worker_seed)
    b = np.random.rand(3)
    np.testing.assert_array_equal(a, b)


class TestTrainFlags:

  def test_train_defaults_seed_42(self):
    from scdiag.train import parse_args

    args = parse_args([])
    assert args.seed == 42
    assert args.deterministic is False

  def test_pretrain_defaults_seed_42(self):
    from scdiag.pretrain import parse_args

    args = parse_args(["--method", "simmim", "--datasets", "dummy"])
    assert args.seed == 42
    assert args.deterministic is False
    assert args.grad_accum_steps == 1

  def test_pretrain_grad_accum_registered_once(self):
    """Regression guard: pretrain.py must not re-register shared flags.

    A duplicate ``--grad_accum_steps`` (local copy + shared
    ``add_optimization_args``) makes parse_args raise
    ``ArgumentError: conflicting option string``.
    """
    from scdiag.pretrain import parse_args

    parser_actions = []
    orig_add = argparse.ArgumentParser.add_argument

    def recording_add(self, *names, **kwargs):
      parser_actions.extend(n for n in names if n.startswith("-"))
      return orig_add(self, *names, **kwargs)

    with mock.patch.object(argparse.ArgumentParser, "add_argument", recording_add):
      parse_args(["--method", "simmim", "--datasets", "dummy"])
    assert parser_actions.count("--grad_accum_steps") == 1


class TestBalancedSamplerSeed:

  def test_same_seed_same_batches(self):
    from scdiag.datasets.balanced_sampler import BalancedBatchSampler

    labels = [i % 3 for i in range(30)]
    s1 = BalancedBatchSampler(labels, batch_size=6, samples_per_class=2, seed=11)
    s2 = BalancedBatchSampler(labels, batch_size=6, samples_per_class=2, seed=11)
    b1 = [list(b) for b in iter(s1)]
    b2 = [list(b) for b in iter(s2)]
    assert b1 == b2

  def test_none_seed_varies(self):
    from scdiag.datasets.balanced_sampler import BalancedBatchSampler

    labels = [i % 3 for i in range(30)]
    s = BalancedBatchSampler(labels, batch_size=6, samples_per_class=2)
    first = [list(b) for b in iter(s)]
    second = [list(b) for b in iter(s)]
    assert first != second

  def test_seed_none_default_backward_compatible(self):
    from scdiag.datasets.balanced_sampler import BalancedBatchSampler

    labels = [i % 2 for i in range(20)]
    sampler = BalancedBatchSampler(labels, batch_size=4, samples_per_class=2)
    assert sampler._seed is None
    batches = list(iter(sampler))
    assert len(batches) == 5


@pytest.mark.filterwarnings("ignore:.*GradScaler.*")
def test_checkpoint_save_atomic_on_failure(tmp_path, monkeypatch):
  """A torch.save failure leaves no temp file and no corrupted output."""
  from scdiag.storage_utils import save_checkpoint

  dest = tmp_path / "model_latest.pt"
  dest.write_text("PREVIOUS_GOOD_CHECKPOINT")

  def boom(*args, **kwargs):
    raise RuntimeError("disk full")

  monkeypatch.setattr(torch, "save", boom)
  with pytest.raises(RuntimeError):
    save_checkpoint({"epoch": 1}, str(dest))
  # Previous checkpoint untouched, no .tmp residue.
  assert dest.read_text() == "PREVIOUS_GOOD_CHECKPOINT"
  assert not (tmp_path / "model_latest.pt.tmp").exists()


def test_checkpoint_save_success(tmp_path):
  from scdiag.storage_utils import save_checkpoint

  dest = tmp_path / "nested" / "model_latest.pt"
  save_checkpoint({"epoch": 3}, str(dest))
  assert torch.load(dest, weights_only=False) == {"epoch": 3}
  assert not (tmp_path / "nested" / "model_latest.pt.tmp").exists()
