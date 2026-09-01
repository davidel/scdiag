"""Tests for the shared checkpointing utilities."""

import os

import torch

from scdiag.checkpointing import (
    filter_state_dict,
    format_count,
    load_checkpoint_weights,
    should_save_periodic,
)
from scdiag.storage_utils import save_checkpoint


class TestFormatCount:

  def test_below_1k(self):
    assert format_count(999) == "999"

  def test_kilobytes(self):
    assert format_count(1024) == "1.00K"

  def test_millions(self):
    assert format_count(25_600_000) == "24.41M"

  def test_gigabytes(self):
    assert format_count(3_221_225_472) == "3.00G"


class TestShouldSavePeriodic:

  def test_fires_at_multiple(self):
    assert should_save_periodic(500, 500) is True
    assert should_save_periodic(1000, 500) is True

  def test_no_fire_off_multiple(self):
    assert should_save_periodic(499, 500) is False
    assert should_save_periodic(501, 500) is False
    assert should_save_periodic(1, 500) is False

  def test_disabled_for_zero_or_negative_interval(self):
    assert should_save_periodic(500, 0) is False
    assert should_save_periodic(500, -1) is False

  def test_never_fires_at_step_zero(self):
    assert should_save_periodic(0, 500) is False
    assert should_save_periodic(0, 0) is False


class TestFilterStateDict:

  def test_identical_keys(self):
    state = {"a": torch.zeros(3), "b": torch.ones(5)}
    filtered, skipped = filter_state_dict(state, state)
    assert filtered == state
    assert skipped == []

  def test_shape_mismatch(self):
    ckpt = {"a": torch.zeros(3), "b": torch.ones(5)}
    model = {"a": torch.zeros(3), "b": torch.ones(10)}
    filtered, skipped = filter_state_dict(ckpt, model)
    assert "a" in filtered
    assert "b" not in filtered
    assert any("b" in s[0] for s in skipped)

  def test_missing_in_model(self):
    ckpt = {"a": torch.zeros(3), "extra": torch.ones(5)}
    model = {"a": torch.zeros(3)}
    filtered, skipped = filter_state_dict(ckpt, model)
    assert "extra" not in filtered
    assert any("extra" in s[0] for s in skipped)


class TestSaveAndLoadCheckpoint:

  def test_roundtrip(self, tmp_path):
    state = {"epoch": 5, "loss": 0.42}
    path = str(tmp_path / "ckpt.pt")
    save_checkpoint(state, path)
    assert os.path.exists(path)
    loaded = torch.load(path, weights_only=False)
    assert loaded["epoch"] == 5
    assert loaded["loss"] == 0.42

  def test_creates_parent_dirs(self, tmp_path):
    path = str(tmp_path / "subdir" / "ckpt.pt")
    save_checkpoint({"x": 1}, path)
    assert os.path.exists(path)


class TestLoadCheckpointWeights:

  def test_loads_matching_weights(self, tmp_path):
    import torch.nn as nn
    model = nn.Linear(10, 5)
    path = str(tmp_path / "model.pt")
    torch.save({"model_state_dict": model.state_dict()}, path)
    new_model = nn.Linear(10, 5)
    load_checkpoint_weights(path, new_model)

  def test_skips_shape_mismatch(self, tmp_path):
    """Keys with wrong shapes are silently skipped by alignment."""
    import torch.nn as nn
    model = nn.Linear(10, 5)
    path = str(tmp_path / "model.pt")
    torch.save({"model_state_dict": model.state_dict()}, path)
    new_model = nn.Linear(10, 8)  # different output dim
    report = load_checkpoint_weights(path, new_model)
    assert report.unused_old or report.unmatched_new
