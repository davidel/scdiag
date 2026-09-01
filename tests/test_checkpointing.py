"""Tests for the shared checkpointing utilities."""

import os

import torch
from torch import nn

from scdiag.checkpointing import (
    CheckpointSaver,
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


class _TinyModel(nn.Module):

  def __init__(self):
    super().__init__()
    self.fc = nn.Linear(2, 3)


class TestCheckpointSaver:

  def _make_saver(self, tmp_path):
    model = _TinyModel()
    optimizer = torch.optim.SGD(model.parameters(), lr=0.01)
    saver = CheckpointSaver(
        model,
        optimizer,
        None,
        root=str(tmp_path / "ckpt"),
        save_every=500,
    )
    return saver, model, optimizer

  def test_should_save_truth_table(self):
    model = _TinyModel()
    optimizer = torch.optim.SGD(model.parameters(), lr=0.01)
    saver = CheckpointSaver(model, optimizer, None, root="/unused", save_every=500)
    assert saver.should_save(500) is True
    assert saver.should_save(1000) is True
    assert saver.should_save(499) is False
    assert saver.should_save(501) is False
    assert saver.should_save(0) is False

  def test_should_save_disabled_by_default(self):
    model = _TinyModel()
    optimizer = torch.optim.SGD(model.parameters(), lr=0.01)
    saver = CheckpointSaver(model, optimizer, None, root="/unused")
    assert saver.should_save(500) is False
    assert saver.should_save(1000) is False

  def test_save_latest_and_best_write_files(self, tmp_path):
    saver, _, _ = self._make_saver(tmp_path)
    path = saver.save_latest(2, global_step=500, best_macro_f1=90.0)
    best = saver.save_best(3, global_step=600, best_macro_f1=91.5)
    assert path.endswith("_latest.pt")
    assert best.endswith("_best.pt")
    assert os.path.exists(path)
    assert os.path.exists(best)

  def test_saved_dict_carries_extra_kv(self, tmp_path):
    saver, _, _ = self._make_saver(tmp_path)
    saver.save_latest(1, global_step=250, best_macro_f1=42.0, custom_kv="hello")
    state = torch.load(str(tmp_path / "ckpt_latest.pt"), weights_only=False)
    assert state["epoch"] == 1
    assert state["global_step"] == 250
    assert state["best_macro_f1"] == 42.0
    assert state["custom_kv"] == "hello"
    assert "fc.weight" in state["model_state_dict"]
    assert "optimizer_state_dict" in state
    assert state["scheduler_state_dict"] is None

  def test_save_latest_is_idempotent_path(self, tmp_path):
    saver, _, _ = self._make_saver(tmp_path)
    first = saver.save_latest(1, global_step=10)
    second = saver.save_latest(1, global_step=20)
    assert first == second
    state = torch.load(str(second), weights_only=False)
    assert state["global_step"] == 20


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
