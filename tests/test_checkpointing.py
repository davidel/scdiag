"""Tests for the shared checkpointing utilities."""

import os
import pytest
import torch

from scdiag.checkpointing import (
    filter_state_dict,
    resume_checkpoint,
    load_checkpoint_weights,
)
from scdiag.gcs_utils import save_checkpoint


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

  def test_excludes_prefixes(self, tmp_path):
    import torch.nn as nn
    model = nn.Sequential(nn.Linear(10, 5), nn.Linear(5, 3))
    state = model.state_dict()
    path = str(tmp_path / "model.pt")
    torch.save(state, path)
    new_model = nn.Sequential(nn.Linear(10, 5), nn.Linear(5, 3))
    # Load only layer.0 weights
    load_checkpoint_weights(path, new_model, exclude_prefixes=["1."])
