"""Tests for auto-detect resume_from_checkpoint logic."""

import glob
import os
from unittest.mock import MagicMock, patch

import pytest


def _find_latest_checkpoint(output_dir):
    """Replicate the production checkpoint-detection logic."""
    checkpoint_dirs = []
    for p in glob.glob(os.path.join(output_dir, "checkpoint-*")):
      try:
        checkpoint_dirs.append((int(os.path.basename(p).split("-")[-1]), p))
      except ValueError:
        continue
    checkpoint_dirs.sort(key=lambda x: x[0])
    if checkpoint_dirs:
      return checkpoint_dirs[-1][1]
    return None


def test_latest_checkpoint_is_chosen(tmp_path):
    """Given checkpoint-10, checkpoint-20, checkpoint-30, picks checkpoint-30."""
    for step in (10, 20, 30):
        (tmp_path / f"checkpoint-{step}").mkdir()

    result = _find_latest_checkpoint(str(tmp_path))
    assert result == str(tmp_path / "checkpoint-30")


def test_no_checkpoint_gives_none(tmp_path):
    """With no checkpoint dirs, resume should be None."""
    result = _find_latest_checkpoint(str(tmp_path))
    assert result is None


def test_single_checkpoint(tmp_path):
    """Single checkpoint is picked up."""
    (tmp_path / "checkpoint-500").mkdir()
    result = _find_latest_checkpoint(str(tmp_path))
    assert result.endswith("checkpoint-500")


def test_checkpoint_sorted_by_step_not_name(tmp_path):
    """Checkpoints with non-sequential steps are sorted numerically."""
    for step in (3, 100, 20, 500):
        (tmp_path / f"checkpoint-{step}").mkdir()

    result = _find_latest_checkpoint(str(tmp_path))
    assert result.endswith("checkpoint-500")


def test_non_checkpoint_dirs_are_ignored(tmp_path):
    """Non-numeric checkpoint-* dirs should not crash or be selected."""
    (tmp_path / "checkpoint-10").mkdir()
    (tmp_path / "not-a-checkpoint").mkdir()
    (tmp_path / "checkpoint-abc").mkdir()

    result = _find_latest_checkpoint(str(tmp_path))
    assert result.endswith("checkpoint-10")


def test_empty_step_number_ignored(tmp_path):
    """checkpoint- (empty suffix) should not crash."""
    (tmp_path / "checkpoint-").mkdir()
    (tmp_path / "checkpoint-42").mkdir()

    result = _find_latest_checkpoint(str(tmp_path))
    assert result.endswith("checkpoint-42")
