"""Tests for the DermoscopyEnsemble dataset."""

import pytest
from PIL import Image

from scdiag.datasets.ensemble import (
    DermoscopyEnsemble,
    _ImageFolderDataset,
)


class TestImageFolderDataset:

  def test_loads_images(self, tmp_path):
    """Create a temporary directory with images and verify loading."""
    for i in range(5):
      img = Image.new("RGB", (64, 64), color=(i * 50, 100, 200))
      img.save(tmp_path / f"img_{i}.png")
    ds = _ImageFolderDataset(str(tmp_path))
    assert len(ds) == 5
    img = ds[0]
    assert isinstance(img, Image.Image)
    assert img.mode == "RGB"

  def test_empty_dir(self, tmp_path):
    ds = _ImageFolderDataset(str(tmp_path))
    assert len(ds) == 0


class TestDermoscopyEnsemble:

  def test_imagefolder_ensemble(self, tmp_path):
    """Ensemble of two local image directories."""
    dir_a = tmp_path / "dataset_a"
    dir_b = tmp_path / "dataset_b"
    dir_a.mkdir()
    dir_b.mkdir()
    for i in range(3):
      Image.new("RGB", (64, 64), color=(100, 100, 100)).save(dir_a / f"{i}.png")
    for i in range(4):
      Image.new("RGB", (64, 64), color=(200, 200, 200)).save(dir_b / f"{i}.png")

    configs = [
        {"name": str(dir_a), "source": "imagefolder"},
        {"name": str(dir_b), "source": "imagefolder"},
    ]
    ensemble = DermoscopyEnsemble(configs)
    assert len(ensemble) == 7
    assert ensemble.num_datasets == 2

    # Verify flat indexing crosses dataset boundaries
    img_a = ensemble[2]   # last image in dir_a
    img_b = ensemble[3]   # first image in dir_b
    assert isinstance(img_a, Image.Image)
    assert isinstance(img_b, Image.Image)

  def test_summary(self, tmp_path):
    d = tmp_path / "data"
    d.mkdir()
    for i in range(2):
      Image.new("RGB", (64, 64)).save(d / f"{i}.png")
    ensemble = DermoscopyEnsemble([{"name": str(d), "source": "imagefolder"}])
    s = ensemble.summary()
    assert "2 images" in s
    assert "1 dataset" in s

  def test_graceful_failure(self, tmp_path):
    """Ensemble should skip datasets that fail to load."""
    configs = [
        {"name": "/nonexistent/path", "source": "imagefolder"},
    ]
    ensemble = DermoscopyEnsemble(configs)
    assert len(ensemble) == 0

  def test_empty_config(self):
    ensemble = DermoscopyEnsemble([])
    assert len(ensemble) == 0

  def test_getitem_out_of_range(self, tmp_path):
    """Out-of-range index should fallback to index 0."""
    d = tmp_path / "data"
    d.mkdir()
    Image.new("RGB", (64, 64)).save(d / "img.png")
    ensemble = DermoscopyEnsemble([{"name": str(d), "source": "imagefolder"}])
    img = ensemble[9999]  # out of range
    assert isinstance(img, Image.Image)
