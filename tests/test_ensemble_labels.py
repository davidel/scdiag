"""Tests for DatasetEnsemble with_labels mode and label validation."""

import os
import tempfile

import numpy as np
import pytest
from datasets import Dataset as HFDataset
from PIL import Image

from scdiag.datasets.ensemble import DatasetEnsemble, _HFDataset
from scdiag.datasets.image_folder import ImageFolderDataset


def _make_fake_hf_dataset(n=20, num_classes=4):
  """Create a small in-memory HF dataset with images and labels."""
  images = [Image.new("RGB", (64, 64), color=(i * 50, 100, 150)) for i in range(n)]
  labels = list(range(num_classes)) * (n // num_classes)
  return HFDataset.from_dict({"image": images, "label": labels})


def _make_patched_init(hf_ds, label_column="label"):
  """Return a patched _HFDataset.__init__ that injects a pre-built dataset."""
  original_init = _HFDataset.__init__

  def patched_init(self, **kwargs):
    self.name = kwargs.get("name", "test")
    self._image_column = kwargs.get("image_column") or "image"
    self._label_col = None
    self._label_names = []
    self._label2id = {}
    self._ds = hf_ds
    self._detect_labels(kwargs.get("label_column") or label_column)

  return original_init, patched_init


class TestImageFolderHasLabels:

  def test_has_labels_is_false(self):
    ds = ImageFolderDataset(root_dir="/nonexistent")
    assert ds.has_labels is False


class TestHFDatasetHasLabels:

  def test_has_labels_true_when_label_column_specified(self):
    hf_ds = _make_fake_hf_dataset()
    ds = _HFDataset.__new__(_HFDataset)
    ds.name = "test_dataset"
    ds._image_column = "image"
    ds._ds = hf_ds
    ds._detect_labels(label_column="label")
    assert ds.has_labels is True
    assert ds.num_labels == 4
    assert len(ds.label_names) == 4

  def test_has_labels_false_when_no_label_column(self):
    hf_ds = _make_fake_hf_dataset()
    hf_ds = hf_ds.remove_columns("label")
    ds = _HFDataset.__new__(_HFDataset)
    ds.name = "test_nolabel"
    ds._image_column = "image"
    ds._ds = hf_ds
    ds._detect_labels(label_column=None)
    assert ds.has_labels is False
    assert ds.num_labels == 0
    assert ds.label_names == []

  def test_label_for_idx(self):
    hf_ds = _make_fake_hf_dataset(n=8, num_classes=2)
    ds = _HFDataset.__new__(_HFDataset)
    ds.name = "test"
    ds._image_column = "image"
    ds._ds = hf_ds
    ds._detect_labels(label_column="label")
    label = ds.label_for_idx(0)
    assert isinstance(label, int)
    assert 0 <= label < 2

  def test_labels_array(self):
    hf_ds = _make_fake_hf_dataset(n=12, num_classes=3)
    ds = _HFDataset.__new__(_HFDataset)
    ds.name = "test"
    ds._image_column = "image"
    ds._ds = hf_ds
    ds._detect_labels(label_column="label")
    arr = ds.labels_array()
    assert isinstance(arr, np.ndarray)
    assert arr.dtype == np.int64
    assert len(arr) == 12


class TestDatasetEnsembleLabels:

  def test_with_labels_false_by_default(self):
    hf_ds = _make_fake_hf_dataset()
    orig, patched = _make_patched_init(hf_ds, label_column=None)
    _HFDataset.__init__ = patched
    try:
      configs = [{"name": "test_mock", "source": "hf"}]
      ensemble = DatasetEnsemble(configs)
      assert ensemble.has_labels is False
      assert ensemble.num_labels == 0
      item = ensemble[0]
      assert isinstance(item, Image.Image)
    finally:
      _HFDataset.__init__ = orig

  def test_with_labels_true(self):
    hf_ds = _make_fake_hf_dataset(n=16, num_classes=4)
    orig, patched = _make_patched_init(hf_ds, label_column="label")
    _HFDataset.__init__ = patched
    try:
      configs = [{"name": "test", "source": "hf", "label_column": "label"}]
      ensemble = DatasetEnsemble(configs, with_labels=True)
      assert ensemble.has_labels is True
      assert ensemble.num_labels == 4
      assert len(ensemble.label_names) == 4
      item = ensemble[0]
      assert isinstance(item, tuple)
      assert len(item) == 2
      image, label = item
      assert isinstance(image, Image.Image)
      assert isinstance(label, int)
    finally:
      _HFDataset.__init__ = orig

  def test_labels_array(self):
    hf_ds = _make_fake_hf_dataset(n=12, num_classes=3)
    orig, patched = _make_patched_init(hf_ds, label_column="label")
    _HFDataset.__init__ = patched
    try:
      configs = [{"name": "test", "source": "hf", "label_column": "label"}]
      ensemble = DatasetEnsemble(configs, with_labels=True)
      arr = ensemble.labels_array
      assert isinstance(arr, np.ndarray)
      assert arr.dtype == np.int64
      assert len(arr) == 12
    finally:
      _HFDataset.__init__ = orig

  def test_imagefolder_rejects_with_labels(self):
    with tempfile.TemporaryDirectory() as tmpdir:
      img = Image.new("RGB", (64, 64), "red")
      img.save(os.path.join(tmpdir, "test.jpg"))
      # with_labels=True but directory has no class subdirectories
      # → ImageFolderDataset raises, ensemble catches and re-raises
      with pytest.raises(RuntimeError, match="No datasets loaded"):
        DatasetEnsemble(
            [{
                "name": tmpdir,
                "source": "imagefolder"
            }],
            with_labels=True,
        )

  def test_summary_with_labels(self):
    hf_ds = _make_fake_hf_dataset(n=16, num_classes=4)
    orig, patched = _make_patched_init(hf_ds, label_column="label")
    _HFDataset.__init__ = patched
    try:
      configs = [{"name": "test", "source": "hf", "label_column": "label"}]
      ensemble = DatasetEnsemble(configs, with_labels=True)
      s = ensemble.summary()
      assert "classes" in s.lower()
    finally:
      _HFDataset.__init__ = orig
