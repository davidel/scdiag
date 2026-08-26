"""Tests for DatasetEnsemble label handling and dict-returning datasets."""

import os
import tempfile

import numpy as np
import pytest
from datasets import Dataset as HFDataset
from PIL import Image

from scdiag.datasets.ensemble import DatasetEnsemble, _HFDataset
from scdiag.datasets.field_dataset import FieldSectorDataset
from scdiag.datasets.image_folder import ImageFolderDataset


def _make_fake_hf_dataset(n=20, num_classes=4):
  """Create a small in-memory HF dataset with images and labels."""
  images = [Image.new("RGB", (64, 64), color=(i * 30 % 256, 128, 64)) for i in range(n)]
  labels = [f"class_{i % num_classes}" for i in range(n)]
  return HFDataset.from_dict({"image": images, "label": labels})


def _make_patched_init(hf_ds, label_column=None):
  """Return (original_init, patched_init) for monkeypatching _HFDataset."""
  orig = _HFDataset.__init__
  _label_col = label_column

  def patched_init(self,
                   name,
                   split="train",
                   image_column=None,
                   label_column=None,
                   cache_dir=None,
                   hf_token=None):
    self.name = name
    self._image_column = image_column or "image"
    self._label_names = []
    self._label2id = {}
    self._label_col = None
    self._ds = hf_ds
    if _label_col is not None:
      self._detect_labels(label_column=_label_col)

  return orig, patched_init


class TestImageFolderDictReturns:

  def test_has_labels_true_for_class_folders(self):
    with tempfile.TemporaryDirectory() as root:
      for cls in ("cat", "dog"):
        cls_dir = os.path.join(root, "train", cls)
        os.makedirs(cls_dir)
        Image.new("RGB", (8, 8)).save(os.path.join(cls_dir, "img.jpg"))
      ds = ImageFolderDataset(root)
      assert ds.has_labels is True
      assert ds.num_labels == 2
      assert sorted(ds.label_names) == ["cat", "dog"]

  def test_dict_with_labels(self):
    with tempfile.TemporaryDirectory() as root:
      for cls in ("A", "B"):
        d = os.path.join(root, "split", cls)
        os.makedirs(d)
        Image.new("RGB", (8, 8)).save(os.path.join(d, "x.jpg"))
      ds = ImageFolderDataset(root)
      item = ds[0]
      assert isinstance(item, dict)
      assert "image" in item
      assert "label" in item
      assert isinstance(item["image"], Image.Image)
      assert isinstance(item["label"], int)

  def test_dict_without_labels(self):
    with tempfile.TemporaryDirectory() as root:
      d = os.path.join(root, "split")
      os.makedirs(d)
      Image.new("RGB", (8, 8)).save(os.path.join(d, "x.jpg"))
      ds = ImageFolderDataset(root)
      item = ds[0]
      assert isinstance(item, dict)
      assert "image" in item
      assert "label" not in item

  def test_labels_array(self):
    with tempfile.TemporaryDirectory() as root:
      for cls in ("A", "B"):
        d = os.path.join(root, "s", cls)
        os.makedirs(d)
        for i in range(3):
          Image.new("RGB", (8, 8)).save(os.path.join(d, f"{i}.jpg"))
      ds = ImageFolderDataset(root)
      arr = ds.labels_array()
      assert isinstance(arr, np.ndarray)
      assert arr.dtype == np.int64
      assert len(arr) == 6

  def test_nonexistent_dir(self):
    ds = ImageFolderDataset("/nonexistent")
    assert ds.has_labels is False
    assert len(ds) == 0

  def test_mixed_layout_fatal(self):
    with tempfile.TemporaryDirectory() as root:
      # split/label/file (3 components)
      d = os.path.join(root, "train", "A")
      os.makedirs(d)
      Image.new("RGB", (8, 8)).save(os.path.join(d, "img.jpg"))
      # split/file (2 components)
      d2 = os.path.join(root, "train")
      Image.new("RGB", (8, 8)).save(os.path.join(d2, "orphan.jpg"))
      with pytest.raises(ValueError, match="Mixed layout"):
        ImageFolderDataset(root)


class TestHFDatasetDictReturns:

  def test_dict_with_labels(self):
    hf_ds = _make_fake_hf_dataset(n=4, num_classes=2)
    ds = _HFDataset.__new__(_HFDataset)
    ds.name = "test"
    ds._image_column = "image"
    ds._ds = hf_ds
    ds._detect_labels(label_column="label")
    item = ds[0]
    assert isinstance(item, dict)
    assert "image" in item
    assert "label" in item
    assert isinstance(item["image"], Image.Image)

  def test_dict_without_labels(self):
    images = [Image.new("RGB", (64, 64)) for _ in range(4)]
    hf_ds = HFDataset.from_dict({"image": images})
    ds = _HFDataset.__new__(_HFDataset)
    ds.name = "test"
    ds._image_column = "image"
    ds._ds = hf_ds
    ds._label_col = None
    ds._label_names = []
    ds._label2id = {}
    item = ds[0]
    assert isinstance(item, dict)
    assert "image" in item
    assert "label" not in item

  def test_labels_array(self):
    hf_ds = _make_fake_hf_dataset(n=12, num_classes=4)
    ds = _HFDataset.__new__(_HFDataset)
    ds.name = "test"
    ds._image_column = "image"
    ds._ds = hf_ds
    ds._detect_labels(label_column="label")
    arr = ds.labels_array()
    assert isinstance(arr, np.ndarray)
    assert arr.dtype == np.int64
    assert len(arr) == 12


class TestFieldSectorDataset:

  def test_extracts_field(self):

    class _FakeDS:

      def __len__(self):
        return 3

      def __getitem__(self, idx):
        return {"image": f"img_{idx}", "label": idx}

    ds = FieldSectorDataset(_FakeDS(), field="image")
    assert ds[0] == "img_0"
    assert len(ds) == 3

  def test_extracts_label(self):

    class _FakeDS:

      def __len__(self):
        return 2

      def __getitem__(self, idx):
        return {"image": f"img_{idx}", "label": idx + 10}

    ds = FieldSectorDataset(_FakeDS(), field="label")
    assert ds[1] == 11


class TestDatasetEnsembleLabels:

  def test_has_labels_detected(self):
    hf_ds = _make_fake_hf_dataset(n=8, num_classes=3)
    orig, patched = _make_patched_init(hf_ds, label_column="label")
    _HFDataset.__init__ = patched
    try:
      configs = [{"name": "test", "source": "hf", "label_column": "label"}]
      ensemble = DatasetEnsemble(configs)
      assert ensemble.has_labels is True
      assert ensemble.num_labels == 3
    finally:
      _HFDataset.__init__ = orig

  def test_no_labels_when_absent(self):
    hf_ds = _make_fake_hf_dataset(n=8, num_classes=3)
    orig, patched = _make_patched_init(hf_ds, label_column=None)
    _HFDataset.__init__ = patched
    try:
      configs = [{"name": "test", "source": "hf"}]
      ensemble = DatasetEnsemble(configs)
      assert ensemble.has_labels is False
    finally:
      _HFDataset.__init__ = orig

  def test_labels_array(self):
    hf_ds = _make_fake_hf_dataset(n=12, num_classes=4)
    orig, patched = _make_patched_init(hf_ds, label_column="label")
    _HFDataset.__init__ = patched
    try:
      configs = [{"name": "test", "source": "hf", "label_column": "label"}]
      ensemble = DatasetEnsemble(configs)
      arr = ensemble.labels_array
      assert isinstance(arr, np.ndarray)
      assert arr.dtype == np.int64
      assert len(arr) == 12
    finally:
      _HFDataset.__init__ = orig

  def test_item_is_dict(self):
    hf_ds = _make_fake_hf_dataset(n=8, num_classes=2)
    orig, patched = _make_patched_init(hf_ds, label_column="label")
    _HFDataset.__init__ = patched
    try:
      configs = [{"name": "test", "source": "hf", "label_column": "label"}]
      ensemble = DatasetEnsemble(configs)
      item = ensemble[0]
      assert isinstance(item, dict)
      assert "image" in item
      assert "label" in item
    finally:
      _HFDataset.__init__ = orig

  def test_summary_with_labels(self):
    hf_ds = _make_fake_hf_dataset(n=16, num_classes=4)
    orig, patched = _make_patched_init(hf_ds, label_column="label")
    _HFDataset.__init__ = patched
    try:
      configs = [{"name": "test", "source": "hf", "label_column": "label"}]
      ensemble = DatasetEnsemble(configs)
      s = ensemble.summary()
      assert "classes" in s.lower()
    finally:
      _HFDataset.__init__ = orig
