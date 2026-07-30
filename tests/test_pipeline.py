"""Integration test for the dataset loading and preprocessing pipeline.

Creates synthetic in-memory datasets to exercise the full
load → detect columns → split → build → sample path without a GPU
and without downloading anything from the Hub.
"""

import tempfile

import pytest
import torch
import numpy as np
import datasets
from PIL import Image
from unittest.mock import patch, MagicMock


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _import_train():
  """Lazy-import train.py so module-level evaluate.load() is mocked."""
  with patch("evaluate.load", return_value=MagicMock()):
    from scdiag import train
  return train


def _make_synthetic_dataset(num_samples=32, num_classes=3,
                            image_col="image", label_col="dx"):
  """Build a tiny in-memory dataset that mimics marmal88/skin_cancer."""
  rng = np.random.RandomState(42)
  images = [Image.fromarray(rng.randint(0, 255, (64, 64, 3), dtype=np.uint8))
            for _ in range(num_samples)]
  labels = [f"class_{i % num_classes}" for i in range(num_samples)]
  data = {image_col: images, label_col: labels}
  feat = datasets.Features({
      image_col: datasets.Image(),
      label_col: datasets.Value("string"),
  })
  return datasets.Dataset.from_dict(data, features=feat)


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------

class TestDetectLabelColumn:

  def test_detects_class_label(self):
    train_mod = _import_train()
    ds = datasets.Dataset.from_dict({
        "image": [Image.fromarray(np.zeros((8, 8, 3), dtype=np.uint8))],
        "label": [0],
    }, features=datasets.Features({
        "image": datasets.Image(),
        "label": datasets.ClassLabel(names=["a", "b"]),
    }))
    assert train_mod._detect_label_column(ds) == "label"

  def test_detects_dx_string_column(self):
    train_mod = _import_train()
    ds = _make_synthetic_dataset(label_col="dx")
    assert train_mod._detect_label_column(ds) == "dx"

  def test_raises_when_no_label_found(self):
    train_mod = _import_train()
    ds = datasets.Dataset.from_dict({
        "image": [Image.fromarray(np.zeros((8, 8, 3), dtype=np.uint8))],
        "data": [1.0],
    }, features=datasets.Features({
        "image": datasets.Image(),
        "data": datasets.Value("float32"),
    }))
    with pytest.raises(ValueError, match="No label column found"):
      train_mod._detect_label_column(ds)


class TestDetectImageColumn:

  def test_detects_image_column(self):
    train_mod = _import_train()
    ds = _make_synthetic_dataset(image_col="image")
    assert train_mod._detect_image_column(ds) == "image"

  def test_detects_img_column(self):
    train_mod = _import_train()
    ds = _make_synthetic_dataset(image_col="img")
    assert train_mod._detect_image_column(ds) == "img"

  def test_raises_when_no_image_found(self):
    train_mod = _import_train()
    ds = datasets.Dataset.from_dict({
        "data": [1.0],
        "label": ["a"],
    }, features=datasets.Features({
        "data": datasets.Value("float32"),
        "label": datasets.ClassLabel(names=["a"]),
    }))
    with pytest.raises(ValueError, match="No image column found"):
      train_mod._detect_image_column(ds)


class TestLoadAndSplit:

  def test_renames_label_and_detects_image(self):
    train_mod = _import_train()
    raw = _make_synthetic_dataset(label_col="dx", image_col="image")

    with patch("scdiag.train.load_dataset", return_value=raw):
      ds, image_col = train_mod.load_and_split_dataset("fake-dataset")

    assert "label" in ds["train"].features
    assert "dx" not in ds["train"].features
    assert image_col == "image"
    assert len(ds["train"]) > 0
    assert len(ds["test"]) > 0

  def test_preserves_label_if_already_named_label(self):
    train_mod = _import_train()
    raw = _make_synthetic_dataset(label_col="label")

    with patch("scdiag.train.load_dataset", return_value=raw):
      ds, image_col = train_mod.load_and_split_dataset("fake-dataset")

    assert "label" in ds["train"].features

  def test_handles_nonstandard_image_column(self):
    train_mod = _import_train()
    raw = _make_synthetic_dataset(image_col="pixel_data", label_col="class")

    with patch("scdiag.train.load_dataset", return_value=raw):
      ds, image_col = train_mod.load_and_split_dataset("fake-dataset")

    assert image_col == "pixel_data"
    assert "label" in ds["train"].features


class TestBuildDatasets:

  @pytest.fixture()
  def loaded(self):
    train_mod = _import_train()
    raw = _make_synthetic_dataset()
    with patch("scdiag.train.load_dataset", return_value=raw):
      ds, image_col = train_mod.load_and_split_dataset("fake-dataset")
    return ds, image_col

  def test_returns_dataset_and_processor(self, loaded):
    train_mod = _import_train()
    ds, image_col = loaded
    with tempfile.TemporaryDirectory() as cache:
      dataset, processor = train_mod.build_datasets(
          ds, "facebook/convnextv2-base-22k-224", 224,
          image_col=image_col, cache_dir=cache)
    assert "train" in dataset
    assert "test" in dataset
    assert processor is not None

  def test_train_sample_shape(self, loaded):
    train_mod = _import_train()
    ds, image_col = loaded
    with tempfile.TemporaryDirectory() as cache:
      dataset, _ = train_mod.build_datasets(
          ds, "facebook/convnextv2-base-22k-224", 224,
          image_col=image_col, cache_dir=cache)

    sample = dataset["train"][0]
    assert "pixel_values" in sample
    assert "labels" in sample
    assert sample["pixel_values"].shape == (3, 224, 224)

  def test_test_sample_shape(self, loaded):
    train_mod = _import_train()
    ds, image_col = loaded
    with tempfile.TemporaryDirectory() as cache:
      dataset, _ = train_mod.build_datasets(
          ds, "facebook/convnextv2-base-22k-224", 224,
          image_col=image_col, cache_dir=cache)

    sample = dataset["test"][0]
    assert sample["pixel_values"].shape == (3, 224, 224)

  def test_batched_access(self, loaded):
    """Simulate what the DataLoader does with __getitems__."""
    train_mod = _import_train()
    ds, image_col = loaded
    with tempfile.TemporaryDirectory() as cache:
      dataset, _ = train_mod.build_datasets(
          ds, "facebook/convnextv2-base-22k-224", 224,
          image_col=image_col, cache_dir=cache)

    n = len(dataset["train"])
    batch_size = min(8, n)
    batch = dataset["train"][:batch_size]
    assert batch["pixel_values"].shape == (batch_size, 3, 224, 224)
    assert batch["labels"].shape == (batch_size,)

  def test_labels_are_integer_tensors(self, loaded):
    train_mod = _import_train()
    ds, image_col = loaded
    with tempfile.TemporaryDirectory() as cache:
      dataset, _ = train_mod.build_datasets(
          ds, "facebook/convnextv2-base-22k-224", 224,
          image_col=image_col, cache_dir=cache)

    labels = dataset["train"][0]["labels"]
    assert isinstance(labels, torch.Tensor)
    assert labels.dtype in (torch.int64, torch.int32)

  def test_pixel_values_in_valid_range(self, loaded):
    """After normalization, values should be roughly in [-4, 4]."""
    train_mod = _import_train()
    ds, image_col = loaded
    with tempfile.TemporaryDirectory() as cache:
      dataset, _ = train_mod.build_datasets(
          ds, "facebook/convnextv2-base-22k-224", 224,
          image_col=image_col, cache_dir=cache)

    pv = dataset["train"][0]["pixel_values"]
    assert torch.isfinite(pv).all()
    assert pv.min() > -5.0
    assert pv.max() < 5.0

  def test_nonstandard_image_column(self, loaded):
    """Ensure transforms work when the image column isn't called 'image'."""
    train_mod = _import_train()
    raw = _make_synthetic_dataset(image_col="custom_img", label_col="dx")
    with patch("scdiag.train.load_dataset", return_value=raw):
      ds, image_col = train_mod.load_and_split_dataset("fake-dataset")

    with tempfile.TemporaryDirectory() as cache:
      dataset, _ = train_mod.build_datasets(
          ds, "facebook/convnextv2-base-22k-224", 224,
          image_col=image_col, cache_dir=cache)

    sample = dataset["train"][0]
    assert sample["pixel_values"].shape == (3, 224, 224)


class TestComputeClassWeights:

  def test_weights_shape_and_positive(self):
    train_mod = _import_train()
    raw = _make_synthetic_dataset(num_classes=3)
    with patch("scdiag.train.load_dataset", return_value=raw):
      ds, _ = train_mod.load_and_split_dataset("fake-dataset")
    weights = train_mod.compute_class_weights(ds, 3)
    assert weights.shape == (3,)
    assert torch.isfinite(weights).all()
    assert (weights > 0).all()

  def test_balanced_dataset_gives_uniform_weights(self):
    train_mod = _import_train()
    # Build a perfectly balanced dataset, bypass train_test_split so
    # class proportions stay even.
    raw = _make_synthetic_dataset(num_samples=30, num_classes=3)
    raw = raw.class_encode_column("dx").rename_column("dx", "label")
    ds = datasets.DatasetDict({"train": raw, "test": raw})
    weights = train_mod.compute_class_weights(ds, 3)
    # All weights should be equal (≈1.0)
    assert torch.allclose(weights, torch.ones(3), atol=1e-5)


class TestParseArgs:

  def test_defaults(self):
    train_mod = _import_train()
    args = train_mod.parse_args([])
    assert args.model == "facebook/convnextv2-base-22k-224"
    assert args.dataset == "marmal88/skin_cancer"
    assert args.epochs == 5
    assert args.image_size == 448

  def test_overrides(self):
    train_mod = _import_train()
    args = train_mod.parse_args([
        "--model", "my-model",
        "--dataset", "my-dataset",
        "--epochs", "3",
        "--image_size", "224",
    ])
    assert args.model == "my-model"
    assert args.dataset == "my-dataset"
    assert args.epochs == 3
    assert args.image_size == 224
