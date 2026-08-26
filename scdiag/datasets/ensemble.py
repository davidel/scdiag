"""Unified skin-lesion image dataset for self-supervised pre-training.

Stitches together multiple HuggingFace datasets (or pre-downloaded image
directories) into a single :class:`torch.utils.data.Dataset` with flat
indexing.  Optionally exposes labels when a label-aware pre-training
method (e.g. supervised contrastive) is used.
"""

import logging

import numpy as np
from PIL import Image

from scdiag.datasets.hf_proxy import HFDatasetProxy
from scdiag.datasets.image_folder import ImageFolderDataset
from scdiag.datasets.retry import getitem_retry
from scdiag.logging_utils import fatal


class _HFDataset:
  """Wrapper around a HuggingFace ``datasets.Dataset``."""

  def __init__(self,
               name,
               split="train",
               image_column=None,
               label_column=None,
               cache_dir=None,
               hf_token=None):
    self.name = name
    self._image_column = image_column
    self._label_col = None
    self._label_names = []
    self._label2id = {}
    from datasets import load_dataset
    logging.info(f"Loading HF dataset '{self.name}' (split={split}) ...")
    ds = load_dataset(self.name, split=split, cache_dir=cache_dir, token=hf_token)
    if self._image_column is None:
      col = HFDatasetProxy.detect_image_column(ds)
      if col is None:
        fatal(
            f"Cannot auto-detect image column in '{self.name}'. "
            f"Columns: {ds.column_names}. Set 'image_column' explicitly.",
            ValueError,
        )
      self._image_column = col
    self._ds = HFDatasetProxy.normalize_image_column(ds, self._image_column)
    self._detect_labels(label_column)

  def _detect_labels(self, label_column=None):
    """Detect and normalize the label column, if any."""
    import datasets as ds_lib
    self._label_names = []
    self._label2id = {}
    self._label_col = label_column
    if self._label_col is None:
      self._label_col = HFDatasetProxy.detect_label_column(self._ds)
    if self._label_col is None:
      return
    if self._label_col not in self._ds.column_names:
      fatal(
          f"Label column '{self._label_col}' not found in '{self.name}'. "
          f"Available: {self._ds.column_names}",
          ValueError,
      )
    feat = self._ds.features[self._label_col]
    if isinstance(feat, ds_lib.ClassLabel):
      self._label_names = list(feat.names)
    else:
      self._label_names = sorted(set(self._ds[self._label_col]))
    self._label2id = {name: i for i, name in enumerate(self._label_names)}
    logging.info(f"  '{self.name}': {len(self._label_names)} classes "
                 f"(label column: '{self._label_col}')")

  @property
  def has_labels(self):
    return self._label_col is not None

  @property
  def num_labels(self):
    return len(self._label_names)

  @property
  def label_names(self):
    return list(self._label_names)

  def label_for_idx(self, idx):
    raw = self._ds[idx][self._label_col]
    return self._label2id.get(raw, raw)

  def labels_array(self):
    """Return a numpy array of integer labels for all samples."""
    raw = np.array(self._ds[self._label_col])
    mapped = np.array([self._label2id.get(v, v) for v in raw])
    return mapped.astype(np.int64)

  def __len__(self):
    return len(self._ds)

  def __getitem__(self, idx):

    def load(i):
      row = self._ds[i]
      image = row[self._image_column]
      if not isinstance(image, Image.Image):
        image = Image.open(image)
      return image.convert("RGB")

    return getitem_retry(idx, load, len(self._ds))


class DatasetEnsemble:
  """Concatenation of multiple skin-lesion image datasets.

  Images are returned as RGB PIL :class:`PIL.Image.Image` objects —
  the caller (typically a :class:`torchvision.transforms.v2.Compose`
  pipeline) handles resizing, tensor conversion, and normalization.

  When ``with_labels=True``, each item is returned as a
  ``(PIL.Image.Image, int)`` tuple.  This mode requires every
  constituent dataset to carry labels — the constructor raises
  ``ValueError`` if any dataset lacks them.

  Args:
      dataset_configs: List of dicts describing each constituent dataset.
          Supported keys:

          - ``"name"`` (str, required): HuggingFace dataset ID **or**
            local directory path.
          - ``"source"`` (str): ``"hf"`` (default), ``"imagefolder"``.
          - ``"split"`` (str): Dataset split (default ``"train"``).
          - ``"image_column"`` (str): Column name for images (auto-detected
            if omitted).
          - ``"label_column"`` (str): Column name for labels (auto-detected
            if omitted).  Only used when ``with_labels=True``.
      cache_dir: HuggingFace cache directory.
      hf_token: HuggingFace token for gated datasets (e.g. Derm1M).
      with_labels: When True, expose ``(image, label)`` tuples and
          validate that every dataset has a label column.
  """

  def __init__(self,
               dataset_configs,
               cache_dir=None,
               hf_token=None,
               strict=False,
               with_labels=False):
    self._datasets = []
    self._offsets = None
    self._with_labels = with_labels
    self._global_label_names = []
    self._global_label2id = {}

    for cfg in dataset_configs:
      name = cfg["name"]
      source = cfg.get("source", "hf")
      try:
        if source == "hf":
          ds = _HFDataset(
              name=name,
              split=cfg.get("split", "train"),
              image_column=cfg.get("image_column"),
              label_column=cfg.get("label_column"),
              cache_dir=cache_dir,
              hf_token=hf_token,
          )
        elif source == "imagefolder":
          ds = ImageFolderDataset(root_dir=name, with_labels=with_labels)
        else:
          logging.warning(f"Unknown source '{source}' for dataset '{name}', skipping")
          continue
        n = len(ds)
        if n == 0:
          logging.warning(f"Dataset '{name}' has 0 images, skipping")
          continue
        logging.info(f"  + {name}: {n:,} images")
        self._datasets.append(ds)
      except Exception:
        if strict:
          fatal(
              f"Failed to initialize dataset '{name}' ({source})",
              RuntimeError,
          )
        logging.exception(
            "  Failed to initialize dataset '%s' (%s); skipping.",
            name,
            source,
        )

    offsets = [0]
    for ds in self._datasets:
      offsets.append(offsets[-1] + len(ds))
    self._offsets = offsets
    if not self._datasets:
      fatal("No datasets loaded successfully", RuntimeError)

    if with_labels:
      self._validate_labels()
      self._build_global_label_space()

  def _validate_labels(self):
    """Raise ValueError if any dataset cannot provide labels."""
    for ds in self._datasets:
      if not ds.has_labels:
        fatal(
            f"Dataset '{ds.name}' ({type(ds).__name__}) does not "
            f"provide labels, but the selected pre-training method "
            f"requires them.  Use a HuggingFace dataset with a "
            f"label column, or switch to a label-free method "
            f"(e.g. --method simmim).",
            ValueError,
        )

  def _build_global_label_space(self):
    """Map per-dataset label names to a shared integer space."""
    all_names = set()
    for ds in self._datasets:
      all_names.update(ds.label_names)
    self._global_label_names = sorted(all_names)
    self._global_label2id = {name: i for i, name in enumerate(self._global_label_names)}

  def _remap_label(self, ds, local_label):
    """Convert a dataset-local label to the global integer id."""
    name = ds.label_names[local_label]
    return self._global_label2id[name]

  @property
  def has_labels(self):
    return self._with_labels

  @property
  def num_labels(self):
    if not self._with_labels:
      return 0
    return len(self._global_label_names)

  @property
  def label_names(self):
    if not self._with_labels:
      return []
    return list(self._global_label_names)

  @property
  def labels_array(self):
    """Flat numpy array of global integer labels for every sample."""
    if not self._with_labels:
      return None
    parts = []
    for ds in self._datasets:
      local = ds.labels_array()
      global_ids = np.array(
          [self._global_label2id[ds.label_names[int(l)]] for l in local],
          dtype=np.int64,
      )
      parts.append(global_ids)
    return np.concatenate(parts)

  def __len__(self):
    return self._offsets[-1] if self._offsets else 0

  def _get_item(self, idx):
    """Resolve a valid global index and load its image (and label)."""
    import bisect
    ds_idx = bisect.bisect_right(self._offsets, idx) - 1
    local_idx = idx - self._offsets[ds_idx]
    ds = self._datasets[ds_idx]
    image = ds[local_idx]
    if self._with_labels:
      local_label = ds.label_for_idx(local_idx)
      return image, self._remap_label(ds, local_label)
    return image

  def __getitem__(self, idx):
    if idx < 0 or idx >= len(self):
      fatal(
          f"Index {idx} out of range for ensemble of length {len(self)}",
          IndexError,
      )
    return self._get_item(idx)

  @property
  def num_datasets(self):
    return len(self._datasets)

  def summary(self):
    """Return a human-readable summary of the ensemble."""
    lines = [(f"DatasetEnsemble: {len(self):,} images from "
              f"{len(self._datasets)} dataset(s)")]
    for i, ds in enumerate(self._datasets):
      tag = f"{len(ds):,} images"
      if self._with_labels and ds.has_labels:
        tag += f", {ds.num_labels} classes"
      elif self._with_labels:
        tag += ", NO LABELS"
      lines.append(f"  [{i}] {ds.name}: {tag}")
    if self._with_labels:
      lines.append(f"  Global label space: {self.num_labels} classes")
    return "\n".join(lines)
