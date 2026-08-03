"""Unified skin-lesion image dataset for self-supervised pre-training.

Stitches together multiple HuggingFace datasets (or pre-downloaded image
directories) into a single :class:`torch.utils.data.Dataset` with flat
indexing.  Only images are exposed — no labels — making this suitable for
SimMIM / MAE pre-training.
"""

import logging
import os
from pathlib import Path

from PIL import Image

from scdiag.datasets.hf_proxy import HFDatasetProxy

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Individual dataset back-ends
# ---------------------------------------------------------------------------


class _HFDataset:
  """Lazy-loading wrapper around a HuggingFace ``datasets.Dataset``."""

  def __init__(self,
               name,
               split="train",
               image_column=None,
               cache_dir=None,
               hf_token=None,
               min_resolution=None):
    self.name = name
    self.split = split
    self.image_column = image_column
    self.cache_dir = cache_dir
    self.hf_token = hf_token
    self.min_resolution = min_resolution
    self._ds = None

  def _load(self):
    from datasets import load_dataset
    logger.info(f"Loading HF dataset '{self.name}' (split={self.split}) …")
    try:
      ds = load_dataset(self.name,
                        split=self.split,
                        cache_dir=self.cache_dir,
                        token=self.hf_token)
    except Exception as exc:
      logger.warning(f"  Failed to load '{self.name}': {exc}")
      return None
    self._detect_and_normalize_image_column(ds)
    return ds

  def _detect_and_normalize_image_column(self, ds):
    """Detect the image column and ensure it returns decoded PIL images."""
    if self.image_column is not None:
      col = self.image_column
    else:
      col = HFDatasetProxy.detect_image_column(ds)
    if col is None:
      raise ValueError(f"Cannot auto-detect image column in '{self.name}'. "
                       f"Columns: {ds.column_names}. Set 'image_column' explicitly.")
    HFDatasetProxy.normalize_image_column(ds, col)
    self.image_column = col

  def _ensure_loaded(self):
    if self._ds is None:
      self._ds = self._load()

  def __len__(self):
    self._ensure_loaded()
    if self._ds is None:
      return 0
    return len(self._ds)

  def __getitem__(self, idx):
    self._ensure_loaded()
    if self._ds is None:
      raise IndexError(f"Dataset '{self.name}' failed to load")
    row = self._ds[idx]
    image = row[self.image_column]
    if not isinstance(image, Image.Image):
      image = Image.open(image)
    image = image.convert("RGB")
    if self.min_resolution is not None:
      if image.width < self.min_resolution or image.height < self.min_resolution:
        raise IndexError(f"Image too small: {image.size}, min={self.min_resolution}")
    return image


class _ImageFolderDataset:
  """Wrapper around a local ImageFolder directory of images."""

  def __init__(self, root_dir, min_resolution=None, cache_dir=None):
    self.name = str(root_dir)
    self.root_dir = root_dir
    self.min_resolution = min_resolution
    self.image_column = "image"
    self._cache_dir = cache_dir
    self._ds = None

  def _load(self):
    from datasets import load_dataset
    logger.info(f"Loading ImageFolder '{self.root_dir}' ...")
    try:
      ds = load_dataset("imagefolder",
                        data_dir=self.root_dir,
                        split="train",
                        cache_dir=self._cache_dir)
    except Exception:
      # No train split — fall back to whatever split exists.
      try:
        ds = load_dataset("imagefolder",
                          data_dir=self.root_dir,
                          cache_dir=self._cache_dir)
        # If dict of splits, take the first one.
        if isinstance(ds, dict):
          ds = next(iter(ds.values()))
      except Exception as exc:
        logger.warning(f"  Failed to load '{self.root_dir}': {exc}")
        return None
    return ds

  def _ensure_loaded(self):
    if self._ds is None:
      self._ds = self._load()

  def __len__(self):
    self._ensure_loaded()
    if self._ds is None:
      return 0
    return len(self._ds)

  def __getitem__(self, idx):
    self._ensure_loaded()
    if self._ds is None:
      raise IndexError(f"Dataset '{self.name}' failed to load")
    row = self._ds[idx]
    image = row[self.image_column]
    if not isinstance(image, Image.Image):
      image = Image.open(image)
    image = image.convert("RGB")
    if self.min_resolution is not None:
      if image.width < self.min_resolution or image.height < self.min_resolution:
        raise IndexError(f"Image too small: {image.size}, min={self.min_resolution}")
    return image


# ---------------------------------------------------------------------------
# Ensemble
# ---------------------------------------------------------------------------


class DermoscopyEnsemble:
  """Concatenation of multiple skin-lesion image datasets.

    Each constituent dataset is loaded lazily.  Images are returned as RGB
    PIL :class:`PIL.Image.Image` objects — the caller (typically a
    :class:`torchvision.transforms.v2.Compose` pipeline) handles resizing,
    tensor conversion, and normalization.

    Args:
        dataset_configs: List of dicts describing each constituent dataset.
            Supported keys:

            - ``"name"`` (str, required): HuggingFace dataset ID **or**
              local directory path.
            - ``"source"`` (str): ``"hf"`` (default), ``"imagefolder"``.
            - ``"split"`` (str): Dataset split (default ``"train"``).
            - ``"image_column"`` (str): Column name for images (auto-detected
              if omitted).
            - ``"min_resolution"`` (int): Skip images smaller than this.
        cache_dir: HuggingFace cache directory.
        hf_token: HuggingFace token for gated datasets (e.g. Derm1M).
    """

  def __init__(self, dataset_configs, cache_dir=None, hf_token=None):
    self._configs = dataset_configs
    self._cache_dir = cache_dir
    self._hf_token = hf_token
    self._datasets = []  # lazily populated
    self._offsets = None  # prefix-sum of lengths

  # -- lazy loading -------------------------------------------------------

  def _ensure_loaded(self):
    if self._datasets:
      return  # already loaded
    for cfg in self._configs:
      name = cfg["name"]
      source = cfg.get("source", "hf")
      try:
        if source == "hf":
          ds = _HFDataset(
              name=name,
              split=cfg.get("split", "train"),
              image_column=cfg.get("image_column"),
              cache_dir=self._cache_dir,
              hf_token=self._hf_token,
              min_resolution=cfg.get("min_resolution"),
          )
        elif source == "imagefolder":
          ds = _ImageFolderDataset(root_dir=name,
                                   min_resolution=cfg.get("min_resolution"),
                                   cache_dir=self._cache_dir)
        else:
          logger.warning(f"Unknown source '{source}' for dataset '{name}', skipping")
          continue
        # Probe length to verify the dataset loads
        n = len(ds)
        if n == 0:
          logger.warning(f"Dataset '{name}' has 0 images, skipping")
          continue
        logger.info(f"  + {name}: {n:,} images")
        self._datasets.append(ds)
      except Exception as exc:
        logger.warning(f"  Failed to initialize dataset '{name}': {exc}")

    # Build prefix-sum offsets for flat indexing
    self._rebuild_offsets()

  def _rebuild_offsets(self):
    offsets = [0]
    for ds in self._datasets:
      offsets.append(offsets[-1] + len(ds))
    self._offsets = offsets

  # -- public API ---------------------------------------------------------

  def __len__(self):
    self._ensure_loaded()
    return self._offsets[-1] if self._offsets else 0

  def __getitem__(self, idx):
    self._ensure_loaded()
    if not self._datasets:
      raise RuntimeError("No datasets loaded")
    # Binary search for the right dataset
    import bisect
    ds_idx = bisect.bisect_right(self._offsets, idx) - 1
    ds_idx = max(0, min(ds_idx, len(self._datasets) - 1))
    local_idx = idx - self._offsets[ds_idx]
    try:
      return self._datasets[ds_idx][local_idx]
    except IndexError:
      # Fallback: return first image from first dataset
      logger.warning(f"Index {idx} out of range, falling back to index 0")
      return self._datasets[0][0]

  @property
  def num_datasets(self):
    self._ensure_loaded()
    return len(self._datasets)

  def summary(self):
    """Return a human-readable summary of the ensemble."""
    self._ensure_loaded()
    lines = [
        f"DermoscopyEnsemble: {len(self):,} images from "
        f"{len(self._datasets)} dataset(s)"
    ]
    for i, ds in enumerate(self._datasets):
      lines.append(f"  [{i}] {ds.name}: {len(ds):,} images")
    return "\n".join(lines)
