"""Unified skin-lesion image dataset for self-supervised pre-training.

Stitches together multiple HuggingFace datasets (or pre-downloaded image
directories) into a single :class:`torch.utils.data.Dataset` with flat
indexing.  Only images are exposed — no labels — making this suitable for
SimMIM / MAE pre-training.
"""

import logging

from PIL import Image

from scdiag.datasets.hf_proxy import HFDatasetProxy
from scdiag.datasets.image_folder import ImageFolderDataset
from scdiag.logging_utils import fatal


class _HFDataset:
  """Wrapper around a HuggingFace ``datasets.Dataset``."""

  def __init__(self,
               name,
               split="train",
               image_column=None,
               cache_dir=None,
               hf_token=None):
    self.name = name
    self._image_column = image_column
    from datasets import load_dataset
    logging.info(f"Loading HF dataset '{self.name}' (split={split}) …")
    ds = load_dataset(self.name,
                      split=split,
                      cache_dir=cache_dir,
                      token=hf_token)
    if self._image_column is None:
      col = HFDatasetProxy.detect_image_column(ds)
      if col is None:
        fatal(
            f"Cannot auto-detect image column in '{self.name}'. "
            f"Columns: {ds.column_names}. Set 'image_column' explicitly.", ValueError)
      self._image_column = col
    self._ds = HFDatasetProxy.normalize_image_column(ds, self._image_column)

  def __len__(self):
    return len(self._ds)

  def __getitem__(self, idx):
    row = self._ds[idx]
    image = row[self._image_column]
    if not isinstance(image, Image.Image):
      image = Image.open(image)
    image = image.convert("RGB")
    return image


class DatasetEnsemble:
  """Concatenation of multiple skin-lesion image datasets.

    Images are returned as RGB PIL :class:`PIL.Image.Image` objects —
    the caller (typically a :class:`torchvision.transforms.v2.Compose`
    pipeline) handles resizing, tensor conversion, and normalization.

    Args:
        dataset_configs: List of dicts describing each constituent dataset.
            Supported keys:

            - ``"name"`` (str, required): HuggingFace dataset ID **or**
              local directory path.
            - ``"source"`` (str): ``"hf"`` (default), ``"imagefolder"``.
            - ``"split"`` (str): Dataset split (default ``"train"``).
            - ``"image_column"`` (str): Column name for images (auto-detected
              if omitted).
        cache_dir: HuggingFace cache directory.
        hf_token: HuggingFace token for gated datasets (e.g. Derm1M).
    """

  def __init__(self, dataset_configs, cache_dir=None, hf_token=None, strict=False):
    self._datasets = []
    self._offsets = None
    for cfg in dataset_configs:
      name = cfg["name"]
      source = cfg.get("source", "hf")
      try:
        if source == "hf":
          ds = _HFDataset(
              name=name,
              split=cfg.get("split", "train"),
              image_column=cfg.get("image_column"),
              cache_dir=cache_dir,
              hf_token=hf_token,
          )
        elif source == "imagefolder":
          ds = ImageFolderDataset(root_dir=name)
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
          fatal(f"Failed to initialize dataset '{name}' ({source})", RuntimeError)
        logging.exception("  Failed to initialize dataset '%s' (%s); skipping.", name,
                          source)

    offsets = [0]
    for ds in self._datasets:
      offsets.append(offsets[-1] + len(ds))
    self._offsets = offsets
    if not self._datasets:
      fatal("No datasets loaded successfully", RuntimeError)

  def __len__(self):
    return self._offsets[-1] if self._offsets else 0

  def _get_item(self, idx):
    """Resolve a valid global index and load its image."""
    import bisect
    ds_idx = bisect.bisect_right(self._offsets, idx) - 1
    local_idx = idx - self._offsets[ds_idx]
    return self._datasets[ds_idx][local_idx]

  def __getitem__(self, idx):
    if not self._datasets:
      fatal("No datasets loaded", RuntimeError)
    if idx < 0 or idx >= len(self):
      fatal(f"Index {idx} out of range for ensemble of length {len(self)}", IndexError)
    return self._get_item(idx)

  @property
  def num_datasets(self):
    return len(self._datasets)

  def summary(self):
    """Return a human-readable summary of the ensemble."""
    lines = [(f"DatasetEnsemble: {len(self):,} images from "
              f"{len(self._datasets)} dataset(s)")]
    for i, ds in enumerate(self._datasets):
      lines.append(f"  [{i}] {ds.name}: {len(ds):,} images")
    return "\n".join(lines)
