"""ImageFolderDataset — fast local-image dataset backed by Path.rglob()."""

import logging
from pathlib import Path

from PIL import Image

# Recognised image file extensions.
_IMAGE_EXTS = {".jpg", ".jpeg", ".png", ".bmp", ".tiff", ".tif", ".webp"}


class ImageFolderDataset:
  """Reads images from a flat (or nested) directory tree.

    This is a lightweight drop-in replacement for
    ``datasets.load_dataset("imagefolder", ...)`` that avoids the
    multi-minute Arrow conversion overhead for large on-disk image
    collections.  Only images are returned — no labels, no metadata —
    making it suitable for self-supervised pre-training (SimMIM / MAE).

    Parameters
    ----------
    root_dir : str | Path
        Root directory containing images (recursively scanned).
    min_resolution : int | None
        If set, images whose width *or* height is smaller than this
        value are silently skipped (an ``IndexError`` is raised so the
        ``DataLoader`` retries the next index).
    """

  def __init__(self, root_dir, min_resolution=None):
    self.name = str(root_dir)
    self.root_dir = Path(root_dir)
    self.min_resolution = min_resolution
    self._paths: list[Path] = []

  # Internal helpers

  def _ensure_loaded(self):
    if self._paths:
      return
    if not self.root_dir.is_dir():
      logging.warning(f"ImageFolderDataset: '{self.root_dir}' is not a directory")
      return
    self._paths = sorted(
        p for p in self.root_dir.rglob("*") if p.suffix.lower() in _IMAGE_EXTS)
    logging.info(
        f"ImageFolderDataset: found {len(self._paths):,} images in '{self.root_dir}'")

  # Dataset protocol

  def __len__(self):
    self._ensure_loaded()
    return len(self._paths)

  def __getitem__(self, idx):
    self._ensure_loaded()
    if idx >= len(self._paths):
      raise IndexError(f"Index {idx} out of range for '{self.name}'")
    path = self._paths[idx]
    image = Image.open(path).convert("RGB")
    if self.min_resolution is not None:
      if image.width < self.min_resolution or image.height < self.min_resolution:
        raise IndexError(f"Image too small: {image.size}, min={self.min_resolution}")
    return image
