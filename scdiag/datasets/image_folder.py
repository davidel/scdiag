"""ImageFolderDataset — fast local-image dataset backed by Path.rglob()."""

import logging
from pathlib import Path

from PIL import Image

from scdiag.logging_utils import fatal

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
    """

  def __init__(self, root_dir):
    self.name = str(root_dir)
    self._root_dir = Path(root_dir)
    self._paths = []
    if not self._root_dir.is_dir():
      logging.warning(f"ImageFolderDataset: '{self._root_dir}' is not a directory")
      return
    self._paths = sorted(
        p for p in self._root_dir.rglob("*") if p.suffix.lower() in _IMAGE_EXTS)
    logging.info(
        f"ImageFolderDataset: found {len(self._paths):,} images in '{self._root_dir}'")

  def __len__(self):
    return len(self._paths)

  def __getitem__(self, idx):
    if idx >= len(self._paths):
      fatal(f"Index {idx} out of range for '{self.name}'", IndexError)
    path = self._paths[idx]
    image = Image.open(path).convert("RGB")
    return image
