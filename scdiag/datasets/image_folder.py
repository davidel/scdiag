"""ImageFolderDataset — fast local-image dataset backed by Path.rglob()."""

import logging
from pathlib import Path

import numpy as np
from PIL import Image

from scdiag.datasets.retry import getitem_retry
from scdiag.logging_utils import fatal

_IMAGE_EXTS = {
    # JPEG family
    ".jpg",
    ".jpeg",
    ".jpe",
    ".jfif",
    # PNG family
    ".png",
    ".apng",
    # GIF
    ".gif",
    # BMP / DIB
    ".bmp",
    ".dib",
    # TIFF
    ".tiff",
    ".tif",
    # WebP
    ".webp",
    # AVIF
    ".avif",
    ".avifs",
    # JPEG 2000
    ".jp2",
    ".j2k",
    # Multi-Picture Object (stereo / 3D)
    ".mpo",
    # PPM family (common in CV research)
    ".pbm",
    ".pgm",
    ".ppm",
    ".pnm",
    # TGA / Targa
    ".tga",
    ".icb",
    ".vda",
    ".vst",
    # DDS (DirectDraw Surface)
    ".dds",
    # PCX
    ".pcx",
    # ICO / Windows icons
    ".ico",
    # X11 bitmaps
    ".xbm",
    ".xpm",
    # Photoshop (flattened layers on load)
    ".psd",
}


class ImageFolderDataset:
  """Fast local-image dataset backed by ``Path.rglob()``.

  When ``with_labels=True`` and the root directory contains class
  subdirectories (e.g. ``train/AK/``, ``train/BCC/``), labels are
  extracted from the parent folder name and ``__getitem__`` returns
  ``(image, label)`` tuples.  Otherwise, returns bare images only.

  Parameters
  ----------
  root_dir : str | Path
      Root directory containing images (recursively scanned).
  with_labels : bool
      If ``True``, treat immediate subdirectories as class folders and
      expose labels via the standard dataset interface (``has_labels``,
      ``label_names``, ``labels_array``).
  """

  def __init__(self, root_dir, with_labels=False):
    self.name = str(root_dir)
    self._root_dir = Path(root_dir)
    self._with_labels = with_labels
    self._label_names = []
    self._label2id = {}
    self._labels = None  # np.ndarray or None
    self._paths = []

    if not self._root_dir.is_dir():
      logging.warning(f"ImageFolderDataset: '{self._root_dir}' is not a directory")
      return

    if self._with_labels:
      self._scan_with_labels()
    else:
      self._scan_flat()

  # ------------------------------------------------------------------
  # Label-free scan (original behaviour)
  # ------------------------------------------------------------------
  def _scan_flat(self):
    self._paths = sorted(p for p in self._root_dir.rglob("*")
                         if p.is_file() and p.suffix.lower() in _IMAGE_EXTS)
    logging.info(f"ImageFolderDataset: found {len(self._paths):,} images "
                 f"in '{self._root_dir}'")

  # ------------------------------------------------------------------
  # Label-aware scan — class subdirectories
  # ------------------------------------------------------------------
  def _scan_with_labels(self):
    # Collect class subdirectories (only immediate children that are dirs)
    class_dirs = sorted(d for d in self._root_dir.iterdir()
                        if d.is_dir() and not d.name.startswith("."))
    if not class_dirs:
      fatal(
          f"with_labels=True but no class subdirectories found in "
          f"'{self._root_dir}'",
          ValueError,
      )

    self._label_names = [d.name for d in class_dirs]
    self._label2id = {name: i for i, name in enumerate(self._label_names)}

    paths = []
    labels = []
    for label_id, class_dir in enumerate(class_dirs):
      class_paths = sorted(p for p in class_dir.rglob("*")
                           if p.is_file() and p.suffix.lower() in _IMAGE_EXTS)
      paths.extend(class_paths)
      labels.extend([label_id] * len(class_paths))

    self._paths = paths
    self._labels = np.array(labels, dtype=np.int64)

    logging.info(f"ImageFolderDataset: found {len(self._paths):,} images, "
                 f"{len(self._label_names)} classes in '{self._root_dir}'")
    for i, name in enumerate(self._label_names):
      count = int((self._labels == i).sum())
      logging.info(f"  class {name}: {count:,} images")

  # ------------------------------------------------------------------
  # Label interface (used by DatasetEnsemble)
  # ------------------------------------------------------------------
  @property
  def has_labels(self):
    return self._with_labels and self._labels is not None

  @property
  def label_names(self):
    return list(self._label_names)

  def labels_array(self):
    """Return a numpy array of integer labels for all samples."""
    return self._labels

  # ------------------------------------------------------------------
  # Core dataset protocol
  # ------------------------------------------------------------------
  def __len__(self):
    return len(self._paths)

  def __getitem__(self, idx):
    if idx >= len(self._paths):
      fatal(f"Index {idx} out of range for '{self.name}'", IndexError)

    def load(i):
      path = self._paths[i]
      return Image.open(path).convert("RGB")

    image = getitem_retry(idx, load, len(self._paths))
    if self._with_labels:
      return image, int(self._labels[idx])
    return image
