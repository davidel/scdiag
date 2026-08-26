"""ImageFolderDataset — fast local-image dataset backed by Path.rglob()."""

import logging
from pathlib import Path

import numpy as np
from PIL import Image

from scdiag.datasets.retry import getitem_retry
from scdiag.logging_utils import fatal

_IMAGE_EXTS = {
    ".jpg",
    ".jpeg",
    ".jpe",
    ".jfif",
    ".png",
    ".apng",
    ".gif",
    ".bmp",
    ".dib",
    ".tiff",
    ".tif",
    ".webp",
    ".avif",
    ".avifs",
    ".jp2",
    ".j2k",
    ".mpo",
    ".pbm",
    ".pgm",
    ".ppm",
    ".pnm",
    ".tga",
    ".icb",
    ".vda",
    ".vst",
    ".dds",
    ".pcx",
    ".ico",
    ".xbm",
    ".xpm",
    ".psd",
}


class ImageFolderDataset:
  """Fast local-image dataset backed by ``Path.rglob()``.

  Labels are auto-detected from relative paths.  Each image's path
  relative to *root_dir* is split into components:

  - 3 components (``split/label/file``) → labels present
  - 2 components (``split/file``) → no labels
  - A mix of both → fatal error

  ``__getitem__`` always returns a dict:

  - With labels: ``{"image": PIL.Image, "label": int}``
  - Without labels: ``{"image": PIL.Image}``
  """

  def __init__(self, root_dir):
    self.name = str(root_dir)
    self._root_dir = Path(root_dir)
    self._label_names = []
    self._label2id = {}
    self._labels = None
    self._paths = []

    if not self._root_dir.is_dir():
      fatal(f"ImageFolderDataset: '{self._root_dir}' is not a directory",
           ValueError)

    self._scan()

  def _scan(self):
    all_paths = sorted(p for p in self._root_dir.rglob("*")
                       if p.is_file() and p.suffix.lower() in _IMAGE_EXTS)

    has_labels = []
    no_labels = []
    for p in all_paths:
      rel = p.relative_to(self._root_dir)
      parts = rel.parts
      if len(parts) == 3:
        has_labels.append(p)
      elif len(parts) in (1, 2):
        no_labels.append(p)
      else:
        fatal(
            f"Unexpected path depth {len(parts)} for '{rel}' "
            f"(expected 1, 2, or 3 components)",
            ValueError
        )

    if has_labels and no_labels:
      fatal(
          f"Mixed layout: {len(has_labels)} images in split/label/file "
          f"but {len(no_labels)} images in split/file format",
          ValueError
      )

    self._paths = has_labels if has_labels else no_labels

    if has_labels:
      label_set = sorted({p.relative_to(self._root_dir).parts[1] for p in has_labels})
      self._label_names = label_set
      self._label2id = {name: i for i, name in enumerate(label_set)}
      self._labels = np.array(
          [self._label2id[p.relative_to(self._root_dir).parts[1]] for p in self._paths],
          dtype=np.int64,
      )

    logging.info(f"ImageFolderDataset: found {len(self._paths):,} images " +
                 (f", {len(self._label_names)} classes " if self._label_names else "") +
                 f"in '{self._root_dir}'")
    for name in self._label_names:
      count = int((self._labels == self._label2id[name]).sum())
      logging.info(f"  class {name}: {count:,} images")

  @property
  def has_labels(self):
    return self._labels is not None

  @property
  def num_labels(self):
    return len(self._label_names)

  @property
  def label_names(self):
    return list(self._label_names)

  def labels_array(self):
    return self._labels

  def __len__(self):
    return len(self._paths)

  def __getitem__(self, idx):
    if idx >= len(self._paths):
      fatal(f"Index {idx} out of range for '{self.name}'", IndexError)

    def load(i):
      return Image.open(self._paths[i]).convert("RGB")

    image, gidx = getitem_retry(idx, load, len(self._paths))
    if self._labels is not None:
      return {"image": image, "label": int(self._labels[gidx])}
    return {"image": image}
