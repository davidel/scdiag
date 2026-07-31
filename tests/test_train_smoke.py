"""Smoke test for train.py end-to-end.

Patches all HuggingFace hub calls and runs training for 1 epoch on a tiny
synthetic dataset with a mock model.  Verifies that checkpoints are written.
"""

import os
from collections import namedtuple
from unittest.mock import patch

import numpy as np
import torch
from datasets import Dataset
from PIL import Image

LogitsOutput = namedtuple("LogitsOutput", ["logits"])


class TinyModel(torch.nn.Module):
  """Minimal torch.nn.Module that accepts pixel_values and returns .logits."""

  def __init__(self, num_labels=3):
    super().__init__()
    self.fc = torch.nn.Linear(3 * 64 * 64, num_labels)

  def forward(self, pixel_values, **kwargs):
    b = pixel_values.shape[0]
    flat = pixel_values.view(b, -1)
    return LogitsOutput(logits=self.fc(flat))


class TinyProcessor:
  """Mimics AutoImageProcessor with .image_mean / .image_std."""

  image_mean = [0.5, 0.5, 0.5]
  image_std = [0.5, 0.5, 0.5]


def _make_dataset(n_total=12, num_classes=3, size=64):
  """Return a raw Dataset with random images and labels."""
  imgs = [
      Image.fromarray(np.random.randint(0, 256, (size, size, 3), dtype=np.uint8))
      for _ in range(n_total)
  ]
  labels = [i % num_classes for i in range(n_total)]
  return Dataset.from_dict({"image": imgs, "label": labels})


def test_train_smoke(tmp_path):
  """Run main() for 1 epoch, verify checkpoint files are created."""
  ds = _make_dataset()
  ckpt_base = str(tmp_path / "ckpts" / "model")

  test_args = [
      "train.py",
      "--model",
      "dummy/tiny-model",
      "--dataset",
      "dummy/dataset",
      "--epochs",
      "1",
      "--batch_size",
      "4",
      "--image_size",
      "64",
      "--checkpoint",
      ckpt_base,
      "--log_every",
      "2",
      "--num_workers",
      "0",
      "--log_level",
      "WARNING",
  ]

  with patch("sys.argv", test_args), \
       patch("scdiag.train.load_dataset", return_value=ds), \
       patch(
           "scdiag.train.AutoImageProcessor.from_pretrained",
           return_value=TinyProcessor(),
       ), \
       patch(
           "scdiag.train.AutoModelForImageClassification.from_pretrained",
           return_value=TinyModel(num_labels=3),
       ):
    from scdiag.train import main
    main()

  latest = ckpt_base + "_latest.pt"
  assert os.path.exists(latest), f"Missing {latest}"

  ckpt = torch.load(latest, map_location="cpu", weights_only=False)
  assert "model_state_dict" in ckpt
  assert "optimizer_state_dict" in ckpt
  assert "scheduler_state_dict" in ckpt
  assert "epoch" in ckpt
  assert "best_top1" in ckpt
