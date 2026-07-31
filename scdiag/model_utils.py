"""Shared model loading and preprocessing utilities."""

import logging

import numpy as np
import torch
from torch.utils.data import DataLoader
from torchvision.transforms import v2
from torchvision.transforms.functional import InterpolationMode
from transformers import AutoImageProcessor, AutoModelForImageClassification

DTYPE_MAP = {
    "float16": torch.float16,
    "bfloat16": torch.bfloat16,
}


def build_val_transform(processor, image_size: int):
  """Validation/inference transform matching train.py's val pipeline."""

  resize = v2.Resize(
      (image_size, image_size),
      interpolation=InterpolationMode.BICUBIC,
  )
  to_image = v2.ToImage()
  to_float = v2.ToDtype(torch.float32, scale=True)
  normalize = v2.Normalize(
      mean=processor.image_mean, std=processor.image_std
  )

  def _transform(image):
    return normalize(to_float(to_image(resize(image))))

  return _transform


def load_model_for_inference(
    model_name: str,
    checkpoint_path: str,
    device: str = "cuda",
    cache_dir: str | None = None,
):
  """Load a fine-tuned model ready for inference.

  Args:
      model_name: HuggingFace model name or local path defining architecture.
      checkpoint_path: Path to our .pt checkpoint containing model_state_dict.
      device: Target device.
      cache_dir: Optional HF cache directory.

  Returns:
      (model, processor) tuple.
  """
  ckpt = torch.load(checkpoint_path, map_location="cpu")
  state_dict = ckpt["model_state_dict"]

  # Infer num_labels from the checkpoint's classifier head.
  num_labels = state_dict["classifier.weight"].shape[0]

  processor = AutoImageProcessor.from_pretrained(
      model_name, cache_dir=cache_dir
  )
  model = AutoModelForImageClassification.from_pretrained(
      model_name, num_labels=num_labels, ignore_mismatched_sizes=True,
      cache_dir=cache_dir
  )
  missing, unexpected = model.load_state_dict(state_dict, strict=False)
  logging.info(
      f"Loaded weights from {checkpoint_path}: "
      f"{len(missing)} missing, {len(unexpected)} unexpected keys"
  )

  # Restore label mapping.  New checkpoints include id2label; old ones
  # don't, so we generate generic LABEL_0..LABEL_N matching the classifier.
  if "id2label" in ckpt:
    model.config.id2label = ckpt["id2label"]
    model.config.label2id = {v: k for k, v in ckpt["id2label"].items()}
  else:
    n = num_labels
    model.config.id2label = {str(i): f"LABEL_{i}" for i in range(n)}
    model.config.label2id = {f"LABEL_{i}": str(i) for i in range(n)}
    logging.warning(
        f"Checkpoint has no id2label — using generic LABEL_0..LABEL_{n - 1}. "
        "Re-train with the updated code to get proper label names."
    )
  model.to(device).eval()
  return model, processor


def extract_features(outputs):
  """Extract features from a HuggingFace model output.

  If pooler_output is available (already pooled to [N, hidden_size]),
  use it directly.  Otherwise, apply global average pooling to
  last_hidden_state.

  Args:
      outputs: Model output object (e.g. BaseModelOutputWithPoolingAndNoAttention).

  Returns:
      torch.Tensor of shape [N, hidden_size].
  """
  if outputs.pooler_output is not None:
    return outputs.pooler_output
  return outputs.last_hidden_state.mean([-2, -1])


def collect_features(model, dataset, device, batch_size=128):
  """Extract backbone features for all samples in a dataset.

  Args:
      model: ConvNextV2ForImageClassification (eval mode, on device).
      dataset: HFDatasetProxy (already has transform applied).
      device: torch device.
      batch_size: batch size for feature extraction (default 128).

  Returns:
      features: np.ndarray of shape [N, hidden_size], dtype float32.
      labels: np.ndarray of shape [N], dtype int64.
  """
  model.eval()
  loader = DataLoader(dataset, batch_size=batch_size, shuffle=False)

  features_list = []
  labels_list = []

  with torch.no_grad():
    for pixel_values, batch_labels in loader:
      pixel_values = pixel_values.to(device)
      outputs = model.convnextv2(pixel_values)
      features = extract_features(outputs)
      features_list.append(features.cpu().numpy())
      labels_list.append(batch_labels.numpy())

  return np.concatenate(features_list), np.concatenate(labels_list)
