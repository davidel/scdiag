"""Shared model loading and preprocessing utilities."""

import logging

import numpy as np
import torch
from torch.utils.data import DataLoader
from torchvision.transforms import v2
from torchvision.transforms.functional import InterpolationMode

from scdiag.models import load_model, load_processor

DTYPE_MAP = {
    "float16": torch.float16,
    "bfloat16": torch.bfloat16,
}


def build_val_transform(processor, image_size):
  """Build validation transform: Resize → CenterCrop → Processor."""
  return v2.Compose([
      v2.Resize(image_size, interpolation=InterpolationMode.BICUBIC),
      v2.CenterCrop(image_size),
      v2.ToImage(),
      v2.ToDtype(torch.float32, scale=True),
      v2.Normalize(
          mean=processor.image_mean,
          std=processor.image_std,
      ),
  ])


def load_model_for_inference(
    model_name,
    checkpoint_path,
    device="cuda",
    cache_dir=None,
):
  """Load a fine-tuned model ready for inference.

    Args:
        model_name: HuggingFace model name or local path defining architecture,
                    or a registered custom model name (e.g. "convvit").
        checkpoint_path: Path to our .pt checkpoint containing model_state_dict.
        device: Target device.
        cache_dir: Optional HF cache directory.

    Returns:
        (model, processor) tuple.
    """
  import os

  ckpt = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
  state_dict = ckpt.get("model_state_dict", ckpt)

  # Infer num_labels from the checkpoint's classifier head.
  num_labels = None
  for key in ("classifier.weight", "head.3.weight", "head.weight"):
    if key in state_dict:
      num_labels = state_dict[key].shape[0]
      break
  if num_labels is None:
    raise ValueError("Cannot infer num_labels from checkpoint. "
                     f"Keys present: {list(state_dict.keys())[:10]}")

  # Load id2label mapping from checkpoint if available.
  id2label = None
  label2id = None
  if "id2label" in ckpt:
    id2label = ckpt["id2label"]
    label2id = {v: k for k, v in id2label.items()}

  # Use the unified registry — it dispatches to HF or custom transparently.
  model = load_model(
      model_name,
      num_labels=num_labels,
      id2label=id2label,
      label2id=label2id,
      image_size=224,
      device=torch.device(device),
      checkpoint_path=checkpoint_path,
      cache_dir=cache_dir,
  )

  # Restore label mapping if not provided via checkpoint.
  if id2label is None:
    n = num_labels
    model.config.id2label = {str(i): f"LABEL_{i}" for i in range(n)}
    model.config.label2id = {f"LABEL_{i}": str(i) for i in range(n)}
    logging.warning(
        f"Checkpoint has no id2label — using generic LABEL_0..LABEL_{n - 1}. "
        "Re-train with the updated code to get proper label names.")
  else:
    model.config.id2label = id2label
    model.config.label2id = label2id

  processor = load_processor(
      model_name,
      image_size=224,
      cache_dir=cache_dir,
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


def extract_backbone_features(model, pixel_values):
  """Extract features from the penultimate layer via a forward hook.

    Supports any model that has a ``.classifier`` attribute (common for HF
    image classifiers and custom wrappers that follow the same convention).

    1. If the model provides an ``extract_backbone_features(pixel_values)``
       method (the scdiag protocol), delegate to it.
    2. Otherwise, attach a one-shot forward-hook to ``model.classifier``
       to capture the input features just before the classification head.

    The hook is removed after extraction to avoid side-effects.

    Raises:
        ValueError: if no ``classifier`` attribute is found and the model
                    does not implement the protocol method.
    """
  # Protocol method — custom models and explicitly wrapped HF models.
  if hasattr(model, "extract_backbone_features"):
    out = model.extract_backbone_features(pixel_values)
    if isinstance(out, torch.Tensor):
      if out.ndim == 4:  # [B, C, H, W] → GAP
        return out.mean(dim=(2, 3))
      return out  # already [B, D]
    if isinstance(out, list):
      last = out[-1]
      return last.mean(dim=(2, 3)) if last.ndim == 4 else last
    return extract_features(out)

  # Fallback: hook-based extraction via ``model.classifier``.
  classifier = getattr(model, "classifier", None)
  if classifier is None:
    raise ValueError("Cannot find a classifier head on this model. "
                     "Implement extract_backbone_features() on the model wrapper.")

  captured = []

  def _hook(module, inp, out):
    captured.append(inp[0])

  handle = classifier.register_forward_hook(_hook)
  try:
    model(pixel_values=pixel_values)
  finally:
    handle.remove()

  if not captured:
    raise ValueError(
        "Hook did not capture any features. "
        "Ensure the model has a classifier head that is called during forward().")

  return captured[0]


def collect_features(model, dataset, device, batch_size=128):
  """Extract backbone features for all samples in a dataset.

    Args:
        model: Any model (HF or custom) in eval mode, on device.
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
      features = extract_backbone_features(model, pixel_values)
      features_list.append(features.cpu().numpy())
      labels_list.append(batch_labels.numpy())

  return np.concatenate(features_list), np.concatenate(labels_list)
