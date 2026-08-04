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

# ---------------------------------------------------------------------------
# Backbone extraction
# ---------------------------------------------------------------------------


def _find_head_module(model):
  """Locate the classification head on a model.

    Searches for common head attribute names in the order most HF and
    custom models use them.  Returns ``(attr_name, module)`` or ``None``.
    """
  for attr in ("classifier", "head", "fc"):
    mod = getattr(model, attr, None)
    if mod is not None and isinstance(mod, torch.nn.Module):
      return attr, mod
  return None


def get_backbone(model):
  """Return the backbone encoder as a standalone ``nn.Module``.

    The returned module should accept raw patch / feature inputs and
    produce the penultimate representations (i.e. everything *before*
    the classification head).

    Resolution order
    ----------------
    1. If *model* exposes a ``get_backbone()`` method (scdiag protocol),
       delegate to it.
    2. If *model* has a ``.model`` attribute (e.g.
       ``ConvViTForClassification``), return ``model.model``.
    3. Otherwise (HF model), find the classification head via
       ``_find_head_module``, replace it with ``Identity``, and return
       the model itself as the backbone.
    """
  # 1. Explicit protocol
  if hasattr(model, "get_backbone") and callable(model.get_backbone):
    return model.get_backbone()

  # 2. Wrapper with .model attribute
  if hasattr(model, "model") and isinstance(model.model, torch.nn.Module):
    return model.model

  # 3. HF-style model — neutralise the head and return the model itself
  head = _find_head_module(model)
  if head is not None:
    attr_name, _ = head
    setattr(model, attr_name, torch.nn.Identity())
    logging.info("Replaced model.%s with Identity for backbone extraction.", attr_name)
  else:
    logging.warning("No classification head found; returning model as-is.")
  return model


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
    model_kwargs=None,
    proc_kwargs=None,
):
  """Load a fine-tuned model ready for inference.

    Args:
        model_name: HuggingFace model name or local path defining architecture,
                    or a registered custom model name (e.g. "convvit").
        checkpoint_path: Path to our .pt checkpoint containing model_state_dict.
        device: Target device.
        cache_dir: Optional HF cache directory.
        model_kwargs: Optional dict of extra kwargs forwarded to
            :func:`load_model` (e.g. ``{"depth": 6}``).
        proc_kwargs: Optional dict of extra kwargs forwarded to
            :func:`load_processor`.

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
      **(model_kwargs or {}),
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
      **(proc_kwargs or {}),
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
  # 1. Scdiag protocol
  if hasattr(model, "extract_backbone_features") and callable(
      model.extract_backbone_features):
    out = model.extract_backbone_features(pixel_values)
    if isinstance(out, torch.Tensor):
      return out.mean(dim=(2, 3)) if out.ndim == 4 else out
    if isinstance(out, list):
      last = out[-1]
      return last.mean(dim=(2, 3)) if last.ndim == 4 else last
    return extract_features(out)

  # 2. HF hidden states
  try:
    out = model(pixel_values=pixel_values, output_hidden_states=True)
    if hasattr(out, "hidden_states") and out.hidden_states:
      last = out.hidden_states[-1]  # (B, N, D) or (B, D, H, W)
      if last.ndim == 4:
        return last.mean(dim=(2, 3))  # spatial average-pool
      if last.ndim == 3:
        return last[:, 0]  # CLS token
      return last
  except TypeError:
    pass  # model doesn't accept output_hidden_states

  # 3. Hook-based fallback via the classification head
  head = _find_head_module(model)
  if head is None:
    raise ValueError("Cannot extract backbone features — no classification head found "
                     "and the model does not implement extract_backbone_features().")
  _, head_module = head
  captured = []

  def _hook(module, inp, out):
    captured.append(inp[0])

  handle = head_module.register_forward_hook(_hook)
  try:
    model(pixel_values=pixel_values)
  finally:
    handle.remove()

  if not captured:
    raise ValueError("Hook did not capture any features. "
                     "Ensure the model's head module is called during forward().")

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
