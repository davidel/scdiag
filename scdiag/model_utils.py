"""Shared model loading and preprocessing utilities."""

import logging
from contextlib import contextmanager

import numpy as np
import torch
from torch.utils.data import DataLoader
from torchvision.transforms import v2
from torchvision.transforms.functional import InterpolationMode

from scdiag.logging_utils import fatal
from scdiag.models import load_model, load_processor


@contextmanager
def model_mode(model, mode):
  """Context manager to temporarily set a model's training mode.

    Args:
        model: A ``torch.nn.Module``.
        mode: Either ``"train"`` or ``"eval"``.

    Yields the model itself so callers can write::

        with model_mode(model, "eval"):
            out = model(x)
  """
  was_training = model.training
  try:
    model.train(mode == "train")
    yield model
  finally:
    if was_training:
      model.train()
    else:
      model.eval()


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


def freeze_model(model, trainable_prefixes):
  """Freeze all parameters except those matching *trainable_prefixes*.

  Each parameter is thawed when *any* dotted segment of its name
  starts with one of the given prefixes.  For example, ``"head"``
  will match ``classifier.head.0.weight``.

  Args:
      model: A ``torch.nn.Module``.
      trainable_prefixes: Tuple of parameter name prefixes that should
          remain trainable (e.g. ``("head", "classifier")``).

  Returns:
      Tuple ``(total_params, trainable_params)`` with counts.
  """
  for p in model.parameters():
    p.requires_grad = False
  thawed = 0
  for name, p in model.named_parameters():
    segments = name.split(".")
    if any(seg.startswith(trainable_prefixes) for seg in segments):
      p.requires_grad = True
      thawed += 1
  total = sum(1 for _ in model.parameters())
  logging.info(f"Froze {total - thawed}/{total} parameters "
               f"(thawed prefixes: {', '.join(trainable_prefixes)})")
  return total, thawed


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
    *,
    device="cuda",
    cache_dir=None,
    model_kwargs=None,
    proc_kwargs=None,
    num_labels=None,
    image_size,
):
  """Load a fine-tuned model ready for inference.

    Args:
        model_name: HuggingFace model name or local path defining architecture,
                    or a registered custom model name (e.g. "convvit").
        checkpoint_path: Path to a raw state dictionary or a wrapped checkpoint
            containing ``model_state_dict``. Only model weights are loaded here.
        device: Target device.
        cache_dir: Optional HF cache directory.
        model_kwargs: Optional dict of extra kwargs forwarded to
            :func:`load_model` (e.g. ``{"depth": 6}``).
        proc_kwargs: Optional dict of extra kwargs forwarded to
            :func:`load_processor`.
        num_labels: Number of output classes.  When ``None`` the function
            recovers it from the checkpoint metadata (``num_labels`` or
            ``id2label``).
        image_size: Input resolution (pixels).

    Returns:
        (model, processor) tuple.
    """
  ckpt = torch.load(checkpoint_path, map_location="cpu", weights_only=True)

  # --- Resolve num_labels (priority: explicit > checkpoint metadata) ---
  if num_labels is None:
    num_labels = ckpt.get("num_labels")
  if num_labels is None and "id2label" in ckpt:
    num_labels = len(ckpt["id2label"])
  if num_labels is None:
    fatal(
        "Cannot infer num_labels from checkpoint. "
        "Expected 'num_labels' or 'id2label' in checkpoint metadata.", ValueError)

  # Load id2label mapping from checkpoint if available.
  id2label = ckpt.get("id2label")
  label2id = None
  if id2label is not None:
    label2id = {v: k for k, v in id2label.items()}

  # Use the unified registry — it dispatches to HF or custom transparently.
  model = load_model(
      model_name,
      num_labels=num_labels,
      id2label=id2label,
      label2id=label2id,
      image_size=image_size,
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
      image_size=image_size,
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

    The model is temporarily placed in eval mode during feature extraction
    so that BatchNorm/Dropout behave deterministically, and restored to
    its original mode afterwards.

    Raises:
        ValueError: if no ``classifier`` attribute is found and the model
                    does not implement the protocol method.
    """
  with model_mode(model, "eval"):
    return _extract_backbone_features_impl(model, pixel_values)


def _extract_backbone_features_impl(model, pixel_values):
  """Internal implementation of backbone feature extraction.

    .. note::

       The caller (``extract_backbone_features``) is responsible for
       setting the model to eval mode.  This function assumes the
       model is already in eval mode.
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
    fatal(
        "Cannot extract backbone features — no classification head found "
        "and the model does not implement extract_backbone_features().", ValueError)
  _, head_module = head
  captured = []

  def hook(module, inp, out):
    captured.append(inp[0])

  handle = head_module.register_forward_hook(hook)
  try:
    model(pixel_values=pixel_values)
  finally:
    handle.remove()

  if not captured:
    fatal(
        "Hook did not capture any features. "
        "Ensure the model's head module is called during forward().", ValueError)

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
