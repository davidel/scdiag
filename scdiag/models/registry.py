"""Model registry — dispatch between HF and custom models."""

import logging

import torch

# ---------------------------------------------------------------------------
# Public registry
# ---------------------------------------------------------------------------

REGISTRY = {}


def register_model(name):
  """Decorator to register a custom model loader under *name*.

    Usage::

        @register_model("convvit")
        def load_convvit(...):
            ...
    """

  def decorator(fn):
    if name in REGISTRY:
      raise ValueError(f"Model name '{name}' is already registered.")
    REGISTRY[name] = fn
    logging.debug("Registered custom model: %s", name)
    return fn

  return decorator


# ---------------------------------------------------------------------------
# Protocol helpers
# ---------------------------------------------------------------------------


class ModelOutput:
  """Lightweight container matching the ``.logits`` interface of
    HuggingFace model outputs.

    Custom models whose ``forward()`` returns a raw tensor should wrap
    it before returning::

        return ModelOutput(logits)
    """

  def __init__(self, logits):
    self.logits = logits


def is_custom_model(model_name):
  """Return ``True`` if *model_name* is a registered custom model."""
  return model_name is not None and model_name in REGISTRY


def load_custom_model(
    model_name,
    *,
    num_labels,
    id2label,
    label2id,
    image_size,
    device,
    checkpoint_path=None,
    **kwargs,
):
  """Load a custom model and its processor from the registry.

    Returns
    -------
    (model, processor)
        *model* must satisfy the protocol:
        - ``forward(pixel_values=images)`` → ``ModelOutput`` with ``.logits``
        - ``config.id2label`` / ``config.label2id`` exist
        - ``extract_backbone_features(pixel_values)`` works
        - ``state_dict()`` / ``load_state_dict()`` work

        *processor* must satisfy:
        - ``__call__(images)`` → ``pixel_values`` tensor ``[B, C, H, W]``
    """
  if model_name not in REGISTRY:
    raise ValueError(f"Unknown custom model '{model_name}'. "
                     f"Available: {list(REGISTRY.keys())}")
  return REGISTRY[model_name](
      num_labels=num_labels,
      id2label=id2label,
      label2id=label2id,
      image_size=image_size,
      device=device,
      checkpoint_path=checkpoint_path,
      **kwargs,
  )
