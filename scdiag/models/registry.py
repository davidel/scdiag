"""Model registry — unified loading for HF and custom models.

The registry exposes two entry points:

- ``load_model(name, ...)``  — returns a ``torch.nn.Module``
- ``load_processor(name, ...)`` — returns a processor object that
  exposes ``image_mean`` and ``image_std`` attributes.

Both functions transparently dispatch to the appropriate backend
(custom or HuggingFace) based on *name*.
"""

import logging

import torch

# Public registry

REGISTRY = {}

# Maps model name → loader fn
_MODEL_REGISTRY = {}
# Maps processor name → loader fn
_PROCESSOR_REGISTRY = {}

# Protocol helpers


class ModelOutput:
  """Lightweight container matching the ``.logits`` interface of
    HuggingFace model outputs.

    Custom models whose ``forward()`` returns a raw tensor should wrap
    it before returning::

        return ModelOutput(logits)
    """

  def __init__(self, logits):
    self.logits = logits


def register_model(name):
  """Decorator to register a custom model loader under *name*.

    Usage::

        @register_model("convvit")
        def load_convvit(num_labels, id2label, label2id, image_size,
                         device, checkpoint_path, **kwargs):
            ...
            return model
    """

  def wrapper(fn):
    if name in _MODEL_REGISTRY:
      raise ValueError(f"Model '{name}' is already registered.")
    _MODEL_REGISTRY[name] = fn
    # Also keep the legacy REGISTRY in sync for backwards compatibility.
    REGISTRY[name] = fn
    return fn

  return wrapper


def register_processor(name):
  """Decorator to register a custom processor loader under *name*.

    The loader must return an object with ``image_mean`` and ``image_std``
    attributes (list of floats).

    Usage::

        @register_processor("convvit")
        def load_convvit_processor(image_size, **kwargs):
            ...
            return processor
    """

  def wrapper(fn):
    if name in _PROCESSOR_REGISTRY:
      raise ValueError(f"Processor '{name}' is already registered.")
    _PROCESSOR_REGISTRY[name] = fn
    return fn

  return wrapper


def is_custom_model(model_name):
  """Return *True* if *model_name* maps to a registered custom model."""
  return model_name in _MODEL_REGISTRY


# Unified loading API


def load_model(
    model_name,
    *,
    num_labels,
    id2label=None,
    label2id=None,
    image_size=224,
    device="cpu",
    checkpoint_path=None,
    cache_dir=None,
    **kwargs,
):
  """Load a model by *model_name* — custom or HuggingFace.

    Custom models are dispatched to the function registered via
    ``@register_model``.  HuggingFace models are loaded via
    ``AutoModelForImageClassification``.

    Returns:
        ``torch.nn.Module``
    """
  from transformers import (
      AutoModelForImageClassification,)  # local to avoid top-level import

  if model_name in _MODEL_REGISTRY:
    logging.info("Loading custom model '%s' from registry.", model_name)
    return _MODEL_REGISTRY[model_name](
        num_labels=num_labels,
        id2label=id2label,
        label2id=label2id,
        image_size=image_size,
        device=device,
        checkpoint_path=checkpoint_path,
        **kwargs,
    )

  # HuggingFace path ---------------------------------------------------
  logging.info("Loading HuggingFace model '%s'.", model_name)
  model = AutoModelForImageClassification.from_pretrained(
      model_name,
      num_labels=num_labels,
      id2label=id2label,
      label2id=label2id,
      cache_dir=cache_dir,
      ignore_mismatched_sizes=True,
  )
  model.to(device)
  return model


def load_processor(
    model_name,
    *,
    image_size=224,
    cache_dir=None,
    **kwargs,
):
  """Load an image processor by *model_name* — custom or HuggingFace.

    Custom processors are dispatched to the function registered via
    ``@register_processor``.  HuggingFace processors are loaded via
    ``AutoImageProcessor``.

    The returned object must expose ``image_mean`` and ``image_std``
    (list of floats) so that ``build_transforms`` can use them.

    Returns:
        processor object with ``image_mean`` / ``image_std`` attributes.
    """
  from transformers import AutoImageProcessor  # local to avoid top-level import

  if model_name in _PROCESSOR_REGISTRY:
    logging.info("Loading custom processor '%s' from registry.", model_name)
    return _PROCESSOR_REGISTRY[model_name](
        image_size=image_size,
        **kwargs,
    )

  # HuggingFace path ---------------------------------------------------
  logging.info("Loading HuggingFace processor '%s'.", model_name)
  return AutoImageProcessor.from_pretrained(model_name, cache_dir=cache_dir)


# Legacy helpers (kept for backwards compatibility)


def load_custom_model(
    model_name,
    *,
    num_labels,
    id2label=None,
    label2id=None,
    image_size=224,
    device="cpu",
    checkpoint_path=None,
    **kwargs,
):
  """Load a custom model by *model_name*.

    Returns:
        ``(model, processor)`` tuple.  Prefer the unified ``load_model``
        and ``load_processor`` entry-points for new code.

    Raises:
        ValueError: if *model_name* is not registered.
    """
  if model_name not in _MODEL_REGISTRY:
    raise ValueError(f"Unknown custom model '{model_name}'. "
                     f"Available: {list(_MODEL_REGISTRY.keys())}")
  model = _MODEL_REGISTRY[model_name](
      num_labels=num_labels,
      id2label=id2label,
      label2id=label2id,
      image_size=image_size,
      device=device,
      checkpoint_path=checkpoint_path,
      **kwargs,
  )

  # Load processor if a custom one is registered; otherwise return None.
  processor = None
  if model_name in _PROCESSOR_REGISTRY:
    processor = _PROCESSOR_REGISTRY[model_name](
        image_size=image_size,
        **kwargs,
    )

  return model, processor
