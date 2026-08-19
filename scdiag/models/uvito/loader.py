"""UVito loader — called by the model registry.

Wraps the raw UVito so that it conforms to the scdiag protocol
(forward → .logits, config.id2label, extract_backbone_features).
"""

import logging
import os
from types import SimpleNamespace

import torch
import torch.nn as nn

from scdiag.checkpointing import load_checkpoint_weights
from scdiag.models.registry import ModelOutput, register_model, register_processor
from scdiag.models.uvito.model import UVito


class UVitoForClassification(nn.Module):
  """Thin wrapper that makes UVito match the scdiag / HF interface.

  * ``forward(pixel_values=)`` → ``ModelOutput`` with ``.logits``
  * ``config.id2label`` / ``config.label2id`` accessible
  * ``extract_backbone_features(pixel_values)`` for XGBoost
  """

  def __init__(self, model, config):
    super().__init__()
    self.model = model
    self.config = config

  def forward(self, pixel_values=None, **kwargs):
    logits = self.model(pixel_values)
    return ModelOutput(logits=logits)

  def extract_backbone_features(self, pixel_values):
    """Return the CLS-flattened representation (before head_norm/mlp_head).

    Returns
    -------
    torch.Tensor
        Shape ``(B, num_cls_tokens * transformer_dim)``.
    """
    with torch.no_grad():
      return self.model._backbone_features(pixel_values)


@register_model("uvito")
def load_uvito(
    *,
    num_labels,
    id2label,
    label2id,
    image_size,
    device,
    checkpoint_path=None,
    **kwargs,
):
  """Instantiate UVito model.

  Parameters
  ----------
  num_labels : int
      Number of output classes.
  id2label : dict
  label2id : dict
  image_size : int
  device : str or torch.device
  checkpoint_path : str or None
  """
  config = SimpleNamespace(
      num_labels=num_labels,
      id2label=id2label,
      label2id=label2id,
      image_size=image_size,
  )

  model = UVito(
      num_classes=num_labels,
      img_size=image_size,
  )

  if checkpoint_path and os.path.isfile(checkpoint_path):
    load_checkpoint_weights(
        path=checkpoint_path,
        model=model,
        device="cpu",
        strict=False,
    )
    logging.info("Loaded UVito checkpoint: %s", checkpoint_path)
  else:
    logging.info("UVito: training from random init (no checkpoint)")

  wrapped = UVitoForClassification(model, config)
  wrapped.to(device)

  logging.info(
      "UVito ready — params: %.1fM",
      sum(p.numel() for p in wrapped.parameters()) / 1e6,
  )

  return wrapped


@register_processor("uvito")
def load_uvito_processor(*, image_size=224, **kwargs):
  """Return a UVitoProcessor for the given *image_size*."""
  from scdiag.models.uvito.processor import UVitoProcessor
  return UVitoProcessor(image_size=image_size)
