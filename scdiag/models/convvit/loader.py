"""ConvViT loader — called by the model registry.

Wraps the raw ConvViT so that it conforms to the scdiag protocol
(forward → .logits, config.id2label, extract_backbone_features).
"""

import logging
import os
from types import SimpleNamespace

import torch
import torch.nn as nn

from scdiag.models.registry import ModelOutput, register_model, register_processor
from scdiag.models.convvit.model import CustomPatchTransformer
from scdiag.models.convvit.processor import ConvViTProcessor

# Wrapped model — protocol adapter


class ConvViTForClassification(nn.Module):
  """Thin wrapper that makes ConvViT match the scdiag / HF interface.

    * ``forward(pixel_values=)`` → ``ModelOutput`` with ``.logits``
    * ``config.id2label`` / ``config.label2id`` accessible
    * ``extract_backbone_features(pixel_values)`` for XGBoost
    """

  def __init__(self, model, config):
    super().__init__()
    self.model = model
    self.config = config

  def forward(self, pixel_values=None, **kwargs):
    logits = self.model(pixel_values)  # [B, num_classes]
    return ModelOutput(logits)

  def extract_backbone_features(self, pixel_values):
    """Return intermediate feature maps from the conv patch embedding.

        The ConvViT uses ``ConvPatchEmbedding`` (a convolutional stem) rather
        than a full ConvNeXtV2 backbone, so we extract the spatial feature
        maps from each convolutional block in the stem.

        Returns
        -------
        list[torch.Tensor]
            One tensor per conv block, each shaped ``[B, C, H, W]``.
        """
    features = []
    # ConvPatchEmbedding stores blocks in self.blocks
    stem = self.model.patch_embed
    x = pixel_values
    for block in stem.blocks:
      x = block(x)
      features.append(x)
    return features


# Registry entry point


@register_model("convvit")
def load_convvit(
    *,
    num_labels,
    id2label,
    label2id,
    image_size,
    device,
    checkpoint_path=None,
    **kwargs,
):
  """Instantiate ConvViT model.

    Parameters
    ----------
    num_labels : int
        Number of output classes.
    id2label : dict
        Mapping ``{int_index: label_name}``.
    label2id : dict
        Mapping ``{label_name: int_index}``.
    image_size : int
        Spatial resolution (pixels).  ConvViT default is 224.
    device : torch.device
        Target device for the model.
    checkpoint_path : str or None
        Optional path to a ``.pt`` checkpoint to load weights from.
    **kwargs
        Extra configuration overrides (e.g. ``depth=6``, ``num_heads=8``,
        ``dropout=0.2``).  Typically passed via ``--model_arg`` from the
        command line.
    """
  # Allow CLI / caller overrides via --model_arg (e.g. depth=6 num_heads=8).
  config = SimpleNamespace(
      image_size=image_size,
      embed_dim=768,
      num_heads=12,
      depth=12,
      dropout=0.0,
      drop_path_rate=0.1,
      num_conv_layers=4,
      num_labels=num_labels,
      id2label=id2label,
      label2id=label2id,
      **kwargs,
  )

  logging.info(
      "Building ConvViT  (image_size=%d, num_labels=%d, depth=%d, "
      "num_heads=%d, embed_dim=%d)",
      config.image_size,
      config.num_labels,
      config.depth,
      config.num_heads,
      config.embed_dim,
  )

  model = CustomPatchTransformer(
      num_classes=config.num_labels,
      img_size=config.image_size,
      embed_dim=config.embed_dim,
      num_heads=config.num_heads,
      depth=config.depth,
      dropout=config.dropout,
      drop_path_rate=config.drop_path_rate,
      num_conv_layers=config.num_conv_layers,
  )

  # Load checkpoint (if provided)
  if checkpoint_path and os.path.isfile(checkpoint_path):
    ckpt = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
    state = ckpt.get("model_state_dict", ckpt)

    # The checkpoint may come from the wrapper (keys prefixed with
    # ``model.``) or from the raw model.  Strip the prefix if present.
    raw_keys = set(model.state_dict().keys())
    if state and not raw_keys.intersection(state):
      # None of the checkpoint keys match — try stripping "model."
      stripped = {k.removeprefix("model."): v for k, v in state.items()}
      if raw_keys.intersection(stripped):
        logging.info("ConvViT checkpoint: stripping 'model.' prefix from keys")
        state = stripped

    missing, unexpected = model.load_state_dict(state, strict=False)
    if missing:
      logging.warning(
          "ConvViT checkpoint: %d missing keys (new/untrained): %s",
          len(missing),
          missing[:5],
      )
    if unexpected:
      logging.warning(
          "ConvViT checkpoint: %d unexpected keys (skipped): %s",
          len(unexpected),
          unexpected[:5],
      )
    logging.info("Loaded ConvViT checkpoint: %s", checkpoint_path)
  else:
    logging.info("ConvViT: training from random init (no checkpoint)")

  wrapped = ConvViTForClassification(model, config)
  wrapped.to(device)

  logging.info(
      "ConvViT ready — params: %.1fM",
      sum(p.numel() for p in wrapped.parameters()) / 1e6,
  )

  return wrapped


@register_processor("convvit")
def load_convvit_processor(*, image_size=224, **kwargs):
  """Return a ConvViTProcessor for the given *image_size*."""
  return ConvViTProcessor(image_size=image_size)
