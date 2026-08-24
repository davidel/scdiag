"""ContrastiveEncoder — backbone + MLP projection head.

Used by supervised contrastive pre-training.  The projection head
is discarded after pre-training; the backbone is kept for downstream
fine-tuning.
"""

import logging

import torch.nn as nn


class ProjectionHead(nn.Module):
  """Two-layer MLP projection head.

  Architecture: Linear -> BatchNorm -> ReLU -> Linear -> BatchNorm.
  """

  def __init__(self, in_dim, hidden_dim, out_dim):
    super().__init__()
    self.net = nn.Sequential(
        nn.Linear(in_dim, hidden_dim),
        nn.BatchNorm1d(hidden_dim),
        nn.ReLU(inplace=True),
        nn.Linear(hidden_dim, out_dim),
        nn.BatchNorm1d(out_dim),
    )

  def forward(self, x):
    return self.net(x)


class ContrastiveEncoder(nn.Module):
  """Backbone encoder with a projection head for contrastive learning.

  During pre-training, ``forward()`` projects backbone features through
  the MLP head.  ``encode()`` returns raw backbone features for
  downstream use.

  Parameters
  ----------
  encoder : nn.Module
      Any registered backbone (ConvViT, HF model, timm, etc.).
  proj_dim : int
      Output dimensionality of the projection head.
  proj_hidden : int
      Hidden dimensionality of the projection MLP.
  backbone_dim : int, optional
      Explicit backbone output dimension.  When ``None``, inferred
      automatically from ``config.hidden_size`` / ``config.d_model``
      / ``classifier.feat_dim``.
  """

  def __init__(self, encoder, proj_dim=256, proj_hidden=2048,
               backbone_dim=None):
    super().__init__()
    self.encoder = encoder
    self._proj_dim = proj_dim
    self._proj_hidden = proj_hidden
    self._backbone_dim = (
        backbone_dim if backbone_dim is not None
        else self._detect_backbone_dim()
    )
    self.projection = ProjectionHead(
        self._backbone_dim,
        proj_hidden,
        proj_dim,
    )

  def _detect_backbone_dim(self):
    """Infer the backbone output feature dimension."""
    # Try common attribute names used by scdiag models.
    for attr in ("config", "classifier"):
      obj = getattr(self.encoder, attr, None)
      if obj is not None:
        for key in ("hidden_size", "d_model", "num_features"):
          val = getattr(obj, key, None)
          if val is not None:
            return val
    # Try classifiers that expose extract_features.
    classifier = getattr(self.encoder, "classifier", None)
    if classifier is not None:
      feat_dim = getattr(classifier, "feat_dim", None)
      if feat_dim is not None:
        return feat_dim
    raise ValueError(
        "Cannot infer backbone output dimension.  "
        "Pass backbone_dim explicitly."
    )

  def encode(self, images):
    """Extract raw backbone features ``(B, D)``."""
    # Try scdiag's protocol first (hooks into classifier, etc.).
    try:
      from scdiag.model_utils import extract_backbone_features
      return extract_backbone_features(self.encoder, images)
    except (ValueError, AttributeError):
      logging.debug(
          "extract_backbone_features failed, using direct forward")
    # Fallback: plain forward pass through the encoder.
    raw = self.encoder(images)
    if hasattr(raw, "logits"):
      # HF model output — take the CLS / pooler.
      if raw.pooler_output is not None:
        return raw.pooler_output
      return raw.last_hidden_state.mean(dim=1)
    return raw

  def forward(self, images):
    """Project backbone features through the MLP head ``(B, proj_dim)``."""
    features = self.encode(images)
    return self.projection(features)
