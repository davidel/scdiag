"""CLS-guided attention pooling classifier head."""

import torch
import torch.nn as nn

from scdiag.classifiers import register_classifier
from scdiag.classifiers.base import BaseClassifier
from scdiag.models.attention_pooling import CLSGuidedAttentionPooling


@register_classifier("cls_attention")
class Classifier(BaseClassifier):
  """Attention-pooled backbone features followed by a linear head.

  Uses the CLS token as a query to cross-attend over a subset of
  spatial tokens from the backbone, producing a single pooled
  representation that feeds a linear classifier.

  Parameters
  ----------
  num_labels : int
      Number of output classes.
  hidden_size : int
      Dimensionality of backbone hidden states (``D``).
  cls_slice : tuple[int, int]
      Start/end indices for the CLS token slice.  Default ``(0, 1)``.
  spc_slice : tuple[int, int | None]
      Start/end indices for the spatial token slice.  Tokens before
      the start index (e.g. register tokens) are dropped.
      Default ``(1, None)`` keeps all spatial tokens after CLS.
  num_heads : int
      Number of attention heads in the pooling layer.
  dropout : float
      Dropout rate inside the pooling layer.
  """

  def __init__(
      self,
      num_labels,
      hidden_size,
      cls_slice=(0, 1),
      spc_slice=(1, None),
      num_heads=8,
      dropout=0.1,
  ):
    super().__init__()
    self.cls_slice = slice(*cls_slice)
    self.spc_slice = slice(*spc_slice)
    self.pool = CLSGuidedAttentionPooling(
        embed_dim=hidden_size,
        num_heads=num_heads,
        dropout=dropout,
    )
    self.head = nn.Linear(hidden_size, num_labels)

  def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
    return self.head(self.extract_features(hidden_states))

  def extract_features(self, hidden_states: torch.Tensor) -> torch.Tensor:
    cls_out = hidden_states[:, self.cls_slice, :]
    spatial_out = hidden_states[:, self.spc_slice, :]
    return self.pool(cls_out, spatial_out)  # (B, D)
