"""CLS-guided attention pooling classifier head."""

import torch.nn as nn

from scdiag.classifiers import register_classifier
from scdiag.models.attention_pooling import CLSGuidedAttentionPooling


@register_classifier("cls_attention")
class Classifier(nn.Module):
  """Attention-pooled backbone features followed by a linear head.

  Uses the CLS token as a query to cross-attend over all spatial tokens
  from the backbone, producing a single pooled representation that is
  then classified by a linear layer.

  Parameters
  ----------
  backbone : torch.nn.Module
      HuggingFace backbone loaded via ``AutoModel``.
  num_labels : int
      Number of output classes.
  num_heads : int
      Attention heads for the pooling layer.
  dropout : float
      Dropout inside the pooling layer.
  """

  def __init__(self, backbone, num_labels, num_heads=8, dropout=0.1):
    super().__init__()
    self.backbone = backbone
    dim = getattr(backbone.config, "hidden_size", 1024)
    self.pool = CLSGuidedAttentionPooling(
        embed_dim=dim,
        num_heads=num_heads,
        dropout=dropout,
    )
    self.head = nn.Linear(dim, num_labels)

  def forward(self, pixel_values):
    out = self.backbone(pixel_values)
    hidden = out.last_hidden_state  # (B, N+1, D)
    cls_out = hidden[:, :1, :]  # (B, 1, D)
    spatial_out = hidden[:, 1:, :]  # (B, N, D)
    features = self.pool(cls_out, spatial_out)  # (B, D)
    return self.head(features)
