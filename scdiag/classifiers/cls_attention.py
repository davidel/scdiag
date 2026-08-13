"""CLS-guided attention pooling classifier head."""

import torch.nn as nn

from scdiag.classifiers import register_classifier
from scdiag.models.attention_pooling import CLSGuidedAttentionPooling


@register_classifier("cls_attention")
class Classifier(nn.Module):
  """Attention-pooled backbone features followed by a linear head.

  Uses the CLS token as a query to cross-attend over a subset of
  spatial tokens from the backbone, producing a single pooled
  representation that is then classified by a linear layer.

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
  cls_slice : tuple
      Slice indices for the CLS token(s) from the transformer output.
      Converted to ``slice(*cls_slice)``.  Default ``(0, 1)`` selects
      the first token.
  spc_slice : tuple
      Slice indices for spatial tokens from the transformer output.
      Converted to ``slice(*spc_slice)``.  Default ``(1, None)`` selects
      all tokens from index 1 onward.
  """

  def __init__(self,
               backbone,
               num_labels,
               num_heads=8,
               dropout=0.1,
               cls_slice=(0, 1),
               spc_slice=(1, None)):
    super().__init__()
    self.backbone = backbone
    dim = getattr(backbone.config, "hidden_size", 1024)
    self.cls_slice = slice(*cls_slice)
    self.spc_slice = slice(*spc_slice)
    self.pool = CLSGuidedAttentionPooling(
        embed_dim=dim,
        num_heads=num_heads,
        dropout=dropout,
    )
    self.head = nn.Linear(dim, num_labels)

  def forward(self, pixel_values):
    out = self.backbone(pixel_values)
    hidden = out.last_hidden_state  # (B, N, D)
    cls_out = hidden[:, self.cls_slice, :]
    spatial_out = hidden[:, self.spc_slice, :]
    features = self.pool(cls_out, spatial_out)  # (B, D)
    return self.head(features)
