"""Two-layer MLP classifier head."""

import torch.nn as nn

from scdiag.classifiers import register_classifier


@register_classifier("mlp")
class Classifier(nn.Module):
  """Two-layer MLP classification head on top of a HF backbone."""

  def __init__(self, backbone, num_labels, hidden=512, dropout=0.3):
    super().__init__()
    self.backbone = backbone
    dim = getattr(backbone.config, "hidden_size", 768)
    self.head = nn.Sequential(
        nn.Linear(dim, hidden),
        nn.GELU(),
        nn.Dropout(dropout),
        nn.Linear(hidden, num_labels),
    )

  def forward(self, pixel_values):
    out = self.backbone(pixel_values)
    features = out.last_hidden_state[:, 0]  # CLS token
    return self.head(features)
