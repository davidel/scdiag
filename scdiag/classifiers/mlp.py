"""Two-layer MLP classifier head."""

import torch
import torch.nn as nn

from scdiag.classifiers import register_classifier
from scdiag.classifiers.base import BaseClassifier


@register_classifier("mlp")
class Classifier(BaseClassifier):
  """Two-layer MLP classification head.

  Receives backbone hidden states ``(B, N, D)`` and classifies the
  CLS token (``[:, 0]``).
  """

  def __init__(self, num_labels, hidden_size, hidden=512, dropout=0.3):
    super().__init__()
    self.head = nn.Sequential(
        nn.Linear(hidden_size, hidden),
        nn.GELU(),
        nn.Dropout(dropout),
        nn.Linear(hidden, num_labels),
    )

  def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
    return self.head(self.extract_features(hidden_states))

  def extract_features(self, hidden_states: torch.Tensor) -> torch.Tensor:
    return hidden_states[:, 0]  # CLS token, (B, D)
