"""ClsModelWrapper — HuggingFace backbone with a custom classifier head."""

import logging

import torch.nn as nn
from transformers import AutoModel

from scdiag.classifiers import build_classifier


class ClsModelWrapper(nn.Module):
  """Wraps a HuggingFace backbone with a custom classifier head.

  Parameters
  ----------
  backbone_name : str
      HuggingFace model name or path.
  num_labels : int
      Number of output classes.
  classifier : str
      Classifier spec: a registered name or a ``.py`` file path.
  classifier_args : dict, optional
      Extra keyword arguments forwarded to the classifier constructor.
  """

  def __init__(self, backbone_name, num_labels, classifier, classifier_args=None):
    super().__init__()
    classifier_args = classifier_args or {}

    base = AutoModel.from_pretrained(backbone_name)
    logging.info(
        "ClsModelWrapper: loaded backbone '%s' (%.1fM params)",
        backbone_name,
        sum(p.numel() for p in base.parameters()) / 1e6,
    )

    self.classifier = build_classifier(
        spec=classifier,
        backbone=base,
        num_labels=num_labels,
        **classifier_args,
    )
    self.config = base.config

  def forward(self, pixel_values):
    return self.classifier(pixel_values)
