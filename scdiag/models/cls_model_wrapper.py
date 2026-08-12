"""HuggingFace backbone + custom classifier head.

Registered as ``cls_model_wrapper`` in the model registry.  Invoked via::

    --model cls_model_wrapper:google/vit-base-patch16-224
    --classifier mlp
    --classifier_args "hidden=512,dropout=0.3"

The text after the ``:`` is the HF model name/path (``base_model``).
"""

import logging

import torch.nn as nn
from transformers import AutoModelForImageClassification

from scdiag.classifiers import build_classifier
from scdiag.models.registry import register_model


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

    base = AutoModelForImageClassification.from_pretrained(
        backbone_name,
        num_labels=0,
        ignore_mismatched_sizes=True,
    )
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


@register_model("cls_model_wrapper")
def load_cls_model_wrapper(*,
                           backbone,
                           num_labels,
                           classifier,
                           classifier_args=None,
                           device="cpu",
                           **kwargs):
  """Instantiate :class:`ClsModelWrapper` and move to *device*."""
  model = ClsModelWrapper(
      backbone_name=backbone,
      num_labels=num_labels,
      classifier=classifier,
      classifier_args=classifier_args,
  )
  model.to(device)

  logging.info(
      "ClsModelWrapper ready — classifier=%s, params=%.1fM",
      classifier,
      sum(p.numel() for p in model.parameters()) / 1e6,
  )
  return model
