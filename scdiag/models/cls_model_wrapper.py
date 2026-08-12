"""HuggingFace backbone + custom classifier head.

Registered as ``cls_model_wrapper`` in the model registry.  Invoked via::

    --model cls_model_wrapper:google/vit-base-patch16-224
    --classifier mlp
    --classifier_args "hidden=512,dropout=0.3"

The text after the ``:`` is the HF model name/path (``base_model``).
"""

import logging

from transformers import AutoModelForImageClassification

from scdiag.classifiers import build_classifier
from scdiag.models.registry import register_model


@register_model("cls_model_wrapper")
def load_cls_model_wrapper(*,
                           backbone,
                           num_labels,
                           classifier,
                           classifier_args=None,
                           device="cpu",
                           **kwargs):
  """Load a HF backbone and wrap it with a custom classifier.

  Parameters
  ----------
  backbone : str
      HuggingFace model name or path (parsed from ``--model <name>:<backbone>``).
  num_labels : int
      Number of output classes.
  classifier : str
      Classifier spec: a registered name or a ``.py`` file path.
  classifier_args : dict, optional
      Extra keyword arguments forwarded to the classifier constructor.
  device : str
      Target device.
  """
  classifier_args = classifier_args or {}

  # Load HF model with num_labels=0 → classifier becomes nn.Identity().
  base = AutoModelForImageClassification.from_pretrained(
      backbone,
      num_labels=0,
      ignore_mismatched_sizes=True,
  )

  logging.info(
      "ClsModelWrapper: loaded backbone '%s' (%.1fM params)",
      backbone,
      sum(p.numel() for p in base.parameters()) / 1e6,
  )

  model = build_classifier(
      spec=classifier,
      backbone=base,
      num_labels=num_labels,
      **classifier_args,
  )
  model.to(device)

  logging.info(
      "ClsModelWrapper: classifier=%s, total=%.1fM params",
      classifier,
      sum(p.numel() for p in model.parameters()) / 1e6,
  )
  return model
