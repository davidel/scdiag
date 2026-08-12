"""Classifier registry — built-in and user-supplied classifier heads.

Usage::

    from scdiag.classifiers import build_classifier

    model = build_classifier("mlp", backbone, num_labels=3, hidden=256)

A classifier spec can be:

* A registered name (e.g. ``"mlp"``).
* A path/URL to a ``.py`` file defining a ``Classifier`` class.
"""

import logging

from scdiag.logging_utils import fatal
from scdiag.script_utils import load_extern

_CLASSIFIERS = {}


def register_classifier(name):
  """Decorator to register a classifier class under *name*."""

  def decorator(cls):
    if name in _CLASSIFIERS:
      fatal(
          f"Classifier {name!r} already registered "
          f"(was {_CLASSIFIERS[name].__name__}).",
          ValueError,
      )
    _CLASSIFIERS[name] = cls
    return cls

  return decorator


def build_classifier(spec, backbone, num_labels, **kwargs):
  """Resolve *spec* and return an instantiated classifier ``nn.Module``.

  Parameters
  ----------
  spec : str
      Either a registered classifier name or a path/URL to a ``.py`` file.
  backbone : nn.Module
      The backbone model (e.g. a HF model with its head stripped).
  num_labels : int
      Number of output classes.
  **kwargs
      Forwarded to the classifier constructor (from ``--classifier_args``).
  """
  if spec.endswith(".py"):
    logging.info("Loading custom classifier from %s", spec)
    cls = load_extern(spec, "Classifier")
  elif spec in _CLASSIFIERS:
    cls = _CLASSIFIERS[spec]
    logging.info("Using registered classifier %r (%s)", spec, cls.__name__)
  else:
    available = sorted(_CLASSIFIERS.keys())
    fatal(
        f"Unknown classifier {spec!r}. Available: {', '.join(available)}. "
        f"Or provide a path to a .py file.",
        ValueError,
    )
  return cls(backbone, num_labels, **kwargs)


# Import built-in classifiers so @register_classifier fires.
from scdiag.classifiers import (  # noqa: F401
    cls_attention, mlp,
)
