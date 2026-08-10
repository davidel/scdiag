"""Pre-training method registry."""
from __future__ import annotations

import logging

from scdiag.logging_utils import fatal

METHODS = {}


def register_method(cls):
  """Class decorator that registers a pre-training method."""
  name = cls.NAME
  if name in METHODS:
    fatal(f"Duplicate pre-training method name '{name}'", RuntimeError)
  METHODS[name] = cls
  logging.debug("Registered pre-training method '%s'", name)
  return cls


def get_method(name):
  """Look up a pre-training method by name."""
  if name not in METHODS:
    available = ", ".join(sorted(METHODS)) or "(none)"
    fatal(
        f"Unknown pre-training method '{name}'. "
        f"Available: {available}",
        ValueError,
    )
  return METHODS[name]


def list_methods():
  """Return sorted list of registered method names."""
  return sorted(METHODS)
