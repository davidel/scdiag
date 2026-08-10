# ruff: noqa
"""Pre-training method registry.

Each pre-training method registers itself via the ``METHODS`` dict.
"""
from scdiag.pretrain_methods.base import PretrainMethod
from scdiag.pretrain_methods.registry import get_method, list_methods

# Import built-in methods to trigger registration.
import scdiag.pretrain_methods.simmim  # noqa: F401
import scdiag.pretrain_methods.ijepa  # noqa: F401

__all__ = ["PretrainMethod", "get_method", "list_methods"]
