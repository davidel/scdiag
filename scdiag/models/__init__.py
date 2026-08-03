"""Custom model registry.

Each custom model registers itself via a @register_model decorator or a
manual REGISTRY dict.  The registry maps model name strings to loader
functions that return ``(model, processor)`` pairs conforming to the
common protocol expected by ``train.py`` and ``infer.py``.
"""

from scdiag.models.registry import (
    REGISTRY,
    ModelOutput,
    is_custom_model,
    load_custom_model,
    register_model,
)

__all__ = [
    "REGISTRY",
    "ModelOutput",
    "is_custom_model",
    "load_custom_model",
    "register_model",
]
