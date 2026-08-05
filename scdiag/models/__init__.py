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
    load_model,
    load_processor,
    register_model,
    register_processor,
)

# Import built-in custom models so their @register_model decorators run.
import scdiag.models.convvit  # noqa: F401

__all__ = [
    "REGISTRY",
    "ModelOutput",
    "is_custom_model",
    "load_model",
    "load_processor",
    "register_model",
    "register_processor",
]
