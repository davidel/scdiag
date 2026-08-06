# ruff: noqa
"""Custom model registry.

Each custom model registers itself via the ``@register_model`` decorator.
The registry maps model name strings to loader functions that return
``(model, processor)`` pairs conforming to the
common protocol expected by ``train.py`` and ``infer.py``.
"""

from scdiag.models.registry import (
    ModelOutput,
    is_custom_model,
    load_model,
    load_processor,
    register_model,
    register_processor,
)

# Import built-in models after the registry API is available so their
# module-level decorators can register ConvViT in the central registries.
# The import is intentionally unused as a name; importing the module performs
# the registration side effect required by model loading.
import scdiag.models.convvit

__all__ = [
    "ModelOutput",
    "is_custom_model",
    "load_model",
    "load_processor",
    "register_model",
    "register_processor",
]
