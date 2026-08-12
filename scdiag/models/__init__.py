# ruff: noqa
"""Custom model registry.

Each custom model registers itself via the ``@register_model`` decorator.
The registry maps model name strings to loader functions that return
``(model, processor)`` pairs conforming to the
common protocol expected by ``train.py`` and ``infer.py``.
"""

from scdiag.models.registry import (
    ModelOutput,
    ParsedModelName,
    is_custom_model,
    load_model,
    load_processor,
    parse_model_name,
    register_model,
    register_processor,
)

# Import bundled custom models so their @register_model /
# @register_processor decorators run.  The imports are intentionally
# unused as names; importing the modules performs the registration side
# effects required by model loading.
import scdiag.models.cls_model_wrapper  # noqa: F401
import scdiag.models.convvit  # noqa: F401

__all__ = [
    "ModelOutput",
    "ParsedModelName",
    "is_custom_model",
    "load_model",
    "load_processor",
    "parse_model_name",
    "register_model",
    "register_processor",
]
