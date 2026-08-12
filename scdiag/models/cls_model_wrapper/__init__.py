"""ClsModelWrapper — HuggingFace backbone + custom classifier head."""

from scdiag.models.cls_model_wrapper.loader import load_cls_model_wrapper
from scdiag.models.cls_model_wrapper.model import ClsModelWrapper

__all__ = [
    "ClsModelWrapper",
    "load_cls_model_wrapper",
]
