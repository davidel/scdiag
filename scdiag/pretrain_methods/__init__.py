"""Pre-training method registry."""
# Import built-in methods to trigger registration.
import scdiag.pretrain_methods.byol
import scdiag.pretrain_methods.dino
import scdiag.pretrain_methods.ijepa
import scdiag.pretrain_methods.simmim
import scdiag.pretrain_methods.supcon  # noqa: F401
from scdiag.pretrain_methods.base import PretrainMethod
from scdiag.pretrain_methods.registry import get_method, list_methods

__all__ = ["PretrainMethod", "get_method", "list_methods"]
