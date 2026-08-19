"""UVito: SMP encoder + Transformer for skin lesion classification."""

from scdiag.models.uvito.loader import UVitoForClassification
from scdiag.models.uvito.model import UVito
from scdiag.models.uvito.processor import UVitoProcessor

__all__ = [
    "UVito",
    "UVitoForClassification",
    "UVitoProcessor",
]
