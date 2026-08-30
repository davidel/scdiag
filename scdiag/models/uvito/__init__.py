"""UVito: SMP encoder + Transformer for skin lesion classification."""

from scdiag.models.uvito.loader import UVitoAdapter
from scdiag.models.uvito.model import UVito
from scdiag.models.uvito.processor import UVitoProcessor

__all__ = [
    "UVito",
    "UVitoAdapter",
    "UVitoProcessor",
]
