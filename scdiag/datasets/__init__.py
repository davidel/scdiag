"""Dataset utilities for scdiag.

Provides :class:`DermoscopyEnsemble` for stitching together multiple
skin-lesion image datasets into a unified pre-training corpus, and
:class:`HFDatasetProxy` for bridging HuggingFace datasets to PyTorch.
"""

from scdiag.datasets.ensemble import DermoscopyEnsemble
from scdiag.datasets.hf_proxy import HFDatasetProxy

__all__ = ["DermoscopyEnsemble", "HFDatasetProxy"]
