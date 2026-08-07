"""Dataset utilities for scdiag.

Provides :class:`DatasetEnsemble` for stitching together multiple
skin-lesion image datasets into a unified pre-training corpus, and
:class:`HFDatasetProxy` for bridging HuggingFace datasets to PyTorch.
"""

from scdiag.datasets.ensemble import DatasetEnsemble
from scdiag.datasets.hf_proxy import HFDatasetProxy

__all__ = ["DatasetEnsemble", "HFDatasetProxy"]
