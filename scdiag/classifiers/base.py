"""Base class for custom classifier heads.

A classifier is an ``nn.Module`` that wraps a backbone model and adds
a classification head.  The only contract is that ``__init__`` must
accept ``backbone`` and ``num_labels`` as its first two arguments,
followed by any keyword arguments forwarded from ``--classifier_args``.
"""
