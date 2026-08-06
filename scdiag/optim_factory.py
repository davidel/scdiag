"""Factory functions for creating optimizers and schedulers.

Used by ``train.py`` and ``pretrain.py`` to replace hardcoded optimizer /
scheduler creation with configurable versions driven by CLI arguments.
"""

import logging
import os
import tempfile
import urllib.request

import torch.optim as optim

from scdiag.logging_utils import fatal


def create_optimizer(params, *, name="AdamW", lr=1e-4, weight_decay=0.01, **kwargs):
  """Create an optimizer by name, forwarding extra keyword arguments.

    Args:
        params: Iterable of parameters (typically ``model.parameters()``).
        name: Name of a ``torch.optim`` optimizer class (e.g. ``"AdamW"``,
            ``"Adam"``, ``"SGD"``).  Must match the class name exactly
            (case-sensitive).
        lr: Learning rate.
        weight_decay: Weight decay (L2 penalty).
        **kwargs: Forwarded to the optimizer constructor (e.g. ``betas``,
            ``momentum``, ``amsgrad``).

    Returns:
        A ``torch.optim.Optimizer`` instance.
    """
  cls = getattr(optim, name, None)
  if cls is None or not isinstance(cls, type):
    available = [
        x for x in dir(optim)
        if not x.startswith("_") and isinstance(getattr(optim, x), type)
    ]
    fatal(f"Unknown optimizer {name!r}. "
          f"Available: {', '.join(sorted(available))}", ValueError)
  logging.info("Creating %s optimizer (lr=%.2e, weight_decay=%.2e)", name, lr,
               weight_decay)
  if kwargs:
    logging.info("  Extra optimizer kwargs: %s", kwargs)
  return cls(params, lr=lr, weight_decay=weight_decay, **kwargs)


def create_scheduler(optimizer, *, name=None, epochs=100, base_lr=1e-4, **kwargs):
  """Create a LR scheduler by name, forwarding extra keyword arguments.

    Three dispatch modes:

    1. **name is None / empty** — returns ``None`` (no scheduler).
    2. **name ends with ``.py``** — loads the script and calls its
       ``create_scheduler(optimizer, **kwargs)`` function (similar to
       custom image processors).
    3. **Otherwise** — resolves *name* via
       ``getattr(torch.optim.lr_scheduler, name)`` (case-sensitive)
       and instantiates it, forwarding ``kwargs``.

    Args:
        optimizer: The optimizer to schedule.
        name: A ``torch.optim.lr_scheduler`` class name (e.g.
            ``"CosineAnnealingLR"``, ``"StepLR"``), a path / URL to a
            Python script, or ``None`` for no scheduling.
        epochs: Total number of training epochs.
        base_lr: The target / peak learning rate.
        **kwargs: Forwarded to the scheduler constructor (e.g.
            ``T_max``, ``eta_min``, ``step_size``, ``gamma``).

    Returns:
        A ``torch.optim.lr_scheduler.LRScheduler`` instance, or
        ``None`` if *name* is ``None``.
    """
  if not name:
    logging.info("No scheduler requested.")
    return None

  if name.endswith(".py"):
    return _load_scheduler_script(optimizer,
                                  name,
                                  epochs=epochs,
                                  base_lr=base_lr,
                                  **kwargs)

  cls = getattr(optim.lr_scheduler, name, None)
  if cls is None or not isinstance(cls, type):
    available = [
        x for x in dir(optim.lr_scheduler)
        if not x.startswith("_") and isinstance(getattr(optim.lr_scheduler, x), type)
    ]
    fatal(f"Unknown scheduler {name!r}. "
          f"Available: {', '.join(sorted(available))}", ValueError)

  logging.info("Creating %s scheduler (kwargs=%s)", name, kwargs)
  return cls(optimizer, **kwargs)


def _load_scheduler_script(optimizer, path_or_url, **extra):
  """Load a custom scheduler from a Python file or URL.

    The script must define a callable
    ``create_scheduler(optimizer, **kwargs)``.
    """
  namespace = {}

  if path_or_url.startswith(("http://", "https://")):
    with urllib.request.urlopen(path_or_url) as resp:
      code = resp.read().decode("utf-8")
    with tempfile.NamedTemporaryFile(mode="w",
                                     suffix=".py",
                                     delete=False,
                                     prefix="sched_") as tmp:
      tmp.write(code)
      tmp_path = tmp.name
    try:
      exec(compile(code, path_or_url, "exec"), namespace)  # noqa: S102
    finally:
      os.unlink(tmp_path)
  else:
    with open(path_or_url) as f:
      code = f.read()
    exec(compile(code, path_or_url, "exec"), namespace)  # noqa: S102

  fn = namespace.get("create_scheduler")
  if fn is None or not callable(fn):
    fatal(
        f"Script {path_or_url!r} does not define a callable "
        "'create_scheduler(optimizer, **kwargs)'.", ValueError)

  logging.info("Loading custom scheduler from %s", path_or_url)
  return fn(optimizer, **extra)
