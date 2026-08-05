"""Factory functions for creating optimizers and schedulers.

Used by ``train.py`` and ``pretrain.py`` to replace hardcoded optimizer /
scheduler creation with configurable versions driven by CLI arguments.
"""

import logging

import torch.optim as optim

from scdiag.logging_utils import fatal

_OPTIMIZER_MAP = {
    "adam": optim.Adam,
    "adamw": optim.AdamW,
    "sgd": optim.SGD,
}


def create_optimizer(params, *, name="adamw", lr=1e-4, weight_decay=0.01, **kwargs):
  """Create an optimizer by name, forwarding extra keyword arguments.

    Args:
        params: Iterable of parameters (typically ``model.parameters()``).
        name: One of ``"adamw"`` (default), ``"adam"``, ``"sgd"``.
        lr: Learning rate.
        weight_decay: Weight decay (L2 penalty).
        **kwargs: Forwarded to the optimizer constructor (e.g. ``betas``,
            ``momentum``, ``amsgrad``).

    Returns:
        A ``torch.optim.Optimizer`` instance.
    """
  cls = _OPTIMIZER_MAP.get(name.lower())
  if cls is None:
    fatal(
        f"Unknown optimizer {name!r}. "
        f"Available: {', '.join(sorted(_OPTIMIZER_MAP))}", ValueError)
  logging.info("Creating %s optimizer (lr=%.2e, weight_decay=%.2e)", name, lr,
               weight_decay)
  if kwargs:
    logging.info("  Extra optimizer kwargs: %s", kwargs)
  return cls(params, lr=lr, weight_decay=weight_decay, **kwargs)


_SCHEDULER_MAP = {
    "cosine": "cosine",
    "cosine_warmup": "cosine_warmup",
    "step": "step",
    "constant": "constant",
}


def create_scheduler(optimizer,
                     *,
                     name="cosine",
                     epochs=100,
                     warmup_epochs=0,
                     base_lr=1e-4,
                     **kwargs):
  """Create a LR scheduler by name, optionally with linear warmup.

    Args:
        optimizer: The optimizer to schedule.
        name: Scheduler type — ``"cosine"`` (default), ``"cosine_warmup"``,
            ``"step"``, or ``"constant"``.
        epochs: Total number of training epochs (used to set ``T_max`` for
            cosine and the warmup schedule endpoint).
        warmup_epochs: Number of warmup epochs at the start.  If > 0 a
            ``LinearLR`` warmup is composed with the main scheduler via
            ``SequentialLR``.
        base_lr: The target / peak learning rate (used as the warmup end
            point).
        **kwargs: Forwarded to the underlying scheduler constructor (e.g.
            ``T_max``, ``eta_min``, ``step_size``, ``gamma``).

    Returns:
        A ``torch.optim.lr_scheduler.LRScheduler`` instance.
    """
  sched_type = _SCHEDULER_MAP.get(name.lower())
  if sched_type is None:
    fatal(
        f"Unknown scheduler {name!r}. "
        f"Available: {', '.join(sorted(_SCHEDULER_MAP))}", ValueError)

  sched = _build_main_scheduler(optimizer, sched_type, epochs, base_lr, kwargs)

  if warmup_epochs > 0 and sched_type in ("cosine", "cosine_warmup", "step"):
    warmup = optim.lr_scheduler.LinearLR(optimizer,
                                         start_factor=0.01,
                                         total_iters=warmup_epochs)
    sched = optim.lr_scheduler.SequentialLR(optimizer, [warmup, sched],
                                            milestones=[warmup_epochs])
    logging.info("Creating %s scheduler with %d-epoch linear warmup", name,
                 warmup_epochs)
  else:
    logging.info("Creating %s scheduler (no warmup)", name)

  return sched


def _build_main_scheduler(optimizer, sched_type, epochs, base_lr, extra):
  """Build the core (non-warmup) scheduler."""
  if sched_type == "cosine":
    T_max = extra.pop("T_max", epochs)
    eta_min = extra.pop("eta_min", base_lr * 0.01)
    return optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=T_max, eta_min=eta_min)

  if sched_type == "cosine_warmup":
    T_max = extra.pop("T_max", epochs)
    eta_min = extra.pop("eta_min", base_lr * 0.01)
    return optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=T_max, eta_min=eta_min)

  if sched_type == "step":
    step_size = extra.pop("step_size", 30)
    gamma = extra.pop("gamma", 0.1)
    return optim.lr_scheduler.StepLR(optimizer, step_size=step_size, gamma=gamma)

  if sched_type == "constant":
    # Constant LR — return a dummy scheduler that never changes LR.
    return optim.lr_scheduler.LambdaLR(optimizer, lr_lambda=lambda epoch: 1.0)

  fatal(f"Unhandled scheduler type: {sched_type!r}", ValueError)
