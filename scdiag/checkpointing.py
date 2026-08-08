"""Shared checkpoint save/load utilities.

Extracted from ``train.py`` to avoid code duplication with ``pretrain.py``.
Both scripts import these functions rather than maintaining separate copies.
"""

import logging
import os
import re

import torch

from scdiag.logging_utils import fatal
from scdiag.param_align import AlignConfig, _report_to_str, align_state_dicts


def rename_keys(state_dict, patterns):
  """Apply regex-based key renaming to a state dict.

    Each pattern is a string ``SEARCH;REPLACE`` where *SEARCH* is a
    Python regex and *REPLACE* is a replacement string that may use
    ``$1``, ``$2``, … for capture groups (``$N`` is automatically
    converted to ``\\g<N>`` for Python's ``re.sub``).

    Patterns are applied in order.  The last pattern wins for any key
    that matches multiple patterns.

    Args:
        state_dict: The state dictionary to rename keys on.
        patterns: List of ``"search;replace"`` strings.

    Returns:
        A new dictionary with renamed keys.
  """
  compiled = []
  for pat_str in patterns:
    if ";" not in pat_str:
      fatal(
          f"Invalid --param_rename pattern {pat_str!r}: "
          "expected 'SEARCH;REPLACE'.", ValueError)
    search, replace = pat_str.split(";", 1)
    # Python re.sub only supports \N backreferences, not $N.
    # Convert user-friendly $N to \g<N> for clarity and safety.
    replace = re.sub(r"\$(\d+)", r"\\g<\1>", replace)
    compiled.append((re.compile(search), replace))

  new_state = {}
  for key, value in state_dict.items():
    new_key = key
    for regex, replacement in compiled:
      new_key = regex.sub(replacement, new_key)
    new_state[new_key] = value
  return new_state


def select_available_checkpoint(root_path):
  """Return the best checkpoint path, falling back to latest.

    Checkpoint files are assumed to follow the ``<root>_best.pt`` /
    ``<root>_latest.pt`` naming convention.

    Args:
        root_path: The checkpoint root path (without ``_best.pt`` or
            ``_latest.pt`` suffix).

    Returns:
        The path to the available checkpoint, or ``None`` if neither
        exists.
  """
  best = root_path + "_best.pt"
  latest = root_path + "_latest.pt"
  if os.path.isfile(best):
    return best
  if os.path.isfile(latest):
    logging.warning(f"Best checkpoint not found ({best}). "
                    f"Falling back to latest: {latest}")
    return latest
  logging.error(f"No checkpoint found at {root_path} "
                f"(tried {best}, {latest}).")
  return None


_VALID_STATE_FLAGS = {"opt", "sched", "amp", "none"}


def _format_bytes(num_bytes):
  """Convert a byte count into a human-readable string (KB, MB, GB)."""
  if num_bytes < 1024:
    return f"{num_bytes} B"
  elif num_bytes < 1024**2:
    return f"{num_bytes / 1024:.2f} KB"
  elif num_bytes < 1024**3:
    return f"{num_bytes / 1024**2:.2f} MB"
  else:
    return f"{num_bytes / 1024**3:.2f} GB"


def log_model_params(model):
  """Log every parameter name, shape, element count, and memory size.

    Finishes with a summary line showing total parameter count and total
    memory consumed by the model's parameters.  All columns are
    dynamically aligned into a tabular layout.
    """
  params = list(model.named_parameters())
  if not params:
    logging.info("Model has no parameters.")
    return

  # Pre-compute display values to determine column widths.
  rows = []
  for name, param in params:
    numel = param.numel()
    param_bytes = numel * param.element_size()
    shape_str = str(tuple(param.shape))
    rows.append((name, shape_str, numel, param_bytes))

  # Column widths (header labels are included in the min width).
  name_w = max(len(r[0]) for r in rows)
  shape_w = max(len(r[1]) for r in rows)
  params_w = max(len(f"{r[2]:,d}") for r in rows)
  size_w = max(len(_format_bytes(r[3])) for r in rows)

  # Header.
  header = (f"  {'Parameter':<{name_w}}  {'Shape':>{shape_w}}  "
            f"{'Params':>{params_w}}  {'Size':>{size_w}}")
  sep = "  " + "-" * (len(header) - 2)

  logging.info("Model parameter details:")
  logging.info(header)
  logging.info(sep)

  total_params = 0
  total_bytes = 0
  for name, shape_str, numel, param_bytes in rows:
    total_params += numel
    total_bytes += param_bytes
    logging.info(f"  {name:<{name_w}}  {shape_str:>{shape_w}}  "
                 f"{numel:>{params_w},d}  {_format_bytes(param_bytes):>{size_w}}")

  logging.info(sep)
  logging.info(f"  {'TOTAL':<{name_w}}  {'':>{shape_w}}  "
               f"{total_params:>{params_w},d}  {_format_bytes(total_bytes):>{size_w}}")


def parse_state_flags(flag_value):
  """Parse a comma-separated state flag string into a set of tokens.

    Returns a set like ``{"opt", "sched", "amp"}``.
    If the string contains ``"none"``, returns an empty set.
    Raises ValueError on invalid tokens or empty input.
    """
  tokens = {t.strip().lower() for t in flag_value.split(",")}
  if not tokens:
    fatal("state flag string must not be empty", ValueError)
  invalid = tokens - _VALID_STATE_FLAGS
  if invalid:
    fatal(f"Invalid state flag(s): {invalid}. Allowed: {_VALID_STATE_FLAGS}",
          ValueError)
  if "none" in tokens:
    return set()
  return tokens


def checkpoint_dict(model,
                    optimizer,
                    scheduler,
                    epoch,
                    states_to_save=None,
                    scaler=None,
                    **extra):
  """Build a standard checkpoint dict.

    ``states_to_save`` is a set like ``{"opt", "sched", "amp"}``.
    If ``None``, everything is saved (backward compat).
    Any additional keyword arguments are merged into the dict as-is.
    """
  d = {
      "model_state_dict": model.state_dict(),
      "epoch": epoch,
  }
  # Persist num_labels so downstream loaders never need to guess.
  if hasattr(model, "config"):
    if hasattr(model.config, "num_labels"):
      d["num_labels"] = model.config.num_labels
    if hasattr(model.config, "id2label"):
      d["id2label"] = model.config.id2label
  if states_to_save is None or "opt" in states_to_save:
    d["optimizer_state_dict"] = optimizer.state_dict()
  if states_to_save is None or "sched" in states_to_save:
    d["scheduler_state_dict"] = scheduler.state_dict(
    ) if scheduler is not None else None
  if "amp" in states_to_save:
    d["scaler_state_dict"] = scaler.state_dict() if scaler is not None else None
  d.update(extra)
  return d


def filter_state_dict(ckpt_state, model_state):
  """Filter a checkpoint state dict to only include keys compatible with the model.

    Skips keys whose tensor shape differs between checkpoint and model.
    Returns ``(filtered_state, skipped)`` where *skipped* is a list of
    ``(key, reason)`` tuples describing why each key was dropped.
    """
  filtered = {}
  skipped = []
  for k, v in ckpt_state.items():
    if k not in model_state:
      skipped.append((k, "missing in model"))
    elif v.shape != model_state[k].shape:
      skipped.append((
          k,
          (f"shape mismatch: checkpoint {list(v.shape)} " \
           f"vs model {list(model_state[k].shape)}"),
      ))
    else:
      filtered[k] = v
  return filtered, skipped


def resume_checkpoint(ckpt_latest, ckpt_best, model, optimizer, scheduler, scaler,
                      device, states_to_load):
  """Resume training state from an existing checkpoint.

    Looks for *ckpt_latest* first, then *ckpt_best*.  Restores model weights
    (filtering out shape-mismatched keys), and conditionally restores
    optimizer, scheduler, and GradScaler states depending on
    *states_to_load*.

    Returns ``(start_epoch, best_metric)``.
    """
  resume_path = None
  if os.path.exists(ckpt_latest):
    resume_path = ckpt_latest
  elif os.path.exists(ckpt_best):
    resume_path = ckpt_best

  if not resume_path:
    return 0, 0.0

  logging.info(f"Resuming from checkpoint: {resume_path}")
  ckpt = torch.load(resume_path, map_location=device, weights_only=False)
  logging.info(f"  Checkpoint keys: {list(ckpt.keys())}")

  # Filter checkpoint to skip keys with shape mismatches
  # (e.g. classifier head when resuming with different num_classes).
  filtered, skipped = filter_state_dict(
      ckpt["model_state_dict"],
      model.state_dict(),
  )
  if skipped:
    for k, reason in skipped:
      logging.warning(f"  Skipped key '{k}': {reason}")

  result = model.load_state_dict(filtered, strict=False)
  logging.info("  Restored model weights")
  if result.missing_keys:
    logging.warning(f"  Missing keys (randomly initialized): "
                    f"{result.missing_keys}")
  if result.unexpected_keys:
    logging.warning(f"  Unexpected keys (ignored): "
                    f"{result.unexpected_keys}")

  log_model_params(model)

  if "opt" in states_to_load and "optimizer_state_dict" in ckpt:
    if skipped:
      logging.warning("  Skipped optimizer restore (model architecture changed)")
    else:
      optimizer.load_state_dict(ckpt["optimizer_state_dict"])
      logging.info("  Restored optimizer state")
  else:
    logging.info("  Skipped optimizer state")

  if "sched" in states_to_load and "scheduler_state_dict" in ckpt:
    if scheduler is None:
      logging.info("  Skipped scheduler restore (no scheduler in current run)")
    elif skipped:
      logging.warning("  Skipped scheduler restore (model architecture changed)")
    else:
      scheduler.load_state_dict(ckpt["scheduler_state_dict"])
      logging.info("  Restored scheduler state")
  else:
    logging.info("  Skipped scheduler state")

  if "amp" in states_to_load:
    scaler_dict = ckpt.get("scaler_state_dict")
    if scaler_dict is not None and scaler is not None:
      scaler.load_state_dict(scaler_dict)
      logging.info("  Restored GradScaler state")
    else:
      logging.info("  Skipped GradScaler state")

  start_epoch = ckpt.get("epoch", -1) + 1
  best_metric = ckpt.get("best_top1", ckpt.get("best_metric", 0.0))
  logging.info(f"  Resumed at epoch {start_epoch}, best_metric={best_metric}")
  return start_epoch, best_metric


def load_checkpoint_weights(path,
                            model,
                            device="cpu",
                            strict=False,
                            param_rename=None,
                            max_distance=None):
  """Load model weights from a source checkpoint, aligned by shape.

    ``align_state_dicts`` matches source keys to model keys by tensor
    shape and weighted token distance.  ``param_rename`` patterns are
    applied *before* alignment so that manual renames take priority.

    Args:
        path: Path to the checkpoint file.
        model: The model to load weights into.
        device: Device to map tensors to.
        strict: If True, raise on missing/unexpected keys.
        param_rename: Optional list of ``"SEARCH;REPLACE"`` patterns
            for renaming checkpoint keys via regex before alignment.
        max_distance: Override for ``AlignConfig.max_distance``.
            ``None`` uses the default (0.25).

    Returns:
        The ``AlignReport`` produced by :func:`align_state_dicts`.
  """
  if not os.path.isfile(path):
    fatal(
        f"Checkpoint file not found: {path}. "
        "Ensure the checkpoint exists before calling load_checkpoint_weights().",
        FileNotFoundError)

  ckpt = torch.load(path, map_location=device, weights_only=True)

  if "model_state_dict" in ckpt:
    state = ckpt["model_state_dict"]
  else:
    logging.warning(
        "Loading raw state dictionary from '%s'; checkpoint metadata is "
        "unavailable.", path)
    state = ckpt

  if param_rename:
    state = rename_keys(state, param_rename)

  config_kwargs = {}
  if max_distance is not None:
    config_kwargs["max_distance"] = max_distance
  config = AlignConfig(**config_kwargs) if config_kwargs else None

  report = align_state_dicts(state, model.state_dict(), config=config)

  logging.info(_report_to_str(report))

  aligned = {}
  for new_key, old_key in report.mapping.items():
    aligned[new_key] = state[old_key]

  result = model.load_state_dict(aligned, strict=strict)
  logging.info(f"  Loaded weights from {path}")
  if result.missing_keys:
    logging.warning(f"  Missing keys: {result.missing_keys}")
  if result.unexpected_keys:
    logging.warning(f"  Unexpected keys: {result.unexpected_keys}")
  return report
