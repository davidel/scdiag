"""Shared checkpoint save/load utilities.

Extracted from ``train.py`` to avoid code duplication with ``pretrain.py``.
Both scripts import these functions rather than maintaining separate copies.
"""

import io
import logging
import os
import re
import tarfile
import tempfile

import torch

from scdiag.logging_utils import fatal
from scdiag.param_align import AlignConfig, align_state_dicts, report_to_str


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


def serialize_lora_state(model):
  """Serialize PEFT adapter state to bytes.

  The adapter files are saved via ``save_pretrained``, packed into a
  tar archive (contents only, no top-level directory), and returned
  as ``bytes``.

  Requires ``peft`` to be installed.  *model* must be a ``PeftModel``.
  """
  from peft import PeftModel

  if not isinstance(model, PeftModel):
    fatal(
        f"Expected a PeftModel instance, got {type(model).__name__}",
        TypeError,
    )

  with tempfile.TemporaryDirectory() as tmpdir:
    model.save_pretrained(tmpdir)
    buf = io.BytesIO()
    with tarfile.open(fileobj=buf, mode="w") as tar:
      for entry in os.listdir(tmpdir):
        tar.add(os.path.join(tmpdir, entry), arcname=entry)
    blob = buf.getvalue()
    logging.info("  LoRA adapter blob: %d bytes", len(blob))
    return blob


def deserialize_lora_state(model, blob):
  """Restore PEFT adapter state from a tar blob.

  *blob* must have been produced by :func:`serialize_lora_state`.
  Works whether *model* is already a ``PeftModel`` or a plain model.

  Returns:
      Tuple of ``(model, loaded_keys)`` where *loaded_keys* is the set of
      state-dict keys that were restored from the blob.
  """
  from peft import PeftModel

  with tempfile.TemporaryDirectory() as tmpdir:
    buf = io.BytesIO(blob)
    with tarfile.open(fileobj=buf, mode="r") as tar:
      tar.extractall(tmpdir, filter="data")
    # When the model is already a PeftModel (e.g. resume_checkpoint called
    # after apply_lora), PeftModel.from_pretrained would double-wrap it,
    # producing mangled keys like "base_model.model.base_model.model.*".
    # Instead, swap the existing adapter weights in place.
    if isinstance(model, PeftModel):
      model.delete_adapter("default")
      before = set(model.state_dict().keys())
      model.load_adapter(tmpdir, adapter_name="default")
      model.set_adapter("default")
    else:
      before = set(model.state_dict().keys())
      model = PeftModel.from_pretrained(model, tmpdir)

  loaded = set(model.state_dict().keys()) - before
  return model, loaded


def select_best_checkpoint(root_path):
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


_VALID_STATE_FLAGS = {"opt", "sched", "amp", "none"}


def _format_count(num, suffixes=("K", "M", "G", "T")):
  """Format an integer count into a human-readable string with a suffix.

    *suffixes* should have entries for 1024¹, 1024², 1024³, and 1024⁴
    respectively.  Values below 1 024 are returned as-is with no suffix.
    """
  if num < 1024:
    return f"{num}"
  for power, sfx in enumerate(suffixes, start=1):
    if num < 1024**(power + 1):
      return f"{num / 1024**power:.2f}{sfx}"
  # Exceeds the largest suffix — use the last one.
  return f"{num / 1024**len(suffixes):.2f}{suffixes[-1]}"


def _format_bytes(num_bytes):
  """Convert a byte count into a human-readable string (B, KB, MB, GB)."""
  if num_bytes < 1024:
    return f"{num_bytes} B"
  return _format_count(num_bytes, suffixes=(" KB", " MB", " GB", " TB"))


def create_model_report(model):
  """Return a full model report string: network structure and parameter details.

    The report includes the ``str(model)`` representation (layer tree), a
    summary line with total/trainable parameter counts, and a detailed
    per-parameter table showing name, shape, element count, and memory
    size.
    """
  parts = []

  parts.append(f"Model structure:\n{model}")

  total = sum(p.numel() for p in model.parameters())
  trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
  parts.append(f"Model params: {_format_count(total)} total, "
               f"{_format_count(trainable)} trainable")

  parts.append(_format_model_param_table(model))
  return "\n".join(parts)


def _format_model_param_table(model):
  """Return a formatted table of every parameter's name, shape, count, and size.

    Finishes with a summary line showing total parameter count and total
    memory consumed by the model's parameters.  All columns are
    dynamically aligned into a tabular layout.
    """
  params = list(model.named_parameters())
  if not params:
    return "Model has no parameters."

  # Pre-compute display values to determine column widths.
  rows = []
  for name, param in params:
    numel = param.numel()
    param_bytes = numel * param.element_size()
    shape_str = str(tuple(param.shape))
    trainable = "yes" if param.requires_grad else "no"
    rows.append((name, shape_str, numel, param_bytes, trainable))

  # Column widths (header labels are included in the min width).
  name_w = max(len(r[0]) for r in rows)
  shape_w = max(len(r[1]) for r in rows)
  params_w = max(len(_format_count(r[2])) for r in rows)
  size_w = max(len(_format_bytes(r[3])) for r in rows)
  trainable_w = max(len(r[4]) for r in rows)
  trainable_w = max(trainable_w, len("Trainable"))

  # Header.
  header = (f"  {'Parameter':<{name_w}}  {'Shape':>{shape_w}}  "
            f"{'Params':>{params_w}}  {'Size':>{size_w}}  "
            f"{'Trainable':>{trainable_w}}")
  sep = "  " + "-" * (len(header) - 2)

  lines = []
  lines.append("Model parameter details:")
  lines.append(header)
  lines.append(sep)

  total_params = 0
  total_bytes = 0
  trainable_params = 0
  for name, shape_str, numel, param_bytes, trainable in rows:
    total_params += numel
    total_bytes += param_bytes
    if trainable == "yes":
      trainable_params += numel
    lines.append(
        f"  {name:<{name_w}}  {shape_str:>{shape_w}}  "
        f"{_format_count(numel):>{params_w}}  {_format_bytes(param_bytes):>{size_w}}  "
        f"{trainable:>{trainable_w}}")

  lines.append(sep)
  lines.append(
      f"  {'TOTAL':<{name_w}}  {'':>{shape_w}}  "
      f"{_format_count(total_params):>{params_w}}  {_format_bytes(total_bytes):>{size_w}}  "
      f"{_format_count(trainable_params):>{trainable_w}}")
  return "\n".join(lines)


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
                    save_frozen=True,
                    **extra):
  """Build a standard checkpoint dict.

    ``states_to_save`` is a set like ``{"opt", "sched", "amp"}``.
    If ``None``, everything is saved (backward compat).
    Any additional keyword arguments are merged into the dict as-is.

    When *save_frozen* is ``False``, only trainable parameters are
    included in the model state dict (via :func:`trainable_state_dict`).
    This drastically reduces checkpoint size for fine-tuning runs with
    a frozen backbone.
    """
  from scdiag.model_utils import trainable_state_dict

  # When a LoRA blob is present the adapter weights live in the blob;
  # only persist the non-adapter part of the state dict here.
  # trainable_state_dict() correctly respects requires_grad, yielding
  # the unfrozen classifier weights (and nothing else).  The lora_
  # keys are then stripped so the same blob can fully restore them.
  if extra.get("lora_state_blob") is not None:
    base = (model.state_dict() if save_frozen else trainable_state_dict(model))
    sd = {k: v for k, v in base.items() if "lora_" not in k}
  else:
    sd = (model.state_dict() if save_frozen else trainable_state_dict(model))
  d = {
      "model_state_dict": sd,
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

    Returns ``(start_epoch, best_metric, extra)`` where *extra* is a dict
    of any non-state-dict keys stored in the checkpoint (e.g.
    ``global_step``).
    """
  _KNOWN_CKPT_KEYS = frozenset({
      "model_state_dict",
      "optimizer_state_dict",
      "scheduler_state_dict",
      "scaler_state_dict",
      "epoch",
      "best_macro_f1",
      "lora_state_blob",
  })

  resume_path = None
  if os.path.exists(ckpt_latest):
    resume_path = ckpt_latest
  elif os.path.exists(ckpt_best):
    resume_path = ckpt_best

  if not resume_path:
    return model, 0, 0.0, {}

  logging.info(f"Resuming from checkpoint: {resume_path}")
  ckpt = torch.load(resume_path, map_location=device, weights_only=False)
  logging.info(f"  Checkpoint keys: {list(ckpt.keys())}")

  ckpt_sd = ckpt.get("model_state_dict", {})
  logging.info(f"  model_state_dict: {len(ckpt_sd)} keys")
  if ckpt_sd:
    for k, v in ckpt_sd.items():
      logging.info(f"    {k}  {tuple(v.shape)}")

  # Restore LoRA adapters first so their keys are present before we load
  # the non-LoRA model weights (this lets us cleanly separate truly
  # missing keys from expected LoRA keys in the load_state_dict report).
  lora_blob = ckpt.get("lora_state_blob")
  lora_keys = set()
  if lora_blob is not None:
    logging.info(f"  lora_state_blob: {len(lora_blob)} bytes")
    model, lora_keys = deserialize_lora_state(model, lora_blob)
    logging.info(f"  Restored {len(lora_keys)} LoRA keys from blob")

  filtered, skipped = filter_state_dict(
      ckpt["model_state_dict"],
      model.state_dict(),
  )
  if skipped:
    for k, reason in skipped:
      logging.warning(f"  Skipped key '{k}': {reason}")

  result = model.load_state_dict(filtered, strict=False)
  matched = len(filtered) - len(result.unexpected_keys)
  logging.info(f"  Restored model weights ({matched}/{len(filtered)} keys)")

  truly_missing = set(result.missing_keys) - lora_keys
  if truly_missing:
    logging.warning(f"  Missing keys ({len(truly_missing)} not loaded"
                    f" from checkpoint):")
    cur_sd = model.state_dict()
    for mk in sorted(truly_missing):
      shape = tuple(cur_sd[mk].shape) if mk in cur_sd else "N/A"
      logging.warning(f"    {mk}  {shape}")
  if result.unexpected_keys:
    logging.warning(f"  Unexpected keys (ignored): "
                    f"{result.unexpected_keys}")

  opt_state = ckpt.get("optimizer_state_dict")
  if "opt" in states_to_load and opt_state is not None:
    if skipped:
      logging.warning("  Skipped optimizer restore (model architecture changed)")
    else:
      optimizer.load_state_dict(opt_state)
      logging.info("  Restored optimizer state")
  else:
    logging.info("  Skipped optimizer state")

  sched_state = ckpt.get("scheduler_state_dict")
  if "sched" in states_to_load and sched_state is not None:
    if scheduler is None:
      logging.info("  Skipped scheduler restore (no scheduler in current run)")
    elif skipped:
      logging.warning("  Skipped scheduler restore (model architecture changed)")
    else:
      scheduler.load_state_dict(sched_state)
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
  best_metric = ckpt.get("best_macro_f1", 0.0)
  extra = {k: v for k, v in ckpt.items() if k not in _KNOWN_CKPT_KEYS}

  logging.info(f"  Resumed at epoch {start_epoch}, best_metric={best_metric:.4f}")
  return model, start_epoch, best_metric, extra


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

  ckpt = torch.load(path, map_location=device, weights_only=False)

  state = ckpt.get("model_state_dict")
  if state is None:
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

  logging.info(report_to_str(report))

  aligned = {}
  for new_key, old_key in report.mapping.items():
    aligned[new_key] = state[old_key]

  result = model.load_state_dict(aligned, strict=strict)
  matched = len(aligned) - len(result.unexpected_keys)
  logging.info(f"  Loaded weights from {path} ({matched}/{len(aligned)} keys)")
  if result.missing_keys:
    logging.warning(f"  Missing keys: {result.missing_keys}")
  if result.unexpected_keys:
    logging.warning(f"  Unexpected keys: {result.unexpected_keys}")
  return report
