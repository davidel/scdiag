"""GPU statistics helpers (copied from conv_vit)."""

import contextlib

import torch


def gpu_stats_str(device):
  """Return a human-readable GPU stats string, or empty string if not CUDA."""
  if device.type != "cuda":
    return ""
  mem_used = torch.cuda.memory_allocated(device) / 1024**2
  mem_reserved = torch.cuda.memory_reserved(device) / 1024**2
  mem_total = torch.cuda.get_device_properties(device).total_memory / 1024**2
  msg = f"GPU: mem={mem_used:.0f}/{mem_total:.0f} res={mem_reserved:.0f}"
  if hasattr(torch.cuda, "utilization"):
    # NVML/pynvml may be missing or unavailable; a stats string must never
    # be able to kill a training run.
    with contextlib.suppress(Exception):
      msg += f" util={torch.cuda.utilization(device):.0f}"
  return msg
