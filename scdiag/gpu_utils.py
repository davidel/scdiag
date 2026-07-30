"""GPU statistics helpers (copied from conv_vit)."""

import torch


def gpu_stats_str(device):
  """Return a human-readable GPU stats string, or empty string if not CUDA."""
  if device.type != "cuda":
    return ""
  mem_used = torch.cuda.memory_allocated(device) / 1024**2
  mem_reserved = torch.cuda.memory_reserved(device) / 1024**2
  mem_total = torch.cuda.get_device_properties(device).total_mem / 1024**2
  msg = (f" | GPU Mem: {mem_used:.0f}/{mem_total:.0f} MB"
         f" (reserved {mem_reserved:.0f} MB)")
  if hasattr(torch.cuda, "utilization"):
    msg += f" | GPU Util: {torch.cuda.utilization(device):.0f}%"
  return msg
