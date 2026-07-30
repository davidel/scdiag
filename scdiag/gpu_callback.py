"""GPU monitoring callback for HuggingFace Trainer."""

import logging

import torch
from transformers import TrainerCallback

from scdiag.gpu_utils import gpu_stats_str


class GPUStatsCallback(TrainerCallback):

  def __init__(self, device):
    self.device = device

  def on_log(self, args, state, control, logs=None, **kwargs):
    if self.device.type != "cuda":
      return
    stats = gpu_stats_str(self.device)
    if stats and logs is not None:
      mem_used = torch.cuda.memory_allocated(self.device) / 1024**2
      mem_reserved = torch.cuda.memory_reserved(self.device) / 1024**2
      logs["gpu_mem_used_mb"] = round(mem_used, 1)
      logs["gpu_mem_reserved_mb"] = round(mem_reserved, 1)

  def on_epoch_end(self, args, state, control, **kwargs):
    if self.device.type != "cuda":
      return
    stats = gpu_stats_str(self.device)
    if stats:
      logging.getLogger(__name__).info(f"GPU stats at epoch {state.epoch:.0f}{stats}")

  def on_train_end(self, args, state, control, **kwargs):
    if self.device.type != "cuda":
      return
    stats = gpu_stats_str(self.device)
    if stats:
      logging.getLogger(__name__).info(f"Training complete.{stats}")
