"""DINO: Self-Distillation with No Labels (Caron et al., ICCV 2021)."""

import torch

from scdiag.augmentations.multicrop import MultiCropTransform
from scdiag.model_utils import set_train_mode
from scdiag.models.dino import DINO
from scdiag.pretrain_methods.base import PretrainMethod
from scdiag.pretrain_methods.registry import register_method


@register_method
class DINOMethod(PretrainMethod):
  """Self-distillation pre-training via DINO with multi-crop."""

  NAME = "dino"
  needs_labels = False

  @classmethod
  def add_args(cls, parser):
    g = parser.add_argument_group("DINO")
    g.add_argument("--dino_proj_dim",
                   type=int,
                   default=256,
                   help="Projection head output dimension.")
    g.add_argument("--dino_proj_hidden",
                   type=int,
                   default=2048,
                   help="Projection head hidden dimension.")
    g.add_argument("--dino_student_temp",
                   type=float,
                   default=0.1,
                   help="Student temperature.")
    g.add_argument("--dino_teacher_temp",
                   type=float,
                   default=0.04,
                   help="Teacher temperature.")
    g.add_argument("--dino_center_momentum",
                   type=float,
                   default=0.9,
                   help="EMA momentum for center update.")
    g.add_argument("--dino_momentum",
                   type=float,
                   default=0.996,
                   help="Initial EMA momentum for teacher encoder.")
    g.add_argument("--dino_final_momentum",
                   type=float,
                   default=1.0,
                   help="Final EMA momentum.")
    g.add_argument("--dino_global_size",
                   type=int,
                   default=224,
                   help="Spatial size of global crops.")
    g.add_argument("--dino_local_size",
                   type=int,
                   default=96,
                   help="Spatial size of local crops.")
    g.add_argument("--dino_local_num",
                   type=int,
                   default=8,
                   help="Number of local crops.")

  def build_transform(self, image_size):
    return MultiCropTransform(
        global_size=min(image_size, self._global_size),
        local_size=self._local_size,
        local_num=self._local_num,
    )

  @property
  def _global_size(self):
    return getattr(self, "_dino_global_size", 224)

  @property
  def _local_size(self):
    return getattr(self, "_dino_local_size", 96)

  @property
  def _local_num(self):
    return getattr(self, "_dino_local_num", 8)

  def build(self, args, encoder, device):
    self._dino_global_size = getattr(args, "dino_global_size", 224)
    self._dino_local_size = getattr(args, "dino_local_size", 96)
    self._dino_local_num = getattr(args, "dino_local_num", 8)
    return DINO(
        encoder,
        proj_dim=args.dino_proj_dim,
        proj_hidden=args.dino_proj_hidden,
        teacher_temp=args.dino_teacher_temp,
        student_temp=args.dino_student_temp,
        center_momentum=args.dino_center_momentum,
    ).to(device)

  def train_step(self, model, images, global_step, *, labels=None):
    set_train_mode(model, "train")
    if isinstance(images, (list, tuple)):
      global_crops = torch.cat(images[:2], dim=0)
      local_crops = torch.cat(images[2:], dim=0)
    else:
      global_crops = images
      local_crops = images
    loss, info = model(global_crops, local_crops)
    momentum = self._current_momentum(global_step, model)
    model.update_momentum(momentum)
    return loss, info

  def _current_momentum(self, global_step, model):
    total = getattr(model, "_total_steps", 0)
    if total <= 0:
      return self._momentum_end()
    ratio = min(global_step / total, 1.0)
    start = self._momentum_start()
    end = self._momentum_end()
    return end + (start - end) * (1.0 - ratio)

  def _momentum_start(self):
    return getattr(self, "_dino_momentum", 0.996)

  def _momentum_end(self):
    return getattr(self, "_dino_final_momentum", 1.0)

  def on_epoch_end(self, model, epoch, writer):
    pass

  def get_checkpoint_state(self, model, args):
    state = {
        "method": "dino",
        "momentum": self._momentum_start(),
        "final_momentum": self._momentum_end(),
    }
    if hasattr(model, "loss"):
      state["center"] = model.loss.center.clone()
    return state

  def load_checkpoint_state(self, model, state, args):
    self._dino_momentum = state.get("momentum", args.dino_momentum)
    self._dino_final_momentum = state.get("final_momentum", args.dino_final_momentum)
    if "center" in state and hasattr(model, "loss"):
      model.loss.center.copy_(state["center"])

  def validate(self, model, images, num_samples):
    return None
