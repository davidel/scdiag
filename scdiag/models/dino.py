"""DINO module — student + teacher networks for self-distillation."""

import copy

import torch
import torch.nn as nn

from scdiag.losses.dino import DINOLoss
from scdiag.models.contrastive import ProjectionHead


class DINO(nn.Module):
  """DINO: student sees all crops, teacher sees only global crops.

  Args:
      encoder: Backbone encoder.
      proj_dim: Output dimension of the projection head.
      proj_hidden: Hidden dimension of the projection head.
      backbone_dim: Explicit backbone output dimension.  Inferred if None.
      teacher_temp: Teacher sharpening temperature.
      student_temp: Student temperature.
      center_momentum: EMA momentum for center update.
  """

  def __init__(self,
               encoder,
               proj_dim=256,
               proj_hidden=2048,
               backbone_dim=None,
               teacher_temp=0.04,
               student_temp=0.1,
               center_momentum=0.9):
    super().__init__()
    self.student = _EncoderWithHead(encoder, proj_dim, proj_hidden, backbone_dim)
    self.teacher = copy.deepcopy(self.student)
    for p in self.teacher.parameters():
      p.requires_grad = False

    if backbone_dim is None:
      backbone_dim = self.student.backbone_dim
    self.loss = DINOLoss(proj_dim, teacher_temp, student_temp, center_momentum)

  @torch.no_grad()
  def update_momentum(self, momentum):
    for sp, tp in zip(self.student.parameters(), self.teacher.parameters()):
      tp.data.mul_(momentum).add_(sp.data, alpha=1.0 - momentum)

  def forward(self, global_crops, local_crops):
    """Compute DINO loss.

    Args:
        global_crops: ``(B, C, H, W)`` — 2 global crops stacked.
        local_crops: ``(N, C, h, w)`` — N local crops stacked.

    Returns:
        (loss, info_dict)
    """
    n_global = global_crops.shape[0]

    s_global = self.student(global_crops)
    s_local = self.student(local_crops)

    with torch.no_grad():
      t_global = self.teacher(global_crops)
      self.loss.update_center(t_global)

    s_all = torch.cat([s_global, s_local], dim=0)
    t_all = t_global.repeat((s_all.shape[0] // n_global) + 1, 1)[:s_all.shape[0]]

    loss = self.loss(s_all, t_all)
    return loss, {"loss": loss.item()}


class _EncoderWithHead(nn.Module):
  """Backbone + projection head."""

  def __init__(self, encoder, proj_dim, proj_hidden, backbone_dim=None):
    super().__init__()
    self.encoder = encoder
    if backbone_dim is None:
      backbone_dim = self._detect_dim()
    self.backbone_dim = backbone_dim
    self.projection = ProjectionHead(backbone_dim, proj_hidden, proj_dim)

  def _detect_dim(self):
    from scdiag.attr_utils import MISSING, get_attribute
    enc = self.encoder
    for path in (
        "config.hidden_size",
        "config.d_model",
        "config.num_features",
        "model.num_features",
        "num_features",
        "head.in_features",
        "model.head.in_features",
        "classifier.in_features",
        "classifier.feat_dim",
    ):
      val = get_attribute(enc, path)
      if val is not MISSING:
        return val
    raise ValueError("Cannot infer backbone output dimension.")

  def encode(self, images):
    from scdiag.model_utils import extract_backbone_features
    try:
      return extract_backbone_features(self.encoder, images)
    except (ValueError, AttributeError, RuntimeError):
      pass
    raw = self.encoder(images)
    if hasattr(raw, "logits"):
      if raw.pooler_output is not None:
        return raw.pooler_output
      return raw.last_hidden_state.mean(dim=1)
    return raw

  def forward(self, images):
    return self.projection(self.encode(images))
