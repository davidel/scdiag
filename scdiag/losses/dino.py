"""DINO loss (Caron et al., ICCV 2021).

Cross-entropy between sharpened/centered teacher logits and student
logits across global crops.
"""

import torch
import torch.nn as nn


class DINOLoss(nn.Module):
  """DINO self-distillation loss with exponential moving average centering.

  Args:
      out_dim: Projection dimension.
      teacher_temp: Temperature for teacher sharpening.
      student_temp: Temperature for student logits.
      center_momentum: EMA momentum for the center.
  """

  def __init__(self, out_dim, teacher_temp=0.04, student_temp=0.1, center_momentum=0.9):
    super().__init__()
    self.teacher_temp = teacher_temp
    self.student_temp = student_temp
    self.center_momentum = center_momentum
    self.register_buffer("center", torch.zeros(1, out_dim))

  def forward(self, student_output, teacher_output):
    """Compute cross-entropy loss.

    Args:
      student_output: ``(B, D)`` student projection for global crops.
      teacher_output: ``(B, D)`` teacher projection for global crops
        (no gradient, centering and sharpening applied).

    Returns:
      Scalar loss tensor.
    """
    teacher_out = (teacher_output - self.center) / self.teacher_temp
    teacher_prob = torch.softmax(teacher_out, dim=1).detach()

    student_out = student_output / self.student_temp
    student_log_prob = torch.log_softmax(student_out, dim=1)

    loss = -(teacher_prob * student_log_prob).sum(dim=1).mean()
    return loss

  @torch.no_grad()
  def update_center(self, teacher_output):
    """Update the center with EMA."""
    batch_center = teacher_output.mean(dim=0, keepdim=True)
    self.center.mul_(self.center_momentum).add_(batch_center,
                                                alpha=1.0 - self.center_momentum)
