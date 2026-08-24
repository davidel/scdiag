"""Supervised Contrastive Loss (Khosla et al., NeurIPS 2020).

Computes NT-Xent loss where same-class pairs are positives and
all other pairs are negatives.
"""

import torch
import torch.nn.functional as F


def supcon_loss(features, labels, temperature=0.07):
  """Supervised contrastive loss.

  Args:
      features: ``(B, D)`` L2-normalized projection features.
      labels: ``(B,)`` integer class labels.
      temperature: Temperature scaling factor.

  Returns:
      Scalar loss tensor.
  """
  device = features.device
  labels = labels.contiguous().view(-1, 1)

  # (B, B) boolean mask: True where two samples share a class.
  mask = torch.eq(labels, labels.T).float().to(device)

  # Pairwise cosine similarity, scaled by temperature.
  features = F.normalize(features, dim=1)
  logits = torch.mm(features, features.T) / temperature

  # Mask out self-similarity on the diagonal.
  logits_mask = torch.ones_like(mask) - torch.eye(mask.shape[0], device=device)
  mask = mask * logits_mask

  # For numerical stability, subtract the row-wise max.
  logits_max, _ = logits.max(dim=1, keepdim=True)
  logits = logits - logits_max.detach()

  # exp(logits) with diagonal masked to zero.
  exp_logits = torch.exp(logits) * logits_mask

  # Log-sum-exp over all non-diagonal entries.
  log_prob = logits - torch.log(exp_logits.sum(dim=1, keepdim=True))

  # Mean of log-prob over positive pairs for each anchor.
  n_positives = mask.sum(dim=1)
  # Guard against classes with only one sample (no positives).
  valid = n_positives > 0
  mean_log_prob = (mask * log_prob).sum(dim=1) / n_positives.clamp(min=1)

  if valid.sum() == 0:
    # No positive pairs exist — return zero loss (gradients still flow).
    return (logits * 0.0).sum()
  loss = -mean_log_prob[valid].mean()
  return loss
