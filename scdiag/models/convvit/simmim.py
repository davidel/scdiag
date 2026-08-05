"""SimMIM pre-training wrapper for ConvViT.

Provides :class:`ConvViTSimMIM` (encoder + mask token + lightweight decoder)
and the patchify / unpatchify / masking utilities required for masked image
modelling on the ConvViT patch grid.

Reference
---------
Z. Xie et al., "SimMIM: A Simple Framework for Masked Image Modeling",
CVPR 2022.  https://arxiv.org/abs/2111.09886
"""

import torch
import torch.nn as nn
import torch.nn.functional as F

def patchify(images, patch_size):
  """Convert images to per-patch pixel vectors.

    ``(B, C, H, W)`` → ``(B, N, patch_size² × C)`` where
    ``N = (H / patch_size) × (W / patch_size)``.
    """
  B, C, H, W = images.shape
  p = patch_size
  h, w = H // p, W // p
  # (B, C, H, W) → (B, C, h, p, w, p)
  x = images.reshape(B, C, h, p, w, p)
  # → (B, h, w, p, p, C)
  x = x.permute(0, 2, 4, 3, 5, 1)
  # → (B, h * w, p * p * C)
  x = x.reshape(B, h * w, p * p * C)
  return x


def unpatchify(patches, patch_size, img_size=448, channels=3):
  """Convert per-patch pixel vectors back to an image.

    ``(B, N, p² × C)`` → ``(B, C, H, W)``.
    """
  B = patches.shape[0]
  p = patch_size
  h = w = img_size // p
  # (B, N, p²C) → (B, h, w, p, p, C)
  x = patches.reshape(B, h, w, p, p, channels)
  # → (B, C, h, p, w, p)
  x = x.permute(0, 5, 1, 3, 2, 4)
  # → (B, C, H, W)
  x = x.reshape(B, channels, h * p, w * p)
  return x


def random_mask(batch_size, num_patches, mask_ratio=0.60, device=None):
  """Generate independent random per-patch masks.

    Returns a boolean tensor of shape ``(B, N)`` where ``True`` = masked.
    Each image has exactly ``round(N * mask_ratio)`` masked patches.
    """
  num_masked = int(num_patches * mask_ratio)
  # (B, N) — all False initially
  mask = torch.zeros(batch_size, num_patches, dtype=torch.bool, device=device)
  for i in range(batch_size):
    idx = torch.randperm(num_patches, device=device)[:num_masked]
    mask[i, idx] = True
  return mask


def simmim_loss(pred, target, mask):
  """SimMIM masked pixel reconstruction loss (MSE).

    Args:
        pred:   ``(B, N, D)`` predicted pixel values for all patches.
        target: ``(B, N, D)`` ground-truth pixel values for all patches.
        mask:   ``(B, N)`` boolean mask, ``True`` = masked.

    Returns:
        Scalar loss (mean over masked patches and channels).
    """
  # Per-patch, per-channel squared error
  loss = (pred - target)**2  # (B, N, D)
  loss = loss.mean(dim=-1)  # (B, N) — average over D
  loss = (loss * mask.float()).sum() / mask.float().sum().clamp(min=1)
  return loss


class ConvViTSimMIM(nn.Module):
  """SimMIM wrapper around the ConvViT encoder.

    Wraps :class:`CustomPatchTransformer` and adds:

    * A single **learned mask token** that replaces masked patch embeddings.
    * A lightweight **decoder MLP** that predicts raw pixel values for masked
      patches.

    During pre-training the full encoder (ConvNet stem + transformer) runs on
    ALL patches.  The only compute saving comes from the decoder being tiny
    and the loss being computed only at masked positions — the encoder itself
    is fully evaluated.

    Args:
        encoder: A :class:`CustomPatchTransformer` instance.
            Its ``head`` and ``cls_guided_pool`` are **not** used during
            pre-training (they are bypassed).
        decoder_dim: Hidden dimension of the decoder MLP.
        decoder_depth: Number of ``Linear → GELU`` layers in the decoder
            (excluding the final linear projection).  ``2`` is the SimMIM
            default.
    """

  def __init__(self, encoder, decoder_dim=768, decoder_depth=2):
    super().__init__()
    # Accept a wrapper (ConvViTForClassification) or raw model.
    # If the wrapper exposes .model, use the raw encoder; otherwise
    # use the object directly (it must have encoder_forward / patch_embed).
    if hasattr(encoder, "model"):
      self.encoder = encoder.model
    else:
      self.encoder = encoder
    embed_dim = self.encoder.pos_embedding.shape[-1]

    # Patch size derived from the conv stem (2 ** num_conv_layers).
    # The decoder output dim must equal patch_size² × 3 (raw pixel values
    # per patch) — NOT embed_dim — to match the reconstruction target.
    ps = self.encoder.patch_embed.patch_size
    self._patch_pixel_dim = ps * ps * 3  # e.g. 16*16*3 = 768

    # Mask token (one learned vector shared across all positions).
    self.mask_token = nn.Parameter(torch.zeros(1, 1, embed_dim))
    nn.init.trunc_normal_(self.mask_token, std=0.02)

    # Decoder MLP.
    layers = []
    in_dim = embed_dim
    for _ in range(decoder_depth):
      layers += [nn.Linear(in_dim, decoder_dim), nn.GELU()]
      in_dim = decoder_dim
    layers.append(nn.Linear(decoder_dim, self._patch_pixel_dim))
    self.decoder = nn.Sequential(*layers)

  def forward(self, images, mask):
    """SimMIM forward pass.

    Args:
        images: ``(B, 3, H, W)`` input images.
        mask:   ``(B, N)`` boolean mask, ``True`` = masked.

    Returns:
        pred:   ``(B, N, D)`` predicted pixel values for **all** patches.
        target: ``(B, N, D)`` raw pixel values for **all** patches.
    """
    B = images.shape[0]

    # 1. Reconstruction target (raw pixels, patchified)
    patch_size = self.encoder.patch_embed.patch_size
    target = patchify(images, patch_size=patch_size)  # (B, N, D)

    # 2. Conv stem — runs on ALL patches (conv needs spatial context)
    x = self.encoder.patch_embed(images)  # (B, N, D)

    # 3. Scatter mask token into masked positions
    mask_tokens = self.mask_token.expand(B, x.shape[1], -1)  # (B, N, D)
    x = torch.where(mask.unsqueeze(-1), mask_tokens, x)

    # 4. Run transformer encoder via non-mutating method
    #    (no need to remove classification heads)
    spatial = self.encoder.encoder_forward(x)  # (B, N, D)

    # 5. Decode
    pred = self.decoder(spatial)  # (B, N, D)

    return pred, target
