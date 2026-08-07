import torch
import torch.nn as nn
import torch.nn.functional as F


def drop_path(x, drop_prob=0.0, training=False):
  """Stochastic Depth implementation."""
  if drop_prob == 0.0 or not training:
    return x
  keep_prob = 1 - drop_prob
  shape = (x.shape[0],) + (1,) * (x.ndim - 1)
  random_tensor = keep_prob + torch.rand(shape, dtype=x.dtype, device=x.device)
  random_tensor.floor_()
  output = x.div(keep_prob) * random_tensor
  return output


class DropPath(nn.Module):
  """Drop paths (Stochastic Depth) per sample."""

  def __init__(self, drop_prob=None):
    super().__init__()
    self.drop_prob = drop_prob

  def forward(self, x):
    return drop_path(x, self.drop_prob, self.training)


class ConvPatchEmbeddingBlock(nn.Module):
  """Single block with stride-1 conv + stride-2 conv + skip connection."""

  def __init__(self, in_channels, mid_channels, out_channels):
    super().__init__()
    # Main path: stride-1 conv + stride-2 conv
    self.conv1 = nn.Conv2d(in_channels, mid_channels, 3, stride=1, padding=1)
    self.bn1 = nn.BatchNorm2d(mid_channels)
    self.conv2 = nn.Conv2d(mid_channels, out_channels, 3, stride=2, padding=1)
    self.bn2 = nn.BatchNorm2d(out_channels)
    # Skip path: avg pool + 1×1 projection
    self.skip_pool = nn.AvgPool2d(2, stride=2)
    self.skip_proj = nn.Conv2d(in_channels, out_channels, 1)
    self.skip_bn = nn.BatchNorm2d(out_channels)

  def forward(self, x):
    identity = x
    x = F.gelu(self.bn1(self.conv1(x)))
    x = self.bn2(self.conv2(x))
    skip = self.skip_bn(self.skip_proj(self.skip_pool(identity)))
    # conv2 (kernel=3, stride=2, pad=1) and AvgPool2d(2, 2) disagree
    # by 1 px on odd spatial dims.  Crop the larger to match.
    if x.shape[-2:] != skip.shape[-2:]:
      x = x[:, :, :skip.shape[-2], :skip.shape[-1]]
    return F.gelu(x + skip)


class ConvPatchEmbedding(nn.Module):
  """Multi-block conv front-end with skip connections."""

  def __init__(self, img_channels, embed_dim, num_blocks, img_size):
    super().__init__()
    # Compute channel progression
    step = (embed_dim - img_channels) / num_blocks
    channels = []
    for i in range(num_blocks):
      if i == num_blocks - 1:
        channels.append(embed_dim)
      else:
        channels.append(int(img_channels + (i + 1) * step))

    # Build blocks
    blocks = []
    in_ch = img_channels
    for out_ch in channels:
      mid_ch = out_ch  # mid = out for this design
      blocks.append(ConvPatchEmbeddingBlock(in_ch, mid_ch, out_ch))
      in_ch = out_ch
    self.blocks = nn.Sequential(*blocks)

    # Compute num_patches deterministically from img_size (no forward side-effects)
    self.num_patches_h = img_size // (2**num_blocks)
    self.num_patches_w = img_size // (2**num_blocks)

  @property
  def patch_size(self):
    """Effective spatial downsampling factor of the conv stem.

    Each block halves the spatial resolution, so
    ``patch_size = 2 ** num_blocks``.
    """
    return 2**len(self.blocks)

  @property
  def num_patches(self):
    return self.num_patches_h * self.num_patches_w

  def forward(self, x):
    x = self.blocks(x)
    x = x.flatten(2).transpose(1, 2)  # (B, N, embed_dim)
    return x


class SwiGLUFFN(nn.Module):
  """SwiGLU feed-forward network."""

  def __init__(self, in_features, hidden_features, dropout=0.0):
    super().__init__()
    self.w12 = nn.Linear(in_features, 2 * hidden_features)
    self.w3 = nn.Linear(hidden_features, in_features)
    self.dropout = nn.Dropout(dropout)

  def forward(self, x):
    x12 = self.w12(x)
    x1, x2 = x12.chunk(2, dim=-1)
    return self.dropout(self.w3(F.silu(x1) * x2))


class Block(nn.Module):
  """Transformer block with pre-norm, attention, and SwiGLU FFN."""

  def __init__(self, embed_dim, num_heads, dropout=0.1, drop_path=0.0):
    super().__init__()
    self.ln1 = nn.LayerNorm(embed_dim)
    self.attn = nn.MultiheadAttention(embed_dim,
                                      num_heads,
                                      dropout=dropout,
                                      batch_first=True)
    self.drop_path = DropPath(drop_path) if drop_path > 0.0 else nn.Identity()
    self.ln2 = nn.LayerNorm(embed_dim)
    hidden_dim = int(embed_dim * 4)
    self.ffn = SwiGLUFFN(embed_dim, hidden_dim, dropout)

  def forward(self, x):
    norm_x = self.ln1(x)
    attn_out, _ = self.attn(norm_x, norm_x, norm_x)
    x = x + self.drop_path(attn_out)
    x = x + self.drop_path(self.ffn(self.ln2(x)))
    return x


class CLSGuidedAttentionPooling(nn.Module):
  """Use the final CLS token as a query to attention-weight the spatial tokens."""

  def __init__(self, embed_dim=512, num_heads=8, dropout=0.1):
    super().__init__()
    self.norm_cls = nn.LayerNorm(embed_dim)
    self.norm_spatial = nn.LayerNorm(embed_dim)
    self.cross_attn = nn.MultiheadAttention(embed_dim,
                                            num_heads,
                                            dropout=dropout,
                                            batch_first=True)

  def forward(self, cls_out, spatial_out):
    """
    cls_out:     [B, 1, D]     — final CLS token from transformer
    spatial_out: [B, N, D]     — final spatial tokens from transformer
    Returns:     [B, D]        — attention-weighted pooling
    """
    attn_out, _ = self.cross_attn(
        query=self.norm_cls(cls_out),
        key=self.norm_spatial(spatial_out),
        value=spatial_out,
    )
    return attn_out.squeeze(1)  # [B, D]


class CustomPatchTransformer(nn.Module):
  """Conv-ViT hybrid model: convolutional patch embedding + Transformer encoder."""

  def __init__(
      self,
      num_classes,
      img_size=320,
      embed_dim=512,
      num_heads=8,
      depth=12,
      dropout=0.1,
      drop_path_rate=0.1,
      num_conv_layers=4,
  ):
    super().__init__()
    self.patch_embed = ConvPatchEmbedding(img_channels=3,
                                          embed_dim=embed_dim,
                                          num_blocks=num_conv_layers,
                                          img_size=img_size)
    self.num_tokens = self.patch_embed.num_patches
    self.pos_embedding = nn.Parameter(
        torch.randn(1, self.num_tokens + 1, embed_dim) * 0.02)
    self.cls_token = nn.Parameter(torch.randn(1, 1, embed_dim) * 0.02)
    self.cls_pos = nn.Parameter(torch.randn(1, 1, embed_dim) * 0.02)
    self.pos_drop = nn.Dropout(p=dropout)
    dpr = [x.item() for x in torch.linspace(0, drop_path_rate, depth)]
    self.transformer_layers = nn.ModuleList([
        Block(
            embed_dim=embed_dim,
            num_heads=num_heads,
            dropout=dropout,
            drop_path=dpr[i],
        ) for i in range(depth)
    ])
    self.ln_norm = nn.LayerNorm(embed_dim)
    self.cls_guided_pool = CLSGuidedAttentionPooling(
        embed_dim=embed_dim,
        num_heads=min(num_heads, 8),
        dropout=dropout,
    )
    self.head = nn.Linear(embed_dim, num_classes)
    self.apply(self._init_weights)

  def _init_weights(self, m):
    if isinstance(m, nn.Linear):
      nn.init.trunc_normal_(m.weight, std=0.02)
      if m.bias is not None:
        nn.init.constant_(m.bias, 0)
    elif isinstance(m, nn.LayerNorm):
      nn.init.constant_(m.bias, 0)
      nn.init.constant_(m.weight, 1.0)

  def _run_transformer(self, patch_embeddings):
    """Run the shared transformer core on patch embeddings.

        Applies CLS token prepend, positional embeddings, all transformer
        layers, and layer norm.  Returns both CLS and spatial outputs so
        that callers can decide how to use them.

        Args:
            patch_embeddings: ``(B, N, D)`` output from the conv stem
            (or modified version with mask tokens injected).

        Returns:
            ``(cls_out, spatial_out)`` where *cls_out* is ``(B, 1, D)``
            and *spatial_out* is ``(B, N, D)``.
        """
    B = patch_embeddings.shape[0]
    x = patch_embeddings

    # Prepend learnable CLS token with its own positional embedding
    cls = self.cls_token.expand(B, -1, -1) + self.cls_pos  # [B, 1, D]
    x = torch.cat([cls, x], dim=1)  # [B, 1+N, D]
    x = x + self.pos_embedding[:, :x.shape[1], :]
    x = self.pos_drop(x)

    # Transformer encoder (all layers, all tokens)
    for layer in self.transformer_layers:
      x = layer(x)
    x = self.ln_norm(x)

    return x[:, :1, :], x[:, 1:, :]  # CLS, spatial

  def forward(self, x):
    embeddings = self.patch_embed(x)  # [B, N, D]
    cls_out, spatial_out = self._run_transformer(embeddings)
    pooled = self.cls_guided_pool(cls_out, spatial_out)  # [B, D]
    return self.head(pooled)

  def encoder_forward(self, patch_embeddings):
    """Run the transformer encoder on pre-computed patch embeddings.

        This method provides a non-mutating way to access the encoder's
        internal representations (for SimMIM pre-training, feature
        extraction, etc.) without removing classification heads.

        The caller is responsible for:

        1. Running the conv stem: ``embeddings = self.patch_embed(images)``
        2. Optionally injecting mask tokens (e.g. for SimMIM) before
           calling this method.

        Args:
            patch_embeddings: ``(B, N, D)`` output from the conv stem
            (or modified version with mask tokens injected).

        Returns:
            ``(B, N, D)`` spatial transformer output (CLS token dropped).
        """
    _, spatial_out = self._run_transformer(patch_embeddings)
    return spatial_out
