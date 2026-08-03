# ConvViT Self-Supervised Pre-Training Plan (SimMIM)

## Table of Contents
- [1. Executive Summary](#1-executive-summary)
- [2. Motivation](#2-motivation)
- [3. Pre-Training Method: SimMIM for ConvViT](#3-pre-training-method-simmim-for-convvit)
- [4. Dataset Ensemble](#4-dataset-ensemble)
- [5. Architecture Details](#5-architecture-details)
- [6. File Layout & Code Design](#6-file-layout--code-design)
- [7. Shared Code: `scdiag/checkpointing.py`](#7-shared-code-scdiagcheckpointingpy)
- [8. Encoder / Decoder / Loss Design](#8-encoder--decoder--loss-design)
- [9. Training Loop](#9-training-loop)
- [10. Checkpointing & Downstream Handoff](#10-checkpointing--downstream-handoff)
- [11. GPU Budget & Hyperparameters](#11-gpu-budget--hyperparameters)
- [12. CLI Interface](#12-cli-interface)
- [13. Testing Strategy](#13-testing-strategy)
- [14. Phased Roll-Out](#14-phased-roll-out)


---

## 1. Executive Summary

Add a **separate** `scdiag/pretrain.py` script that performs SimMIM
self-supervised pre-training on the ConvViT backbone using a stitched-together
multi-source skin-lesion image dataset. The pre-trained encoder weights are
exported as a standalone checkpoint that `scdiag/train.py` can load for
fine-tuning.

**Key design principles:**
1. The pre-training script is intentionally decoupled from `train.py`. The only
   shared artifact is a model checkpoint file and a small checkpointing utility
   module.
2. Shared checkpoint/resume logic is extracted into `scdiag/checkpointing.py`
   to avoid code duplication. Argument parsing stays script-specific since the
   two scripts have fundamentally different CLI interfaces.
3. Default image size is **448×448** (configurable via CLI).

---

## 2. Motivation

### Why pre-train?

Our ConvViT is currently trained from random initialization on ~10k HAM10000
images. With ~86M parameters, the model is heavily over-parameterized for that
dataset size. Pre-training on a larger corpus of dermoscopy images should:

1. **Reduce downstream sample complexity** — the backbone learns general
   dermatological features (texture, color variation, border irregularity) before
   seeing any labels.
2. **Improve melanoma recall** — our current worst-performing class (55.8%)
   likely suffers from the model not learning fine-grained visual patterns due to
   limited training data.
3. **Enable transfer across lesion types** — features learned from 300K+ images
   across multiple sources capture more intra-class variation.

### Why SimMIM instead of MAE?

| Concern | MAE | SimMIM (chosen) |
|---|---|---|
| **Conv stem compatibility** | Awkward — conv stem must see all patches, masking must happen after it | Natural — all patches go through encoder, mask tokens injected before transformer |
| **Implementation complexity** | High — need to separate visible/hidden tokens, custom transformer forward | Low — standard forward pass, mask token replacement |
| **Compute** | Cheaper (only visible tokens reach transformer) | ~3× more expensive (all patches processed) |
| **ConvViT fit** | The conv stem outputs 768-dim tokens; MAE must split these awkwardly | SimMIM replaces masked tokens with a learned mask token before positional embedding — clean |
| **Decoder** | Must handle variable-length visible tokens | Simple — fixed-length sequence, lightweight 2-layer MLP |

**SimMIM's simplicity wins for our hybrid architecture.** The compute cost is
acceptable at 448px with batch_size=32 on FP16 (~8–10 GB VRAM).

### Why separate from `train.py`?

| Concern | Reason |
|---|---|
| **Separation of concerns** | SimMIM requires a reconstruction loss, a decoder, masking logic, and mask tokens — none of which belong in supervised fine-tuning. |
| **Dependency hygiene** | `train.py` should not import SimMIM-specific losses or decoders. |
| **Reusability** | The pre-trained checkpoint is a plain `.pt` file. Any downstream script can load it. |
| **Simpler debugging** | A focused script is easier to debug than a monolith with branching logic. |

---

## 3. Pre-Training Method: SimMIM for ConvViT

### 3.1 SimMIM Overview (Xie et al., 2021)

SimMIM is the simplest masked image modeling framework:

1. **Random mask** a fraction (e.g. 60–70%) of input patches
2. **Replace** masked patches with a single **learned mask token** (one shared
   vector for all masked positions)
3. Run the **full encoder** on all tokens (visible + mask tokens)
4. A **lightweight decoder** predicts the raw RGB pixel values of masked patches
5. Loss is **MSE between predicted and actual pixel values** at masked positions

Key SimMIM properties:
- The encoder runs on ALL patches — no conditional computation, no variable
  sequence lengths
- Masking ratio of 60% is the standard (vs. 75% for MAE)
- The decoder is a 2-layer MLP (extremely lightweight)
- Random per-patch masking works well (no block-wise masking needed)

### 3.2 SimMIM for ConvViT — Detailed Forward Pass

Here is exactly how the forward pass works with ConvViT's architecture:

```
Input image: (B, 3, 448, 448)

Step 1: Patchify for loss target
  reshape → (B, 784, 768)   # 28×28 = 784 patches, each 16×16×3 = 768 values
  This is the reconstruction target (raw pixel values per patch).

Step 2: Patch embedding (ConvNet stem — runs on ALL patches)
  ConvPatchEmbedding(image) → (B, 784, 768)
  This is a 4-stage ConvNet with stride-2 convolutions:
    Stage 1: Conv(3→128, k3, s1) + Conv(128→160, k3, s2) → 112×112
    Stage 2: Conv(160→192, k3, s1) + Conv(192→256, k3, s2) → 56×56
    Stage 3: Conv(256→384, k3, s1) + Conv(384→512, k3, s2) → 28×28
    Stage 4: Conv(512→640, k3, s1) + Conv(640→768, k3, s2) → 28×28
  Output: (B, 28, 28, 768) → flatten → (B, 784, 768)

  CRITICAL: The conv stem sees the FULL image. It uses 3×3 convolutions
  that require spatial neighbors. This is fine — we don't mask before the
  conv stem. Masking happens AFTER.

Step 3: Random masking
  Generate random mask: (B, 784) boolean, True = masked
  At 60% masking ratio: ~470 masked, ~314 visible per image

Step 4: Replace masked tokens with mask token
  mask_token: nn.Parameter(torch.zeros(1, 1, 768))  — ONE learned vector
  For each image in the batch:
    visible_tokens = patch_tokens[~mask]  # ~314 tokens
    masked_positions = patch_tokens[mask]  # ~470 tokens
    masked_positions = mask_token.expand_as(masked_positions)  # replace with mask token

  Result: (B, 784, 768) — same shape as input, but masked positions
  now contain the learned mask token instead of real features.

Step 5: Add positional embedding + CLS token
  cls = self.cls_token.expand(B, -1, -1) + self.cls_pos  # (B, 1, 768)
  x = torch.cat([cls, x], dim=1)  # (B, 785, 768)
  x = x + self.pos_embedding[:, :785, :]
  x = self.pos_drop(x)

Step 6: Transformer encoder (ALL 12 layers, ALL 785 tokens)
  for layer in self.transformer_layers:
      x = layer(x)
  x = self.ln_norm(x)
  # x shape: (B, 785, 768)

Step 7: Extract spatial tokens (drop CLS)
  spatial_out = x[:, 1:, :]  # (B, 784, 768)

Step 8: Lightweight decoder
  decoder = nn.Sequential(
      nn.Linear(768, 768),
      nn.GELU(),
      nn.Linear(768, 16*16*3),  # predict 768 pixel values per patch
  )
  pred = decoder(spatial_out)  # (B, 784, 768)

Step 9: Compute loss (MSE on masked patches only)
  pred_masked = pred[mask]    # (B*470, 768)
  target_masked = target[mask]  # (B*470, 768)
  loss = F.mse_loss(pred_masked, target_masked)
```

### 3.3 Why This Works Well for ConvViT

1. **The conv stem is preserved exactly as-is.** No architectural modifications.
   It does what it was designed to do: produce spatially-aware patch embeddings.

2. **The mask token is learned alongside the transformer.** The transformer must
   learn to "fill in" missing patch information using context from visible patches
   — exactly the skill needed for downstream classification of partially-occluded
   or ambiguous lesions.

3. **The decoder is extremely lightweight** (2-layer MLP, ~1.2M params). It exists
   only to provide a training signal — it's discarded after pre-training.

4. **At 448×448, the patch grid is 28×28 = 784 patches.** This is a manageable
   sequence length for the 12-layer transformer (784² attention per layer is
   feasible on FP16 with ~8 GB VRAM at batch_size=32).

### 3.4 Masking Ratio: 60% (SimMIM default)

SimMIM's paper shows that 60% masking is optimal across a range of models.
Unlike MAE (which uses 75%), SimMIM benefits from a lower ratio because:
- The encoder processes all patches anyway — there's no compute incentive for
  higher masking
- Too high a ratio makes the task unrecoverable for the lightweight decoder

We will use 60% as the default, but make it configurable via CLI.

### 3.5 Why NOT Block-Wise Masking

SimMIM's paper explicitly shows that **random per-patch masking** works as well
as (or better than) block-wise masking. Since we're not worried about conv stem
leakage (the mask token replaces features, not pixels, after the conv stem
already ran), random masking is simpler and equally effective.

---

## 4. Dataset Ensemble

### 4.1 Goal

Create a unified, lazy-loading dataset of **dermoscopy images only** suitable for
unsupervised pre-training. Labels are ignored during SimMIM training — only
images matter.

### 4.2 Candidate Datasets

| Dataset | Images | Source | HF/Access | Notes |
|---|---|---|---|---|
| **HAM10000** | 10,015 | ISIC Archive | `bejakobic/ham10000` or ISIC API | 7 classes, dermoscopy. Our primary dataset. |
| **ISIC 2018 Task 1–2** | ~10,000 | ISIC Archive | ISIC CLI or API | Segmentation + classification. |
| **ISIC 2019** | ~25,331 | ISIC Archive | ISIC CLI or API | Multi-label, dermoscopy. |
| **ISIC 2020** | ~33,126 | ISIC Archive | `mrbrobot/isic-2024` (contains ISIC2020 images) | Binary melanoma vs. benign. |
| **Derm1M** | 403,563 unique | PubMed + ISIC | `redlessone/Derm1M` (gated, CC BY-NC 4.0) | **Largest available.** Vision-language pairs, 390 conditions. |
| **PH²** | 200 | Manual download | Not on HF | Too small to matter. |
| `ahmed-ai/skin-lesions-classification-dataset` | ~12,000 | HF | Public | HAM10000 + MSLD v2.0 merged. |

### 4.3 Recommended Ensemble

**Tier 1 — Definitely include:**

| Dataset | Est. Unique Images | Access Method |
|---|---|---|
| HAM10000 | 10,015 | `datasets.load_dataset()` |
| ISIC 2019 + 2020 (via ISIC CLI) | ~25,000 (after dedup with HAM) | `isic-cli` pre-download + `ImageFolder` |
| Derm1M (images only, ignore text) | ~300,000 (after dedup) | `datasets.load_dataset()` with `HF_TOKEN` |

**Estimated total: ~335,000 images** — sufficient for meaningful SimMIM pre-training.

**Tier 2 — Optional:**

| Dataset | Notes |
|---|---|
| `ahmed-ai/skin-lesions-classification-dataset` | Some overlap with HAM10000. |
| DERM12345 (12,345 images, 38 subclasses) | Newer (2024), may not be on HF yet. |

### 4.4 ISIC Data Access Strategy

The ISIC Archive provides a **61 GB bulk data snapshot** via AWS Open Data
(free download) and an official CLI tool (`isic-cli`) for selective filtering.

**Recommended approach:** Use `isic-cli` to pre-download ISIC challenge images
into a local directory, then load them via HuggingFace's `ImageFolder` or a
simple `Dataset` wrapper.

```bash
# One-time preparation (run manually before pre-training)
pip install isic-cli
isic image download --cases-dir ./data/isic2019 --challenge ISIC_2019
isic image download --cases-dir ./data/isic2020 --challenge ISIC_2020
```

This avoids on-the-fly API calls during training (which would be slow and
rate-limited). The `ISICArchiveDataset` class wraps the pre-downloaded
directory.

### 4.5 Derm1M Access

Derm1M is gated behind a CC BY-NC 4.0 license on HuggingFace. The user has
accepted the license. The dataset module must:

1. Accept an optional `HF_TOKEN` environment variable or `--hf_token` CLI arg
2. Pass it to `datasets.load_dataset("redlessone/Derm1M", token=hf_token)`
3. If access is denied (401/403), log a warning and skip the dataset gracefully
4. Never hard-fail on a single dataset — the ensemble must degrade gracefully

**Derm1M image format note:** Derm1M pairs PubMed journal images with clinical
text descriptions. Image quality may vary (thumbnails from papers vs. full-res
dermoscopy). The ensemble should filter out images below a minimum resolution
(e.g. 224×224) to avoid low-quality patches dominating the pre-training signal.

### 4.6 Dataset Ensemble Module: `scdiag/datasets/ensemble.py`

```python
class DermoscopyEnsemble(torch.utils.data.Dataset):
    """Concatenation of multiple skin-lesion image datasets for pre-training.

    Each constituent dataset is loaded lazily via HuggingFace datasets.
    Images are converted to RGB PIL Images on-the-fly.
    No labels are exposed — this is for pre-training only.
    """

    def __init__(self, dataset_configs, image_size=448, transform=None,
                 cache_dir=None, hf_token=None):
        """
        Args:
            dataset_configs: List of dicts, each with keys:
                - "name": HuggingFace dataset ID or local path prefix
                - "source": "hf" | "imagefolder" | "isic_cli"
                - "split": Which split to use (default: "train")
                - "image_column": Column name (auto-detected if None)
            image_size: Target image size
            transform: torchvision transforms to apply
            cache_dir: HF cache directory
            hf_token: HuggingFace token for gated datasets
        """
        ...

    def __len__(self):
        return sum(len(ds) for ds in self._datasets)

    def __getitem__(self, idx):
        # Flat index → (dataset_index, local_index) via prefix-sum
        ds_idx, local_idx = self._map_index(idx)
        image = self._datasets[ds_idx][local_idx]  # PIL Image → RGB
        if self.transform:
            image = self.transform(image)
        return image  # No label needed for SimMIM
```

**Key design decisions:**

1. **Flat indexing** — The ensemble presents a single `__len__` / `__getitem__`
   interface. A flat index is mapped to `(dataset_index, local_index)` via a
   prefix-sum lookup table computed at init time.

2. **Lazy loading** — Each constituent dataset is loaded on first access via
   `datasets.load_dataset()` (not at init). This avoids loading all datasets
   into RAM simultaneously.

3. **No labels** — For SimMIM pre-training, we only need images. The dataset
   returns `(transformed_image_tensor,)` without labels.

4. **No deduplication** — Duplicate images across datasets are acceptable for
   SimMIM (they act as implicit oversampling). Avoids expensive perceptual hash
   computations.

5. **Graceful degradation** — If any dataset fails to load (access denied,
   network error), it's skipped with a warning. The ensemble still works with
   the remaining datasets.

### 4.7 Pre-training Transform Pipeline

```python
def build_pretrain_transform(image_size=448):
    """SimMIM pre-training augmentations.

    SimMIM already provides strong regularization via masking, so we only
    need basic spatial/color augmentations.
    """
    return v2.Compose([
        v2.Resize(image_size, interpolation=InterpolationMode.BICUBIC),
        v2.CenterCrop(image_size),
        v2.RandomHorizontalFlip(p=0.5),
        v2.RandomVerticalFlip(p=0.5),
        v2.ToTensor(),  # [0, 1] float tensor
    ])
```

No normalization is applied at this stage — SimMIM reconstructs raw pixel
values. Normalization is applied only during supervised fine-tuning.

---

## 5. Architecture Details

### 5.1 ConvViT Architecture Recap

From `scdiag/models/convvit/model.py`:

```
CustomPatchTransformer:
  patch_embed: ConvPatchEmbedding(img_channels=3, embed_dim=768, num_blocks=4)
    → 4× ConvBlock with stride-1 + stride-2 + skip connections
    → Output: (B, N, 768) where N = (img_size / 16)²

  cls_token: (1, 1, 768)
  cls_pos: (1, 1, 768)
  pos_embedding: (1, N+1, 768)
  transformer_layers: 12× Block(embed_dim=768, num_heads=12, dropout=0.1)
  ln_norm: LayerNorm(768)
  cls_guided_pool: CLSGuidedAttentionPooling(768, num_heads=8)
  head: Linear(768, num_classes)
```

At **448×448 input**, the patch grid is **28×28 = 784 patches**. The positional
embedding must be resized from the default (typically 14×14 or 20×20). This is
handled by the ConvViT loader via the `image_size` parameter.

### 5.2 SimMIM Wrapper

```python
class ConvViTSimMIM(nn.Module):
    """SimMIM wrapper around the ConvViT encoder.

    Wraps CustomPatchTransformer and adds:
    - mask_token: learned replacement for masked patch tokens
    - decoder: lightweight MLP to predict masked pixel values
    """

    def __init__(self, encoder, decoder_dim=768, decoder_depth=2):
        super().__init__()
        self.encoder = encoder  # CustomPatchTransformer (without head)
        embed_dim = encoder.pos_embedding.shape[-1]

        # Mask token — ONE learned vector shared across all positions
        self.mask_token = nn.Parameter(torch.zeros(1, 1, embed_dim))
        nn.init.trunc_normal_(self.mask_token, std=0.02)

        # Lightweight decoder (SimMIM default: 2-layer MLP)
        self.decoder = nn.Sequential(
            nn.Linear(embed_dim, decoder_dim),
            nn.GELU(),
            nn.Linear(decoder_dim, 16 * 16 * 3),  # reconstruct patch pixels
        )

    def forward(self, images, mask):
        """
        Args:
            images: (B, 3, 448, 448) input images
            mask: (B, 784) boolean mask, True = masked
        Returns:
            pred: (B, 784, 768) predicted pixel values for all patches
            target: (B, 784, 768) target pixel values (patchified input)
        """
        # 1. Patchify for loss target (raw pixels)
        target = patchify(images, patch_size=16)  # (B, 784, 768)

        # 2. Patch embedding (conv stem — runs on ALL patches)
        x = self.encoder.patch_embed(images)  # (B, 784, 768)

        # 3. Replace masked positions with mask token
        mask_tokens = self.mask_token.expand_as(x[mask.reshape(x.shape[0], -1)
                                                    .unsqueeze(-1)
                                                    .expand(-1, -1, x.shape[-1])])
        # Simpler: scatter mask token into masked positions
        B, N, D = x.shape
        mask_flat = mask.reshape(B, N)  # (B, 784)
        mask_tokens = self.mask_token.expand(B, N, -1)  # (B, 784, 768)
        x = torch.where(mask_flat.unsqueeze(-1), mask_tokens, x)

        # 4. Add CLS + positional embedding
        cls = self.encoder.cls_token.expand(B, -1, -1) + self.encoder.cls_pos
        x = torch.cat([cls, x], dim=1)  # (B, 785, 768)
        x = x + self.encoder.pos_embedding[:, :x.shape[1], :]
        x = self.encoder.pos_drop(x)

        # 5. Transformer encoder (ALL layers, ALL tokens)
        for layer in self.encoder.transformer_layers:
            x = layer(x)
        x = self.encoder.ln_norm(x)

        # 6. Extract spatial tokens (drop CLS)
        spatial = x[:, 1:, :]  # (B, 784, 768)

        # 7. Decode
        pred = self.decoder(spatial)  # (B, 784, 768)

        return pred, target
```

### 5.3 `patchify` and `unpatchify`

```python
def patchify(images, patch_size=16):
    """Convert images to patch tokens.

    (B, C, H, W) → (B, N, patch_size² × C)
    where N = (H/patch_size) × (W/patch_size)
    """
    B, C, H, W = images.shape
    p = patch_size
    h, w = H // p, W // p
    # (B, C, H, W) → (B, C, h, p, w, p)
    x = images.reshape(B, C, h, p, w, p)
    # → (B, h, w, p, p, C)
    x = x.permute(0, 2, 4, 3, 5, 1)
    # → (B, h*w, p*p*C)
    x = x.reshape(B, h * w, p * p * C)
    return x

def unpatchify(patches, patch_size=16, img_size=448, channels=3):
    """Convert patch tokens back to images.

    (B, N, p²×C) → (B, C, H, W)
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
```

### 5.4 Loss Function

**Mean Squared Error (MSE) on raw pixel values at masked positions:**

```python
def simmim_loss(pred, target, mask):
    """SimMIM reconstruction loss.

    Args:
        pred: (B, N, p²×3) predicted pixel values
        target: (B, N, p²×3) target pixel values (raw, [0,1])
        mask: (B, N) boolean mask, True = masked
    Returns:
        scalar loss
    """
    loss = (pred - target) ** 2  # (B, N, p²×3)
    loss = loss.mean(dim=-1)     # (B, N) — mean per patch
    loss = (loss * mask.float()).sum() / mask.float().sum()  # mean over masked
    return loss
```

No per-patch normalization is applied (unlike MAE). SimMIM's paper shows that
raw pixel MSE works well and is simpler.

### 5.5 Conv Stem Frozen or Fine-Tuned?

**Decision: Train the full encoder (including conv stem) during SimMIM.**

Rationale:
- The conv stem has only ~0.5M parameters — tiny compared to the transformer's
  ~85M. Freezing it saves negligible compute.
- The conv stem should learn to produce features that are useful for the
  transformer to "fill in" missing patches. This is only possible if it's
  jointly optimized.
- SimMIM's paper trains all encoder parameters end-to-end.

### 5.6 Encoder Initialization

**Decision: Random initialization (current default).**

The ConvViT encoder starts from random weights (as it does today for supervised
training). We do NOT use ImageNet-pretrained weights for the conv stem or
transformer layers.

Rationale:
- SimMIM pre-training IS the initialization step — the encoder learns its
  representation from scratch on dermoscopy images.
- Starting from ImageNet weights would inject natural-image biases that may not
  transfer well to dermoscopy (different color distributions, texture patterns).
- SimMIM's paper shows strong results from random init given sufficient data.
  With ~335K images, we have enough data for this.

---

## 6. File Layout & Code Design

### 6.1 New Files

```
scdiag/
├── pretrain.py                    # SimMIM pre-training script (NEW)
├── checkpointing.py               # Shared checkpoint utilities (NEW, extracted from train.py)
├── datasets/                      # New package for dataset ensemble
│   ├── __init__.py
│   ├── ensemble.py                # DermoscopyEnsemble dataset
│   └── isic_archive.py           # ISIC Archive pre-downloaded dataset loader
├── models/
│   └── convvit/
│       ├── model.py               # Existing — no changes
│       ├── mae.py                 # SimMIM wrapper + patchify/unpatchify (NEW)
│       └── ...                    # loader.py, processor.py, config.py unchanged
├── train.py                       # Existing — imports from checkpointing.py (MINOR REFACTOR)
└── ...                            # Everything else unchanged
```

### 6.2 What Lives Where

| Component | Location | Rationale |
|---|---|---|
| SimMIM wrapper | `scdiag/models/convvit/mae.py` | Architecturally coupled to ConvViT (uses same `Block` class, same patch grid) |
| Patchify utilities | `scdiag/models/convvit/mae.py` | Operate on the ConvViT's patch grid (16×16) |
| Dataset ensemble | `scdiag/datasets/ensemble.py` | General-purpose, reusable for supervised training too |
| ISIC archive loader | `scdiag/datasets/isic_archive.py` | ISIC pre-downloaded images |
| Checkpoint utilities | `scdiag/checkpointing.py` | Shared between train.py and pretrain.py |
| Pre-training script | `scdiag/pretrain.py` | Standalone entry point |

---

## 7. Shared Code: `scdiag/checkpointing.py`

### 7.1 Functions to Extract from `train.py`

The following functions are currently in `train.py` and should be moved to
`scdiag/checkpointing.py`:

| Function | Lines in train.py | Used by pretrain.py? | Notes |
|---|---|---|---|
| `filter_state_dict(ckpt_state, model_state)` | 143–164 | ✅ Yes | Loading partial weights (e.g. encoder-only from SimMIM checkpoint) |
| `resume_checkpoint(...)` | 184–257 | ✅ Yes | Checkpoint resume logic with graceful degradation |
| `save_checkpoint(...)` | 93–137 (inline) | ✅ Yes | Checkpoint saving with GCS upload |

### 7.2 What Stays in `train.py`

| Function | Stays? | Reason |
|---|---|---|
| `parse_args()` | ✅ Yes | Script-specific CLI args (dataset, model, focal loss, etc.) |
| `parse_class_multipliers()` | ✅ Yes | Supervised-training-specific |
| `parse_state_flags()` | ✅ Yes | Supervised-training-specific |
| `load_augmentation_script()` | ✅ Yes | Supervised-training-specific |
| `build_transforms()` | ✅ Yes | Supervised-training-specific |
| `load_and_split_dataset()` | ✅ Yes | Supervised-training-specific |
| `compute_class_weights()` | ✅ Yes | Supervised-training-specific |
| `mixup_data()` | ✅ Yes | Supervised-training-specific |
| `train_one_epoch()` | ✅ Yes | Supervised-training-specific |
| `evaluate_performance()` | ✅ Yes | Supervised-training-specific |
| `train_xgboost_on_backbone()` | ✅ Yes | Supervised-training-specific |
| `main()` | ✅ Yes | Supervised-training-specific |

### 7.3 `train.py` Refactor

The refactor is minimal — `train.py` moves 3 functions to `checkpointing.py`
and imports them back:

```python
# train.py (after refactor)
from scdiag.checkpointing import filter_state_dict, resume_checkpoint, save_checkpoint

# All existing code works unchanged — the functions are just imported
# from a different location now.
```

### 7.4 `checkpointing.py` API

```python
"""Shared checkpoint save/load utilities."""

def filter_state_dict(ckpt_state, model_state):
    """Filter checkpoint state dict to skip shape-mismatched keys."""
    ...

def save_checkpoint(state, path, gcs_uri=None):
    """Save checkpoint to local path, optionally upload to GCS."""
    ...

def resume_checkpoint(ckpt_latest, ckpt_best, model, optimizer=None,
                      scheduler=None, scaler=None, states_to_load="all",
                      device="cpu"):
    """Resume training state from an existing checkpoint.

    Returns (start_epoch, best_metric).
    """
    ...
```

---

## 8. Encoder / Decoder / Loss Design

### 8.1 SimMIM Wrapper Details

The `ConvViTSimMIM` class (see Section 5.2) wraps the ConvViT encoder. Key
properties:

1. **Encoder:** The full `CustomPatchTransformer` — `ConvPatchEmbedding` +
   `TransformerLayer×12` + `LayerNorm`. The CLS token, CLSGuidedPooling, and
   classification head are NOT used during SimMIM training.

2. **Mask token:** A single learned `nn.Parameter` of shape `(1, 1, 768)`.
   Expanded to `(B, N, 768)` and scattered into masked positions via
   `torch.where`.

3. **Decoder:** A 2-layer MLP: `Linear(768→768) → GELU → Linear(768→768)`.
   This is the SimMIM default. The output dimension is 768 = 16×16×3 (one
   pixel value per channel per position in a patch).

4. **Loss:** MSE on raw pixel values at masked positions only.

### 8.2 Decoder Sizing

At 448×448 with patch_size=16:
- Number of patches: 28×28 = 784
- Each patch: 16×16×3 = 768 pixel values
- Decoder output: (B, 784, 768) — same dimension as encoder hidden state

The default 2-layer MLP decoder has ~1.2M parameters — negligible compared to
the encoder's ~86M. The decoder depth is configurable via `--decoder_depth`
(default=2). Increasing to 4–8 would add ~10–20M params, which may improve
representation quality at the cost of longer pre-training. We start with the
proven 2-layer default and tune if needed.

### 8.3 Masking Implementation

```python
def random_mask(batch_size, num_patches, mask_ratio=0.60):
    """Generate random per-patch mask.

    Args:
        batch_size: B
        num_patches: N (e.g. 784 for 28×28)
        mask_ratio: fraction of patches to mask (default 0.60)
    Returns:
        mask: (B, N) boolean tensor, True = masked
    """
    num_masked = int(num_patches * mask_ratio)
    mask = torch.zeros(batch_size, num_patches, dtype=torch.bool)
    for i in range(batch_size):
        masked_indices = torch.randperm(num_patches)[:num_masked]
        mask[i, masked_indices] = True
    return mask
```

This is simpler than MAE's block-wise masking and works just as well for SimMIM.

---

## 9. Training Loop

### 9.1 Structure

```python
def main():
    args = parse_args()

    # 1. Build dataset
    dataset = build_pretrain_dataset(args)
    loader = DataLoader(dataset, batch_size=args.batch_size,
                       shuffle=True, num_workers=args.num_workers,
                       pin_memory=True, drop_last=True)

    # 2. Build SimMIM model (encoder + decoder)
    encoder = load_encoder(args)  # ConvViT without head
    model = ConvViTSimMIM(encoder, decoder_dim=args.decoder_dim,
                          decoder_depth=args.decoder_depth)
    model = model.to(device)

    # 3. Optimizer — single param group, uniform LR
    optimizer = optim.AdamW(model.parameters(), lr=args.lr,
                            weight_decay=args.weight_decay,
                            betas=(0.9, 0.95))
    scheduler = optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=args.epochs, eta_min=args.lr * 0.01
    )

    # 4. Resume from checkpoint if available
    start_epoch, _ = resume_checkpoint(
        args.checkpoint + "_latest.pt",
        args.checkpoint + "_best.pt",
        model, optimizer, scheduler, device=device,
        states_to_load="all",
    )

    # 5. Training loop
    for epoch in range(start_epoch, args.epochs):
        model.train()
        for step, images in enumerate(loader):
            images = images.to(device)

            # Generate mask
            mask = random_mask(images.shape[0], num_patches=784,
                              mask_ratio=args.mask_ratio)
            mask = mask.to(device)

            # Forward
            pred, target = model(images, mask)
            loss = simmim_loss(pred, target, mask)

            # Backward
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()
            optimizer.zero_grad()

            # Log
            if step % args.log_every == 0:
                writer.add_scalar("pretrain/loss", loss.item(), global_step)

        scheduler.step()

        # Save checkpoint
        save_checkpoint(
            {"model_state_dict": model.state_dict(),
             "optimizer_state_dict": optimizer.state_dict(),
             "scheduler_state_dict": scheduler.state_dict(),
             "epoch": epoch,
             "loss": avg_loss,
             "config": {**encoder_config}},
            args.checkpoint + "_latest.pt",
            gcs_uri=args.gcs_checkpoint,
        )

        # Visualization (optional)
        if epoch % args.vis_every == 0:
            log_reconstruction_visualization(model, loader, writer, epoch)
```

### 9.2 Logging

- **TensorBoard:** `pretrain/loss`, `pretrain/lr`, `pretrain/epoch`
- **Console:** Every `log_every` steps, log running loss
- **Reconstruction visualization:** Every `vis_every` epochs, save a grid of
  (original | masked | reconstructed) images to TensorBoard. Critical for
  debugging — if reconstructions look wrong, the pre-training is failing.

### 9.3 Mixed Precision

Use `torch.amp.autocast("cuda", dtype=torch.float16)` for the forward pass.
SimMIM's MSE loss is well-suited for FP16.

### 9.4 Gradient Clipping

`torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)` — SimMIM
training can occasionally produce large gradients.

---

## 10. Checkpointing & Downstream Handoff

### 10.1 Pre-Training Checkpoint Format

```python
checkpoint = {
    "model_state_dict": model.state_dict(),    # Full SimMIM model
    "optimizer_state_dict": optimizer.state_dict(),
    "scheduler_state_dict": scheduler.state_dict(),
    "epoch": epoch,
    "loss": avg_loss,
    "config": {                                 # ConvViT config for reconstruction
        "image_size": 448,
        "patch_size": 16,
        "embed_dims": [128, 256, 512, 768],
        "vit_hidden_dim": 768,
        "vit_num_layers": 12,
        ...
    },
}
```

### 10.2 Loading Pre-Trained Weights for Fine-Tuning

The supervised `train.py` already supports `--checkpoint` for resuming. We add
a new flag `--pretrained_encoder` that loads encoder-only weights from the
SimMIM checkpoint, skipping shape-mismatched keys (the decoder, head, etc.):

```bash
scdiag-train \
    --model convvit \
    --dataset HAM10000 \
    --image_size 448 \
    --pretrained_encoder convvit_simmim_pretrained.pt \
    --epochs 100
```

This requires a **small addition to `train.py`** (~15 lines) in the checkpoint
loading section that extracts `model_state_dict` from the SimMIM checkpoint
and filters to only encoder keys. The existing `filter_state_dict()` from
`checkpointing.py` handles the shape-mismatch filtering.

### 10.3 What Transfers

| Component | Transfers? | Notes |
|---|---|---|
| ConvPatchEmbedding weights | ✅ | The core representation |
| Transformer layer weights | ✅ | The main benefit |
| Positional embedding | ✅ | Learned spatial layout |
| CLS token | ✅ | Though its SimMIM role was different (context for mask prediction) |
| CLSGuidedAttentionPooling | ❌ | Not present in SimMIM. Random init during fine-tuning. |
| Classification head | ❌ | Random init. Different for each downstream task. |
| Decoder | ❌ | Discarded after pre-training. |

---

## 11. GPU Budget & Hyperparameters

### 11.1 Memory Estimation (Single GPU, FP16)

| Component | Parameters | FP32 Memory |
|---|---|---|
| ConvViT Encoder | ~86M | ~344 MB |
| SimMIM Decoder (2-layer MLP) | ~1.2M | ~5 MB |
| Optimizer (AdamW) | — | ~347 MB |
| **Activation memory (batch_size=32, 784 patches)** | — | ~6 GB |
| **Total estimated** | — | **~7 GB** |

With mixed precision (FP16), this drops to **~4–5 GB**, well within consumer GPU
budgets (e.g. RTX 3080 = 10 GB, RTX 4090 = 24 GB).

### 11.2 Default Hyperparameters

| Parameter | Value | Rationale |
|---|---|---|
| `image_size` | **448** | Avoids losing fine-grained medical details; configurable via CLI |
| `patch_size` | 16 | Matches ConvViT config → 28×28 = 784 patches |
| `mask_ratio` | 0.60 | SimMIM standard |
| `batch_size` | 32 | Fits in ~5 GB VRAM with FP16 |
| `epochs` | 200 | With 335K images, this is ~67M images seen |
| `lr` | 1e-4 | Standard for SimMIM (slightly lower than MAE due to full forward) |
| `weight_decay` | 0.05 | Standard for AdamW |
| `lr_min` | 1e-6 | Cosine schedule minimum (1% of peak) |
| `warmup_epochs` | 10 | Linear warmup |
| `decoder_dim` | 768 | Same as encoder dim (SymMIM default) |
| `decoder_depth` | 2 | SimMIM default: 2-layer MLP |
| `num_workers` | 4 | Dataset I/O bound |
| `grad_clip` | 1.0 | Prevents occasional large gradients |

### 11.3 Training Time Estimation

| Dataset Size | Batch Size | Steps/Epoch | Time/Step (est.) | Time/Epoch |
|---|---|---|---|---|
| 335K images | 32 | ~10,469 | ~0.8s (FP16) | ~140 min |
| 335K images | 16 | ~20,938 | ~1.2s | ~420 min |

Total (batch_size=32): **~200 epochs × 140 min = ~470 hours** (~20 days).

This is feasible for overnight / multi-day runs on a consumer GPU.

### 11.4 Compute vs. MAE

SimMIM processes ALL patches through the encoder, while MAE processes only
visible patches. At 60% masking:
- SimMIM: 784 patches × 12 layers = 9,408 patch-layer operations
- MAE: ~314 patches × 12 layers = 3,768 patch-layer operations

SimMIM is ~2.5× more expensive. This is the trade-off for architectural
simplicity.

---

## 12. CLI Interface

```bash
scdiag-pretrain \
    --image_size 448 \
    --mask_ratio 0.60 \
    --batch_size 32 \
    --epochs 200 \
    --lr 1e-4 \
    --weight_decay 0.05 \
    --warmup_epochs 10 \
    --decoder_dim 768 \
    --decoder_depth 2 \
    --num_workers 4 \
    --output_dir ./checkpoints/pretrain \
    --log_every 50 \
    --vis_every 10 \
    --gcs_checkpoint gs://my-bucket/pretrain/checkpoints \
    --hf_token $HF_TOKEN \
    \
    --datasets \
        "HAM10000" \
        "isic_cli:./data/isic2019" \
        "isic_cli:./data/isic2020" \
        "redlessone/Derm1M" \
    --cache_dir ~/.cache/huggingface
```

### 12.1 Entry Point

Add to `pyproject.toml`:
```toml
[project.scripts]
scdiag-pretrain = "scdiag.pretrain:main"
```

---

## 13. Testing Strategy

### 13.1 Unit Tests

| Test | What it verifies |
|---|---|
| `test_patchify_roundtrip` | `unpatchify(patchify(x)) == x` |
| `test_random_mask_ratio` | Masked patches are exactly `mask_ratio` fraction |
| `test_simmim_loss_finite` | Loss is finite and non-negative for random inputs |
| `test_decoder_output_shape` | Decoder output shape matches `(B, N, p²×3)` |
| `test_simmim_forward_end_to_end` | Full forward pass with ConvViT encoder produces a scalar loss |
| `test_dermoscopy_ensemble_len` | `len(ensemble) == sum(len(ds) for ds in datasets)` |
| `test_dermoscopy_ensemble_getitem` | Returns a `(3, 448, 448)` tensor |
| `test_checkpoint_save_load` | Save + load checkpoint preserves model weights |
| `test_resume_checkpoint` | Resume from checkpoint restores epoch and optimizer state |
| `test_filter_state_dict` | Shape-mismatched keys are skipped gracefully |

### 13.2 Integration Test

```bash
# Smoke test: 1 epoch on HAM10000 only, batch_size=4
scdiag-pretrain \
    --datasets "HAM10000" \
    --batch_size 4 \
    --epochs 1 \
    --output_dir /tmp/pretrain_test
# Assert: checkpoint file exists, loss decreased
```

### 13.3 Files

```
tests/
├── test_simmim.py                 # Unit tests for SimMIM components
├── test_datasets_ensemble.py      # Unit tests for dataset ensemble
├── test_checkpointing.py          # Unit tests for shared checkpoint utilities
└── test_pretrain_smoke.py         # Integration test
```

---

## 14. Phased Roll-Out

### Phase 1: Infrastructure (no training yet)

1. Create `scdiag/checkpointing.py` — extract `filter_state_dict`,
   `resume_checkpoint`, `save_checkpoint` from `train.py`
2. Update `train.py` to import from `checkpointing.py` (verify all tests pass)
3. Create `scdiag/models/convvit/mae.py` — `ConvViTSimMIM`, `patchify()`,
   `unpatchify()`, `random_mask()`, `simmim_loss()`
4. Create `scdiag/datasets/__init__.py`, `ensemble.py`, `isic_archive.py`
5. Write unit tests for all of the above
6. Verify existing 138 tests still pass

### Phase 2: Pre-training script

1. Create `scdiag/pretrain.py` with full training loop
2. Add `--pretrained_encoder` flag to `scdiag/train.py`
3. Add entry point to `pyproject.toml`
4. Smoke test: 1 epoch on HAM10000 only

### Phase 3: Dataset assembly

1. Pre-download ISIC 2019 + 2020 images via `isic-cli`
2. Add Derm1M to the ensemble (test gated access)
3. Test with full dataset ensemble
4. Verify image counts and format consistency

### Phase 4: Full pre-training run

1. Run full 200-epoch pre-training
2. Monitor reconstruction quality via TensorBoard visualizations
3. Export final encoder checkpoint
4. Fine-tune on HAM10000 and compare against random-init baseline

---


