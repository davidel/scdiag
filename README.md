# scdiag

A training and inference toolkit for skin-lesion image classification. Supports
self-supervised pre-training, supervised fine-tuning, and XGBoost ensemble
inference — all from the command line.

## Why scdiag?

Medical imaging models face two practical problems: **labeled data is scarce**
and **off-the-shelf models are not domain-specific**. A ViT pre-trained on
ImageNet can classify cats and dogs, but dermatoscopic images look nothing
like natural photos — the feature distributions are fundamentally different.

scdiag solves this with a two-stage pipeline:

1. **Pre-train** on large, often unlabeled dermoscopy datasets (HAM10000,
   Derm1M, ISIC challenges) to learn skin-lesion-specific visual features.
2. **Fine-tune** on your smaller labeled dataset, starting from those
   pre-trained features instead of random initialization.

This consistently outperforms training from scratch, especially when your
labeled dataset has fewer than ~5 000 images. The tool also supports ensemble
inference with XGBoost on top of the learned features, which can squeeze out
additional performance for deployment.

## How it works

```
┌─────────────────────────────────────────────────────────────────┐
│                        Pre-Training                             │
│  Unlabeled/labeled dermoscopy images                            │
│  ──────────────────────────────────►  Encoder with learned      │
│  SimMIM / I-JEPA / SupCon              visual features          │
└────────────────────────────┬────────────────────────────────────┘
                             │  encoder weights
                             ▼
┌─────────────────────────────────────────────────────────────────┐
│                       Fine-Tuning                               │
│  Small labeled dataset  ──────────►  Trained classifier         │
│  + pre-trained encoder                for your task             │
└────────────────────────────┬────────────────────────────────────┘
                             │  backbone features
                             ▼
┌─────────────────────────────────────────────────────────────────┐
│                     (Optional) Ensemble                         │
│  Backbone features  ──────────►  XGBoost on top of the         │
│                                   learned representations      │
└─────────────────────────────────────────────────────────────────┘
```

**Pre-training** teaches the model to understand skin-lesion images — textures,
boundaries, colour patterns, and spatial relationships. **Fine-tuning** adapts
that understanding to your specific classification task (e.g. melanoma vs.
benign nevus). **Ensemble inference** (optional) trains a tree-based model on
the same features, which sometimes generalises better than a linear head for
small datasets.

## Installation

```bash
pip install -e .

# With timm model support:
pip install -e ".[timm]"

# With GCS checkpoint sync:
pip install -e ".[gcs]"

# With LoRA fine-tuning:
pip install -e ".[lora]"
```

**Requirements:** Python ≥ 3.9, PyTorch, torchvision, transformers, datasets,
NumPy, scikit-learn ≥ 1.3, XGBoost ≥ 2.0.

## Quick Start

The fastest way to get started:

```bash
# Fine-tune a ViT on a skin cancer dataset (5 epochs, ~2 minutes on GPU)
scdiag-train --model google/vit-base-patch16-224 \
             --dataset marmal88/skin_cancer \
             --epochs 5 \
             --batch_size 32 \
             --lr 3e-5 \
             --image_size 448
```

This downloads the model and dataset from HuggingFace, trains for 5 epochs,
and saves `scdiag_latest.pt` and `scdiag_best.pt`. See
[Pre-Training Guide](#pre-training-guide) below for the full pipeline
(starting with pre-training before fine-tuning).

---

## Pre-Training Guide

Pre-training learns general visual features from large datasets *before* you
fine-tune on your specific task. This is especially valuable in medical
imaging, where labeled data is expensive to obtain but raw images are often
available in bulk.

scdiag supports three pre-training methods, each with different strengths:

### Choosing Your Method

| Method | Needs labels? | Best when… | Key idea |
|---|---|---|---|
| **SimMIM** | No | You have large unlabeled datasets; want a simple, proven approach | Mask 60% of image patches, train the model to reconstruct the raw pixels |
| **I-JEPA** | No | You want faster training and better downstream transfer than SimMIM | Predict *representations* of masked regions, not raw pixels — avoids learning noise |
| **SupCon** | Yes | You have labels and want representations that cluster by class | Pull same-class images together, push different classes apart in feature space |

### How Each Method Works

**SimMIM** (Masked Image Modelling): Randomly masks ~60% of image patches and
trains a lightweight decoder to reconstruct the original pixels. The encoder
must learn to understand textures, boundaries, and spatial context from just
40% of the image. Think of it as a "fill in the blanks" exercise for vision
models. Good default choice when you have lots of unlabeled images.

**I-JEPA** (Joint-Embedding Predictive Architecture): Also masks patches, but
instead of reconstructing pixels, it predicts the *latent representation* of
the masked region from the visible context. This avoids wasting capacity on
pixel-level noise (e.g. exact JPEG compression artifacts) and learns more
transferable features. Uses a teacher–student setup with EMA momentum ramping.

**SupCon** (Supervised Contrastive Learning): Uses labels to define "positive"
pairs (same class) and "negative" pairs (different classes). The loss pulls
features of same-class images together and pushes different-class features
apart. Produces a feature space where similar lesions naturally cluster.
Requires a `ContrastiveEncoder` (backbone + projection head) and balanced
batch sampling to ensure each batch has enough same-class pairs.

### Typical Hyperparameters for Dermoscopy

These are reasonable starting points. Tune from here based on your dataset
size and GPU memory:

| Parameter | SimMIM | I-JEPA | SupCon |
|---|---|---|---|
| `--image_size` | 448 | 448 | 448 |
| `--batch_size` | 32 | 32 | 64 |
| `--lr` | 1e-4 | 1e-4 | 1e-4 |
| `--epochs` | 200 | 200 | 100 |
| `--scheduler` | CosineAnnealingLR | CosineAnnealingLR | CosineAnnealingLR |
| `--amp_dtype` | bfloat16 | bfloat16 | bfloat16 |
| `--mask_ratio` | 0.6 | — | — |
| `--teacher_momentum` | — | 0.996→1.0 | — |
| `--temperature` | — | — | 0.07 |
| `--samples_per_class` | — | — | 16 |

**Tips:**
- Start with 200 epochs for SimMIM/I-JEPA. SupCon converges faster (~100).
- `--temperature 0.07` is the standard from the original SupCon paper.
  Lower = sharper contrastive distribution; try 0.05–0.1.
- `--samples_per_class 16` with `--batch_size 64` gives 4 classes per batch
  on HAM10000 (7 classes). Adjust so batch_size is divisible by
  samples_per_class × num_classes.
- Use `--amp_dtype bfloat16` if your GPU supports it (Ampere+). Otherwise
  `float16` with GradScaler works too.

### Example: Full Pre-Training Pipeline

```bash
# Step 1: Pre-train with SimMIM on two large datasets
scdiag-pretrain --method simmim \
                --model convvit \
                --datasets HAM10000 "redlessone/Derm1M" \
                --cache_dir /tmp/pretrain_cache \
                --hf_token hf_XXXX \
                --image_size 448 \
                --batch_size 32 \
                --epochs 200 \
                --lr 1e-4 \
                --scheduler CosineAnnealingLR \
                --sched_arg T_max=200 --sched_arg eta_min=1e-6 \
                --amp_dtype bfloat16 \
                --checkpoint ./checkpoints/convvit_simmim

# Step 2: Fine-tune on your labeled dataset
scdiag-train --model convvit \
             --dataset marmal88/skin_cancer \
             --source_checkpoint ./checkpoints/convvit_simmim_latest.pt \
             --epochs 100 \
             --lr 3e-5 \
             --batch_size 32 \
             --amp_dtype bfloat16
```

### Example: Supervised Contrastive Pre-Training

```bash
scdiag-pretrain --method supcon \
                --model convvit \
                --datasets HAM10000 \
                --cache_dir /tmp/pretrain_cache \
                --hf_token hf_XXXX \
                --image_size 448 \
                --batch_size 64 \
                --samples_per_class 16 \
                --proj_dim 128 \
                --temperature 0.07 \
                --epochs 100 \
                --lr 1e-4 \
                --amp_dtype bfloat16 \
                --checkpoint ./checkpoints/convvit_supcon

# Then fine-tune as above with --source_checkpoint ./checkpoints/convvit_supcon_latest.pt
```

### Pre-Training CLI Reference

| Argument | Default | Description |
|---|---|---|
| `--method` | `simmim` | Pre-training method. Choices: `simmim`, `ijepa`, `supcon`. |
| `--model` | `convvit` | Model name registered in scdiag or HuggingFace model ID. |
| `--datasets` | (required) | Space-separated dataset names or local paths. |
| `--cache_dir` | `None` | HuggingFace cache directory for downloads. |
| `--hf_token` | `None` | HuggingFace token for gated datasets (or set `HF_TOKEN` env var). |
| `--image_column` | auto-detected | Explicit HF image column name. |
| `--label_column` | `None` | Explicit HF label column name. Required by `--method supcon` if non-standard. |
| `--strict_datasets` | `False` | Abort on first dataset-loading failure instead of skipping. |
| `--image_size` | `448` | Input image size (square). |
| `--batch_size` | `32` | Per-GPU batch size. |
| `--epochs` | `200` | Total pre-training epochs. |
| `--lr` | `1e-4` | Peak learning rate for AdamW. |
| `--amp_dtype` | `None` | Mixed precision: `float16` or `bfloat16`. Omit to disable. |
| `--num_workers` | `4` | DataLoader worker processes. |
| `--resume` | `True` | Auto-resume from latest checkpoint. Use `--no-resume` to disable. |
| `--state_save` | `opt,sched` | States to save: `opt`, `sched`, `amp`, `none`. |
| `--state_load` | `opt,sched` | States to restore on resume: `opt`, `sched`, `amp`, `none`. |
| `--checkpoint` | (required) | Checkpoint path prefix (saves `_latest.pt` and `_best.pt`). |
| `--log_level` | `INFO` | Minimum logging level. |
| `--grad_monitor` | `-1` | Log gradient statistics every N steps; `-1` disables. See [Gradient Monitor](#gradient-monitor). |
| `--norm_history` | `0` | Keep last N norm snapshots per parameter for trend analysis. |
| `--trend_top_n` | `10` | Show top N params in trend table by abs change %. `0` = show all. |
| `--grad_clip` | `1.0` | Maximum gradient norm for clipping. `0` disables. |
| `--lr_group` | `None` | Per-parameter-group learning rates (repeatable). Format: `"REGEX=LR"`. |
| `--llrd_decay` | `None` | Layer-wise learning rate decay factor. |
| `--vis_every` | `0` | Log reconstruction visualisation every N steps (SimMIM only). |
| `--model_arg` | `{}` | Override model configuration (repeatable). |
| `--proc_arg` | `{}` | Override processor configuration (repeatable). |
| `--optimizer` | `AdamW` | `torch.optim` optimizer class name or `.py` script path. |
| `--opt_arg` | `{}` | Extra optimizer kwargs (repeatable). |
| `--scheduler` | `None` | `torch.optim.lr_scheduler` class name or `.py` script path. |
| `--sched_arg` | `{}` | Extra scheduler kwargs (repeatable). |
| `--source_checkpoint` | `None` | Path to source checkpoint to absorb parameters from. |
| `--param_rename` | `None` | Regex-based key rename patterns (`SEARCH;REPLACE`). |

**SupCon-specific arguments:**

| Argument | Default | Description |
|---|---|---|
| `--proj_dim` | `128` | Output dimensionality of the projection head. |
| `--proj_hidden` | `None` | Hidden layer size of the projection MLP. `None` = single linear layer. |
| `--temperature` | `0.07` | NT-Xent temperature. Lower = sharper contrastive distribution. |
| `--samples_per_class` | `16` | Samples per class in each batch. Batch size should be divisible by this. |

### Dataset Ensemble

`scdiag-pretrain` stitches multiple datasets into a single pre-training
corpus. This is useful because no single dermoscopy dataset is large enough
for effective pre-training on its own.

Supported dataset types:
- **HuggingFace datasets** — any HF dataset ID that returns decoded image
  data (e.g. `HAM10000`). Gated datasets require `--hf_token` or `HF_TOKEN`.
- **Local image directories** — pass a path to a folder of images
  (ImageFolder format).

Datasets are loaded lazily (only when first accessed). By default, datasets
that fail to load are logged and skipped (best-effort mode). Use
`--strict_datasets` to abort on the first failure.

Images that cannot be decoded are skipped with a warning — this prevents a
single corrupted file from blocking an entire pre-training run.

#### Label Validation

When using `--method supcon` (or any future label-aware method), the ensemble
validates that every dataset supports labels. Datasets without a label column
cause a clear error *before* training begins, not a cryptic runtime failure
mid-epoch.

Labels are automatically remapped to a shared global label space across all
datasets, so mixing HAM10000 (with its label column) and a different dataset
with overlapping but differently-named classes works transparently.

### Preparing Datasets

Some datasets (like Derm1M) store images inside zip archives and require a
preparation step:

```bash
python scripts/prepare_derm1m.py --output_dir ./derm1m_images --token hf_XXX
```

Then use the extracted directory as a local dataset:

```bash
scdiag-pretrain --datasets ./derm1m_images /content/ham10000_grouped \
                --image_size 448 --batch_size 32 ...
```

See `scripts/prepare_ham10000.py` for another example that prepares the
HAM10000 dataset with lesion-id-grouped splits.

---

## Fine-Tuning Guide

After pre-training (or directly, if you skip pre-training), fine-tune a
classifier on your labeled dataset.

### Basic Fine-Tuning

```bash
scdiag-train --model google/vit-base-patch16-224 \
             --dataset marmal88/skin_cancer \
             --epochs 5 \
             --batch_size 32 \
             --lr 3e-5 \
             --image_size 448
```

### With a Custom Classifier Head

Replace the default linear head with a custom MLP or attention-based
classifier:

```bash
# Freeze backbone, train only the custom head
scdiag-train --model cls_model_wrapper:google/vit-base-patch16-224 \
             --dataset marmal88/skin_cancer \
             --classifier mlp \
             --classifier_args hidden=512 dropout=0.3 \
             --freeze ".*\.(head|pool)"
```

### With LoRA (Parameter-Efficient Fine-Tuning)

Freeze the entire backbone and train only small low-rank adapter matrices.
Reduces trainable parameters by ~97% while often matching full fine-tuning:

```bash
scdiag-train \
    --model cls_model_wrapper:facebook/dinov2-with-registers-large \
    --lora --lora_r 16 --lora_alpha 32 \
    --lora_target_modules "query,key,value" \
    --freeze "classifier\.(head|pool|encoder)" \
    --lr 3e-5 \
    --dataset marmal88/skin_cancer \
    --epochs 20
```

### From a Pre-Trained Checkpoint

Load encoder weights from a pre-training run (SimMIM, I-JEPA, or SupCon):

```bash
scdiag-train --model convvit \
             --dataset marmal88/skin_cancer \
             --source_checkpoint ./checkpoints/convvit_simmim_latest.pt \
             --epochs 100
```

The backbone weights are loaded automatically; the classifier head is
reinitialised (different `num_classes`). Use `--state_load none` to avoid
carrying over old optimizer/scheduler states.

### Hyperparameter Guidance

| Scenario | `--lr` | `--epochs` | `--batch_size` | Notes |
|---|---|---|---|---|
| Large dataset (>10k images) | 3e-5 | 20–50 | 32–64 | Standard fine-tuning |
| Small dataset (<1k images) | 1e-5 | 50–100 | 16–32 | Consider LoRA, stronger augmentation |
| From pre-trained checkpoint | 3e-5 | 50–100 | 32 | Lower LR than from scratch |
| Custom classifier head only | 1e-3 | 50–200 | 32 | Higher LR since only head trains |

**Tips:**
- Start with `--lr 3e-5` for full fine-tuning, `1e-3` for head-only training.
- Use `--mixup_alpha 0.2` for small datasets — it helps prevent overfitting.
- `--focal_gamma 2.0` down-weights easy examples, useful when classes are
  imbalanced.
- `--class_multipliers "melanoma=3.0"` increases the loss weight for
  clinically critical classes.

### Fine-Tuning CLI Reference

| Argument | Default | Description |
|---|---|---|
| `--model` | `google/vit-base-patch16-224` | HuggingFace model name, local path, or custom model (e.g. `convvit`, `timm:<name>`). |
| `--dataset` | `marmal88/skin_cancer` | HuggingFace dataset name or `imagefolder/PATH` for local data. |
| `--image_column` | auto-detected | Explicit HF image column name. |
| `--label_column` | auto-detected | Explicit HF label column name. |
| `--image_size` | `448` | Augmentation crop size (processor handles final resize). |
| `--epochs` | `5` | Number of training epochs. |
| `--batch_size` | `32` | Batch size. |
| `--lr` | `3e-5` | Peak learning rate. |
| `--weight_decay` | `0.01` | Weight decay. |
| `--label_smoothing` | `0.0` | Label smoothing factor. |
| `--focal_gamma` | `0.0` | Focal loss gamma (`0` = disabled). Down-weights easy examples. |
| `--class_multipliers` | `""` | Per-class severity multipliers. Example: `"melanoma=3.0,nevus=1.0"`. |
| `--sampler` | `none` | Training sampler: `none` (shuffle) or `weighted` (WeightedRandomSampler for class imbalance). |
| `--sampler_weights` | `frequency` | Weight mode for `--sampler weighted`: `frequency` (inverse-freq), `multipliers` (--class_multipliers), or `combined` (freq × multipliers). |
| `--mixup_alpha` | `0.0` | Mixup alpha (`0` = disabled; recommended: `0.2`). |
| `--grad_accum_steps` | `1` | Gradient accumulation steps (effective batch = batch_size × steps). |
| `--amp_dtype` | `None` | Mixed precision: `float16` or `bfloat16`. |
| `--lr_group` | `None` | Per-parameter-group learning rates (repeatable). Format: `"REGEX=LR"`. |
| `--llrd_decay` | `None` | Layer-wise LR decay factor per depth level. Example: `--llrd_decay 0.85`. |
| `--checkpoint` | `scdiag` | Checkpoint base path (`_latest.pt` / `_best.pt` appended). |
| `--log_every` | `20` | Log every N steps. |
| `--grad_monitor` | `-1` | Log gradient statistics every N steps. See [Gradient Monitor](#gradient-monitor). |
| `--norm_history` | `0` | Keep last N norm snapshots for trend analysis. |
| `--trend_top_n` | `10` | Show top N params in trend table. `0` = show all. |
| `--grad_clip` | `1.0` | Max gradient norm for clipping. `0` disables. |
| `--save_every` | `500` | Save checkpoint every N steps. |
| `--num_workers` | `2` | DataLoader worker processes. |
| `--log_level` | `INFO` | Logging level. |
| `--log_dir` | `None` | TensorBoard log directory (default: `<checkpoint_dir>/logs`). |
| `--cache_dir` | `None` | HuggingFace cache directory. |
| `--remote_checkpoint` | `None` | Remote URI for checkpoint sync (`gs://BUCKET/PREFIX` or `r2://BUCKET/PREFIX`). |
| `--source_checkpoint` | `None` | Path to source checkpoint to absorb parameters from. |
| `--param_rename` | `None` | Regex-based key rename patterns (`SEARCH;REPLACE`). |
| `--classifier` | `None` | Classifier head spec: registered name (e.g. `mlp`) or `.py` path. |
| `--classifier_args` | `{}` | Extra classifier kwargs (repeatable). Example: `hidden=512 dropout=0.3`. |
| `--freeze` | `None` | Regex patterns for parameters to keep trainable. All others frozen. |
| `--lora` | `False` | Enable LoRA via PEFT. Requires `pip install scdiag[lora]`. |
| `--lora_r` | `8` | LoRA rank. |
| `--lora_alpha` | `16` | LoRA alpha (scaling = `alpha / r`). |
| `--lora_dropout` | `0.0` | Dropout on LoRA layers. |
| `--lora_target_modules` | `None` | Comma-separated module names for LoRA (e.g. `"query,key,value"`). |
| `--optimizer` | `AdamW` | `torch.optim` optimizer class name or `.py` script path. |
| `--opt_arg` | `{}` | Extra optimizer kwargs (repeatable). |
| `--scheduler` | `None` | `torch.optim.lr_scheduler` class name or `.py` script path. |
| `--sched_arg` | `{}` | Extra scheduler kwargs (repeatable). |
| `--state_save` | `opt,sched,amp` | States to save: `opt`, `sched`, `amp`, `none`. |
| `--state_load` | `opt,sched,amp` | States to restore on resume. |
| `--xgboost_model` | `None` | Output path for XGBoost model (trains after PyTorch). |
| `--xgb_*` | various | XGBoost hyperparameters (see `--help` for full list). |
| `--model_arg` | `{}` | Override model configuration (repeatable). |
| `--proc_arg` | `{}` | Override processor configuration (repeatable). |
| `--train_augmentation_script` | `None` | Custom augmentation script. Must define `create_train_transform()`. |

Training automatically resumes from an existing `_latest.pt` or `_best.pt`
checkpoint if one exists at the `--checkpoint` path.

### LoRA Details

Low-Rank Adaptation freezes the pre-trained backbone and injects small
trainable low-rank matrices into attention layers. The LoRA output is
`ΔW = (alpha/r) × B @ A`, where A and B are the low-rank matrices.

| r | alpha | alpha/r | Use case |
|---|---|---|---|
| 8 | 16 | 2.0 | Conservative, very few params |
| 16 | 32 | 2.0 | Good default for medium datasets |
| 16 | 64 | 4.0 | Larger updates (helpful under bfloat16) |
| 32 | 64 | 2.0 | More capacity, same scaling |

LoRA can be combined with a custom classifier:

```bash
scdiag-train \
    --model cls_model_wrapper:facebook/dinov2-with-registers-large \
    --classifier cls_attention \
    --classifier_args 'num_encoder_layers=2' \
    --lora --lora_r 16 --lora_alpha 32 \
    --freeze 'classifier\.(head|pool|encoder)' \
    --lr_group 'backbone.*=1e-5' 'classifier.*=3e-4' \
    ...
```

### Cross-Dataset Resume

Switch from one dataset to another while keeping backbone weights:

```bash
scdiag-train --model facebook/convnextv2-base-22k-224 \
             --dataset ahmed-ai/skin-lesions-classification-dataset \
             --checkpoint scdiag \
             --state_load none \
             --epochs 10 \
             --batch_size 16 \
             --lr 3e-5 \
             --mixup_alpha 0.2 \
             --amp_dtype bfloat16
```

Backbone weights load via `strict=False`; the classifier head (different
`num_classes`) is reinitialised. `--state_load none` prevents carrying over
old optimizer/scheduler states.

---

## Inference Guide

Run inference on individual images:

```bash
scdiag-infer --model facebook/convnextv2-base-22k-224 \
             --checkpoint scdiag_best.pt \
             path/to/image.jpg path/to/other_image.png
```

Output is JSON with per-class probabilities:

```json
{
  "source": "image.jpg",
  "predictions": [
    {"label": "melanoma", "probability": 0.435},
    {"label": "benign_keratosis", "probability": 0.281}
  ]
}
```

### Inference CLI Reference

| Flag | Default | Description |
|---|---|---|
| `--model` | (required) | HuggingFace model name or custom model. |
| `--checkpoint` | (required) | Path to state dict or wrapped checkpoint. |
| `--top_k` | `None` | Show top-K predictions; omit for all classes. |
| `--output` | `None` | Write JSON results to file. |
| `--device` | `None` | Force PyTorch device (`cuda`, `cpu`). Auto-detected if omitted. |
| `--cache_dir` | `None` | HuggingFace cache directory. |
| `--xgboost_model` | `None` | XGBoost model path. If provided, runs XGBoost alongside PyTorch. |

Wrapped checkpoints (containing `model_state_dict` and metadata) are preferred
over raw state dictionaries, which produce a metadata warning.

### XGBoost Inference

When `--xgboost_model` is provided, the output includes both predictions:

```json
{
  "source": "image.jpg",
  "predictions": [
    {"label": "melanoma", "probability": 0.435}
  ],
  "xgboost_predictions": [
    {"label": "melanoma", "probability": 0.612}
  ]
}
```

---

## Gradient Monitor

`--grad_monitor N` logs a per-parameter gradient report every *N* training
steps. This helps diagnose training instability (exploding / vanishing
gradients, imbalanced parameter updates) before it shows up in the loss.

### Summary Line

```
[Step 29400] Gradient Report: 202 params | grad_rms: mean=5.68e-01 max=3.41e+00 min=1.66e-05 | grad/param: mean=2.94e-01
```

All norms are **RMS** (root mean square): L2 norm divided by `sqrt(numel)`.
This makes them independent of tensor shape and directly comparable across
parameters of different sizes.

| Field | Meaning |
|---|---|
| `params` | Total number of trainable parameters. |
| `grad_rms: mean/max/min` | RMS of the gradient tensor for each parameter, then aggregated. `max` is the single most aggressive gradient — the one most likely to cause instability. |
| `grad/param: mean` | Average gradient-to-parameter ratio. Healthy: < 0.1. Concerning: > 1.0. |

### Per-Parameter Columns

| Column | Symbol | What to look for |
|---|---|---|
| **g_rms** | `‖∇L‖/√N` | Compare across params. One param with g_rms 100× higher is a problem. |
| **p_rms** | `‖W‖/√N` | Per-element scale context. With `std=0.02` init, expect ~0.02. |
| **g/p** | `‖∇L‖ / (‖W‖ + ε)` | **Most useful column.** Healthy: < 0.1. Concerning: > 1.0 (update overshoots). Dangerous: > 5.0. |
| **g_max** | `max|∇L|` | Highlights individual neurons with extreme gradients. |
| **sparse** | `% zero` | High sparsity (> 50%) = most neurons not receiving signal. |
| **status** | | `OK` = healthy. `STL` = stalled. `OVF` = exploding. `IMB` = imbalanced. `GPR` = high g/p ratio. |

### Reading the Report

**Healthy training:** g/p < 0.1, `grad/param: mean` in 0.01–0.1 range, g_rms within ~10× across layers.

**Exploding gradients:** One or more params with `OVF`, g/p > 5.0, `grad_rms: max` >> `mean`. Fix: lower LR, add `--grad_clip 1.0`, or use warmup.

**Vanishing gradients:** Many params with `STL`, g_rms near 1e-7, high sparsity. Fix: increase LR, check for dead neurons.

**Imbalanced updates:** Some params `IMB`, large g/p disparity between layers. Fix: use `--lr_group` for different rates, or freeze the dominant component.

### Norm Trend History

`--norm_history N` (requires `--grad_monitor`) keeps the last N snapshots
per parameter. A trend summary table is appended to each report showing
direction (`UP`/`DOWN`/`---`), percentage change, and min/max values.

---

## Custom Models

scdiag supports any HuggingFace `AutoModelForImageClassification` model, any
timm model via `timm:<name>`, and custom architectures registered in
`scdiag.models`.

### Built-in Custom Models

| Name | Description | `--model` value |
|---|---|---|
| timm | Any model from [timm](https://github.com/huggingface/pytorch-image-models) | `timm:<model_name>` |
| ConvViT | Multi-block conv stem + ViT encoder with CLS-guided attention pooling | `convvit` |
| UVito | Frozen SMP encoder + learnable patch projection + Transformer encoder | `uvito` |
| ClsModelWrapper | HuggingFace backbone + custom classifier head | `cls_model_wrapper:<hf_name>` |
| ContrastiveEncoder | Backbone + projection head for contrastive pre-training | `contrastive_encoder:<hf_name>` |

### Adding a Custom Model

1. Create `scdiag/models/{name}/` with `model.py`, `processor.py`, `loader.py`,
   and `__init__.py`.
2. Add the import to `scdiag/models/__init__.py`.
3. The model must expose `.forward(pixel_values=images)` → object with `.logits`,
   and `config.id2label` / `config.label2id`.
4. CLI overrides via `--model_arg KEY=VALUE` are forwarded to the loader.

### Custom Classifiers

`ClsModelWrapper` lets you replace the default HF classification head:

```python
class Classifier(nn.Module):
    def __init__(self, num_labels, hidden_size, **kwargs):
        super().__init__()
        self.head = nn.Linear(hidden_size, num_labels)

    def forward(self, hidden_states):          # (B, N, D) tensor
        features = hidden_states[:, 0]         # CLS token
        return self.head(features)

    def extract_features(self, hidden_states): # (B, N, D) → (B, D)
        return hidden_states[:, 0]
```

The `extract_features` method is used by `--xgboost_model` for XGBoost
training on backbone features.

---

## Development

```bash
pip install -e ".[dev]"
pip install -e ".[timm]"   # optional: timm model support
pytest
```

Code is formatted with [yapf](https://github.com/google/yapf) using the
project's `.style.yapf` (Google style, 2-space indent).

## License

Apache-2.0
