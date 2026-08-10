# scdiag

Fine-tune HuggingFace image-classification models for skin-lesion classification
(and other image datasets) with a hand-rolled PyTorch training loop. Supports
self-supervised pre-training (SimMIM, I-JEPA) on multi-source dermoscopy
datasets.

## Features

### Supervised Fine-Tuning (`scdiag-train`)

- **Any HuggingFace image classification dataset** — auto-detects image and
  label columns, handles filepath-based images, ClassLabel casting, and
  pre-split datasets. Use `--image_column` and `--label_column` when a dataset
  uses non-standard or ambiguous column names.
- **Mixed precision** — optional AMP with `float16` (with GradScaler) or
  `bfloat16`.
- **Gradient accumulation** — effective batch size = `batch_size * grad_accum_steps`.
- **Configurable optimizer and scheduler** — use any `torch.optim` optimizer
  (e.g. `AdamW`, `SGD`) and any `torch.optim.lr_scheduler` scheduler
  (e.g. `CosineAnnealingLR`, `StepLR`) via CLI arguments. Custom scheduler
  scripts supported for complex schedules.
- **Class-weighted loss** with inverse-frequency weighting and label smoothing.
- **Cost-sensitive focal loss** — combined focal modulation and per-class
  clinical severity multipliers (`--focal_gamma`, `--class_multipliers`) for
  prioritizing rare or clinically critical classes (e.g. melanoma detection).
- **Mixup** — optional Mixup regularization (`--mixup_alpha`) for reducing
  overfitting on small datasets.
- **Source checkpoint absorption** — load weights from any source checkpoint via
  `--source_checkpoint`. Keys are automatically aligned by shape and name,
  so renames across architectures just work.
- **Custom model registry** — use any HuggingFace `AutoModelForImageClassification`
  model, or register custom architectures via the `scdiag.models` registry.
  First custom model: **ConvViT** (multi-block conv stem + ViT encoder with
  CLS-guided attention pooling). Use `--model convvit` to select it.
- **Hook-based feature extraction** — XGBoost backbone features are extracted
  via a forward hook on the classifier head, making feature extraction
  architecture-agnostic (works for ViT, ResNet, Swin, etc.).
- **Strong augmentations** — random rotation, elastic deformation, aggressive
  cropping, and color jitter tuned for dermoscopy images.
- **Checkpointing** — saves `_latest.pt` and `_best.pt` (by validation accuracy).
- **Auto-resume** — automatically resumes from an existing checkpoint if found.
  Supports **cross-dataset resume** (backbone weights transfer, classifier
  head reinitialised).
- **XGBoost classifier** — optionally train an XGBoost model on backbone features
  after PyTorch training completes (`--xgboost_model`). Compare linear head vs
  tree-based classifier performance.
- **TensorBoard** logging.
- **GCS sync** — optional checkpoint upload to Google Cloud Storage.

### Self-Supervised Pre-Training (`scdiag-pretrain`)

- **Pluggable pre-training methods** — select the algorithm via `--method`.
  Currently supported:
  - **`simmim`** (default) — masked image modelling. Masks ~60% of patches,
    reconstructs raw pixel values via a lightweight MLP decoder.
  - **`ijepa`** — joint-embedding predictive architecture (Assran et al.,
    CVPR 2023). Predicts latent representations of masked blocks using a
    student–teacher setup with EMA momentum ramping.
  Each method adds its own CLI arguments (e.g. `--mask_ratio` for SimMIM,
  `--teacher_momentum` for I-JEPA). New methods can be added by
  implementing the `PretrainMethod` interface in `scdiag/pretrain_methods/`.
- **Model-agnostic** — use `--model` to select any backbone registered in the
  model registry (e.g. `convvit`, or any HuggingFace model ID).
- **Multi-source dataset ensemble** — stitches together multiple HuggingFace
  datasets (e.g. HAM10000, ISIC challenges, Derm1M) into a single unified
  pre-training corpus with flat indexing and lazy loading.
- **Mixed precision** — AMP support (`float16` with GradScaler, or `bfloat16`).
- **Reconstruction visualisation** — periodic TensorBoard logging of
  method-specific validation images for qualitative monitoring (e.g.
  original/masked/reconstructed for SimMIM).

## Installation

```bash
pip install -e .

# With GCS support:
pip install -e ".[gcs]"
```

### Requirements

- Python >= 3.9
- PyTorch
- torchvision
- transformers
- datasets
- NumPy
- scikit-learn >= 1.3
- XGBoost >= 2.0

## Quick Start

```bash
scdiag-train --model google/vit-base-patch16-224 \
             --dataset marmal88/skin_cancer \
             --epochs 5 \
             --batch_size 32 \
             --lr 3e-5 \
             --image_size 448
```

## CLI Arguments

| Argument | Default | Description |
|---|---|---|
| `--model` | `google/vit-base-patch16-224` | HuggingFace model name, local path, or custom model name (e.g. `convvit`) |
| `--dataset` | `marmal88/skin_cancer` | HuggingFace dataset name or `imagefolder/PATH` for a local ImageFolder dataset |
| `--image_column` | auto-detected | Explicit HuggingFace image column name |
| `--label_column` | auto-detected | Explicit HuggingFace label column name |
| `--image_size` | `448` | Augmentation crop size (processor handles final resize) |
| `--train_augmentation_script` | `None` | Path or URL to a Python script defining `create_train_transform(image_size, **kwargs)` returning a list of v2 transforms. The fixed tail (ToImage, ToDtype, Normalize) is appended automatically. |
| `--epochs` | `5` | Number of training epochs |
| `--batch_size` | `32` | Batch size |
| `--lr` | `3e-5` | Peak learning rate |
| `--weight_decay` | `0.01` | Weight decay |
| `--label_smoothing` | `0.0` | Label smoothing factor |
| `--focal_gamma` | `0.0` | Focal loss gamma (`0` = disabled). Down-weights easy examples so the optimizer focuses on hard-to-classify samples. |
| `--class_multipliers` | `""` | Comma-separated `NAME=VALUE` pairs overriding per-class clinical severity multipliers. `NAME` is a label string or integer index; `VALUE` is a float. Unspecified classes default to `1.0`. Example: `"melanoma=3.0,melanocytic_Nevi=1.0"` |
| `--grad_accum_steps` | `1` | Gradient accumulation steps |
| `--amp_dtype` | `None` | Mixed precision: `float16` or `bfloat16` |
| `--checkpoint` | `scdiag` | Checkpoint base path (`_latest.pt` / `_best.pt` appended) |
| `--log_every` | `20` | Log every N steps |
| `--grad_monitor` | `-1` | Log gradient statistics every N steps; `-1` disables gradient monitoring |
| `--save_every` | `500` | Save checkpoint every N steps |
| `--num_workers` | `2` | DataLoader worker processes |
| `--log_level` | `INFO` | Logging level |
| `--log_dir` | `None` | TensorBoard log directory (default: `<checkpoint_dir>/logs`) |
| `--cache_dir` | `None` | HuggingFace cache directory |
| `--remote_checkpoint` | `None` | Remote URI for checkpoint sync (`gs://BUCKET/PREFIX` or `r2://BUCKET/PREFIX`). |
| `--mixup_alpha` | `0.0` | Mixup alpha (0 = disabled, recommended: `0.2`) |
| `--state_save` | `opt,sched,amp` | Comma-separated states to save: `opt`, `sched`, `amp`, `none` |
| `--state_load` | `opt,sched,amp` | Comma-separated states to restore on resume: `opt`, `sched`, `amp`, `none` |
| `--xgboost_model` | `None` | Output path for XGBoost model. If set, train XGBoost on backbone features after training. |
| `--xgb_max_depth` | `6` | XGBoost max tree depth |
| `--xgb_n_estimators` | `200` | XGBoost number of trees |
| `--source_checkpoint` | `None` | Path to a source checkpoint to absorb parameters from. Keys are aligned by shape and name before loading. |
| `--param_rename` | `None` | Regex-based key rename patterns for `--source_checkpoint`. Each pattern is `SEARCH;REPLACE` where SEARCH is a Python regex and REPLACE may use `$1`, `$2`, etc. Applied before shape-based alignment. |
| `--xgb_learning_rate` | `0.1` | XGBoost learning rate |
| `--xgb_subsample` | `0.8` | XGBoost row sampling ratio |
| `--xgb_colsample_bytree` | `0.8` | XGBoost column sampling ratio |
| `--xgb_min_child_weight` | `1` | XGBoost min child weight |
| `--xgb_gamma` | `0.0` | XGBoost min split loss |
| `--xgb_reg_alpha` | `0.0` | XGBoost L1 regularization |
| `--xgb_use_gpu` | `False` | Use GPU for XGBoost training (requires xgboost with CUDA support). |
| `--model_arg` | `{}` | Override model configuration (repeatable). Example: `--model_arg depth=6 num_heads=8` |
| `--proc_arg` | `{}` | Override processor configuration (repeatable). Example: `--proc_arg image_size=384` |
| `--optimizer` | `AdamW` | `torch.optim` optimizer class name (case-sensitive). Examples: `AdamW`, `Adam`, `SGD` |
| `--opt_arg` | `{}` | Extra optimizer kwargs (repeatable). Example: `--opt_arg betas=0.9,0.999 momentum=0.9` |
| `--scheduler` | `None` | `torch.optim.lr_scheduler` class name (case-sensitive), or a `.py` script path. Examples: `CosineAnnealingLR`, `StepLR`. Default: no scheduler |
| `--sched_arg` | `{}` | Extra scheduler kwargs (repeatable). Example: `--sched_arg T_max=50 eta_min=1e-6` |

Training automatically resumes from an existing `_latest.pt` or `_best.pt`
checkpoint if one exists at the `--checkpoint` path.

### Cross-Dataset Resume

To fine-tune backbone weights from a previous run on a different dataset
(e.g. switching from `marmal88/skin_cancer` to `ahmed-ai/skin-lesions-classification-dataset`):

```bash
scdiag-train --model facebook/convnextv2-base-22k-224 \
             --dataset ahmed-ai/skin-lesions-classification-dataset \
             --checkpoint scdiag \
             --state_load none \
             --epochs 10 \
             --batch_size 16 \
             --lr 3e-5 \
             --mixup_alpha 0.2 \
             --amp_dtype bfloat16 \
             --remote_checkpoint gs://YOUR_BUCKET/scdiag
```

The checkpoint's backbone weights are loaded (`strict=False`), while the
classifier head (different `num_classes`) is reinitialised and trained from
scratch.  `--state_load none` ensures the old optimizer, scheduler, and
scaler states are not carried over.

## Pre-Training

Pre-train any registered model's encoder on unlabeled dermoscopy images before
fine-tuning. Use `--method` to select the algorithm (default: `simmim`).

```bash
scdiag-pretrain --method simmim \
                --model convvit \
                --datasets HAM10000 "redlessone/Derm1M" \
                --cache_dir ~/.cache/huggingface \
                --hf_token hf_XXXX \
                --image_size 448 \
                --batch_size 32 \
                --epochs 200 \
                --lr 1e-4 \
                --scheduler CosineAnnealingLR \
                --sched_arg T_max=200 --sched_arg eta_min=1e-6 \
                --amp_dtype bfloat16 \
                --checkpoint ./checkpoints/convvit_simmim
```

Then load the pre-trained encoder during supervised fine-tuning:

```bash
scdiag-train --model convvit \
             --dataset marmal88/skin_cancer \
             --source_checkpoint ./checkpoints/convvit_simmim_latest.pt \
             --epochs 100
```

| Argument | Default | Description |
|---|---|---|
| `--method` | `simmim` | Pre-training method. Choices: `simmim`, `ijepa`. Method-specific args follow. |
| `--model` | `convvit` | Model name registered in scdiag (e.g. `convvit`) or HuggingFace model ID. |
| `--datasets` | (required) | Space-separated dataset names or local paths. HuggingFace IDs or directories. |
| `--cache_dir` | `None` | HuggingFace cache directory for dataset and model downloads. |
| `--hf_token` | `None` | HuggingFace token for gated datasets (or set `HF_TOKEN` env var). |
| `--image_column` | auto-detected | Explicit HF image column for pretraining datasets |
| `--strict_datasets` | `False` | Abort instead of skipping a dataset that fails to load |
| `--image_size` | `448` | Input image size (square). |
| `--batch_size` | `32` | Per-GPU batch size. |
| `--epochs` | `200` | Total pre-training epochs. |
| `--lr` | `1e-4` | Peak learning rate for AdamW. |
| `--amp_dtype` | `None` | Mixed precision dtype. Omit to disable; use `float16` or `bfloat16` to enable. |
| `--num_workers` | `4` | DataLoader worker processes. |
| `--resume` | `True` | Auto-resume from latest checkpoint if one exists. Use `--no-resume` to disable. |
| `--state_save` | `opt,sched` | Comma-separated states to save: `opt`, `sched`, `amp`, `none`. |
| `--state_load` | `opt,sched` | Comma-separated states to restore on resume: `opt`, `sched`, `amp`, `none`. |
| `--checkpoint` | (required) | Checkpoint path prefix (saves `_latest.pt` and `_best.pt`). |
| `--log_level` | `INFO` | Minimum logging level. |
| `--grad_monitor` | `-1` | Log gradient statistics every N steps; `-1` disables gradient monitoring. |
| `--grad_clip` | `1.0` | Maximum gradient norm for clipping. |
| `--vis_every` | `0` | Log reconstruction visualisation to TensorBoard every N steps. `0` disables visualisation logging. |
| `--model_arg` | `{}` | Override model configuration (repeatable). Example: `--model_arg depth=6 num_heads=8` |
| `--proc_arg` | `{}` | Override processor configuration (repeatable). |
| `--optimizer` | `AdamW` | `torch.optim` optimizer class name (case-sensitive). Examples: `AdamW`, `Adam`, `SGD` |
| `--opt_arg` | `{}` | Extra optimizer kwargs (repeatable). Example: `--opt_arg betas=0.9,0.999` |
| `--scheduler` | `None` | `torch.optim.lr_scheduler` class name (case-sensitive), or a `.py` script path. Examples: `CosineAnnealingLR`, `StepLR`. Default: no scheduler |
| `--sched_arg` | `{}` | Extra scheduler kwargs (repeatable). Example: `--sched_arg T_max=50` |
| `--source_checkpoint` | `None` | Path to a source checkpoint to absorb parameters from. Useful for continuing from a prior run or loading weights from a different architecture. |
| `--param_rename` | `None` | Regex-based key rename patterns for `--source_checkpoint`. Each pattern is `SEARCH;REPLACE` where SEARCH is a Python regex and REPLACE may use `$1`, `$2`, etc. Applied before shape-based alignment. |

### Dataset Ensemble

`scdiag-pretrain` stitches multiple datasets into a single pre-training corpus:

- **HuggingFace datasets** — any HF dataset ID that returns decoded image
  data (e.g. `HAM10000`).  Gated datasets require `--hf_token` or `HF_TOKEN`
  env var.
- **Local image directories** — pass a path to a folder of images
  (ImageFolder format).

Datasets are loaded lazily; source initialization can occur when `len()` is
first requested. By default, datasets that fail to load are logged and skipped
(best-effort mode), and the run fails if no dataset loads successfully. Use
`--strict_datasets` to abort on the first dataset-loading failure.

Images that cannot be decoded are skipped with a warning. This avoids blocking
pre-training on corrupted files while keeping the dataset pipeline fast.

### Preparing Datasets

Some datasets (like Derm1M) store images inside zip archives and require a
preparation step before they can be used for pre-training:

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

## Inference

```bash
scdiag-infer --model facebook/convnextv2-base-22k-224 \
             --checkpoint scdiag_best.pt \
             path/to/image.jpg path/to/other_image.png
```

| Flag | Default | Description |
|---|---|---|
| `--model` | (required) | HuggingFace model name |
| `--checkpoint` | (required) | Path to a raw state dictionary or wrapped checkpoint |
| `--top_k` | `None` | Show top-K predictions; omitted to return all class probabilities |
| `--output` | `None` | Write JSON results to file |
| `--device` | `None` | Force the PyTorch device (for example, `cuda` or `cpu`). Auto-detected if omitted. |
| `--cache_dir` | `None` | HuggingFace cache directory |
| `--xgboost_model` | `None` | XGBoost model path. If provided, run XGBoost alongside PyTorch. |
| `--model_arg` | `{}` | Override model configuration (repeatable). Example: `--model_arg depth=6` |
| `--proc_arg` | `{}` | Override processor configuration (repeatable). |

Inference accepts either a raw model state dictionary or a wrapped checkpoint
containing `model_state_dict`; wrapped checkpoints are preferred because they
can include model metadata. Raw state dictionaries produce a warning because
metadata is unavailable. Inference uses safe weights-only checkpoint loading.
Resume-training checkpoints contain additional state and must come from a
trusted source because full PyTorch deserialization is required.

### XGBoost Inference

When `--xgboost_model` is provided at inference time, the output includes both
PyTorch and XGBoost predictions:

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

## Custom Models

scdiag supports any HuggingFace `AutoModelForImageClassification` model out of
the box.  You can also register custom model architectures:

```bash
# Use a HuggingFace model
scdiag-train --model facebook/convnextv2-base-22-22k-384 ...

# Use the built-in ConvViT (conv stem + ViT encoder)
scdiag-train --model convvit --image_size 224 ...

# Override ConvViT architecture from the CLI
scdiag-train --model convvit --model_arg depth=6 num_heads=8 dropout=0.2
```

### Adding a custom model

1. Create `scdiag/models/{name}/` with:
   - `model.py` — the `nn.Module` architecture
   - `processor.py` — image processor (must have `__call__(images)` → tensor)
   - `loader.py` — `@register_model("{name}")` decorated loader function
     whose `**kwargs` accept arbitrary overrides (forwarded from `--model_arg`)
   - `__init__.py` — re-export public symbols
2. Add the import to `scdiag/models/__init__.py`
3. The model must satisfy the protocol:
   - `forward(pixel_values=images)` → object with `.logits`
   - `config.id2label` / `config.label2id` accessible
   - `extract_backbone_features(pixel_values)` for XGBoost (optional —
     hook-based fallback works for HF models automatically)
4. CLI overrides via `--model_arg KEY=VALUE` are forwarded as `**kwargs`
   to the registered loader.  The loader is responsible for applying them
   (e.g. by merging into a default config via `setattr`).

### Built-in custom models

| Name | Description | `--model` value |
|---|---|---|
| ConvViT | Multi-block conv stem (4 blocks → patch_size 16) + 12-layer ViT encoder with CLS-guided attention pooling | `convvit` |

ConvViT supports SimMIM self-supervised pre-training via `scdiag-pretrain`.
Pre-trained encoder weights can be loaded into the supervised training pipeline
via `--source_checkpoint`. Use `--param_rename` for manual key rewriting before
automatic shape-based alignment.

## Development

```bash
pip install -e ".[dev]"
pytest
```

Code is formatted with [yapf](https://github.com/google/yapf) using the
project's `.style.yapf` (Google style, 2-space indent).

## License

Apache-2.0
