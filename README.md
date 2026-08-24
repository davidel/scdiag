# scdiag

Train, pre-train, and evaluate image-classification models for skin-lesion
diagnosis and other medical imaging tasks.  Supports HuggingFace and
[timm](https://github.com/huggingface/pytorch-image-models) models, custom
architectures (ConvViT, UVito), and pluggable classifier heads with a
hand-rolled PyTorch training loop.  Includes self-supervised pre-training
(SimMIM, I-JEPA) and supervised contrastive learning (SupCon), XGBoost
ensemble inference, and dataset preparation utilities for common dermoscopy
benchmarks.

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
  (e.g. `CosineAnnealingLR`, `StepLR`) via CLI arguments. Custom optimizer
  and scheduler scripts supported (pass a `.py` path instead of a class name).
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
- **timm integration** — load any model from
  [timm](https://github.com/huggingface/pytorch-image-models) with
  `--model timm:<model_name>`.  Supports all timm model families (ResNet,
  ViT, ConvNeXt, EVA-02, EfficientNet, etc.) with model-specific
  preprocessing resolved automatically.  Install with `pip install scdiag[timm]`.
- **ClsModelWrapper** — wrap any HuggingFace backbone with a custom
  classifier head. Use `--model cls_model_wrapper:<hf_name>` with
  `--classifier` to select a registered classifier or a `.py` script.
- **Parameter freezing** — freeze all backbone parameters except the
  classifier head with `--freeze ".*\\.(head|pool)"`.
- **LoRA fine-tuning** — parameter-efficient fine-tuning via Low-Rank
  Adaptation (PEFT). Freezes the backbone and trains low-rank adapter
  matrices instead, reducing trainable parameters by ~97%. Enable with
  `--lora`. Requires `pip install scdiag[lora]`.
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
  - **`supcon`** — supervised contrastive learning (Khosla et al., NeurIPS
    2020). Learns representations by pulling together features from the
    same class and pushing apart features from different classes. Uses a
    `ContrastiveEncoder` (backbone + projection head) and a
    `BalancedBatchSampler` for controlled class sampling.
  Each method adds its own CLI arguments (e.g. `--mask_ratio` for SimMIM,
  `--teacher_momentum` for I-JEPA, `--proj_dim` and `--temperature` for
  SupCon). New methods can be added by implementing the `PretrainMethod`
  interface in `scdiag/pretrain_methods/`.
- **Model-agnostic** — use `--model` to select any backbone registered in the
  model registry (e.g. `convvit`, or any HuggingFace model ID).
- **Multi-source dataset ensemble** — stitches together multiple HuggingFace
  datasets (e.g. HAM10000, ISIC challenges, Derm1M) into a single unified
  pre-training corpus with flat indexing and lazy loading.
- **Label-aware pre-training** — methods that require labels (e.g. SupCon)
  validate dataset compatibility, remap labels across datasets, and use
  a balanced batch sampler for controlled class distribution.
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

# Fine-tune with a custom MLP classifier on top of a ViT backbone
scdiag-train --model cls_model_wrapper:google/vit-base-patch16-224 \
             --dataset marmal88/skin_cancer \
             --classifier mlp \
             --classifier_args hidden=512 dropout=0.3 \
             --freeze ".*\\.(head|pool)"

# Fine-tune with a custom classifier from a .py file
scdiag-train --model cls_model_wrapper:google/vit-base-patch16-224 \
             --dataset marmal88/skin_cancer \
             --classifier /path/to/my_classifier.py \
             --freeze ".*\\.(head|pool)"
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
| `--lr_group` | `None` | Per-parameter-group learning rates (repeatable). Format: `"REGEX=LR"`. Regexes matched against `named_parameters()`; first match wins. Unmatched trainable params use `--lr`. Example: `"backbone.*=1e-5" "classifier.*=1e-3"` |
| `--llrd_decay` | `None` | Layer-wise learning rate decay factor. When set, learning rates decay by this factor per depth level (shallow layers get lower LR). Depth is inferred from numeric segments in parameter names (e.g. `blocks.0`, `blocks.11`). Example: `--llrd_decay 0.85` |
| `--checkpoint` | `scdiag` | Checkpoint base path (`_latest.pt` / `_best.pt` appended) |
| `--log_every` | `20` | Log every N steps |
| `--grad_monitor` | `-1` | Log gradient statistics every N steps; `-1` disables. See [Gradient Monitor](#gradient-monitor) for column meanings. |
| `--norm_history` | `0` | Keep last N norm snapshots per parameter for trend analysis. Requires `--grad_monitor`. See [Norm trend history](#norm-trend-history). |
| `--trend_top_n` | `10` | Show top N params in trend table by abs change %. `0` = show all. |
| `--grad_clip` | `1.0` | Maximum gradient norm for clipping. `0` disables clipping. |
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
| `--classifier` | `None` | Classifier head spec: a registered name (e.g. `mlp`) or a path to a `.py` file. Only used with `--model cls_model_wrapper:<hf_name>`. |
| `--classifier_args` | `{}` | Extra classifier kwargs (repeatable). Example: `--classifier_args hidden=512 dropout=0.3`. For `cls_attention`: `cls_slice=(0,1) spc_slice=(1,None)`. |
| `--freeze` | `None` | Comma-separated list of regex patterns (`re.match`) for parameter names to keep trainable. All other parameters are frozen. Each pattern is anchored at the start of the name. Example: `".*\\.(head|pool)"`. If omitted, all parameters are trainable. |
| `--lora` | `False` | Enable LoRA (Low-Rank Adaptation) via PEFT. Freezes the backbone and trains low-rank adapter matrices. Requires `pip install scdiag[lora]`. |
| `--lora_r` | `8` | LoRA rank (dimensionality of the low-rank decomposition). |
| `--lora_alpha` | `16` | LoRA alpha (scaling factor = `alpha / r`). Higher values increase the magnitude of the LoRA update. |
| `--lora_dropout` | `0.0` | Dropout probability applied to LoRA layers during training. |
| `--lora_target_modules` | `None` | Comma-separated module names to apply LoRA to (e.g. `"query,key,value"`). Auto-detects attention layers if omitted. |
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

### Gradient Monitor

`--grad_monitor N` logs a per-parameter gradient report every *N* training
steps.  The report helps diagnose training instability (exploding / vanishing
gradients, imbalanced parameter updates) before it shows up in the loss.

#### Summary line

```
[Step 29400] Gradient Report: 202 params | grad_rms: mean=5.68e-01 max=3.41e+00 min=1.66e-05 | grad/param: mean=2.94e-01
```

All norms are **RMS** (root mean square): L2 norm divided by `sqrt(numel)`.
This makes them independent of tensor shape and directly comparable across
parameters of different sizes. A freshly initialized `nn.Linear(1024, 1024)`
with `std=0.02` would show `p_rms ≈ 0.02`.

| Field | Meaning |
|---|---|
| `params` | Total number of trainable parameters (those with `requires_grad=True`). |
| `grad_rms: mean/max/min` | RMS of the gradient tensor for each parameter, then aggregated across all trainable params. `max` is the single most aggressive gradient — the one most likely to cause instability. |
| `grad/param: mean` | Average of the gradient-to-parameter ratio (see *g/p* below). Healthy models typically show mean g/p < 0.1. Values > 1.0 mean updates are larger than the weights themselves. |

#### Per-parameter columns

Each row reports one trainable parameter (or parameter tensor). Columns:

| Column | Symbol | How it's computed | What to look for |
|---|---|---|---|
| **g_rms** | `‖∇L‖/√N` | `torch.norm(grad) / sqrt(numel)` — RMS of the gradient tensor. | Compare across params. One param with g_rms 100× higher than others is a problem. |
| **p_rms** | `‖W‖/√N` | `torch.norm(param) / sqrt(numel)` — RMS of the parameter tensor. | Gives per-element scale context. With `trunc_normal_(std=0.02)`, expect `p_rms ≈ 0.02`. |
| **g/p** | `‖∇L‖ / (‖W‖ + ε)` | g_rms divided by p_rms (with ε = 10⁻¹² to avoid division by zero). | The single most useful column. Tells you the **relative size of the update**. Healthy: < 0.1. Concerning: > 1.0 (update overshoots the parameter). Dangerous: > 5.0 (parameter is being thrown around randomly). |
| **g_max** | `max\|∇L\|` | `grad.abs().max()` — largest absolute gradient value anywhere in the tensor. | Highlights individual neurons that are getting extreme gradients even when the overall norm looks reasonable. |
| **sparse** | `% zero` | Fraction of gradient elements with `\|grad\| < 1e-7`. | High sparsity (> 50%) means most neurons in this layer aren't receiving gradient signal. Can indicate dead neurons or a learning rate that's too low. |
| **status** | | Anomaly detection flag (see below). | `OK` = healthy. Anything else deserves investigation. |

#### Status flags

| Flag | Full name | Condition |
|---|---|---|
| `OK` | Normal | No anomaly detected. |
| `STL` | Stalled | g_rms has been below `norm_floor` (1e-7) for `stall_window` consecutive log steps (default: 50) — the parameter has stopped learning. |
| `OVF` | Exploding | g_rms exceeds `norm_ceiling` (1.0) — the gradient is dangerously large. |
| `IMB` | Imbalanced | g_rms exceeds 100× the median gradient RMS across all params — this parameter is being updated much more aggressively than others. |
| `GPR` | G/P Ratio | g/p (gradient-to-parameter ratio) exceeds `gpr_ceiling` (default: 1.0) — the update step is larger than the weight itself, meaning the parameter is being pushed further than its current scale in a single step. |

The thresholds (`norm_floor`, `norm_ceiling`, `stall_window`, `imbalance_factor`,
`gpr_ceiling`) are configurable on the `GradMonitor` constructor but not
exposed via CLI flags.

#### Norm trend history

`--norm_history N` (requires `--grad_monitor`) keeps the last N norm snapshots
per parameter. A trend summary table is appended to each report:

```
  Norm Trends (last 10 snapshots, 82 params):
  Param      g_dir   g_chg%     g_min     g_max p_dir   p_chg%     p_min     p_max
  -----------------------------------------------------------------------------------
  fc1.weight    ---     -2.3%  5.37e-01  5.50e-01   ---    -0.0%  2.41e+00  2.41e+00
```

| Column | Meaning |
|---|---|
| `g_dir` / `p_dir` | `UP` (>+10% change), `DOWN` (<-10%), or `---` (stable). |
| `g_chg%` / `p_chg%` | Percentage change from first to last snapshot in the window. |
| `g_min` / `g_max` | Min/max g_rms observed in the window. |
| `p_min` / `p_max` | Min/max p_rms observed in the window. |

#### Reading the report

**Healthy training** looks like:
- g/p ratios well below 1.0 for all params
- `grad/param: mean` in the 0.01–0.1 range
- g_rms values within ~10× of each other across layers
- No `OVF` or `IMB` flags

**Overfitting** (high train accuracy, low val accuracy):
- g/p may still look healthy — the model is learning, just not generalising.
- Look at the loss curves instead; grad monitor won't directly show overfitting.

**Exploding gradients** (loss spikes, NaN):
- One or more params with `OVF` status
- g/p > 5.0 on some layers
- `grad_rms: max` orders of magnitude larger than `mean`
- Fix: lower LR, add gradient clipping (`--grad_clip 1.0`), or use a warmup schedule.

**Vanishing gradients** (loss flat, no learning):
- Many params with `STL` status
- g_rms values near 1e-7 or below
- High sparsity (> 50%)
- Fix: increase LR, check for dead neurons, verify the loss is connected to the parameters being monitored.

**Imbalanced updates** (oscillating validation metrics):
- Some params flagged `IMB` while others are `OK`
- Large disparity in g/p between classifier and backbone layers
- Fix: use `--lr_group` to assign different learning rates to different param groups, or freeze the dominant component.

### LoRA Fine-Tuning

Low-Rank Adaptation (LoRA) freezes the pre-trained backbone and injects
small trainable low-rank matrices into the attention layers.  This reduces
trainable parameters by ~97 % while often matching full fine-tuning
accuracy.

```bash
scdiag-train \
    --model cls_model_wrapper:facebook/dinov2-with-registers-large \
    --lora \
    --lora_r 16 \
    --lora_alpha 32 \
    --lora_target_modules "query,key,value" \
    --freeze "classifier\.(head|pool|encoder)" \
    --lr 3e-5 \
    ...
```

#### Key parameters

| Flag | Default | Notes |
|---|---|---|
| `--lora` | off | Enables LoRA. Requires `pip install scdiag[lora]`. |
| `--lora_r` | `8` | Rank of the low-rank decomposition. Higher = more capacity, more VRAM. |
| `--lora_alpha` | `16` | Controls update magnitude via the scaling factor `alpha / r`. |
| `--lora_dropout` | `0.0` | Dropout on LoRA layers (useful for small datasets). |
| `--lora_target_modules` | auto | Comma-separated names. Auto-detects attention Q/K/V if omitted. |

#### Scaling factor (`alpha / r`)

The LoRA output is `ΔW = (alpha/r) × B @ A`, where `A` and `B` are the
low-rank matrices.  The ratio `alpha/r` scales the update relative to the
base weights.  Common settings:

| r | alpha | alpha/r | Use case |
|---|---|---|---|
| 8 | 16 | 2.0 | Conservative, very few params |
| 16 | 32 | 2.0 | Good default for medium datasets |
| 16 | 64 | 4.0 | Larger updates (helpful under bfloat16) |
| 32 | 64 | 2.0 | More capacity, same scaling |

If your LoRA params barely appear in the gradient monitor, try increasing
`--lora_alpha` before changing `--lora_r`.

#### Combining LoRA with a custom classifier

Use `--lora` together with `--model cls_model_wrapper:<hf_name>` and
`--classifier` to pair LoRA backbone adaptation with a custom classifier
head.  Use `--freeze` to keep the classifier encoder trainable while LoRA
handles the backbone:

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

Pre-train any registered model's encoder before fine-tuning. Use `--method` to
select the algorithm (default: `simmim`). Methods that need labels (e.g.
SupCon) automatically detect and validate label support across dataset ensembles.

```bash
# Masked image modelling (SimMIM)
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

# Supervised contrastive learning (SupCon)
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
```

Then load the pre-trained encoder during supervised fine-tuning:

```bash
scdiag-train --model convvit \
             --dataset marmal88/skin_cancer \
             --source_checkpoint ./checkpoints/convvit_simmim_latest.pt \
             --epochs 100
```

### Pre-Training CLI Arguments

| Argument | Default | Description |
|---|---|---|
| `--method` | `simmim` | Pre-training method. Choices: `simmim`, `ijepa`, `supcon`. Method-specific args follow. |
| `--model` | `convvit` | Model name registered in scdiag (e.g. `convvit`) or HuggingFace model ID. |
| `--datasets` | (required) | Space-separated dataset names or local paths. HuggingFace IDs or directories. |
| `--cache_dir` | `None` | HuggingFace cache directory for dataset and model downloads. |
| `--hf_token` | `None` | HuggingFace token for gated datasets (or set `HF_TOKEN` env var). |
| `--image_column` | auto-detected | Explicit HF image column for pretraining datasets. |
| `--label_column` | `None` | Explicit HF label column for pretraining datasets. Required by `--method supcon` if the dataset uses non-standard label column names. |
| `--strict_datasets` | `False` | Abort instead of skipping a dataset that fails to load. |
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
| `--grad_monitor` | `-1` | Log gradient statistics every N steps; `-1` disables. See [Gradient Monitor](#gradient-monitor) for column meanings. |
| `--norm_history` | `0` | Keep last N norm snapshots per parameter for trend analysis. Requires `--grad_monitor`. See [Norm trend history](#norm-trend-history). |
| `--trend_top_n` | `10` | Show top N params in trend table by abs change %. `0` = show all. |
| `--grad_clip` | `1.0` | Maximum gradient norm for clipping. `0` disables clipping. |
| `--lr_group` | `None` | Per-parameter-group learning rates (repeatable). Format: `"REGEX=LR"`. Regexes matched against `named_parameters()`; first match wins. Unmatched trainable params use `--lr`. Example: `"backbone.*=1e-5" "classifier.*=1e-3"` |
| `--llrd_decay` | `None` | Layer-wise learning rate decay factor. When set, learning rates decay by this factor per depth level (shallow layers get lower LR). Depth is inferred from numeric segments in parameter names (e.g. `blocks.0`, `blocks.11`). Example: `--llrd_decay 0.85` |
| `--vis_every` | `0` | Log reconstruction visualisation to TensorBoard every N steps. `0` disables visualisation logging. Only applicable to reconstruction-based methods (SimMIM). |
| `--model_arg` | `{}` | Override model configuration (repeatable). Example: `--model_arg depth=6 num_heads=8` |
| `--proc_arg` | `{}` | Override processor configuration (repeatable). |
| `--optimizer` | `AdamW` | `torch.optim` optimizer class name (case-sensitive), or a `.py` script path. Examples: `AdamW`, `Adam`, `SGD` |
| `--opt_arg` | `{}` | Extra optimizer kwargs (repeatable). Example: `--opt_arg betas=0.9,0.999` |
| `--scheduler` | `None` | `torch.optim.lr_scheduler` class name (case-sensitive), or a `.py` script path. Examples: `CosineAnnealingLR`, `StepLR`. Default: no scheduler |
| `--sched_arg` | `{}` | Extra scheduler kwargs (repeatable). Example: `--sched_arg T_max=50` |
| `--source_checkpoint` | `None` | Path to a source checkpoint to absorb parameters from. Useful for continuing from a prior run or loading weights from a different architecture. |
| `--param_rename` | `None` | Regex-based key rename patterns for `--source_checkpoint`. Each pattern is `SEARCH;REPLACE` where SEARCH is a Python regex and REPLACE may use `$1`, `$2`, etc. Applied before shape-based alignment. |

### SupCon-Specific Arguments

These arguments are only used by `--method supcon`:

| Argument | Default | Description |
|---|---|---|
| `--proj_dim` | `128` | Output dimensionality of the projection head. |
| `--proj_hidden` | `None` | Hidden layer size of the projection MLP. `None` uses a single linear layer (no hidden). |
| `--temperature` | `0.07` | NT-Xent temperature. Lower values sharpen the contrastive distribution. |
| `--samples_per_class` | `16` | Number of samples to draw per class in each batch. Batch size should be divisible by this value. |

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

#### Label Validation

When using `--method supcon` (or any future label-aware method), the ensemble
validates that every dataset supports labels. Datasets without a label column
cause an error before training begins, so you get a clear message instead of a
runtime failure mid-epoch. Labels are remapped to a shared global label space
across all datasets, so mixing HAM10000 (with its label column) and a
different dataset with overlapping but differently-named classes works
transparently.

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
the box.  It also supports models from
[timm](https://github.com/huggingface/pytorch-image-models) via the `timm:`
prefix, and you can register custom model architectures:

```bash
# Use a HuggingFace model
scdiag-train --model facebook/convnextv2-base-22-22k-384 ...

# Use a timm model (pretrained on ImageNet-21k via MIM)
scdiag-train --model timm:hf_hub:timm/eva02_base_patch14_224.mim_in22k \
    --num_labels 8 --image_size 224 ...

# Use a timm model with native weights (no hf_hub: prefix needed)
scdiag-train --model timm:resnet18 --image_size 224 ...

# Use the built-in ConvViT (conv stem + ViT encoder)
scdiag-train --model convvit --image_size 224 ...

# Override ConvViT architecture from the CLI
scdiag-train --model convvit --model_arg depth=6 num_heads=8 dropout=0.2
```

### Adding a custom model

1. Create `scdiag/models/{name}/` with:
   - `model.py` — the `nn.Module` architecture
   - `processor.py` — image processor (must expose `.image_mean`, `.image_std` properties; registered via `@register_processor`)
   - `loader.py` — `@register_model("{name}")` decorated loader function
     whose `**kwargs` accept arbitrary overrides (forwarded from `--model_arg`)
   - `__init__.py` — re-export public symbols
2. Add the import to `scdiag/models/__init__.py`
3. The model must satisfy the protocol:
   - `forward(pixel_values=images)` → object with `.logits`
   - `config.id2label` / `config.label2id` accessible
   - `extract_backbone_features(pixel_values)` for XGBoost feature
     extraction (ClsModelWrapper implements this natively)
4. CLI overrides via `--model_arg KEY=VALUE` are forwarded as `**kwargs`
   to the registered loader.  The loader is responsible for applying them
   (e.g. by merging into a default config via `setattr`).

### Built-in custom models

| Name | Description | `--model` value |
|---|---|---|
| timm | Any model from the [timm](https://github.com/huggingface/pytorch-image-models) library (ResNet, ViT, ConvNeXt, EVA-02, EfficientNet, …) | `timm:<model_name>` |
| ConvViT | Multi-block conv stem (4 blocks → patch_size 16) + 12-layer ViT encoder with CLS-guided attention pooling | `convvit` |
| UVito | Frozen SMP encoder (e.g. ResNet50) + learnable patch projection + Transformer encoder with CLS tokens | `uvito` |
| ClsModelWrapper | HuggingFace backbone + custom classifier head | `cls_model_wrapper:<hf_name>` |
| ContrastiveEncoder | Backbone + projection head for contrastive pre-training (used automatically by `--method supcon`) | `contrastive_encoder:<hf_name>` |

### Custom Classifiers

`ClsModelWrapper` lets you replace the default HF classification head with a
custom classifier. The backbone lives on the wrapper; the classifier receives
plain `(B, N, D)` hidden-state tensors and does **not** own the backbone.

Use `--model cls_model_wrapper:<hf_name>` with `--classifier`:

```bash
# Use the built-in MLP classifier (freeze backbone, train head only)
scdiag-train --model cls_model_wrapper:google/vit-base-patch16-224 \
             --classifier mlp \
             --classifier_args hidden=512 dropout=0.3 \
             --freeze "classifier\.(head|pool)"

# Use the CLS-guided attention pooling classifier
scdiag-train --model cls_model_wrapper:facebook/dinov2-with-registers-large \
             --classifier cls_attention \
             --classifier_args cls_slice=0,1 spc_slice=5,None \
             --freeze "classifier\.(head|pool)"
```

A custom classifier `.py` file must define a `Classifier` class that receives
`num_labels`, `hidden_size`, and any `**kwargs` from `--classifier_args`:

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

The classifier's `extract_features` method returns the feature vector fed to
the head — used by `--xgboost_model` for XGBoost training on backbone features.

ConvViT supports SimMIM self-supervised pre-training via `scdiag-pretrain`.
Pre-trained encoder weights can be loaded into the supervised training pipeline
via `--source_checkpoint`. Use `--param_rename` for manual key rewriting before
automatic shape-based alignment.

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
