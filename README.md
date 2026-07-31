# scdiag

Fine-tune HuggingFace image-classification models for skin-lesion classification
(and other image datasets) with a simple hand-rolled PyTorch training loop.

## Features

- **Any HuggingFace image classification dataset** — auto-detects image and
  label columns, handles filepath-based images, ClassLabel casting, and
  pre-split datasets.
- **Mixed precision** — optional AMP with `float16` (with GradScaler) or
  `bfloat16`.
- **Gradient accumulation** — effective batch size = `batch_size * grad_accum_steps`.
- **Linear warmup + cosine annealing** learning rate schedule.
- **Class-weighted loss** with inverse-frequency weighting and label smoothing.
- **Checkpointing** — saves `_latest.pt` and `_best.pt` (by validation accuracy).
- **Resume** — auto-resume from latest checkpoint with `--resume`.
- **TensorBoard** logging.
- **GCS sync** — optional checkpoint upload to Google Cloud Storage.

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
| `--model` | `google/vit-base-patch16-224` | HuggingFace model name or local path |
| `--dataset` | `marmal88/skin_cancer` | HuggingFace dataset name |
| `--image_size` | `448` | Resize images to this size |
| `--epochs` | `5` | Number of training epochs |
| `--batch_size` | `32` | Batch size per GPU |
| `--lr` | `3e-5` | Peak learning rate |
| `--weight_decay` | `0.01` | Weight decay |
| `--warmup_epochs` | `2` | Linear warmup epochs |
| `--label_smoothing` | `0.1` | Label smoothing factor |
| `--grad_accum_steps` | `1` | Gradient accumulation steps |
| `--amp_dtype` | `None` | Mixed precision: `float16` or `bfloat16` |
| `--resume` | `False` | Auto-resume from checkpoint |
| `--checkpoint` | `scdiag` | Checkpoint base path |
| `--log_every` | `20` | Log every N steps |
| `--save_every` | `500` | Save checkpoint every N steps |
| `--num_workers` | `2` | DataLoader worker processes |
| `--log_level` | `INFO` | Logging level |
| `--log_dir` | `None` | TensorBoard log directory |
| `--cache_dir` | `None` | HuggingFace cache directory |
| `--gcs_checkpoint` | `None` | GCS URI for checkpoint sync |
| `--num_labels` | `None` | Override number of labels |
| `--ignore_optimizer_ckpt` | `False` | Skip restoring optimizer state |
| `--ignore_scheduler_ckpt` | `False` | Skip restoring scheduler state |

## Development

```bash
pip install -e ".[dev]"
pytest
```

Code is formatted with [yapf](https://github.com/google/yapf) using the
project's `.style.yapf` (Google style, 2-space indent).

## License

Apache-2.0
