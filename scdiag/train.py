"""Fine-tune a HuggingFace image-classification model."""

import argparse
import logging
import os
import time

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import datasets
from datasets import load_dataset
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter
from torchvision.transforms import v2
from transformers import AutoImageProcessor, AutoModelForImageClassification

from scdiag.gpu_utils import gpu_stats_str
from scdiag.logging_utils import setup_logging
from scdiag.gcs_utils import save_checkpoint, checkpoint_dict

from scdiag.hf_proxy import HFDatasetProxy


def build_transforms(processor, image_size):
  """Create train / val augmentation pipelines.

    Stochastic augmentations only.  The processor handles deterministic
    resize-to-model-size and normalization.
    """
  train_augmentations = v2.Compose([
      v2.RandomResizedCrop(size=(image_size, image_size),
                           scale=(0.85, 1.0),
                           antialias=True),
      v2.RandomHorizontalFlip(p=0.5),
      v2.RandomVerticalFlip(p=0.5),
      v2.ColorJitter(brightness=0.15, contrast=0.15, saturation=0.1),
      v2.ToImage(),
      v2.ToDtype(torch.float32, scale=True),
  ])

  val_augmentations = v2.Compose([
      v2.ToImage(),
      v2.ToDtype(torch.float32, scale=True),
  ])

  return train_augmentations, val_augmentations


def load_and_split_dataset(dataset_name, cache_dir=None, test_size=0.2, seed=42):
  """Load a HuggingFace dataset, return ``(train_proxy, val_proxy)``."""
  raw = load_dataset(dataset_name, cache_dir=cache_dir)

  # Single split: validate, split and wrap.
  if isinstance(raw, datasets.Dataset):
    if HFDatasetProxy.detect_image_column(raw) is None:
      raise ValueError(f"No image column detected in {dataset_name}. "
                       f"Columns: {list(raw.features.keys())}")
    split = raw.train_test_split(test_size=test_size, seed=seed)
    return (HFDatasetProxy(split["train"]), HFDatasetProxy(split["test"]))

  # DatasetDict: validate and ensure train/test splits exist.
  for split_name in raw:
    if HFDatasetProxy.detect_image_column(raw[split_name]) is None:
      raise ValueError(f"No image column in split '{split_name}' of {dataset_name}. "
                       f"Columns: {list(raw[split_name].features.keys())}")

  splits = set(raw.keys())
  if "train" not in splits or "test" not in splits:
    if "train" in splits:
      split = raw["train"].train_test_split(test_size=test_size, seed=seed)
      raw = datasets.DatasetDict(split)
    elif len(splits) == 1:
      only = list(splits)[0]
      split = raw[only].train_test_split(test_size=test_size, seed=seed)
      raw = datasets.DatasetDict(split)
    else:
      names = list(raw.keys())
      raw = datasets.DatasetDict({"train": raw[names[0]], "test": raw[names[1]]})

  return HFDatasetProxy(raw["train"]), HFDatasetProxy(raw["test"])


def compute_class_weights(train_dataset, num_labels):
  """Compute inverse-frequency class weights as a CPU tensor.

    *train_dataset* is a raw HF ``Dataset`` (before ``set_transform`` is
    applied) so that column access does not trigger any registered
    transforms.
    """
  feat = train_dataset.features["label"]
  raw_labels = train_dataset["label"]
  if isinstance(feat, datasets.ClassLabel):
    labels = np.array(
        feat.str2int(raw_labels) if isinstance(raw_labels[0], str) else raw_labels,
        dtype=np.int64,
    )
  else:
    labels = np.array(raw_labels, dtype=np.int64)

  actual_labels = np.unique(labels)
  if len(actual_labels) > num_labels:
    raise ValueError(
        f"Dataset has {len(actual_labels)} unique labels but num_labels={num_labels}")
  counts = np.bincount(labels, minlength=num_labels).astype(np.float64)
  counts = np.maximum(counts, 1.0)
  weights = 1.0 / counts
  weights = weights / weights.sum() * num_labels
  return torch.tensor(weights, dtype=torch.float32)


def parse_args(argv=None):
  parser = argparse.ArgumentParser(
      description="Fine-tune a HuggingFace image-classification model.")

  parser.add_argument(
      "--model",
      type=str,
      default="google/vit-base-patch16-224",
      help="HuggingFace model name or path (default: "
      "%(default)s)",
  )
  parser.add_argument(
      "--dataset",
      type=str,
      default="marmal88/skin_cancer",
      help="HuggingFace dataset name (default: %(default)s)",
  )
  parser.add_argument(
      "--image_size",
      type=int,
      default=448,
      help="Resize images to this size (default: %(default)s)",
  )
  parser.add_argument(
      "--num_labels",
      type=int,
      default=None,
      help="Override number of labels (auto-detected by "
      "default)",
  )

  parser.add_argument(
      "--epochs",
      type=int,
      default=5,
      help="Number of training epochs (default: %(default)s)",
  )
  parser.add_argument("--batch_size",
                      type=int,
                      default=32,
                      help="Batch size (default: %(default)s)")
  parser.add_argument(
      "--lr",
      type=float,
      default=3e-5,
      help="Peak learning rate (default: %(default)s)",
  )
  parser.add_argument(
      "--weight_decay",
      type=float,
      default=0.01,
      help="Weight decay (default: %(default)s)",
  )
  parser.add_argument(
      "--warmup_epochs",
      type=int,
      default=2,
      help="Linear warmup epochs (default: %(default)s)",
  )
  parser.add_argument(
      "--label_smoothing",
      type=float,
      default=0.1,
      help="Label smoothing (default: %(default)s)",
  )
  parser.add_argument(
      "--grad_accum_steps",
      type=int,
      default=1,
      help="Gradient accumulation steps (default: %(default)s)",
  )
  parser.add_argument(
      "--amp_dtype",
      type=str,
      default=None,
      choices=["float16", "bfloat16"],
      help="AMP dtype for mixed precision (default: None = "
      "disabled). float16 requires GradScaler; bfloat16 "
      "is recommended for Ampere+ GPUs.",
  )

  parser.add_argument(
      "--checkpoint",
      type=str,
      default="scdiag",
      help="Base path for checkpoints. '_latest.pt' and "
      "'_best.pt' are appended automatically "
      "(default: %(default)s)",
  )

  parser.add_argument(
      "--log_dir",
      type=str,
      default=None,
      help="TensorBoard log directory (default: "
      "<dir_of_latest_ckpt>/logs)",
  )
  parser.add_argument(
      "--log_every",
      type=int,
      default=20,
      help="Log every N steps (default: %(default)s)",
  )
  parser.add_argument(
      "--save_every",
      type=int,
      default=500,
      help="Save checkpoint every N steps "
      "(default: %(default)s)",
  )
  parser.add_argument(
      "--num_workers",
      type=int,
      default=2,
      help="DataLoader worker processes (default: %(default)s)",
  )
  parser.add_argument(
      "--log_level",
      type=str,
      default="INFO",
      choices=["DEBUG", "INFO", "WARNING", "ERROR"],
      help="Minimum logging level (default: %(default)s)",
  )

  parser.add_argument(
      "--ignore_optimizer_ckpt",
      action=argparse.BooleanOptionalAction,
      default=False,
      help="Skip restoring the optimizer state from checkpoint",
  )
  parser.add_argument(
      "--ignore_scheduler_ckpt",
      action=argparse.BooleanOptionalAction,
      default=False,
      help="Skip restoring the scheduler from checkpoint",
  )

  parser.add_argument(
      "--cache_dir",
      type=str,
      default=None,
      help="Cache directory for downloaded datasets",
  )
  parser.add_argument(
      "--gcs_checkpoint",
      type=str,
      default=None,
      help="GCS URI to sync checkpoints to "
      "(format: gs://BUCKET/PREFIX)",
  )

  return parser.parse_args(argv)


def train_one_epoch(
    model,
    dataloader,
    criterion,
    optimizer,
    scaler,
    scheduler,
    device,
    amp_dtype,
    epoch,
    best_top1,
    args,
    writer=None,
):
  """Train for one epoch.

    If ``args.grad_accum_steps > 1``, gradients are accumulated over that many
    micro-batches before stepping the optimizer.
    """
  model.train()
  total_loss, correct_top1, total_samples = 0.0, 0, 0
  total_batches = len(dataloader)
  start_time = time.time()
  last_log_time = time.time()
  window_samples = 0

  for batch_idx, (images, targets) in enumerate(dataloader):
    images, targets = images.to(device), targets.to(device)

    with torch.amp.autocast(
        "cuda",
        dtype=amp_dtype,
        enabled=(amp_dtype is not None and device.type == "cuda"),
    ):
      outputs = model(pixel_values=images)
      logits = outputs.logits
      loss = criterion(logits, targets) / args.grad_accum_steps

    if amp_dtype == torch.float16 and scaler is not None:
      scaler.scale(loss).backward()
    else:
      loss.backward()

    # Step optimizer only every grad_accum_steps batches (or at end of epoch).
    if (batch_idx + 1) % args.grad_accum_steps == 0 or (batch_idx + 1) == total_batches:
      if amp_dtype == torch.float16 and scaler is not None:
        scaler.unscale_(optimizer)
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        scaler.step(optimizer)
        scaler.update()
      else:
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        optimizer.step()
      optimizer.zero_grad(set_to_none=True)

    with torch.no_grad():
      correct_top1 += (logits.argmax(dim=1) == targets).sum().item()

    batch_size = targets.size(0)
    total_loss += loss.item() * batch_size * args.grad_accum_steps
    total_samples += batch_size
    window_samples += batch_size

    # Periodic step-level logging.
    if (batch_idx + 1) % args.log_every == 0 or (batch_idx + 1) == total_batches:
      global_step = epoch * total_batches + batch_idx
      elapsed = time.time() - last_log_time
      throughput = window_samples / elapsed if elapsed > 0 else 0
      lr_now = optimizer.param_groups[0]["lr"]
      avg_loss = total_loss / total_samples
      top1 = (correct_top1 / total_samples) * 100.0
      msg = (f"  [Step {batch_idx + 1}/{total_batches}] "
             f"loss={avg_loss:.4f} top1={top1:.2f}% "
             f"lr={lr_now:.2e} {throughput:.0f} img/s"
             f"{gpu_stats_str(device)}")
      logging.info(msg)
      if writer is not None:
        writer.add_scalar("Train/loss", avg_loss, global_step)
        writer.add_scalar("Train/top1", top1, global_step)
        writer.add_scalar("Train/lr", lr_now, global_step)
        writer.add_scalar("Train/throughput", throughput, global_step)
        if device.type == "cuda":
          writer.add_scalar(
              "GPU/memory_MB",
              torch.cuda.memory_allocated(device) / 1024**2,
              global_step,
          )
          if hasattr(torch.cuda, "utilization"):
            writer.add_scalar(
                "GPU/utilization_pct",
                torch.cuda.utilization(device),
                global_step,
            )
      last_log_time = time.time()
      window_samples = 0

  avg_loss = total_loss / total_samples
  top1 = (correct_top1 / total_samples) * 100.0
  elapsed = time.time() - start_time
  logging.info(f"  Train stats -> loss: {avg_loss:.4f} | top1: {top1:.2f}%"
               f" | time: {elapsed:.1f}s")
  return avg_loss, top1


def evaluate_performance(model, dataloader, criterion, device, amp_dtype):
  """Evaluate on a validation/test set.

    Returns ``(eval_loss, top1_acc_pct)``.
    """
  model.eval()
  eval_loss, correct_top1, total_samples = 0.0, 0, 0
  with torch.no_grad():
    for images, targets in dataloader:
      images, targets = images.to(device), targets.to(device)
      with torch.amp.autocast(
          "cuda",
          dtype=amp_dtype,
          enabled=(amp_dtype is not None and device.type == "cuda"),
      ):
        outputs = model(pixel_values=images)
        logits = outputs.logits
        loss = criterion(logits, targets)

      eval_loss += loss.item() * images.size(0)
      total_samples += targets.size(0)
      correct_top1 += (logits.argmax(dim=1) == targets).sum().item()

  return (eval_loss / total_samples, (correct_top1 / total_samples) * 100.0)


def main():
  args = parse_args()

  # Convert string amp_dtype to torch.dtype.
  _dtype_map = {"float16": torch.float16, "bfloat16": torch.bfloat16}
  args.amp_dtype = _dtype_map.get(args.amp_dtype, args.amp_dtype)

  setup_logging(args.log_level)

  device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
  logging.info(f"Using device: {device}")

  log_dir = args.log_dir or os.path.join(
      os.path.dirname(args.checkpoint) or ".", "logs")
  os.makedirs(log_dir, exist_ok=True)
  writer = SummaryWriter(log_dir=log_dir)

  processor = AutoImageProcessor.from_pretrained(args.model, cache_dir=args.cache_dir)
  train_transforms, val_transforms = build_transforms(processor, args.image_size)

  train_proxy, val_proxy = load_and_split_dataset(args.dataset,
                                                  cache_dir=args.cache_dir)

  num_labels = args.num_labels or train_proxy.num_labels
  logging.info(f"num_labels: {num_labels}")

  class_weights = compute_class_weights(train_proxy.dataset, num_labels).to(device)
  logging.info(f"Class weights: {class_weights.tolist()}")

  processor_size = {"height": args.image_size, "width": args.image_size}
  train_proxy.transform = train_transforms
  train_proxy.processor = processor
  train_proxy.processor_size = processor_size
  val_proxy.processor = processor
  val_proxy.processor_size = processor_size

  if len(train_proxy) < args.batch_size:
    raise ValueError(f"Training set ({len(train_proxy)} samples) is smaller than "
                     f"batch_size ({args.batch_size}). Reduce --batch_size.")
  train_loader = DataLoader(
      train_proxy,
      batch_size=args.batch_size,
      shuffle=True,
      num_workers=args.num_workers,
      pin_memory=(device.type == "cuda"),
      drop_last=True,
  )
  val_loader = DataLoader(
      val_proxy,
      batch_size=args.batch_size,
      shuffle=False,
      num_workers=args.num_workers,
      pin_memory=(device.type == "cuda"),
  )

  model = AutoModelForImageClassification.from_pretrained(
      args.model,
      num_labels=num_labels,
      id2label=train_proxy.id2label,
      label2id=train_proxy.label2id,
      ignore_mismatched_sizes=True,
      cache_dir=args.cache_dir,
  )
  model.to(device)

  total_params = sum(p.numel() for p in model.parameters())
  trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
  logging.info(f"Model params: {total_params:,} total, {trainable:,} trainable")
  logging.info(f"Model structure:\n{model}")

  criterion = nn.CrossEntropyLoss(weight=class_weights,
                                  label_smoothing=args.label_smoothing)

  optimizer = optim.AdamW(model.parameters(),
                          lr=args.lr,
                          weight_decay=args.weight_decay)

  scaler = (torch.amp.GradScaler("cuda")
            if args.amp_dtype == torch.float16 and device.type == "cuda" else None)

  if args.warmup_epochs > 0:
    scheduler_warmup = optim.lr_scheduler.LinearLR(optimizer,
                                                   start_factor=0.01,
                                                   total_iters=args.warmup_epochs)
    scheduler_cosine = optim.lr_scheduler.CosineAnnealingLR(optimizer,
                                                            T_max=args.epochs -
                                                            args.warmup_epochs)
    scheduler = optim.lr_scheduler.SequentialLR(
        optimizer,
        [scheduler_warmup, scheduler_cosine],
        milestones=[args.warmup_epochs],
    )
  else:
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.epochs)

  ckpt_latest = args.checkpoint + "_latest.pt"
  ckpt_best = args.checkpoint + "_best.pt"
  start_epoch = 0
  best_top1 = 0.0

  resume_path = None
  if os.path.exists(ckpt_latest):
    resume_path = ckpt_latest
  elif os.path.exists(ckpt_best):
    resume_path = ckpt_best

  if resume_path:
    logging.info(f"Resuming from checkpoint: {resume_path}")
    ckpt = torch.load(resume_path, map_location=device, weights_only=False)
    ckpt_keys = list(ckpt.keys())
    logging.info(f"  Checkpoint keys: {ckpt_keys}")
    model.load_state_dict(ckpt["model_state_dict"])
    logging.info("  Restored model weights")
    if not args.ignore_optimizer_ckpt and "optimizer_state_dict" in ckpt:
      optimizer.load_state_dict(ckpt["optimizer_state_dict"])
      logging.info("  Restored optimizer state")
    else:
      logging.info("  Skipped optimizer state")
    if not args.ignore_scheduler_ckpt and "scheduler_state_dict" in ckpt:
      scheduler.load_state_dict(ckpt["scheduler_state_dict"])
      logging.info("  Restored scheduler state")
    else:
      logging.info("  Skipped scheduler state")
    if scaler is not None and "scaler_state_dict" in ckpt:
      scaler.load_state_dict(ckpt["scaler_state_dict"])
      logging.info("  Restored GradScaler state")
    start_epoch = ckpt.get("epoch", -1) + 1
    best_top1 = ckpt.get("best_top1", 0.0)
    logging.info(f"  Resumed at epoch {start_epoch}, best_top1={best_top1:.2f}%")

  completed_epoch = start_epoch - 1  # last fully completed (-1 = none yet)
  try:
    for epoch in range(start_epoch, args.epochs):
      effective_batch = args.batch_size * args.grad_accum_steps
      logging.info(f"=== Epoch {epoch + 1}/{args.epochs} "
                   f"(eff_batch={effective_batch}) ===")

      train_loss, train_t1 = train_one_epoch(
          model,
          train_loader,
          criterion,
          optimizer,
          scaler,
          scheduler,
          device,
          args.amp_dtype,
          epoch,
          best_top1,
          args,
          writer=writer,
      )

      scheduler.step()
      writer.add_scalar("Epoch/Loss_Train", train_loss, epoch)
      writer.add_scalar("Epoch/Accuracy_Train_Top1", train_t1, epoch)

      v_loss, v_t1 = evaluate_performance(model, val_loader, criterion, device,
                                          args.amp_dtype)
      writer.add_scalar("Epoch/Loss_Val", v_loss, epoch)
      writer.add_scalar("Epoch/Accuracy_Val_Top1", v_t1, epoch)
      logging.info(f"Epoch {epoch + 1} Results -> "
                   f"Val Loss: {v_loss:.4f} | Top1: {v_t1:.2f}%")

      if v_t1 > best_top1:
        best_top1 = v_t1
        save_checkpoint(
            checkpoint_dict(
                model,
                optimizer,
                scheduler,
                epoch,
                extra={
                    "best_top1":
                        best_top1,
                    "scaler_state_dict":
                        (scaler.state_dict() if scaler is not None else None)
                },
            ),
            args.checkpoint + "_best.pt",
            gcs_uri=args.gcs_checkpoint,
        )
        logging.info(f"New best Top1, checkpoint saved: {best_top1:.2f}%")

      completed_epoch = epoch
  except KeyboardInterrupt:
    logging.warning("Interrupt detected!")
  finally:
    save_checkpoint(
        checkpoint_dict(
            model,
            optimizer,
            scheduler,
            completed_epoch,
            extra={
                "best_top1":
                    best_top1,
                "scaler_state_dict":
                    (scaler.state_dict() if scaler is not None else None)
            },
        ),
        args.checkpoint + "_latest.pt",
        gcs_uri=args.gcs_checkpoint,
    )
    logging.info("Checkpoint saved on exit.")
    writer.close()


if __name__ == "__main__":
  main()
