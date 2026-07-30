"""Fine-tune a HuggingFace image-classification model."""

import argparse
import logging

import evaluate
import numpy as np
import torch
import torch.nn as nn
import datasets
from datasets import load_dataset
from torchvision.transforms import v2
from transformers import (
    AutoImageProcessor,
    AutoModelForImageClassification,
    Trainer,
    TrainingArguments,
)

from scdiag.gpu_callback import GPUStatsCallback
from scdiag.logging_utils import setup_logging


def parse_args(argv=None):
  """Configure the CLI for fine-tuning an image-classification model.

  Accepted flags select the pretrained backbone, dataset, image resolution,
  optimiser settings, and output directory.  *argv* defaults to
  ``sys.argv[1:]`` when not provided (i.e. when called from ``main()``).
"""
  parser = argparse.ArgumentParser(
      description="Fine-tune a HuggingFace vision model for image classification.")

  parser.add_argument(
      "--model",
      default="facebook/convnextv2-base-22k-224",
      help="HuggingFace model id or local path (default: %(default)s)")
  parser.add_argument("--num_labels",
                      type=int,
                      default=None,
                      help="Number of classes. If omitted, inferred from the dataset.")
  parser.add_argument("--dataset",
                      default="marmal88/skin_cancer",
                      help="HuggingFace dataset id (default: %(default)s)")
  parser.add_argument("--image_size",
                      type=int,
                      default=448,
                      help="Resize images to this square size (default: %(default)s)")
  parser.add_argument("--epochs",
                      type=int,
                      default=5,
                      help="Number of training epochs (default: %(default)s)")
  parser.add_argument("--batch_size",
                      type=int,
                      default=32,
                      help="Per-device batch size (default: %(default)s)")
  parser.add_argument("--lr",
                      type=float,
                      default=4e-5,
                      help="Peak learning rate (default: %(default)s)")
  parser.add_argument("--weight_decay",
                      type=float,
                      default=0.01,
                      help="Weight decay (default: %(default)s)")
  parser.add_argument("--lr_scheduler_type",
                      default="cosine",
                      help="Learning-rate scheduler type (default: %(default)s)")
  parser.add_argument("--warmup_ratio",
                      type=float,
                      default=0.1,
                      help="Fraction of steps used for linear warmup (default: %(default)s)")
  parser.add_argument("--max_grad_norm",
                      type=float,
                      default=1.0,
                      help="Max gradient norm for clipping (default: %(default)s)")
  parser.add_argument("--eval_every",
                      default="epoch",
                      choices=["epoch", "step"],
                      help="Run evaluation every epoch or step (default: %(default)s)")
  parser.add_argument("--save_every",
                      default="epoch",
                      choices=["epoch", "step"],
                      help="Save checkpoint every epoch or step (default: %(default)s)")
  parser.add_argument(
      "--eval_steps",
      type=int,
      default=500,
      help="Evaluate every N steps when --eval_every=step (default: %(default)s)")
  parser.add_argument(
      "--save_steps",
      type=int,
      default=500,
      help="Save every N steps when --save_every=step (default: %(default)s)")
  parser.add_argument("--logging_steps",
                      type=int,
                      default=20,
                      help="Log every N steps (default: %(default)s)")
  parser.add_argument("--dataloader_num_workers",
                      type=int,
                      default=2,
                      help="DataLoader worker processes (default: %(default)s)")
  parser.add_argument("--output_dir",
                      default="./results",
                      help="Where to save checkpoints and logs (default: %(default)s)")
  parser.add_argument("--cache_dir",
                      default=None,
                      help="HuggingFace cache directory for models and datasets (default: system default)")
  parser.add_argument("--tb_logdir",
                      default=None,
                      help="TensorBoard log directory. Enables TensorBoard if set.")
  parser.add_argument("--logging_level",
                      default="INFO",
                      choices=["DEBUG", "INFO", "WARNING", "ERROR"],
                      help="Python logging level (default: %(default)s)")
  return parser.parse_args(argv)


def _detect_column(dataset, candidates, *, feature_types=None, col_name):
  """Return the first column in *candidates* that exists in *dataset*.

  *feature_types* optionally restricts the check to specific feature types.
  Raises ``ValueError`` if nothing matches.
  """
  for name, feat in dataset.features.items():
    if name.lower() in candidates:
      if feature_types is None or isinstance(feat, feature_types):
        return name
  raise ValueError(
      f"No {col_name} column found in dataset. "
      f"Available features: {list(dataset.features.keys())}")


def _detect_label_column(dataset):
  """Return the name of the label column in *dataset*.

  Looks for an existing ``ClassLabel`` column first.  If none is found,
  falls back to a ``Value("string")`` column whose name matches common
  label conventions (``label``, ``dx``, ``diagnosis``, ``class``,
  ``category``).
  """
  try:
    return _detect_column(
        dataset,
        {"label", "dx", "diagnosis", "class", "category"},
        feature_types=(datasets.ClassLabel,),
        col_name="label",
    )
  except ValueError:
    return _detect_column(
        dataset,
        {"label", "dx", "diagnosis", "class", "category"},
        feature_types=(datasets.Value,),
        col_name="label",
    )


def _detect_image_column(dataset):
  """Return the name of the image column in *dataset*.

  Checks for common names (``image``, ``img``, ``pixel_values``,
  ``pixel_values_float``) and verifies the feature is an ``Image`` type.
  """
  _IMAGE_CANDIDATES = {"image", "img", "pixel_values", "pixel_values_float"}
  for name, feat in dataset.features.items():
    if name.lower() in _IMAGE_CANDIDATES and isinstance(feat, datasets.Image):
      return name
  # Fall back: any column that is an Image feature.
  for name, feat in dataset.features.items():
    if isinstance(feat, datasets.Image):
      return name
  raise ValueError(
      f"No image column found in dataset. "
      f"Available features: {list(dataset.features.keys())}")


def load_and_split_dataset(dataset_id, test_size=0.2, seed=42, cache_dir=None):
  """Load a dataset with a single train split and split it into train/test.

  Returns ``(dataset_dict, image_col)`` where *image_col* is the name of
  the column containing the PIL images.
  """
  raw = load_dataset(dataset_id, split="train", trust_remote_code=True, cache_dir=cache_dir)

  image_col = _detect_image_column(raw)

  label_col = _detect_label_column(raw)
  # Cast to ClassLabel if needed so the rest of the pipeline works.
  if not isinstance(raw.features[label_col], datasets.ClassLabel):
    raw = raw.class_encode_column(label_col)
  if label_col != "label":
    raw = raw.rename_column(label_col, "label")

  return raw.train_test_split(test_size=test_size, seed=seed), image_col


def compute_class_weights(dataset, num_labels):
  """Compute inverse-frequency class weights as a CPU tensor.

  *dataset* is a raw HF ``DatasetDict`` (before ``set_transform`` is
  applied) so that column access does not trigger any registered
  transforms.
  """
  feat = dataset["train"].features["label"]
  raw_labels = dataset["train"]["label"]
  if isinstance(feat, datasets.ClassLabel):
    labels = np.array(feat.str2int(raw_labels) if isinstance(raw_labels[0], str)
                      else raw_labels, dtype=np.int64)
  else:
    # Fallback: encode arbitrary values to 0..N-1.
    unique = sorted(set(raw_labels))
    mapping = {v: i for i, v in enumerate(unique)}
    labels = np.array([mapping[v] for v in raw_labels], dtype=np.int64)
    num_labels = len(unique)
  counts = np.bincount(labels, minlength=num_labels).astype(np.float64)
  counts = np.maximum(counts, 1.0)
  total = counts.sum()
  weights = total / (num_labels * counts)
  return torch.tensor(weights, dtype=torch.float32)


class WeightedTrainer(Trainer):

  def __init__(self, class_weights=None, **kwargs):
    super().__init__(**kwargs)
    self.class_weights = class_weights

  def log(self, logs, start_time=None):
    loss = logs.get("loss")
    lr = logs.get("learning_rate")
    epoch = logs.get("epoch")
    mem_used = logs.get("gpu_mem_used_mb")
    mem_reserved = logs.get("gpu_mem_reserved_mb")
    parts = []
    if loss is not None:
      parts.append(f"loss={loss:.3f}")
    if lr is not None:
      parts.append(f"lr={lr:.2e}")
    if epoch is not None:
      parts.append(f"epoch={epoch:.4f}")
    if mem_used is not None and mem_reserved is not None:
      parts.append(f"gpu_mem={mem_used:.0f}/{mem_reserved:.0f} MB")
    logging.info(" | ".join(parts))
    super().log(logs, start_time)

  def compute_loss(self, model, inputs, return_outputs=False, num_items_in_batch=None):
    labels = inputs["labels"]
    outputs = model(**inputs)
    logits = outputs.logits
    if self.class_weights is not None:
      loss_fn = nn.CrossEntropyLoss(
          weight=self.class_weights.to(device=logits.device, dtype=logits.dtype))
    else:
      loss_fn = nn.CrossEntropyLoss()
    loss = loss_fn(logits, labels)
    return (loss, outputs) if return_outputs else loss


_METRIC_ACCURACY = evaluate.load("accuracy")
_METRIC_F1 = evaluate.load("f1")


def compute_metrics(eval_pred):
  logits, labels = eval_pred
  preds = np.argmax(logits, axis=-1)
  acc = _METRIC_ACCURACY.compute(predictions=preds, references=labels)["accuracy"]
  f1_score = _METRIC_F1.compute(predictions=preds, references=labels, average="macro")["f1"]
  return {"accuracy": acc, "macro_f1": f1_score}


def build_datasets(ds, model_name, image_size, image_col="image", cache_dir=None):
  """Attach preprocessing transforms to *ds* and return ``(ds, processor)``.

  *ds* is a ``DatasetDict`` with train/test splits already loaded.
  *image_col* is the name of the column holding the PIL images.

  The processor is loaded only to obtain its normalization constants;
  all spatial transforms (resize, flip, jitter) and normalization are
  performed by a single ``torchvision.transforms.v2`` pipeline so there
  is no double-processing overhead.
  """
  processor = AutoImageProcessor.from_pretrained(model_name, cache_dir=cache_dir)
  mean, std = processor.image_mean, processor.image_std

  train_augmentations = v2.Compose([
      v2.RandomResizedCrop(size=(image_size, image_size), scale=(0.85, 1.0), antialias=True),
      v2.RandomHorizontalFlip(p=0.5),
      v2.RandomVerticalFlip(p=0.5),
      v2.ColorJitter(brightness=0.15, contrast=0.15, saturation=0.1),
      v2.ToImage(),
      v2.ToDtype(torch.float32, scale=True),
      v2.Normalize(mean=mean, std=std),
  ])

  val_augmentations = v2.Compose([
      v2.Resize(size=(image_size, image_size), antialias=True),
      v2.ToImage(),
      v2.ToDtype(torch.float32, scale=True),
      v2.Normalize(mean=mean, std=std),
  ])

  def train_transform(examples):
    return {
        "pixel_values": torch.stack(
            [train_augmentations(img.convert("RGB")) for img in examples[image_col]]),
        "labels": torch.tensor(examples["label"]),
    }

  def val_transform(examples):
    return {
        "pixel_values": torch.stack(
            [val_augmentations(img.convert("RGB")) for img in examples[image_col]]),
        "labels": torch.tensor(examples["label"]),
    }

  ds["train"].set_transform(train_transform)
  ds["test"].set_transform(val_transform)

  return ds, processor


def main(argv=None):
  args = parse_args(argv)
  setup_logging(level=args.logging_level)
  device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
  logging.info(f"Using device: {device}")

  if torch.cuda.is_available():
    torch.set_float32_matmul_precision("high")

  # Load the raw dataset first so we can inspect features and compute class
  # weights before any transforms are attached.
  raw, image_col = load_and_split_dataset(args.dataset, cache_dir=args.cache_dir)

  labels = raw["train"].features["label"].names
  num_labels = len(labels) if args.num_labels is None else args.num_labels
  label2id = {label: str(i) for i, label in enumerate(labels)}
  id2label = {str(i): label for i, label in enumerate(labels)}
  logging.info(f"num_labels: {num_labels}")

  class_weights = compute_class_weights(raw, num_labels).to(device)
  logging.info(f"Class weights: {class_weights.tolist()}")

  # Now attach preprocessing transforms.
  dataset, processor = build_datasets(raw, args.model, args.image_size,
                                      image_col=image_col)

  model = AutoModelForImageClassification.from_pretrained(
      args.model,
      num_labels=num_labels,
      id2label=id2label,
      label2id=label2id,
      ignore_mismatched_sizes=True,
      cache_dir=args.cache_dir,
  )
  model.to(device)

  total_params = sum(p.numel() for p in model.parameters())
  trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
  logging.info(f"Model params: {total_params:,} total, {trainable:,} trainable")

  report_to = ["tensorboard"] if args.tb_logdir else ["none"]

  training_args = TrainingArguments(
      output_dir=args.output_dir,
      num_train_epochs=args.epochs,
      per_device_train_batch_size=args.batch_size,
      per_device_eval_batch_size=args.batch_size,
      learning_rate=args.lr,
      weight_decay=args.weight_decay,
      lr_scheduler_type=args.lr_scheduler_type,
      warmup_ratio=args.warmup_ratio,
      max_grad_norm=args.max_grad_norm,
      eval_strategy=args.eval_every,
      save_strategy=args.save_every,
      eval_steps=args.eval_steps if args.eval_every == "step" else None,
      save_steps=args.save_steps if args.save_every == "step" else None,
      logging_steps=args.logging_steps,
      load_best_model_at_end=True,
      metric_for_best_model="macro_f1",
      bf16=torch.cuda.is_available(),
      dataloader_num_workers=args.dataloader_num_workers,
      remove_unused_columns=False,
      report_to=report_to,
      **({"logging_dir": args.tb_logdir} if args.tb_logdir else {}),
  )

  if args.tb_logdir:
    logging.info(f"TensorBoard logging to: {args.tb_logdir}")

  trainer = WeightedTrainer(
      class_weights=class_weights,
      model=model,
      args=training_args,
      train_dataset=dataset["train"],
      eval_dataset=dataset["test"],
      compute_metrics=compute_metrics,
      callbacks=[GPUStatsCallback(device)],
  )

  logging.info("Starting training pipeline...")
  try:
    trainer.train()
  except KeyboardInterrupt:
    trainer.save_model(args.output_dir)
    logging.info("Interrupted — checkpoint saved.")


if __name__ == "__main__":
  main()
