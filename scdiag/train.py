"""Fine-tune a HuggingFace image-classification model."""

import argparse
import logging

import evaluate
import numpy as np
import torch
import torch.nn as nn
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
  """Parse command-line arguments. *argv* defaults to ``sys.argv[1:]``."""
  parser = argparse.ArgumentParser(
      description="Fine-tune a HuggingFace vision model for image classification.")

  parser.add_argument(
      "--model",
      required=True,
      help="HuggingFace model id or local path (e.g. google/vit-base-patch16-224)")
  parser.add_argument("--num-labels",
                      type=int,
                      default=None,
                      help="Number of classes. If omitted, inferred from the dataset.")
  parser.add_argument("--dataset",
                      default="bentrevett/ham10k",
                      help="HuggingFace dataset id (default: %(default)s)")
  parser.add_argument("--image-size",
                      type=int,
                      default=224,
                      help="Resize images to this square size (default: %(default)s)")
  parser.add_argument("--epochs",
                      type=int,
                      default=10,
                      help="Number of training epochs (default: %(default)s)")
  parser.add_argument("--batch-size",
                      type=int,
                      default=32,
                      help="Per-device batch size (default: %(default)s)")
  parser.add_argument("--lr",
                      type=float,
                      default=5e-5,
                      help="Peak learning rate (default: %(default)s)")
  parser.add_argument("--weight-decay",
                      type=float,
                      default=0.01,
                      help="Weight decay (default: %(default)s)")
  parser.add_argument("--lr-scheduler-type",
                      default="cosine",
                      help="Learning-rate scheduler type (default: %(default)s)")
  parser.add_argument("--warmup-ratio",
                      type=float,
                      default=0.1,
                      help="Fraction of steps used for linear warmup (default: %(default)s)")
  parser.add_argument("--max-grad-norm",
                      type=float,
                      default=1.0,
                      help="Max gradient norm for clipping (default: %(default)s)")
  parser.add_argument("--eval-every",
                      default="epoch",
                      choices=["epoch", "step"],
                      help="Run evaluation every epoch or step (default: %(default)s)")
  parser.add_argument("--save-every",
                      default="epoch",
                      choices=["epoch", "step"],
                      help="Save checkpoint every epoch or step (default: %(default)s)")
  parser.add_argument(
      "--eval-steps",
      type=int,
      default=500,
      help="Evaluate every N steps when --eval-every=step (default: %(default)s)")
  parser.add_argument(
      "--save-steps",
      type=int,
      default=500,
      help="Save every N steps when --save-every=step (default: %(default)s)")
  parser.add_argument("--logging-steps",
                      type=int,
                      default=10,
                      help="Log every N steps (default: %(default)s)")
  parser.add_argument("--dataloader-num-workers",
                      type=int,
                      default=2,
                      help="DataLoader worker processes (default: %(default)s)")
  parser.add_argument("--output-dir",
                      default="./results",
                      help="Where to save checkpoints and logs (default: %(default)s)")
  parser.add_argument("--logging-level",
                      default="INFO",
                      choices=["DEBUG", "INFO", "WARNING", "ERROR"],
                      help="Python logging level (default: %(default)s)")
  parser.add_argument("--dry-run",
                      action="store_true",
                      help="Prepare dataset & model but skip training.")

  return parser.parse_args(argv)


def get_num_labels_from_dataset(dataset):
  """Infer the number of classes from the dataset's ClassLabel feature."""
  return dataset["train"].features["label"].num_classes


def compute_class_weights(dataset, num_labels):
  """Compute inverse-frequency weights for imbalanced classes."""
  labels = np.array(dataset["train"]["label"])
  counts = np.bincount(labels, minlength=num_labels).astype(np.float64)
  counts = np.maximum(counts, 1.0)
  total = counts.sum()
  weights = total / (num_labels * counts)
  return weights.astype(np.float32)


class WeightedTrainer(Trainer):

  def __init__(self, class_weights=None, **kwargs):
    super().__init__(**kwargs)
    self.class_weights = class_weights

  def compute_loss(self, model, inputs, return_outputs=False, **kwargs):
    labels = inputs.pop("labels")
    outputs = model(**inputs)
    logits = outputs.logits
    if self.class_weights is not None:
      weight = torch.tensor(self.class_weights,
                            device=logits.device,
                            dtype=logits.dtype)
      loss_fn = nn.CrossEntropyLoss(weight=weight)
    else:
      loss_fn = nn.CrossEntropyLoss()
    loss = loss_fn(logits, labels)
    return (loss, outputs) if return_outputs else loss


_accuracy_metric = evaluate.load("accuracy")


def compute_metrics(eval_pred):
  logits, labels = eval_pred
  preds = np.argmax(logits, axis=-1)
  return _accuracy_metric.compute(predictions=preds, references=labels)


def build_datasets(dataset_id, image_size, pixel_mean, pixel_std):
  """Load a HuggingFace image dataset, preprocess, and return splits."""
  ds = load_dataset(dataset_id, trust_remote_code=True)

  train_img_transform = v2.Compose([
      v2.Resize((image_size, image_size)),
      v2.RandomHorizontalFlip(),
      v2.RandomVerticalFlip(),
      v2.RandomRotation(30),
      v2.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2),
      v2.ToImage(),
      v2.ToDtype(torch.float32, scale=True),
      v2.Normalize(mean=pixel_mean.tolist(), std=pixel_std.tolist()),
  ])

  eval_img_transform = v2.Compose([
      v2.Resize((image_size, image_size)),
      v2.ToImage(),
      v2.ToDtype(torch.float32, scale=True),
      v2.Normalize(mean=pixel_mean.tolist(), std=pixel_std.tolist()),
  ])

  def preprocess_train(examples):
    images = [img.convert("RGB") for img in examples["image"]]
    pixel_values = torch.stack([train_img_transform(img) for img in images])
    return {
        "pixel_values": pixel_values,
        "labels": torch.tensor(examples["label"], dtype=torch.long),
    }

  def preprocess_eval(examples):
    images = [img.convert("RGB") for img in examples["image"]]
    pixel_values = torch.stack([eval_img_transform(img) for img in images])
    return {
        "pixel_values": pixel_values,
        "labels": torch.tensor(examples["label"], dtype=torch.long),
    }

  ds["train"] = ds["train"].map(
      preprocess_train,
      batched=True,
      batch_size=256,
      remove_columns=ds["train"].column_names,
  )
  ds["test"] = ds["test"].map(
      preprocess_eval,
      batched=True,
      batch_size=256,
      remove_columns=ds["test"].column_names,
  )

  return ds


def main(argv=None):
  args = parse_args(argv)
  setup_logging(level=args.logging_level)
  log = logging.getLogger(__name__)

  device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
  log.info(f"Using device: {device}")

  processor = AutoImageProcessor.from_pretrained(args.model)
  log.info(f"Image processor type: {processor.__class__.__name__}")

  ds_raw = load_dataset(args.dataset, trust_remote_code=True)
  small = ds_raw["train"].select(range(min(200, len(ds_raw["train"]))))
  tmp = torch.stack([
      v2.ToTensor()(v2.Resize((args.image_size, args.image_size))(img.convert("RGB")))
      for img in small["image"]
  ])
  pixel_mean = tmp.mean(dim=[0, 2, 3])
  pixel_std = tmp.std(dim=[0, 2, 3]).clamp(min=1e-6)
  log.info(f"Pixel mean: {pixel_mean}")
  log.info(f"Pixel std:  {pixel_std}")
  del ds_raw, small, tmp

  dataset = build_datasets(args.dataset, args.image_size, pixel_mean, pixel_std)

  if args.num_labels is None:
    num_labels = get_num_labels_from_dataset(dataset)
    log.info(f"Inferred num_labels from dataset: {num_labels}")
  else:
    num_labels = args.num_labels

  class_weights = compute_class_weights(dataset, num_labels=num_labels)
  log.info(f"Class weights: {class_weights}")

  id2label = {i: str(i) for i in range(num_labels)}
  label2id = {str(i): i for i in range(num_labels)}
  model = AutoModelForImageClassification.from_pretrained(
      args.model,
      num_labels=num_labels,
      id2label=id2label,
      label2id=label2id,
      ignore_mismatched_sizes=True,
  )
  model.to(device)

  total_params = sum(p.numel() for p in model.parameters())
  trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
  log.info(f"Model params: {total_params:,} total, {trainable:,} trainable")

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
      metric_for_best_model="accuracy",
      fp16=torch.cuda.is_available(),
      dataloader_num_workers=args.dataloader_num_workers,
      report_to="none",
  )

  if args.dry_run:
    log.info("Dry run - model and dataset prepared, skipping training.")
    return

  trainer = WeightedTrainer(
      class_weights=class_weights.tolist(),
      model=model,
      args=training_args,
      train_dataset=dataset["train"],
      eval_dataset=dataset["test"],
      compute_metrics=compute_metrics,
      callbacks=[GPUStatsCallback(device)],
  )

  log.info("Starting training pipeline...")
  trainer.train()


if __name__ == "__main__":
  main()
