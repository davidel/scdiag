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
      help="HuggingFace model id or local path (e.g. facebook/resnext50_32x4d)")
  parser.add_argument("--num_labels",
                      type=int,
                      default=None,
                      help="Number of classes. If omitted, inferred from the dataset.")
  parser.add_argument("--dataset",
                      default="mrtg/ham10000",
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


def load_and_split_dataset(dataset_id, test_size=0.2, seed=42, cache_dir=None):
  """Load a dataset with a single train split and split it into train/test."""
  raw = load_dataset(dataset_id, split="train", trust_remote_code=True, cache_dir=cache_dir)
  return raw.train_test_split(test_size=test_size, seed=seed)


def get_num_labels_from_dataset(dataset):
  """Infer the number of classes from the dataset's ClassLabel feature."""
  return dataset["train"].features["label"].num_classes


def compute_class_weights(dataset, num_labels, device):
  """Compute inverse-frequency weights and return as tensor on *device*."""
  labels = np.array(dataset["train"]["label"])
  counts = np.bincount(labels, minlength=num_labels).astype(np.float64)
  counts = np.maximum(counts, 1.0)
  total = counts.sum()
  weights = total / (num_labels * counts)
  return torch.tensor(weights, dtype=torch.float32).to(device)


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


def compute_metrics(eval_pred):
  logits, labels = eval_pred
  preds = np.argmax(logits, axis=-1)
  accuracy = evaluate.load("accuracy")
  f1 = evaluate.load("f1")
  acc = accuracy.compute(predictions=preds, references=labels)["accuracy"]
  f1_score = f1.compute(predictions=preds, references=labels, average="macro")["f1"]
  return {"accuracy": acc, "macro_f1": f1_score}


def build_datasets(dataset_id, image_size, cache_dir=None):
  """Load a HuggingFace image dataset, preprocess, and return splits."""
  ds = load_and_split_dataset(dataset_id, cache_dir=cache_dir)

  processor = AutoImageProcessor.from_pretrained(
      "facebook/resnext50_32x4d", size={"height": image_size, "width": image_size})

  augmentations = v2.Compose([
      v2.RandomResizedCrop(size=(image_size, image_size), scale=(0.85, 1.0), antialias=True),
      v2.RandomHorizontalFlip(p=0.5),
      v2.RandomVerticalFlip(p=0.5),
      v2.ColorJitter(brightness=0.15, contrast=0.15, saturation=0.1),
      v2.ToImage(),
      v2.ToDtype(torch.float32, scale=True),
  ])

  def train_transform(examples):
    images = [augmentations(img.convert("RGB")) for img in examples["image"]]
    inputs = processor(images, return_tensors="pt")
    inputs["labels"] = examples["label"]
    return inputs

  def val_transform(examples):
    inputs = processor([img.convert("RGB") for img in examples["image"]],
                       return_tensors="pt")
    inputs["labels"] = examples["label"]
    return inputs

  ds["train"].set_transform(train_transform)
  ds["test"].set_transform(val_transform)

  return ds


def main(argv=None):
  args = parse_args(argv)
  setup_logging(level=args.logging_level)
  log = logging.getLogger(__name__)

  device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
  log.info(f"Using device: {device}")

  if torch.cuda.is_available():
    torch.set_float32_matmul_precision("high")

  dataset = build_datasets(args.dataset, args.image_size, cache_dir=args.cache_dir)

  labels = dataset["train"].features["label"].names
  num_labels = len(labels) if args.num_labels is None else args.num_labels
  label2id = {label: str(i) for i, label in enumerate(labels)}
  id2label = {str(i): label for i, label in enumerate(labels)}
  log.info(f"num_labels: {num_labels}")

  class_weights = compute_class_weights(dataset, num_labels, device)
  log.info(f"Class weights: {class_weights.tolist()}")

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
  log.info(f"Model params: {total_params:,} total, {trainable:,} trainable")

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
      report_to=report_to,
      **({"logging_dir": args.tb_logdir} if args.tb_logdir else {}),
  )

  if args.tb_logdir:
    log.info(f"TensorBoard logging to: {args.tb_logdir}")

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
  try:
    trainer.train()
  except KeyboardInterrupt:
    trainer.save_model(args.output_dir)
    log.info("Interrupted — checkpoint saved.")


if __name__ == "__main__":
  main()
