"""Fine-tune a HuggingFace image-classification model for skin lesions."""

import argparse
import os
import logging

from scdiag.logging_utils import setup_logging


# ──────────────────────────────────────────────
#  Constants
# ──────────────────────────────────────────────
NUM_LABELS = 7
MODEL_CHECKPOINT = (
    "microsoft/resnet-50"
    if os.environ.get("SKIN_USE_RESNET", "")
    else "google/vit-base-patch16-224"
)


# ──────────────────────────────────────────────
#  Argument parsing
# ──────────────────────────────────────────────
def parse_args(argv=None):
    """Parse command-line arguments. *argv* defaults to ``sys.argv[1:]``."""
    parser = argparse.ArgumentParser(
        description="Fine-tune a ViT/ResNet model for skin-lesion classification."
    )
    parser.add_argument("--model", default=MODEL_CHECKPOINT,
                        help="HuggingFace model id or local path (default: %(default)s)")
    parser.add_argument("--output-dir", default="./results",
                        help="Where to save checkpoints and logs (default: %(default)s)")
    parser.add_argument("--epochs", type=int, default=10,
                        help="Number of training epochs (default: %(default)s)")
    parser.add_argument("--batch-size", type=int, default=32,
                        help="Per-device batch size (default: %(default)s)")
    parser.add_argument("--lr", type=float, default=5e-5,
                        help="Learning rate (default: %(default)s)")
    parser.add_argument("--weight-decay", type=float, default=0.01,
                        help="Weight decay (default: %(default)s)")
    parser.add_argument("--num-labels", type=int, default=NUM_LABELS,
                        help="Number of classes (default: %(default)s)")
    parser.add_argument("--logging-level", default="INFO",
                        choices=["DEBUG", "INFO", "WARNING", "ERROR"],
                        help="Python logging level (default: %(default)s)")
    parser.add_argument("--dry-run", action="store_true",
                        help="Prepare dataset & model but skip training.")
    return parser.parse_args(argv)


# ──────────────────────────────────────────────
#  Data helpers
# ──────────────────────────────────────────────
def compute_class_weights(dataset, num_labels=NUM_LABELS):
    """Compute inverse-frequency weights for imbalanced classes."""
    import numpy as np

    labels = np.array(dataset["train"]["label"])
    counts = np.bincount(labels, minlength=num_labels).astype(np.float64)
    counts = np.maximum(counts, 1.0)
    total = counts.sum()
    weights = total / (num_labels * counts)
    return weights.astype(np.float32)


# ──────────────────────────────────────────────
#  Custom Trainer with weighted loss
# ──────────────────────────────────────────────
class WeightedTrainer:
    """Placeholder class; the real ``WeightedTrainer`` is built inside
    ``main()`` so that ``transformers`` is only imported when needed."""

    pass


def _build_weighted_trainer_class():
    """Return the real ``WeightedTrainer`` (a ``transformers.Trainer``
    subclass) after ``transformers`` has been imported."""
    import torch.nn as nn
    from transformers import Trainer

    class _WeightedTrainer(Trainer):
        def __init__(self, class_weights=None, **kwargs):
            super().__init__(**kwargs)
            self.class_weights = class_weights

        def compute_loss(self, model, inputs, return_outputs=False, **kwargs):
            import torch

            labels = inputs.pop("labels")
            outputs = model(**inputs)
            logits = outputs.logits
            if self.class_weights is not None:
                weight = torch.tensor(
                    self.class_weights, device=logits.device, dtype=logits.dtype
                )
                loss_fn = nn.CrossEntropyLoss(weight=weight)
            else:
                loss_fn = nn.CrossEntropyLoss()
            loss = loss_fn(logits, labels)
            return (loss, outputs) if return_outputs else loss

    return _WeightedTrainer


# ──────────────────────────────────────────────
#  Build transforms & datasets
# ──────────────────────────────────────────────
def build_datasets(processor, num_labels, pixel_mean, pixel_std):
    """Load *ham10k*, preprocess, and return HuggingFace dataset splits."""
    import numpy as np
    import torch
    from datasets import load_dataset
    from torchvision.transforms import v2

    ds = load_dataset("bentrevett/ham10k", trust_remote_code=True)

    train_img_transform = v2.Compose([
        v2.Resize((224, 224)),
        v2.RandomHorizontalFlip(),
        v2.RandomVerticalFlip(),
        v2.RandomRotation(30),
        v2.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2),
        v2.ToImage(),
        v2.ToDtype(torch.float32, scale=True),
        v2.Normalize(mean=pixel_mean.tolist(), std=pixel_std.tolist()),
    ])

    eval_img_transform = v2.Compose([
        v2.Resize((224, 224)),
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
        preprocess_train, batched=True, batch_size=256,
        remove_columns=ds["train"].column_names,
    )
    ds["test"] = ds["test"].map(
        preprocess_eval, batched=True, batch_size=256,
        remove_columns=ds["test"].column_names,
    )

    return ds


# ──────────────────────────────────────────────
#  Metric callback
# ──────────────────────────────────────────────
def _build_compute_metrics():
    """Return ``compute_metrics`` after ``evaluate`` has been imported."""
    import evaluate
    import numpy as np

    _accuracy_metric = evaluate.load("accuracy")

    def compute_metrics(eval_pred):
        logits, labels = eval_pred
        preds = np.argmax(logits, axis=-1)
        return _accuracy_metric.compute(predictions=preds, references=labels)

    return compute_metrics


# ──────────────────────────────────────────────
#  Main
# ──────────────────────────────────────────────
def main(argv=None):
    # Heavy imports deferred to main()
    import numpy as np
    import torch
    import torch.nn as nn
    from datasets import load_dataset
    from torchvision.transforms import v2
    from transformers import (
        AutoImageProcessor,
        AutoModelForImageClassification,
        TrainingArguments,
    )

    args = parse_args(argv)
    setup_logging(level=args.logging_level)
    log = logging.getLogger(__name__)

    # --- Device -------------------------------------------------
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    log.info(f"Using device: {device}")

    # --- Processor & pixel stats --------------------------------
    processor = AutoImageProcessor.from_pretrained(args.model)
    log.info(f"Image processor type: {processor.__class__.__name__}")

    # Quick pixel-stat estimation from a tiny subset
    ds_raw = load_dataset("bentrevett/ham10k", trust_remote_code=True)
    small = ds_raw["train"].select(range(min(200, len(ds_raw["train"]))))
    tmp = torch.stack([
        v2.ToTensor()(v2.Resize((224, 224))(img.convert("RGB")))
        for img in small["image"]
    ])
    pixel_mean = tmp.mean(dim=[0, 2, 3])
    pixel_std = tmp.std(dim=[0, 2, 3]).clamp(min=1e-6)
    log.info(f"Pixel mean: {pixel_mean}")
    log.info(f"Pixel std:  {pixel_std}")
    del ds_raw, small, tmp

    # --- Full dataset -------------------------------------------
    dataset = build_datasets(processor, args.num_labels, pixel_mean, pixel_std)
    class_weights = compute_class_weights(dataset, num_labels=args.num_labels)
    log.info(f"Class weights: {class_weights}")

    # --- Model --------------------------------------------------
    id2label = {i: str(i) for i in range(args.num_labels)}
    label2id = {str(i): i for i in range(args.num_labels)}
    model = AutoModelForImageClassification.from_pretrained(
        args.model,
        num_labels=args.num_labels,
        id2label=id2label,
        label2id=label2id,
        ignore_mismatched_sizes=True,
    )
    model.to(device)

    total_params = sum(p.numel() for p in model.parameters())
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    log.info(f"Model params: {total_params:,} total, {trainable:,} trainable")

    # --- Training args ------------------------------------------
    training_args = TrainingArguments(
        output_dir=args.output_dir,
        num_train_epochs=args.epochs,
        per_device_train_batch_size=args.batch_size,
        per_device_eval_batch_size=args.batch_size,
        learning_rate=args.lr,
        weight_decay=args.weight_decay,
        eval_strategy="epoch",
        save_strategy="epoch",
        load_best_model_at_end=True,
        metric_for_best_model="accuracy",
        fp16=torch.cuda.is_available(),
        dataloader_num_workers=2,
        report_to="none",
    )

    if args.dry_run:
        log.info("Dry run – model and dataset prepared, skipping training.")
        return

    # --- Trainer -----------------------------------------------
    WeightedTrainer = _build_weighted_trainer_class()
    compute_metrics = _build_compute_metrics()

    trainer = WeightedTrainer(
        class_weights=class_weights.tolist(),
        model=model,
        args=training_args,
        train_dataset=dataset["train"],
        eval_dataset=dataset["test"],
        compute_metrics=compute_metrics,
    )

    log.info("Starting training pipeline...")
    trainer.train()


if __name__ == "__main__":
    main()
