"""Fine-tune a HuggingFace image-classification model."""

import argparse
import gc
import logging
import os
import time

import datasets
import numpy as np
import torch
import torch.nn as nn
from datasets import load_dataset
from sklearn.metrics import f1_score, precision_recall_fscore_support
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter
from torchvision.transforms import v2
from torchvision.transforms.v2 import InterpolationMode

from scdiag.checkpointing import (
    checkpoint_dict,
    parse_state_flags,
    resume_checkpoint,
)
from scdiag.cli_utils import KVPairAction
from scdiag.datasets.hf_proxy import HFDatasetProxy
from scdiag.gpu_utils import gpu_stats_str
from scdiag.grad_monitor import GradMonitor
from scdiag.logging_utils import fatal, setup_logging
from scdiag.models import load_model, load_processor
from scdiag.optim_factory import create_optimizer, create_scheduler
from scdiag.script_utils import load_extern
from scdiag.storage_utils import save_checkpoint


def parse_class_multipliers(s, num_labels, label2id):
  """Parse a ``--class_multipliers`` string into a ``[num_labels]`` tensor.

    *s* is a comma-separated string of ``NAME=VALUE`` pairs where *NAME* is
    a label string (e.g. ``melanoma``) or an integer label index and *VALUE*
    is a float multiplier.  Unspecified classes default to ``1.0``.

    Returns a ``torch.Tensor`` of shape ``[num_labels]``.
    """
  import torch

  m = torch.ones(num_labels)
  if not s or not s.strip():
    return m
  for pair in s.split(","):
    pair = pair.strip()
    if not pair:
      continue
    if "=" not in pair:
      fatal(
          f"Invalid --class_multipliers entry: '{pair}'. "
          "Expected NAME=VALUE (e.g. melanoma=4.0).", ValueError)
    name, val = pair.split("=", 1)
    name, val = name.strip(), val.strip()
    if name.isdigit():
      idx = int(name)
    else:
      if name not in label2id:
        fatal(
            f"Unknown class name '{name}' in --class_multipliers. "
            f"Available: {list(label2id.keys())}", ValueError)
      idx = label2id[name]
    if not (0 <= idx < num_labels):
      fatal(f"Label index {idx} out of range [0, {num_labels})", ValueError)
    m[idx] = float(val)
  return m


class CombinedFocalLoss(nn.Module):
  """Cost-sensitive focal loss for multi-class classification.

    Combines inverse-frequency weights, clinical severity multipliers,
    and focal loss modulation into a single loss function.

    Builds on PyTorch's numerically stable cross-entropy implementation
    (log-sum-exp trick) rather than reimplementing log_softmax manually.
    """

  def __init__(self, weights, gamma=2.0, label_smoothing=0.0, reduction="mean"):
    """
        Args:
            weights: [num_classes] W_final = W_freq x M_c, pre-computed at
                     construction time (constant per class, not per-batch).
            gamma: Focal loss focusing parameter (higher = more focus on
                   hard examples).  0.0 disables focal modulation and reduces
                   to standard weighted cross-entropy.
            label_smoothing: Label smoothing factor (0.0 = disabled).
                             Passed through to F.cross_entropy.
            reduction: 'mean', 'sum', or 'none'.
        """
    super().__init__()
    self.register_buffer("weights", weights)
    self.gamma = gamma
    self.label_smoothing = label_smoothing
    self.reduction = reduction

  def forward(self, inputs, targets):
    """
        Args:
            inputs: [batch_size, num_classes] raw logits
            targets: [batch_size] ground truth class indices
        Returns:
            Scalar loss (or per-sample losses if reduction='none').
        """
    # Use PyTorch's numerically stable CE with reduction='none'.
    ce_loss = nn.functional.cross_entropy(
        inputs,
        targets,
        weight=self.weights,
        label_smoothing=self.label_smoothing,
        reduction="none",
    )

    if self.gamma == 0:
      # No focal modulation — standard weighted CE.
      if self.reduction == "mean":
        return ce_loss.mean()
      elif self.reduction == "sum":
        return ce_loss.sum()
      return ce_loss

    # Focal modulation: down-weight easy examples.
    # p_t is the model's predicted probability for the true class.
    # NOTE: The focal weight is computed with gradients enabled so that
    # the full focal loss contributes correct gradient signals during
    # learning-rate warmup and fine-tuning.  Earlier, this was wrapped
    # in torch.no_grad() which silently disconnected the focal
    # probability from the gradient graph.
    prob = torch.nn.functional.softmax(inputs, dim=-1)
    p_t = prob.gather(1, targets.unsqueeze(1)).squeeze(1)
    focal_weight = (1 - p_t)**self.gamma

    loss = focal_weight * ce_loss

    if self.reduction == "mean":
      return loss.mean()
    elif self.reduction == "sum":
      return loss.sum()
    return loss


def load_augmentation_script(path_or_url):
  """Load a Python script and return its ``create_train_transform`` callable.

    The script must define a ``create_train_transform(image_size, **kwargs)``
    function that returns a list of ``torchvision.transforms.v2`` transforms.

    Args:
        path_or_url: Local file path or HTTP/HTTPS URL to the script.

    Returns:
        The ``create_train_transform`` callable.

    Raises:
        FileNotFoundError: If a local path does not exist.
        ValueError: If the script does not define a callable
            ``create_train_transform``.
  """
  return load_extern(path_or_url, "create_train_transform")


def build_transforms(processor, image_size, train_aug_fn=None):
  """Create train / val augmentation pipelines.

    If *train_aug_fn* is ``None`` the default (hand-picked) augmentation
    list is used.  Otherwise *train_aug_fn(image_size)* must return a
    list of ``v2`` transforms; the fixed preprocessing tail is appended
    automatically.

    Includes the processor's normalization (mean / std) so images arrive
    at the model ready for inference.
    """
  tail = [
      v2.ToImage(),
      v2.ToDtype(torch.float32, scale=True),
      v2.Normalize(mean=processor.image_mean, std=processor.image_std),
  ]

  if train_aug_fn is not None:
    user_transforms = train_aug_fn(image_size)
    if not isinstance(user_transforms, list):
      fatal(
          "create_train_transform() must return a list of transforms, "
          f"got {type(user_transforms).__name__}", TypeError)
    train_augmentations = v2.Compose(user_transforms + tail)
  else:
    train_augmentations = v2.Compose([
        v2.RandomResizedCrop(size=(image_size, image_size),
                             scale=(0.2, 1.0),
                             antialias=True),
        v2.RandomHorizontalFlip(p=0.5),
        v2.RandomVerticalFlip(p=0.5),
        v2.RandomRotation(degrees=360),
        v2.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.15, hue=0.05),
        v2.ElasticTransform(alpha=50.0, sigma=5.0),
        *tail,
    ])

  val_augmentations = v2.Compose([
      v2.Resize(image_size, interpolation=InterpolationMode.BICUBIC),
      v2.CenterCrop(image_size),
      *tail,
  ])

  return train_augmentations, val_augmentations


def load_and_split_dataset(
    dataset_name,
    cache_dir=None,
    test_size=0.2,
    seed=42,
    train_transform=None,
    val_transform=None,
    image_column=None,
    label_column=None,
):
  """Load a HuggingFace dataset, return ``(train_proxy, val_proxy)``."""
  if dataset_name.startswith("imagefolder/"):
    data_dir = dataset_name.split("/", 1)[1]
    raw = load_dataset("imagefolder", data_dir=data_dir, cache_dir=cache_dir)
  else:
    raw = load_dataset(dataset_name, cache_dir=cache_dir)

  # Single split: validate, split and wrap.
  if isinstance(raw, datasets.Dataset):
    detected_image_column = (image_column or HFDatasetProxy.detect_image_column(raw))
    if detected_image_column is None:
      fatal(
          f"No image column detected in {dataset_name}. "
          f"Columns: {list(raw.features.keys())}", ValueError)
    split = raw.train_test_split(test_size=test_size, seed=seed)
    return (
        HFDatasetProxy(split["train"],
                       transform=train_transform,
                       image_column=image_column,
                       label_column=label_column),
        HFDatasetProxy(split["test"],
                       transform=val_transform,
                       image_column=image_column,
                       label_column=label_column),
    )

  # DatasetDict: validate and ensure train/test splits exist.
  for split_name in raw:
    detected_image_column = (image_column or
                             HFDatasetProxy.detect_image_column(raw[split_name]))
    if detected_image_column is None:
      fatal(
          f"No image column in split '{split_name}' of {dataset_name}. "
          f"Columns: {list(raw[split_name].features.keys())}", ValueError)

  splits = set(raw.keys())
  if "train" not in splits or "test" not in splits:
    if "train" in splits:
      split = raw["train"].train_test_split(test_size=test_size, seed=seed)
      raw = datasets.DatasetDict(split)
    elif len(splits) == 1:
      only = next(iter(splits))
      split = raw[only].train_test_split(test_size=test_size, seed=seed)
      raw = datasets.DatasetDict(split)
    else:
      names = list(raw.keys())
      raw = datasets.DatasetDict({"train": raw[names[0]], "test": raw[names[1]]})

  logging.info("Using image column: %s%s", image_column or "auto-detected",
               " (explicit)" if image_column else "")
  logging.info("Using label column: %s%s", label_column or "auto-detected",
               " (explicit)" if label_column else "")
  return (
      HFDatasetProxy(raw["train"],
                     transform=train_transform,
                     image_column=image_column,
                     label_column=label_column),
      HFDatasetProxy(raw["test"],
                     transform=val_transform,
                     image_column=image_column,
                     label_column=label_column),
  )


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
    fatal(
        f"Dataset has {len(actual_labels)} unique labels but "
        f"num_labels={num_labels}", ValueError)
  counts = np.bincount(labels, minlength=num_labels).astype(np.float64)
  counts = np.maximum(counts, 1.0)
  weights = 1.0 / counts
  weights = weights / weights.sum() * num_labels
  return torch.tensor(weights, dtype=torch.float32)


def parse_args(argv=None):
  parser = argparse.ArgumentParser(
      description="Fine-tune a HuggingFace image-classification model.",
      formatter_class=argparse.ArgumentDefaultsHelpFormatter,
  )

  parser.add_argument(
      "--model",
      type=str,
      default="google/vit-base-patch16-224",
      help="HuggingFace model name or path.",
  )
  parser.add_argument(
      "--dataset",
      type=str,
      default="marmal88/skin_cancer",
      help=
      "HuggingFace dataset name, or 'imagefolder/PATH' for local ImageFolder datasets.",
  )
  parser.add_argument(
      "--image_column",
      type=str,
      help="Image column name; auto-detected when omitted.",
  )
  parser.add_argument(
      "--label_column",
      type=str,
      help="Label column name; auto-detected when omitted.",
  )
  parser.add_argument(
      "--image_size",
      type=int,
      default=448,
      help="Resize images to this size.",
  )
  parser.add_argument(
      "--train_augmentation_script",
      type=str,
      help="Path or URL to a Python script defining "
      "create_train_transform(image_size, **kwargs) -> list of v2 "
      "transforms. The fixed tail (ToImage, ToDtype, Normalize) is "
      "appended automatically.",
  )
  parser.add_argument(
      "--epochs",
      type=int,
      default=5,
      help="Number of training epochs.",
  )
  parser.add_argument("--batch_size", type=int, default=32, help="Batch size.")
  parser.add_argument(
      "--lr",
      type=float,
      default=3e-5,
      help="Peak learning rate.",
  )
  parser.add_argument(
      "--weight_decay",
      type=float,
      default=0.01,
      help="Weight decay.",
  )

  parser.add_argument(
      "--label_smoothing",
      type=float,
      default=0.0,
      help="Label smoothing.",
  )
  parser.add_argument(
      "--focal_gamma",
      type=float,
      default=0.0,
      help="Focal loss gamma. 0.0 disables focal modulation "
      "(standard weighted CE).",
  )
  parser.add_argument(
      "--class_multipliers",
      type=str,
      default="",
      help="Comma-separated NAME=VALUE pairs to override per-class "
      "clinical severity multipliers (M_c). NAME is a label string "
      "(e.g. melanoma) or integer label index. VALUE is a float. "
      "Unspecified classes default to 1.0. Example: "
      "'melanoma=4.0,melanocytic_Nevi=0.5'.",
  )
  parser.add_argument(
      "--grad_accum_steps",
      type=int,
      default=1,
      help="Gradient accumulation steps.",
  )
  parser.add_argument(
      "--amp_dtype",
      type=str,
      choices=["float16", "bfloat16"],
      help="AMP dtype for mixed precision. Omit to disable AMP. "
      "float16 requires GradScaler; bfloat16 is recommended for "
      "Ampere+ GPUs.",
  )

  parser.add_argument(
      "--checkpoint",
      type=str,
      default="scdiag",
      help="Base path for checkpoints. '_latest.pt' and "
      "'_best.pt' are appended automatically.",
  )

  parser.add_argument(
      "--log_dir",
      type=str,
      help="TensorBoard log directory. Defaults to "
      "<dir_of_latest_ckpt>/logs.",
  )
  parser.add_argument(
      "--log_every",
      type=int,
      default=20,
      help="Log every N steps.",
  )
  parser.add_argument(
      "--grad_monitor",
      type=int,
      default=-1,
      help="Log gradient stats every N steps. -1 (default) = disabled.",
  )
  parser.add_argument(
      "--save_every",
      type=int,
      default=500,
      help="Save checkpoint every N steps.",
  )
  parser.add_argument(
      "--num_workers",
      type=int,
      default=2,
      help="DataLoader worker processes.",
  )
  parser.add_argument(
      "--log_level",
      type=str,
      default="INFO",
      choices=["DEBUG", "INFO", "WARNING", "ERROR"],
      help="Minimum logging level.",
  )

  parser.add_argument(
      "--state_save",
      type=str,
      default="opt,sched,amp",
      help="Comma-separated list of states to save in "
      "checkpoints. One or more of: opt, sched, amp, none.",
  )
  parser.add_argument(
      "--state_load",
      type=str,
      default="opt,sched,amp",
      help="Comma-separated list of states to restore from checkpoint "
      "on resume. One or more of: opt, sched, amp, none.",
  )

  parser.add_argument(
      "--mixup_alpha",
      type=float,
      default=0.0,
      help="Mixup alpha. Recommended: 0.2 for skin lesion classification.",
  )

  parser.add_argument(
      "--cache_dir",
      type=str,
      help="Cache directory for downloaded datasets.",
  )
  parser.add_argument(
      "--source_checkpoint",
      type=str,
      help="Path to a source checkpoint to absorb parameters from. "
      "Keys are aligned by shape and name before loading. "
      "Typically produced by scdiag-pretrain.",
  )
  parser.add_argument(
      "--param_rename",
      nargs="+",
      default=None,
      help="Regex-based key rename patterns for --source_checkpoint. "
      "Each pattern is 'SEARCH;REPLACE' where SEARCH is a Python regex "
      "and REPLACE may use $1, $2, … for capture groups. "
      "Applied before shape-based alignment. "
      "Example: 'encoder\\\\.(.*);model\\\\.$1'.",
  )

  parser.add_argument(
      "--remote_checkpoint",
      type=str,
      help="Remote URI to sync checkpoints to "
      "(format: gs://BUCKET/PREFIX or r2://BUCKET/PREFIX).",
  )

  xgb_group = parser.add_argument_group("xgboost")
  xgb_group.add_argument(
      "--xgboost_model",
      help="Output path for XGBoost model. If set, train XGBoost on "
      "backbone features after training completes.",
  )
  xgb_group.add_argument(
      "--xgb_max_depth",
      type=int,
      default=6,
      help="XGBoost max tree depth.",
  )
  xgb_group.add_argument(
      "--xgb_n_estimators",
      type=int,
      default=200,
      help="XGBoost number of trees.",
  )
  xgb_group.add_argument(
      "--xgb_learning_rate",
      type=float,
      default=0.1,
      help="XGBoost learning rate.",
  )
  xgb_group.add_argument(
      "--xgb_subsample",
      type=float,
      default=0.8,
      help="XGBoost row sampling ratio.",
  )
  xgb_group.add_argument(
      "--xgb_colsample_bytree",
      type=float,
      default=0.8,
      help="XGBoost column sampling ratio.",
  )
  xgb_group.add_argument(
      "--xgb_min_child_weight",
      type=int,
      default=1,
      help="XGBoost min child weight.",
  )
  xgb_group.add_argument(
      "--xgb_gamma",
      type=float,
      default=0.0,
      help="XGBoost min split loss.",
  )
  xgb_group.add_argument(
      "--xgb_reg_alpha",
      type=float,
      default=0.0,
      help="XGBoost L1 regularization.",
  )
  xgb_group.add_argument(
      "--xgb_use_gpu",
      action="store_true",
      default=False,
      help="Use GPU for XGBoost training (requires xgboost with CUDA support).",
  )

  parser.add_argument(
      "--model_arg",
      nargs="+",
      action=KVPairAction,
      default={},
      metavar="KEY=VALUE",
      help="Override model configuration (repeatable). "
      "Example: --model_arg depth=6 num_heads=8",
  )
  parser.add_argument(
      "--proc_arg",
      nargs="+",
      action=KVPairAction,
      default={},
      metavar="KEY=VALUE",
      help="Override processor configuration (repeatable).",
  )
  parser.add_argument(
      "--optimizer",
      type=str,
      default="AdamW",
      help="torch.optim optimizer class name (case-sensitive). "
      "Examples: AdamW (default), Adam, SGD.",
  )
  parser.add_argument(
      "--opt_arg",
      nargs="+",
      action=KVPairAction,
      default={},
      metavar="KEY=VALUE",
      help="Extra optimizer kwargs (repeatable). "
      "Example: --opt_arg betas=0.9,0.999 momentum=0.9",
  )
  parser.add_argument(
      "--scheduler",
      type=str,
      default=None,
      help="torch.optim.lr_scheduler class name (case-sensitive), "
      "or a path to a custom .py script. "
      "Examples: CosineAnnealingLR, StepLR. "
      "Default: None (no scheduler).",
  )
  parser.add_argument(
      "--sched_arg",
      nargs="+",
      action=KVPairAction,
      default={},
      metavar="KEY=VALUE",
      help="Extra scheduler kwargs (repeatable). "
      "Example: --sched_arg T_max=50 eta_min=1e-6",
  )

  return parser.parse_args(argv)


def train_xgboost_on_backbone(args,
                              train_ds,
                              val_ds,
                              device,
                              num_labels=None,
                              batch_size=32):
  """Train XGBoost on backbone features after PyTorch training completes.

    Args:
        args: Parsed CLI args (contains xgb_* hyperparameters, checkpoint paths, etc.)
        train_ds: Training HF Dataset (raw, before proxy wrapping).
        val_ds: Validation HF Dataset (raw, before proxy wrapping).
        device: torch device.
        num_labels: Number of output classes (forwarded to model loader).
        batch_size: Feature-extraction batch size.
    """
  from scdiag.datasets.hf_proxy import HFDatasetProxy
  from scdiag.model_utils import (
      build_val_transform,
      collect_features,
      load_model_for_inference,
  )
  from scdiag.xgb_utils import eval_xgboost, train_xgboost

  logging.info("=" * 60)
  logging.info("XGBoost training on backbone features")
  logging.info("=" * 60)

  # 1. Load the best checkpoint into a fresh model
  from scdiag.checkpointing import select_available_checkpoint

  ckpt_path = select_available_checkpoint(args.checkpoint)
  if ckpt_path is not None:
    logging.info(f"Loading checkpoint: {ckpt_path}")
    model_best, xgb_processor = load_model_for_inference(args.model,
                                                         ckpt_path,
                                                         device="cpu",
                                                         cache_dir=args.cache_dir,
                                                         num_labels=num_labels,
                                                         image_size=args.image_size,
                                                         proc_kwargs=args.proc_arg)
    model_best = model_best.to(device)

    # 2. Rebuild train and val datasets with val transforms (not train augs)
    val_transform = build_val_transform(xgb_processor, args.image_size)
    train_proxy = HFDatasetProxy(train_ds, transform=val_transform)
    val_proxy = HFDatasetProxy(val_ds, transform=val_transform)

    # 3. Collect features
    logging.info("Extracting train features...")
    train_features, train_labels = collect_features(model_best,
                                                    train_proxy,
                                                    device,
                                                    batch_size=batch_size)
    logging.info(f"  Train features shape: {train_features.shape}")

    logging.info("Extracting val features...")
    val_features, val_labels = collect_features(model_best,
                                                val_proxy,
                                                device,
                                                batch_size=batch_size)
    logging.info(f"  Val features shape: {val_features.shape}")

    # 4. Free the model — XGBoost doesn't need it anymore
    del model_best
    gc.collect()
    torch.cuda.empty_cache()

    # 5. Train XGBoost
    xgb_model = train_xgboost(
        train_features,
        train_labels,
        max_depth=args.xgb_max_depth,
        n_estimators=args.xgb_n_estimators,
        learning_rate=args.xgb_learning_rate,
        subsample=args.xgb_subsample,
        colsample_bytree=args.xgb_colsample_bytree,
        min_child_weight=args.xgb_min_child_weight,
        gamma=args.xgb_gamma,
        reg_alpha=args.xgb_reg_alpha,
        use_gpu=args.xgb_use_gpu,
    )

    # 6. Evaluate on val set
    val_metrics = eval_xgboost(xgb_model,
                               val_features,
                               val_labels,
                               id2label=train_proxy.id2label)
    logging.info(f"XGBoost val accuracy: {val_metrics['accuracy']:.2%}")
    for cls, acc in val_metrics["per_class_accuracy"].items():
      logging.info(f"  {cls}: {acc:.2%}")
    logging.info(f"Classification report:\n"
                 f"{val_metrics['classification_report']}")
    logging.info(f"Confusion matrix:\n{val_metrics['confusion_matrix']}")

    # 7. Save the XGBoost model
    xgb_model.save_model(args.xgboost_model)
    logging.info(f"XGBoost model saved: {args.xgboost_model}")


def mixup_data(x, y, alpha=0.2):
  """Apply Mixup to a batch: returns mixed images, and two label sets + lambda.

    Returns ``(mixed_x, y_a, y_b, lam)`` where ``lam`` is the interpolation
    coefficient sampled from ``Beta(alpha, alpha)``.  When ``alpha <= 0`` the
    function is a no-op and returns the originals unchanged.
    """
  if alpha <= 0:
    return x, y, y, 1.0
  lam = np.random.beta(alpha, alpha)
  lam = max(lam, 1.0 - lam)  # keep lambda > 0.5 for consistency
  batch_size = x.size(0)
  index = torch.randperm(batch_size, device=x.device)
  mixed_x = lam * x + (1.0 - lam) * x[index]
  return mixed_x, y, y[index], lam


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
    monitor=None,
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
  window_correct = 0
  window_loss = 0.0
  window_preds = []
  window_labels = []

  for batch_idx, (images, targets) in enumerate(dataloader):
    images, targets = images.to(device), targets.to(device)

    use_mixup = args.mixup_alpha > 0 and images.size(0) >= 2
    if use_mixup:
      images, targets_a, targets_b, lam = mixup_data(images,
                                                     targets,
                                                     alpha=args.mixup_alpha)

    with torch.amp.autocast(
        "cuda",
        dtype=amp_dtype,
        enabled=(amp_dtype is not None and device.type == "cuda"),
    ):
      outputs = model(pixel_values=images)
      logits = outputs.logits
      if use_mixup:
        loss = lam * criterion(logits, targets_a) + (1.0 - lam) * criterion(
            logits, targets_b)
        loss = loss / args.grad_accum_steps
      else:
        loss = criterion(logits, targets) / args.grad_accum_steps

    if amp_dtype == torch.float16 and scaler is not None:
      scaler.scale(loss).backward()
    else:
      loss.backward()

    # Step optimizer only every grad_accum_steps batches (or at end of epoch).
    if (batch_idx + 1) % args.grad_accum_steps == 0 or (batch_idx + 1) == total_batches:
      if amp_dtype == torch.float16 and scaler is not None:
        scaler.unscale_(optimizer)
        # Gradient monitor must read AFTER unscale_() so it sees the true
        # gradient magnitudes, not the scaled values.
        if monitor is not None:
          monitor.step(epoch * total_batches + batch_idx)
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        scaler.step(optimizer)
        scaler.update()
      else:
        if monitor is not None:
          monitor.step(epoch * total_batches + batch_idx)
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        optimizer.step()
      optimizer.zero_grad(set_to_none=True)

    with torch.no_grad():
      orig_targets = (targets if not use_mixup else
                      (targets_a if lam >= 0.5 else targets_b))
      correct_top1 += (logits.argmax(dim=1) == orig_targets).sum().item()

    batch_size = orig_targets.size(0)
    total_loss += loss.item() * batch_size * args.grad_accum_steps
    total_samples += batch_size
    window_samples += batch_size
    window_loss += loss.item() * batch_size * args.grad_accum_steps
    window_correct += (logits.argmax(dim=1) == orig_targets).sum().item()
    window_preds.extend(logits.argmax(dim=1).cpu().tolist())
    window_labels.extend(orig_targets.cpu().tolist())

    # Periodic step-level logging.
    if (batch_idx + 1) % args.log_every == 0 or (batch_idx + 1) == total_batches:
      global_step = epoch * total_batches + batch_idx
      elapsed = time.time() - last_log_time
      throughput = window_samples / elapsed if elapsed > 0 else 0
      lr_now = optimizer.param_groups[0]["lr"]
      w_loss = window_loss / window_samples if window_samples > 0 else 0.0
      w_top1 = ((window_correct / window_samples) *
                100.0 if window_samples > 0 else 0.0)
      # Compute window macro F1.
      w_macro_f1 = 0.0
      if window_preds:
        w_macro_f1 = (
            f1_score(window_labels, window_preds, average="macro", zero_division=0) *
            100.0)

      avg_loss = total_loss / total_samples
      top1 = (correct_top1 / total_samples) * 100.0
      gpu = gpu_stats_str(device)
      msg = (f"  [Step {batch_idx + 1}/{total_batches}]"
             f" loss={w_loss:.4f} ({avg_loss:.4f})"
             f" top1={w_top1:.2f}% ({top1:.2f}%)"
             f" macro_f1={w_macro_f1:.2f}%"
             f" lr={lr_now:.2e} img/s={throughput:.0f}")
      logging.info(msg)
      if gpu:
        logging.info(f"  [Step {batch_idx + 1}/{total_batches}] {gpu}")
      if writer is not None:
        writer.add_scalar("Train/loss", w_loss, global_step)
        writer.add_scalar("Train/top1", w_top1, global_step)
        writer.add_scalar("Train/macro_f1", w_macro_f1, global_step)
        writer.add_scalar("Train/loss_avg", avg_loss, global_step)
        writer.add_scalar("Train/top1_avg", top1, global_step)
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
      window_correct = 0
      window_loss = 0.0
      window_preds = []
      window_labels = []

  avg_loss = total_loss / total_samples
  top1 = (correct_top1 / total_samples) * 100.0
  elapsed = time.time() - start_time
  logging.info(f"  Train stats -> loss: {avg_loss:.4f} | top1: {top1:.2f}%"
               f" | time: {elapsed:.1f}s")
  return avg_loss, top1


def evaluate_performance(model,
                         dataloader,
                         criterion,
                         device,
                         amp_dtype,
                         id2label=None):
  """Evaluate on a validation/test set.

    Returns ``(eval_loss, top1_acc_pct, macro_f1, per_class_f1)`` where
    *per_class_f1* is a dict ``{class_name: f1_score}`` (or ``{}`` if
    *id2label* is not supplied).
    """
  model.eval()
  eval_loss, correct_top1, total_samples = 0.0, 0, 0
  all_preds = []
  all_labels = []
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
      all_preds.extend(logits.argmax(dim=1).cpu().tolist())
      all_labels.extend(targets.cpu().tolist())

  avg_loss = eval_loss / total_samples
  top1 = (correct_top1 / total_samples) * 100.0

  # Per-class and macro F1.
  _, _, f1s, _ = precision_recall_fscore_support(all_labels,
                                                 all_preds,
                                                 average=None,
                                                 zero_division=0)
  macro_f1 = f1_score(all_labels, all_preds, average="macro", zero_division=0) * 100.0

  per_class_f1 = {}
  if id2label:
    for idx, score in enumerate(f1s):
      name = id2label.get(str(idx), id2label.get(idx, str(idx)))
      per_class_f1[name] = score * 100.0

  return avg_loss, top1, macro_f1, per_class_f1


def main():
  args = parse_args()
  setup_logging(args.log_level)

  # Convert string amp_dtype to torch.dtype.
  args.amp_dtype = getattr(torch, args.amp_dtype, None) if args.amp_dtype else None

  states_to_save = parse_state_flags(args.state_save)
  states_to_load = parse_state_flags(args.state_load)

  device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
  logging.info(f"Using device: {device}")

  log_dir = args.log_dir or os.path.join(
      os.path.dirname(args.checkpoint) or ".", "logs")
  os.makedirs(log_dir, exist_ok=True)
  writer = SummaryWriter(log_dir=log_dir)

  # Load processor — unified registry dispatches to HF or custom.
  processor = load_processor(
      args.model,
      image_size=args.image_size,
      cache_dir=args.cache_dir,
      **args.proc_arg,
  )

  # Resolve custom augmentation script to a callable, if provided.
  train_aug_fn = None
  if args.train_augmentation_script:
    train_aug_fn = load_augmentation_script(args.train_augmentation_script)
    logging.info(f"Using custom augmentation script: "
                 f"{args.train_augmentation_script}")

  train_transforms, val_transforms = build_transforms(processor, args.image_size,
                                                      train_aug_fn)

  train_proxy, val_proxy = load_and_split_dataset(
      args.dataset,
      cache_dir=args.cache_dir,
      train_transform=train_transforms,
      val_transform=val_transforms,
      image_column=args.image_column,
      label_column=args.label_column,
  )

  num_labels = train_proxy.num_labels
  logging.info(f"num_labels: {num_labels}")

  w_freq = compute_class_weights(train_proxy.dataset, num_labels)
  logging.info(f"Inverse-frequency weights: {w_freq.tolist()}")

  # Convert proxy label2id (str values) to int values for parse_class_multipliers.
  label2id_int = {k: int(v) for k, v in train_proxy.label2id.items()}

  # Apply clinical severity multipliers from --class_multipliers.
  clinical_m = parse_class_multipliers(
      args.class_multipliers,
      num_labels,
      label2id_int,
  )
  logging.info(f"Clinical multipliers (M_c): {clinical_m.tolist()}")

  class_weights = (w_freq * clinical_m).to(device)
  logging.info(f"Final class weights (W_freq x M_c): {class_weights.tolist()}")

  if len(train_proxy) < args.batch_size:
    fatal(
        f"Training set ({len(train_proxy)} samples) is smaller than "
        f"batch_size ({args.batch_size}). Reduce --batch_size.", ValueError)
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

  model = load_model(
      args.model,
      num_labels=num_labels,
      id2label=train_proxy.id2label,
      label2id=train_proxy.label2id,
      image_size=args.image_size,
      device=device,
      checkpoint_path=args.checkpoint,
      cache_dir=args.cache_dir,
      **args.model_arg,
  )

  # Optionally load weights from a source checkpoint
  if args.source_checkpoint:
    from scdiag.checkpointing import load_checkpoint_weights

    logging.info(f"Loading source checkpoint: {args.source_checkpoint}")
    if args.param_rename:
      logging.info(f"  Key renames: {args.param_rename}")
    load_checkpoint_weights(
        args.source_checkpoint,
        model,
        device=device,
        param_rename=args.param_rename,
    )

  total_params = sum(p.numel() for p in model.parameters())
  trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
  logging.info(f"Model params: {total_params:,} total, {trainable:,} trainable")
  logging.info(f"Model structure:\n{model}")

  if args.focal_gamma > 0 and args.label_smoothing > 0:
    logging.warning(
        "Both --focal_gamma (%.1f) and --label_smoothing (%.2f) are > 0. "
        "Focal loss and label smoothing conflict. Proceeding anyway — "
        "monitor for instability.",
        args.focal_gamma,
        args.label_smoothing,
    )

  criterion = CombinedFocalLoss(
      weights=class_weights,
      gamma=args.focal_gamma,
      label_smoothing=args.label_smoothing,
  )

  optimizer = create_optimizer(
      model.parameters(),
      name=args.optimizer,
      lr=args.lr,
      weight_decay=args.weight_decay,
      **args.opt_arg,
  )

  scaler = (torch.amp.GradScaler("cuda")
            if args.amp_dtype == torch.float16 and device.type == "cuda" else None)

  scheduler = create_scheduler(
      optimizer,
      name=args.scheduler,
      epochs=args.epochs,
      base_lr=args.lr,
      **args.sched_arg,
  )

  ckpt_latest = args.checkpoint + "_latest.pt"
  ckpt_best = args.checkpoint + "_best.pt"

  start_epoch, best_top1 = resume_checkpoint(
      ckpt_latest,
      ckpt_best,
      model,
      optimizer,
      scheduler,
      scaler,
      device,
      states_to_load,
  )

  completed_epoch = start_epoch - 1  # last fully completed (-1 = none yet)
  grad_monitor = None
  if args.grad_monitor >= 0:
    grad_monitor = GradMonitor(model, log_every=args.grad_monitor)
    logging.info(f"Gradient monitoring enabled (every {args.grad_monitor} steps).")
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
          monitor=grad_monitor,
      )

      if scheduler is not None:
        scheduler.step()
      writer.add_scalar("Epoch/Loss_Train", train_loss, epoch)
      writer.add_scalar("Epoch/Accuracy_Train_Top1", train_t1, epoch)

      v_loss, v_t1, v_macro_f1, v_per_class_f1 = evaluate_performance(
          model,
          val_loader,
          criterion,
          device,
          args.amp_dtype,
          id2label=train_proxy.id2label,
      )
      writer.add_scalar("Epoch/Loss_Val", v_loss, epoch)
      writer.add_scalar("Epoch/Accuracy_Val_Top1", v_t1, epoch)
      writer.add_scalar("Epoch/Macro_F1_Val", v_macro_f1, epoch)
      logging.info(f"Epoch {epoch + 1} Results -> "
                   f"Val Loss: {v_loss:.4f} | Top1: {v_t1:.2f}%"
                   f" | Macro F1: {v_macro_f1:.2f}%")
      if v_per_class_f1:
        for cls_name, f1_val in v_per_class_f1.items():
          writer.add_scalar(f"Epoch/F1_Val/{cls_name}", f1_val, epoch)
          logging.info(f"  {cls_name}: F1={f1_val:.2f}%")

      if v_t1 > best_top1:
        best_top1 = v_t1
        save_checkpoint(
            checkpoint_dict(
                model,
                optimizer,
                scheduler,
                epoch,
                states_to_save=states_to_save,
                scaler=scaler,
                best_top1=best_top1,
            ),
            args.checkpoint + "_best.pt",
            remote_uri=args.remote_checkpoint,
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
            states_to_save=states_to_save,
            scaler=scaler,
            best_top1=best_top1,
        ),
        args.checkpoint + "_latest.pt",
        remote_uri=args.remote_checkpoint,
    )
    logging.info("Checkpoint saved on exit.")
    writer.close()

    # Free training model VRAM before XGBoost block.
    del model
    del optimizer
    del scaler
    del scheduler
    del train_loader
    del val_loader
    gc.collect()
    torch.cuda.empty_cache()

    if args.xgboost_model:
      # Access raw HF datasets (before proxy wrapping) for XGBoost.
      train_xgboost_on_backbone(args,
                                train_proxy.dataset,
                                val_proxy.dataset,
                                device,
                                num_labels=num_labels,
                                batch_size=args.batch_size)


if __name__ == "__main__":
  main()
