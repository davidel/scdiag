"""Self-supervised pre-training for ConvViT.

Supports multiple pre-training algorithms via the ``--method`` flag:
``simmim`` (masked-image modeling) and ``ijepa`` (joint-embedding
predictive architecture).

Usage::

    scdiag-pretrain \
        --method simmim \
        --datasets "HAM10000" "redlessone/Derm1M" \
        --image_size 448 \
        --batch_size 32 \
        --epochs 200 \
        --output_dir ./checkpoints/pretrain

After training, the encoder weights can be loaded into a classification
model via ``--source_checkpoint`` in ``scdiag-train``.
"""

import argparse
import logging
import os
import time

import torch
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter
from torchvision.transforms import v2
from torchvision.transforms.functional import InterpolationMode

from scdiag.checkpointing import (
    checkpoint_dict,
    create_model_report,
    parse_state_flags,
    restore_training_state,
    resume_checkpoint,
)
from scdiag.cli_utils import KVPairAction
from scdiag.datasets.ensemble import DatasetEnsemble
from scdiag.gpu_utils import gpu_stats_str
from scdiag.grad_monitor import GradMonitor
from scdiag.logging_utils import fatal, setup_logging
from scdiag.model_utils import set_train_mode
from scdiag.models.registry import load_model
from scdiag.optim_factory import (
    build_param_groups,
    create_optimizer,
    create_scheduler,
    report_lr,
)
from scdiag.pretrain_methods import get_method, list_methods
from scdiag.storage_utils import save_checkpoint


def build_pretrain_transform(image_size=448):
  """Augmentations for SimMIM pre-training."""
  return v2.Compose([
      v2.Resize(image_size, interpolation=InterpolationMode.BICUBIC),
      v2.CenterCrop(image_size),
      v2.RandomHorizontalFlip(p=0.5),
      v2.RandomVerticalFlip(p=0.5),
      v2.ToImage(),
      v2.ToDtype(torch.float32, scale=True),
  ])


class _TransformWrapper:
  """Apply a transform to a PIL image returned by DatasetEnsemble."""

  def __init__(self, dataset, transform):
    self._dataset = dataset
    self._transform = transform

  def __len__(self):
    return len(self._dataset)

  def __getitem__(self, idx):
    return self._transform(self._dataset[idx])


def build_pretrain_dataset(args):
  """Build the DatasetEnsemble + transform pipeline."""
  configs = []
  for name in args.datasets:
    name = name.strip()
    if not name:
      continue
    if name.startswith("imagefolder/"):
      data_dir = name.split("/", 1)[1]
      configs.append({"name": data_dir, "source": "imagefolder"})
    elif os.path.isdir(name):
      configs.append({"name": name, "source": "imagefolder"})
    else:
      configs.append({
          "name": name,
          "source": "hf",
          "split": "train",
          "image_column": args.image_column,
      })

  if not configs:
    fatal("No datasets specified. Use --datasets <name1> <name2> ...", ValueError)

  ensemble = DatasetEnsemble(
      configs,
      cache_dir=args.cache_dir,
      hf_token=args.hf_token,
      strict=args.strict_datasets,
  )
  transform = build_pretrain_transform(args.image_size)
  dataset = _TransformWrapper(ensemble, transform)
  logging.info(ensemble.summary())
  return dataset


def log_validation_images(method,
                          model,
                          loader,
                          writer,
                          global_step,
                          device,
                          num_samples=8):
  """Log method-specific validation images to TensorBoard.

  Calls ``method.validate()`` which returns optional reconstruction
  images.  If the method returns ``None`` (e.g. DINOv2), nothing is
  logged.
  """
  model.eval()
  images = next(iter(loader))[:num_samples].to(device)
  with torch.no_grad():
    recon = method.validate(model, images, num_samples)
  if recon is None:
    return
  # recon is (N, C, H, W) — log first sample.
  writer.add_image("recon/original", images[0], global_step)
  writer.add_image("recon/reconstructed", recon[0].clamp(0, 1), global_step)
  set_train_mode(model, 'train')


def train_one_epoch(
    method,
    model,
    loader,
    optimizer,
    device,
    amp_dtype,
    epoch,
    global_step,
    writer,
    log_every=50,
    vis_every=0,
    monitor=None,
    grad_accum_steps=1,
    scaler=None,
):
  """Run one epoch of self-supervised pre-training.

    If *grad_accum_steps* > 1, gradients are accumulated over that many
    micro-batches before the optimizer steps and gradients are zeroed.

    When *scaler* is provided (float16 AMP), gradient scaling is applied
    to avoid underflow of small gradients.

    Returns ``(avg_loss, global_step)``.
  """
  set_train_mode(model, 'train')
  total_loss = 0.0
  total_samples = 0
  start_time = time.time()
  last_log_time = start_time
  window_samples = 0
  window_loss = 0.0

  total_batches = len(loader)

  for step, images in enumerate(loader):
    images = images.to(device, non_blocking=True)

    with torch.amp.autocast(
        "cuda",
        dtype=amp_dtype,
        enabled=(amp_dtype is not None and device.type == "cuda"),
    ):
      loss, _info = method.train_step(model, images, global_step)

    loss = loss / grad_accum_steps
    if amp_dtype == torch.float16 and scaler is not None:
      scaler.scale(loss).backward()
    else:
      loss.backward()

    # Step optimizer only every grad_accum_steps batches (or at end of epoch).
    if (step + 1) % grad_accum_steps == 0 or (step + 1) == total_batches:
      if amp_dtype == torch.float16 and scaler is not None:
        scaler.unscale_(optimizer)
        # Gradient monitor must read AFTER unscale_() so it sees the true
        # gradient magnitudes, not the scaled values.
        if monitor is not None:
          monitor.step(global_step)
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        scaler.step(optimizer)
        scaler.update()
      else:
        if monitor is not None:
          monitor.step(global_step)
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        optimizer.step()
      optimizer.zero_grad(set_to_none=True)

      global_step += 1

    bs = images.shape[0]
    total_loss += loss.item() * bs * grad_accum_steps
    total_samples += bs
    window_samples += bs
    window_loss += loss.item() * bs * grad_accum_steps

    elapsed = time.time() - last_log_time
    if global_step > 0 and global_step % log_every == 0:
      throughput = window_samples / elapsed if elapsed > 0 else 0
      w_loss = window_loss / window_samples if window_samples > 0 else 0.0
      avg_loss_so_far = total_loss / total_samples
      gpu = gpu_stats_str(device)
      lr_str = report_lr(optimizer, writer=writer, step=global_step)
      msg = (f"  [Step {step + 1}/{total_batches}]"
             f" loss={w_loss:.4f} ({avg_loss_so_far:.4f})"
             f" {lr_str}"
             f" img/s={throughput:.0f}")
      logging.info(msg)
      if gpu:
        logging.info(f"  [Step {step + 1}/{total_batches}] {gpu}")
      if writer is not None:
        writer.add_scalar("Train/loss_step", w_loss, global_step)
        writer.add_scalar("Train/loss_avg", avg_loss_so_far, global_step)
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
      window_loss = 0.0

    if (vis_every > 0 and global_step % vis_every == 0 and writer is not None):
      log_validation_images(method, model, loader, writer, global_step, device)

  avg_loss = total_loss / max(total_samples, 1)
  elapsed = time.time() - start_time
  logging.info(f"  Train stats -> loss: {avg_loss:.4f}"
               f" | time: {elapsed:.1f}s")
  return avg_loss, global_step


def parse_args(argv=None):
  parser = argparse.ArgumentParser(
      description="Self-supervised pre-training for ConvViT",
      formatter_class=argparse.ArgumentDefaultsHelpFormatter,
  )

  available = ", ".join(list_methods()) or "(none)"
  parser.add_argument(
      "--method",
      type=str,
      default="simmim",
      choices=list_methods(),
      help=f"Pre-training method (available: {available}).",
  )
  parser.add_argument(
      "--model",
      type=str,
      default="convvit",
      help="Model name registered in the scdiag registry "
      "(e.g. 'convvit', a HuggingFace model ID, or "
      "'timm:<model_name>' for timm models).",
  )
  parser.add_argument(
      "--datasets",
      nargs="+",
      required=True,
      help="Dataset names or local paths to include in ensemble. "
      "Use HuggingFace IDs (e.g. 'HAM10000') or directories.",
  )
  parser.add_argument(
      "--image_column",
      type=str,
      help="Image column name for HF datasets; auto-detected when omitted.",
  )
  parser.add_argument(
      "--strict_datasets",
      action=argparse.BooleanOptionalAction,
      default=False,
      help="Abort instead of skipping a dataset that fails to load.",
  )
  parser.add_argument(
      "--cache_dir",
      type=str,
      help="HuggingFace datasets cache directory",
  )
  parser.add_argument(
      "--hf_token",
      type=str,
      help="HuggingFace token for gated datasets (or set HF_TOKEN "
      "env var)",
  )
  parser.add_argument("--image_size",
                      type=int,
                      default=448,
                      help="Input image size (square)")
  parser.add_argument("--num_workers", type=int, default=4, help="DataLoader workers")

  parser.add_argument(
      "--batch_size",
      type=int,
      default=32,
      help="Per-GPU batch size. Reduce if OOM on "
      "consumer GPUs at 448px.",
  )
  parser.add_argument(
      "--grad_accum_steps",
      type=int,
      default=1,
      help="Gradient accumulation steps. Effective batch "
      "size = batch_size * grad_accum_steps.",
  )
  parser.add_argument("--epochs",
                      type=int,
                      default=200,
                      help="Total pre-training epochs.")
  parser.add_argument(
      "--lr",
      type=float,
      default=1e-4,
      help="Peak learning rate for AdamW. Linear warmup "
      "from 1%% of this value.",
  )
  parser.add_argument("--weight_decay",
                      type=float,
                      default=0.05,
                      help="AdamW weight decay.")
  parser.add_argument(
      "--lr_group",
      nargs="+",
      default=None,
      metavar="REGEX=LR",
      help="Per-parameter-group learning rates. "
      "E.g. --lr_group 'backbone.*=1e-5' 'classifier.*=1e-3'. "
      "Regexes matched against named_parameters(); first match wins. "
      "Unmatched trainable params use --lr.",
  )
  parser.add_argument("--grad_clip",
                      type=float,
                      default=1.0,
                      help="Max gradient norm for clipping.")
  parser.add_argument(
      "--amp_dtype",
      type=str,
      choices=["float16", "bfloat16"],
      help="AMP dtype for mixed precision. Omit or use --amp_dtype "
      "without a value to disable AMP. float16 requires GradScaler; "
      "bfloat16 is recommended for Ampere+ GPUs.",
  )

  parser.add_argument(
      "--checkpoint",
      type=str,
      default="./checkpoints/convvit_simmim",
      help="Checkpoint path prefix (without extension). "
      "_latest.pt and _best.pt suffixes are appended "
      "automatically.",
  )
  parser.add_argument(
      "--remote_checkpoint",
      type=str,
      help="Remote URI to sync checkpoints to "
      "(format: gs://BUCKET/PREFIX or r2://BUCKET/PREFIX).",
  )
  parser.add_argument(
      "--resume",
      action=argparse.BooleanOptionalAction,
      default=True,
      help="Resume training from latest checkpoint if one "
      "exists.",
  )
  parser.add_argument(
      "--state_save",
      type=str,
      default="opt,sched",
      help="Comma-separated states to save in checkpoints. "
      "Allowed: opt, sched, amp, none.",
  )
  parser.add_argument(
      "--state_load",
      type=str,
      default="opt,sched",
      help="Comma-separated states to restore from checkpoints. "
      "Allowed: opt, sched, amp, none.",
  )

  parser.add_argument(
      "--log_level",
      type=str,
      default="INFO",
      choices=["DEBUG", "INFO", "WARNING", "ERROR"],
      help="Minimum logging level.",
  )
  parser.add_argument(
      "--log_dir",
      type=str,
      help="TensorBoard log directory. Defaults to "
      "<checkpoint_dir>/logs if not specified.",
  )
  parser.add_argument(
      "--log_every",
      type=int,
      default=50,
      help="Log training metrics every N optimization steps.",
  )
  parser.add_argument(
      "--grad_monitor",
      type=int,
      default=-1,
      help="Log gradient stats every N steps. "
      "-1 (default) = disabled.",
  )
  parser.add_argument(
      "--vis_every",
      type=int,
      default=0,
      help="Log reconstruction visualisation to TensorBoard "
      "every N steps. 0 (default) = disabled.",
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
  parser.add_argument(
      "--source_checkpoint",
      type=str,
      help="Path to a source checkpoint to absorb parameters from. "
      "Keys are aligned by shape and name before loading. "
      "Useful for continuing from a prior pre-training run "
      "or loading weights from a different architecture.",
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

  # Two-pass parse: first to get --method, then add its args.
  known, _ = parser.parse_known_args(argv)
  method_cls = get_method(known.method)
  method_cls().add_args(parser)

  args = parser.parse_args(argv)

  if args.log_dir is None:
    args.log_dir = os.path.dirname(args.checkpoint) or "."
    args.log_dir = os.path.join(args.log_dir, "logs")

  if args.hf_token is None:
    args.hf_token = os.environ.get("HF_TOKEN")

  return args


def main(argv=None):
  args = parse_args(argv)
  setup_logging(args.log_level)

  args.amp_dtype = getattr(torch, args.amp_dtype, None) if args.amp_dtype else None

  method_cls = get_method(args.method)
  method = method_cls()

  logging.info("=" * 60)
  logging.info(f"Pre-training method: {args.method}")
  logging.info("=" * 60)
  logging.info(f"Args: {vars(args)}")

  device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
  logging.info(f"Using device: {device}")
  if torch.cuda.is_available():
    logging.info(gpu_stats_str(device))

  logging.info("Building dataset ...")
  dataset = build_pretrain_dataset(args)
  logging.info(f"Total images: {len(dataset):,}")
  if len(dataset) == 0:
    fatal("No images loaded from any dataset. "
          "Check --datasets, --hf_token, and --cache_dir.")
  loader = DataLoader(
      dataset,
      batch_size=args.batch_size,
      shuffle=True,
      num_workers=args.num_workers,
      pin_memory=(device.type == "cuda"),
      drop_last=True,
  )

  logging.info("Loading model '%s' via registry ...", args.model)
  base_model = load_model(
      args.model,
      num_labels=0,
      id2label={},
      label2id={},
      image_size=args.image_size,
      cache_dir=args.cache_dir,
      device=device,
      **args.model_arg,
  )

  model = method.build(args, base_model, device)
  method.load_checkpoint_state(model, {}, args)  # initialize method state

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

  logging.info(f"Effective batch size: {args.batch_size} x {args.grad_accum_steps}"
               f" = {args.batch_size * args.grad_accum_steps}")

  start_epoch = 0
  ckpt_extra = {}
  if args.resume:
    ckpt_latest = args.checkpoint + "_latest.pt"
    ckpt_best = args.checkpoint + "_best.pt"
    model, start_epoch, _, ckpt_extra = resume_checkpoint(
        ckpt_latest,
        ckpt_best,
        model,
        device,
    )
    # Restore method-specific state from checkpoint.
    method_state = ckpt_extra.get("method_state", {})
    method.load_checkpoint_state(model, method_state, args)

  param_groups = build_param_groups(
      dict(model.named_parameters()),
      lr=args.lr,
      weight_decay=args.weight_decay,
      lr_groups=args.lr_group,
  )
  optimizer = create_optimizer(
      param_groups,
      name=args.optimizer,
      **args.opt_arg,
  )

  # Only use GradScaler with float16 AMP (not bfloat16 which has native
  # wider dynamic range and doesn't need loss scaling).
  scaler = (torch.amp.GradScaler("cuda")
            if args.amp_dtype == torch.float16 and device.type == "cuda" else None)
  if scaler is not None:
    logging.info("GradScaler enabled for float16 AMP stability.")

  scheduler = create_scheduler(
      optimizer,
      name=args.scheduler,
      epochs=args.epochs,
      base_lr=args.lr,
      **args.sched_arg,
  )

  states_to_save = parse_state_flags(args.state_save)
  states_to_load = parse_state_flags(args.state_load)

  restore_training_state(
      ckpt_extra,
      optimizer,
      scheduler,
      scaler,
      states_to_load,
  )

  os.makedirs(args.log_dir, exist_ok=True)
  writer = SummaryWriter(log_dir=args.log_dir)

  completed_epoch = start_epoch - 1  # last fully completed (-1 = none yet)
  global_step = ckpt_extra.get(
      "global_step",
      start_epoch * (len(loader) // args.grad_accum_steps),
  )
  del ckpt_extra
  grad_monitor = None
  if args.grad_monitor >= 0:
    grad_monitor = GradMonitor(model, log_every=args.grad_monitor)
    logging.info(f"Gradient monitoring enabled (every {args.grad_monitor} steps).")

  # Report the final model state after checkpoint restoration and all training
  # initialization, immediately before pre-training begins.
  logging.info(create_model_report(model))

  try:
    for epoch in range(start_epoch, args.epochs):
      avg_loss, global_step = train_one_epoch(
          method,
          model,
          loader,
          optimizer,
          device,
          args.amp_dtype,
          epoch,
          global_step,
          writer,
          log_every=args.log_every,
          vis_every=args.vis_every,
          monitor=grad_monitor,
          grad_accum_steps=args.grad_accum_steps,
          scaler=scaler,
      )
      writer.add_scalar("Train/loss_epoch", avg_loss, epoch)
      if scheduler is not None:
        scheduler.step()
      completed_epoch = epoch
      method.on_epoch_end(model, epoch, writer)

      method_state = method.get_checkpoint_state(model, args)
      save_checkpoint(
          checkpoint_dict(
              model,
              optimizer,
              scheduler,
              completed_epoch,
              states_to_save=states_to_save,
              loss=avg_loss,
              global_step=global_step,
              method_state=method_state,
          ),
          args.checkpoint + "_latest.pt",
          remote_uri=args.remote_checkpoint,
      )

  except KeyboardInterrupt:
    logging.warning("Interrupt detected!")
  finally:
    method_state = method.get_checkpoint_state(model, args)
    save_checkpoint(
        checkpoint_dict(
            model,
            optimizer,
            scheduler,
            completed_epoch,
            states_to_save=states_to_save,
            loss=avg_loss if "avg_loss" in dir() else 0.0,
            global_step=global_step,
            method_state=method_state,
        ),
        args.checkpoint + "_latest.pt",
        remote_uri=args.remote_checkpoint,
    )
    logging.info("Checkpoint saved on exit.")
    writer.close()


if __name__ == "__main__":
  main()
