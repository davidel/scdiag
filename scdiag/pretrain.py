"""SimMIM self-supervised pre-training for ConvViT.

Usage::

    scdiag-pretrain \\
        --datasets "HAM10000" "redlessone/Derm1M" \\
        --image_size 448 \\
        --batch_size 32 \\
        --epochs 200 \\
        --output_dir ./checkpoints/pretrain

The script trains the full ConvViT encoder (ConvNet stem + transformer) to
reconstruct masked image patches.  After training, the encoder weights can
be loaded into a classification model via ``--pretrained_encoder`` in
``scdiag-train``.
"""

import argparse
import gc
import logging
import os
import time

import numpy as np
import torch
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter
from torchvision.transforms import v2
from torchvision.transforms.functional import InterpolationMode

from scdiag.checkpointing import (
    checkpoint_dict,
    parse_state_flags,
    resume_checkpoint,
)
from scdiag.cli_utils import KVPairAction
from scdiag.datasets.ensemble import DermoscopyEnsemble
from scdiag.gcs_utils import save_checkpoint
from scdiag.gpu_utils import gpu_stats_str
from scdiag.logging_utils import fatal, setup_logging
from scdiag.model_utils import DTYPE_MAP, get_backbone
from scdiag.grad_monitor import GradMonitor
from scdiag.optim_factory import create_optimizer, create_scheduler
from scdiag.models.convvit.simmim import (
    ConvViTSimMIM,
    patchify,
    random_mask,
    simmim_loss,
)
from scdiag.models.registry import load_model


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
  """Apply a transform to a PIL image returned by DermoscopyEnsemble."""

  def __init__(self, dataset, transform):
    self._dataset = dataset
    self._transform = transform

  def __len__(self):
    return len(self._dataset)

  def __getitem__(self, idx):
    return self._transform(self._dataset[idx])


def build_pretrain_dataset(args):
  """Build the DermoscopyEnsemble + transform pipeline."""
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
          "min_resolution": 224,
      })

  if not configs:
    raise ValueError("No datasets specified. Use --datasets <name1> <name2> ...")

  ensemble = DermoscopyEnsemble(
      configs,
      cache_dir=args.cache_dir,
      hf_token=args.hf_token,
  )
  transform = build_pretrain_transform(args.image_size)
  dataset = _TransformWrapper(ensemble, transform)
  logging.info(ensemble.summary())
  return dataset


def log_reconstruction(model, loader, writer, epoch, device, num_samples=8):
  """Log original / masked / reconstructed image grids to TensorBoard."""
  model.eval()
  images = next(iter(loader))[:num_samples].to(device)
  _, _, H, W = images.shape
  patch_size = 16
  num_patches = (H // patch_size) * (W // patch_size)
  mask = random_mask(images.shape[0],
                     num_patches,
                     mask_ratio=getattr(model, "_last_mask_ratio", 0.60),
                     device=device)

  with torch.no_grad():
    pred, target = model(images, mask)

  from scdiag.models.convvit.simmim import unpatchify
  target_imgs = unpatchify(target, patch_size=patch_size, img_size=H)
  pred_imgs = unpatchify(pred, patch_size=patch_size, img_size=H)

  masked = images.clone()
  p = patch_size
  mask_expanded = mask.unsqueeze(-1).expand(-1, -1, p * p * 3)
  mask_expanded = mask_expanded.reshape(images.shape[0], H // p, W // p, p, p, 3)
  mask_expanded = mask_expanded.permute(0, 5, 1, 3, 2, 4).reshape_as(masked)
  masked[mask_expanded.bool()] = 0.0

  writer.add_image("recon/original", images[0], epoch)
  writer.add_image("recon/masked", masked[0], epoch)
  writer.add_image("recon/reconstructed", pred_imgs[0].clamp(0, 1), epoch)
  model.train()


def train_one_epoch(model,
                    loader,
                    optimizer,
                    device,
                    amp_dtype,
                    epoch,
                    global_step,
                    writer,
                    log_every=50,
                    monitor=None,
                    grad_accum_steps=1):
  """Run one epoch of SimMIM pre-training.

    If *grad_accum_steps* > 1, gradients are accumulated over that many
    micro-batches before the optimizer steps and gradients are zeroed.

    Returns ``(avg_loss, global_step)``.
  """
  model.train()
  total_loss = 0.0
  total_samples = 0
  start_time = time.time()
  last_log_time = start_time
  window_samples = 0
  window_loss = 0.0

  num_patches = model.encoder.patch_embed.num_patches
  mask_ratio = getattr(model, "_mask_ratio", 0.60)
  total_batches = len(loader)

  for step, images in enumerate(loader):
    images = images.to(device, non_blocking=True)

    mask = random_mask(images.shape[0],
                       num_patches,
                       mask_ratio=mask_ratio,
                       device=device)

    with torch.amp.autocast(
        "cuda",
        dtype=amp_dtype,
        enabled=(amp_dtype is not None and device.type == "cuda"),
    ):
      pred, target = model(images, mask)
      loss = simmim_loss(pred, target, mask)

    loss = loss / grad_accum_steps
    loss.backward()

    # Step optimizer only every grad_accum_steps batches (or at end of epoch).
    if (step + 1) % grad_accum_steps == 0 or (step + 1) == total_batches:
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
      lr_now = optimizer.param_groups[0]["lr"]
      w_loss = window_loss / window_samples if window_samples > 0 else 0.0
      avg_loss_so_far = total_loss / total_samples
      gpu = gpu_stats_str(device)
      msg = (f"  [Step {step + 1}/{total_batches}]"
             f" loss={w_loss:.4f} ({avg_loss_so_far:.4f})"
             f" lr={lr_now:.2e} img/s={throughput:.0f}"
             f" mask_ratio={mask_ratio:.2f}"
             f" accum={grad_accum_steps}")
      logging.info(msg)
      if gpu:
        logging.info(f"  [Step {step + 1}/{total_batches}] {gpu}")
      if writer is not None:
        writer.add_scalar("Train/loss_step", w_loss, global_step)
        writer.add_scalar("Train/loss_avg", avg_loss_so_far, global_step)
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
      window_loss = 0.0

  avg_loss = total_loss / max(total_samples, 1)
  elapsed = time.time() - start_time
  logging.info(f"  Train stats -> loss: {avg_loss:.4f}"
               f" | time: {elapsed:.1f}s")
  return avg_loss, global_step


def parse_args(argv=None):
  parser = argparse.ArgumentParser(
      description="SimMIM pre-training for ConvViT",
      formatter_class=argparse.ArgumentDefaultsHelpFormatter,
  )

  parser.add_argument("--model",
                      type=str,
                      default="convvit",
                      help="Model name registered in the scdiag registry "
                      "(e.g. 'convvit' or an HuggingFace model ID).")
  parser.add_argument("--datasets",
                      nargs="+",
                      required=True,
                      help="Dataset names or local paths to include in ensemble. "
                      "Use HuggingFace IDs (e.g. 'HAM10000') or directories.")
  parser.add_argument("--cache_dir",
                      type=str,
                      default=None,
                      help="HuggingFace datasets cache directory")
  parser.add_argument("--hf_token",
                      type=str,
                      default=None,
                      help="HuggingFace token for gated datasets (or set HF_TOKEN "
                      "env var)")
  parser.add_argument("--image_size",
                      type=int,
                      default=448,
                      help="Input image size (square)")
  parser.add_argument("--num_workers", type=int, default=4, help="DataLoader workers")

  parser.add_argument("--mask_ratio",
                      type=float,
                      default=0.60,
                      help="Fraction of patches to mask (SimMIM default: 0.60)")
  parser.add_argument("--decoder_dim",
                      type=int,
                      default=768,
                      help="Decoder hidden dimension")
  parser.add_argument("--decoder_depth",
                      type=int,
                      default=2,
                      help="Number of Linear->GELU layers in decoder")

  parser.add_argument("--batch_size",
                      type=int,
                      default=32,
                      help="Per-GPU batch size. Reduce if OOM on "
                      "consumer GPUs at 448px.")
  parser.add_argument("--grad_accum_steps",
                      type=int,
                      default=1,
                      help="Gradient accumulation steps. Effective batch "
                      "size = batch_size * grad_accum_steps.")
  parser.add_argument("--epochs",
                      type=int,
                      default=200,
                      help="Total pre-training epochs.")
  parser.add_argument("--lr",
                      type=float,
                      default=1e-4,
                      help="Peak learning rate for AdamW. Linear warmup "
                      "from 1%% of this value.")
  parser.add_argument("--weight_decay",
                      type=float,
                      default=0.05,
                      help="AdamW weight decay.")
  parser.add_argument("--warmup_epochs",
                      type=int,
                      default=10,
                      help="Linear warmup epochs before cosine schedule.")
  parser.add_argument("--grad_clip",
                      type=float,
                      default=1.0,
                      help="Max gradient norm for clipping.")
  parser.add_argument("--amp_dtype",
                      type=str,
                      default="float16",
                      choices=["float16", "bfloat16", "none"],
                      help="Mixed precision dtype. Use 'none' to disable AMP.")

  parser.add_argument("--checkpoint",
                      type=str,
                      default="./checkpoints/convvit_simmim",
                      help="Checkpoint path prefix (without extension). "
                      "_latest.pt and _best.pt suffixes are appended "
                      "automatically.")
  parser.add_argument("--gcs_checkpoint",
                      type=str,
                      default=None,
                      help="GCS URI to sync checkpoints to "
                      "(format: gs://BUCKET/PREFIX). Requires "
                      "google-cloud-storage package.")
  parser.add_argument("--resume",
                      action=argparse.BooleanOptionalAction,
                      default=True,
                      help="Resume training from latest checkpoint if one "
                      "exists.")
  parser.add_argument("--state_save",
                      type=str,
                      default="opt,sched",
                      help="Comma-separated states to save in checkpoints. "
                      "Allowed: opt, sched, amp, none.")
  parser.add_argument("--state_load",
                      type=str,
                      default="opt,sched",
                      help="Comma-separated states to restore from checkpoints. "
                      "Allowed: opt, sched, amp, none.")

  parser.add_argument("--log_level",
                      type=str,
                      default="INFO",
                      choices=["DEBUG", "INFO", "WARNING", "ERROR"],
                      help="Minimum logging level.")
  parser.add_argument("--log_dir",
                      type=str,
                      default=None,
                      help="TensorBoard log directory. Defaults to "
                      "<checkpoint_dir>/logs if not specified.")
  parser.add_argument("--log_every",
                      type=int,
                      default=50,
                      help="Log training metrics every N optimization steps.")
  parser.add_argument("--grad_monitor",
                      type=int,
                      default=-1,
                      help="Log gradient stats every N steps. "
                      "-1 (default) = disabled.")
  parser.add_argument("--vis_every",
                      type=int,
                      default=10,
                      help="Log reconstruction visualisation to TensorBoard "
                      "every N epochs.")

  parser.add_argument("--model_arg",
                      nargs="+",
                      action=KVPairAction,
                      default={},
                      metavar="KEY=VALUE",
                      help="Override model configuration (repeatable). "
                      "Example: --model_arg depth=6 num_heads=8")
  parser.add_argument("--proc_arg",
                      nargs="+",
                      action=KVPairAction,
                      default={},
                      metavar="KEY=VALUE",
                      help="Override processor configuration (repeatable).")
  parser.add_argument("--optimizer",
                      type=str,
                      default="adamw",
                      help="Optimizer name: adamw (default), adam, sgd.")
  parser.add_argument("--opt_arg",
                      nargs="+",
                      action=KVPairAction,
                      default={},
                      metavar="KEY=VALUE",
                      help="Extra optimizer kwargs (repeatable). "
                      "Example: --opt_arg betas=0.9,0.999 momentum=0.9")
  parser.add_argument("--scheduler",
                      type=str,
                      default="cosine",
                      help="Scheduler name: cosine (default), "
                      "cosine_warmup, step, constant.")
  parser.add_argument("--sched_arg",
                      nargs="+",
                      action=KVPairAction,
                      default={},
                      metavar="KEY=VALUE",
                      help="Extra scheduler kwargs (repeatable). "
                      "Example: --sched_arg T_max=50 eta_min=1e-6")

  args = parser.parse_args(argv)

  if args.log_dir is None:
    args.log_dir = os.path.dirname(args.checkpoint) or "."
    args.log_dir = os.path.join(args.log_dir, "logs")

  if args.hf_token is None:
    args.hf_token = os.environ.get("HF_TOKEN")

  return args


def main(argv=None):
  args = parse_args(argv)
  args.amp_dtype = DTYPE_MAP.get(args.amp_dtype, args.amp_dtype)

  setup_logging(args.log_level)
  logging.info("=" * 60)
  logging.info("SimMIM pre-training for ConvViT")
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
  encoder = get_backbone(base_model)

  model = ConvViTSimMIM(
      encoder,
      decoder_dim=args.decoder_dim,
      decoder_depth=args.decoder_depth,
  ).to(device)
  model._mask_ratio = args.mask_ratio
  model._last_mask_ratio = args.mask_ratio

  num_params = sum(p.numel() for p in model.parameters()) / 1e6
  enc_params = sum(p.numel() for p in encoder.parameters()) / 1e6
  effective_batch = args.batch_size * args.grad_accum_steps
  logging.info(
      f"Model params: {num_params:.1f}M "
      f"(encoder: {enc_params:.1f}M + decoder: {num_params - enc_params:.1f}M)")
  logging.info(f"Effective batch size: {args.batch_size} x {args.grad_accum_steps}"
               f" = {effective_batch}")

  optimizer = create_optimizer(
      model.parameters(),
      name=args.optimizer,
      lr=args.lr,
      weight_decay=args.weight_decay,
      **args.opt_arg,
  )

  scheduler = create_scheduler(
      optimizer,
      name=args.scheduler,
      epochs=args.epochs,
      warmup_epochs=args.warmup_epochs,
      base_lr=args.lr,
      **args.sched_arg,
  )

  states_to_save = parse_state_flags(args.state_save)
  states_to_load = parse_state_flags(args.state_load)

  start_epoch = 0
  if args.resume:
    ckpt_latest = args.checkpoint + "_latest.pt"
    ckpt_best = args.checkpoint + "_best.pt"
    start_epoch, _ = resume_checkpoint(
        ckpt_latest,
        ckpt_best,
        model,
        optimizer,
        scheduler,
        scaler=None,
        device=device,
        states_to_load=states_to_load,
    )

  os.makedirs(args.log_dir, exist_ok=True)
  writer = SummaryWriter(log_dir=args.log_dir)

  completed_epoch = start_epoch - 1  # last fully completed (-1 = none yet)
  global_step = start_epoch * (len(loader) // args.grad_accum_steps)
  grad_monitor = None
  if args.grad_monitor >= 0:
    grad_monitor = GradMonitor(model, log_every=args.grad_monitor)
    logging.info(f"Gradient monitoring enabled (every {args.grad_monitor} steps).")
  try:
    for epoch in range(start_epoch, args.epochs):
      avg_loss, global_step = train_one_epoch(
          model,
          loader,
          optimizer,
          device,
          args.amp_dtype,
          epoch,
          global_step,
          writer,
          log_every=args.log_every,
          monitor=grad_monitor,
          grad_accum_steps=args.grad_accum_steps,
      )
      writer.add_scalar("Train/loss_epoch", avg_loss, epoch)
      scheduler.step()
      completed_epoch = epoch

      save_checkpoint(
          checkpoint_dict(
              model,
              optimizer,
              scheduler,
              completed_epoch,
              states_to_save=states_to_save,
              loss=avg_loss,
          ),
          args.checkpoint + "_latest.pt",
          gcs_uri=args.gcs_checkpoint,
      )

      if epoch % args.vis_every == 0:
        log_reconstruction(model, loader, writer, epoch, device)

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
            loss=avg_loss if "avg_loss" in dir() else 0.0,
        ),
        args.checkpoint + "_latest.pt",
        gcs_uri=args.gcs_checkpoint,
    )
    logging.info("Checkpoint saved on exit.")
    writer.close()


if __name__ == "__main__":
  main()
