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

from scdiag.checkpointing import resume_checkpoint
from scdiag.datasets.ensemble import DermoscopyEnsemble
from scdiag.gcs_utils import save_checkpoint
from scdiag.gpu_utils import gpu_stats_str
from scdiag.logging_utils import setup_logging
from scdiag.model_utils import DTYPE_MAP, get_backbone
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
      v2.ToTensor(),
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
    if os.path.isdir(name):
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
                    log_every=50):
  """Run one epoch of SimMIM pre-training.

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

    loss.backward()
    torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
    optimizer.step()
    optimizer.zero_grad(set_to_none=True)

    bs = images.shape[0]
    total_loss += loss.item() * bs
    total_samples += bs
    window_samples += bs
    window_loss += loss.item() * bs
    global_step += 1

    elapsed = time.time() - last_log_time
    if global_step % log_every == 0 and global_step > 0:
      throughput = window_samples / elapsed if elapsed > 0 else 0
      lr_now = optimizer.param_groups[0]["lr"]
      w_loss = window_loss / window_samples if window_samples > 0 else 0.0
      avg_loss_so_far = total_loss / total_samples
      gpu = gpu_stats_str(device)
      msg = (f"  [Step {step + 1}/{total_batches}]"
             f" loss={w_loss:.4f} ({avg_loss_so_far:.4f})"
             f" lr={lr_now:.2e} img/s={throughput:.0f}"
             f" mask_ratio={mask_ratio:.2f}")
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
                      action="store_true",
                      default=True,
                      help="Resume training from latest checkpoint if one "
                      "exists.")
  parser.add_argument("--no_resume",
                      dest="resume",
                      action="store_false",
                      help="Start training from scratch, ignoring any "
                      "existing checkpoints.")

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
  parser.add_argument("--vis_every",
                      type=int,
                      default=10,
                      help="Log reconstruction visualisation to TensorBoard "
                      "every N epochs.")

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
    logging.info(gpu_stats_str())

  logging.info("Building dataset ...")
  dataset = build_pretrain_dataset(args)
  logging.info(f"Total images: {len(dataset):,}")
  if len(dataset) == 0:
    logging.error("No images loaded from any dataset. "
                  "Check --datasets, --hf_token, and --cache_dir.")
    return
  loader = DataLoader(
      dataset,
      batch_size=args.batch_size,
      shuffle=True,
      num_workers=args.num_workers,
      pin_memory=(device.type == "cuda"),
      drop_last=True,
  )

  logging.info("Loading model '%s' via registry ...", args.model)
  base_model, _ = load_model(
      args.model,
      num_labels=0,
      id2label={},
      label2id={},
      image_size=args.image_size,
      cache_dir=args.cache_dir,
      device=device,
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
  logging.info(
      f"Model params: {num_params:.1f}M "
      f"(encoder: {enc_params:.1f}M + decoder: {num_params - enc_params:.1f}M)")

  optimizer = optim.AdamW(
      model.parameters(),
      lr=args.lr,
      weight_decay=args.weight_decay,
      betas=(0.9, 0.95),
  )

  if args.warmup_epochs > 0:
    scheduler_warmup = optim.lr_scheduler.LinearLR(
        optimizer,
        start_factor=0.01,
        total_iters=args.warmup_epochs,
    )
    scheduler_cosine = optim.lr_scheduler.CosineAnnealingLR(
        optimizer,
        T_max=args.epochs - args.warmup_epochs,
        eta_min=args.lr * 0.01,
    )
    scheduler = optim.lr_scheduler.SequentialLR(
        optimizer,
        [scheduler_warmup, scheduler_cosine],
        milestones=[args.warmup_epochs],
    )
  else:
    scheduler = optim.lr_scheduler.CosineAnnealingLR(
        optimizer,
        T_max=args.epochs,
        eta_min=args.lr * 0.01,
    )

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
        states_to_load={"opt", "sched"},
    )

  os.makedirs(args.log_dir, exist_ok=True)
  writer = SummaryWriter(log_dir=args.log_dir)

  global_step = start_epoch * len(loader)
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
      )
      writer.add_scalar("Train/loss_epoch", avg_loss, epoch)
      scheduler.step()

      save_checkpoint(
          {
              "model_state_dict": model.state_dict(),
              "optimizer_state_dict": optimizer.state_dict(),
              "scheduler_state_dict": scheduler.state_dict(),
              "epoch": epoch,
              "loss": avg_loss,
          },
          args.checkpoint + "_latest.pt",
          gcs_uri=args.gcs_checkpoint,
      )

      if epoch % args.vis_every == 0:
        log_reconstruction(model, loader, writer, epoch, device)

  except KeyboardInterrupt:
    logging.warning("Interrupt detected!")
  finally:
    save_checkpoint(
        {
            "model_state_dict": model.state_dict(),
            "optimizer_state_dict": optimizer.state_dict(),
            "scheduler_state_dict": scheduler.state_dict(),
            "epoch": epoch if "epoch" in dir() else 0,
            "loss": avg_loss if "avg_loss" in dir() else 0.0,
        },
        args.checkpoint + "_latest.pt",
        gcs_uri=args.gcs_checkpoint,
    )
    logging.info("Checkpoint saved on exit.")
    writer.close()


if __name__ == "__main__":
  main()
