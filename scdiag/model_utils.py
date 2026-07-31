"""Shared model loading and preprocessing utilities."""

import logging

import torch
from torchvision.transforms import v2
from torchvision.transforms.functional import InterpolationMode
from transformers import AutoImageProcessor, AutoModelForImageClassification

DTYPE_MAP = {
    "float16": torch.float16,
    "bfloat16": torch.bfloat16,
}


def build_val_transform(processor, image_size: int):
  """Validation/inference transform matching train.py's val pipeline."""

  resize = v2.Resize(
      (image_size, image_size),
      interpolation=InterpolationMode.BICUBIC,
  )
  to_image = v2.ToImage()
  to_float = v2.ToDtype(torch.float32, scale=True)
  normalize = v2.Normalize(
      mean=processor.image_mean, std=processor.image_std
  )

  def _transform(image):
    return normalize(to_float(to_image(resize(image))))

  return _transform


def load_model_for_inference(
    model_name: str,
    checkpoint_path: str,
    device: str = "cuda",
    cache_dir: str | None = None,
):
  """Load a fine-tuned model ready for inference.

  Args:
      model_name: HuggingFace model name or local path defining architecture.
      checkpoint_path: Path to our .pt checkpoint containing model_state_dict.
      device: Target device.
      cache_dir: Optional HF cache directory.

  Returns:
      (model, processor) tuple.
  """
  ckpt = torch.load(checkpoint_path, map_location="cpu")
  state_dict = ckpt["model_state_dict"]

  # Infer num_labels from the checkpoint's classifier head.
  num_labels = state_dict["classifier.weight"].shape[0]

  processor = AutoImageProcessor.from_pretrained(
      model_name, cache_dir=cache_dir
  )
  model = AutoModelForImageClassification.from_pretrained(
      model_name, num_labels=num_labels, ignore_mismatched_sizes=True,
      cache_dir=cache_dir
  )
  missing, unexpected = model.load_state_dict(state_dict, strict=False)
  logging.info(
      f"Loaded weights from {checkpoint_path}: "
      f"{len(missing)} missing, {len(unexpected)} unexpected keys"
  )

  # Use checkpoint's id2label if available (new checkpoints), otherwise
  # keep the pretrained model's default id2label (generic LABEL_0..N).
  if "id2label" in ckpt:
    model.config.id2label = ckpt["id2label"]
    model.config.label2id = {v: k for k, v in ckpt["id2label"].items()}
  model.to(device).eval()
  return model, processor
