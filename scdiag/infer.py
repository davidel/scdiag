"""Inference script: predict labels for images using a fine-tuned model."""

import argparse
import json
import logging
import os

import numpy as np
import torch
from PIL import Image

from scdiag.logging_utils import setup_logging
from scdiag.model_utils import build_val_transform, load_model_for_inference


def parse_args(argv=None):
  parser = argparse.ArgumentParser(
      description="Classify images with a fine-tuned HF model."
  )
  parser.add_argument(
      "--model",
      required=True,
      help="HuggingFace model name or local HF path (defines architecture + processor).",
  )
  parser.add_argument(
      "--checkpoint",
      required=True,
      help="Path to .pt checkpoint (must contain 'model_state_dict').",
  )
  parser.add_argument(
      "--image_size",
      type=int,
      default=448,
      help="Image resize target (default: 448).",
  )
  parser.add_argument(
      "--output",
      type=str,
      default=None,
      help="Path to write JSON results.",
  )
  parser.add_argument(
      "--top_k",
      type=int,
      default=None,
      help="Only show top-K predictions (default: all).",
  )
  parser.add_argument(
      "--device",
      type=str,
      default=None,
      help="Device: cuda or cpu (default: auto-detect).",
  )
  parser.add_argument(
      "--cache_dir",
      type=str,
      default=None,
      help="HuggingFace cache directory.",
  )
  parser.add_argument(
      "--log_level",
      default="INFO",
      choices=["DEBUG", "INFO", "WARNING", "ERROR"],
      help="Logging level.",
  )
  parser.add_argument(
      "--xgboost_model",
      type=str,
      default=None,
      help="Optional XGBoost model path. If provided, run XGBoost alongside "
      "the PyTorch classifier.",
  )
  parser.add_argument(
      "images",
      nargs="+",
      help="Image file paths and/or URLs.",
  )
  return parser.parse_args(argv)


def open_image(source):
  """Open an image from a file path or URL."""
  if source.startswith(("http://", "https://")):
    import urllib.request
    with urllib.request.urlopen(source) as resp:
      return Image.open(resp).convert("RGB")
  return Image.open(source).convert("RGB")


def predict_single(model, transform, image, device):
  """Run inference on one image. Returns (logits, pixel_values) tensor."""
  pixel_values = transform(image).unsqueeze(0).to(device)
  with torch.no_grad():
    logits = model(pixel_values).logits
  probs = torch.softmax(logits, dim=-1).squeeze()
  return probs, pixel_values


def main(argv=None):
  args = parse_args(argv)
  setup_logging(args.log_level)

  # Device
  if args.device:
    device = args.device
  else:
    device = "cuda" if torch.cuda.is_available() else "cpu"
  logging.info(f"Using device: {device}")

  # Load model
  model, processor = load_model_for_inference(
      model_name=args.model,
      checkpoint_path=args.checkpoint,
      device=device,
      cache_dir=args.cache_dir,
  )
  transform = build_val_transform(processor, args.image_size)

  # Load XGBoost if requested
  xgb_model = None
  if args.xgboost_model:
    from xgboost import XGBClassifier
    xgb_model = XGBClassifier()
    xgb_model.load_model(args.xgboost_model)
    logging.info(f"Loaded XGBoost model: {args.xgboost_model}")

  # Classify each image
  results = []
  for source in args.images:
    image = open_image(source)
    probs, pixel_values = predict_single(model, transform, image, device)

    # Build sorted prediction list
    predictions = []
    for idx in probs.argsort(descending=True).tolist():
      predictions.append({
          "label": model.config.id2label[str(idx)],
          "probability": round(probs[idx].item(), 4),
      })
    if args.top_k:
      predictions = predictions[:args.top_k]

    # Log results
    logging.info(f"\n{source}")
    for p in predictions:
      logging.info(f"  {p['probability']:.1%}  {p['label']}")

    result = {
        "source": source,
        "predictions": predictions,
    }

    # XGBoost inference
    if xgb_model is not None:
      from scdiag.model_utils import extract_features
      with torch.no_grad():
        backbone_out = model.convnextv2(pixel_values)
        features = extract_features(backbone_out)
      xgb_probs = xgb_model.predict_proba(features.cpu().numpy())[0]
      xgb_predictions = []
      for idx in np.argsort(xgb_probs)[::-1].tolist():
        xgb_predictions.append({
            "label": model.config.id2label[str(idx)],
            "probability": round(float(xgb_probs[idx]), 4),
        })
      if args.top_k:
        xgb_predictions = xgb_predictions[:args.top_k]
      result["xgboost_predictions"] = xgb_predictions
      logging.info("  XGBoost:")
      for p in xgb_predictions:
        logging.info(f"  {p['probability']:.1%}  {p['label']}")

    results.append(result)

  # Optional JSON output
  if args.output:
    with open(args.output, "w") as f:
      json.dump(results, f, indent=2)
    logging.info(f"Results written to {args.output}")


if __name__ == "__main__":
  main()
