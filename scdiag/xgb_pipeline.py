"""XGBoost-on-backbone pipeline.

Extracted from ``train.py`` to reduce module size.  Orchestrates loading
the best checkpoint, extracting backbone features, and training / saving
an XGBoost classifier on top.
"""

import gc
import logging

import torch

from scdiag.datasets.hf_proxy import HFDatasetProxy


def train_xgboost_on_backbone(args,
                              train_ds,
                              val_ds,
                              device,
                              num_labels=None,
                              batch_size=32):
  """Train XGBoost on backbone features after PyTorch training completes.

  Args:
      args: Parsed CLI args (contains xgb_* hyperparameters, checkpoint
          paths, etc.)
      train_ds: Training HF Dataset (raw, before proxy wrapping).
      val_ds: Validation HF Dataset (raw, before proxy wrapping).
      device: torch device.
      num_labels: Number of output classes (forwarded to model loader).
      batch_size: Feature-extraction batch size.
  """
  from scdiag.checkpointing import select_best_checkpoint
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
  ckpt_path = select_best_checkpoint(args.checkpoint)
  if ckpt_path is not None:
    logging.info(f"Loading checkpoint: {ckpt_path}")
    model_best, xgb_processor = load_model_for_inference(
        args.model,
        ckpt_path,
        device="cpu",
        cache_dir=args.cache_dir,
        num_labels=num_labels,
        image_size=args.image_size,
        proc_kwargs=args.proc_arg,
    )
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
