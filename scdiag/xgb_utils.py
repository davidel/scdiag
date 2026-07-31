"""XGBoost training and evaluation helpers."""

import logging

import numpy as np
from xgboost import XGBClassifier


def train_xgboost(train_features, train_labels, args):
  """Train an XGBoost classifier on backbone features.

  Args:
      train_features: np.ndarray of shape [N, hidden_size].
      train_labels: np.ndarray of shape [N].
      args: Parsed CLI args containing xgb_* hyperparameters.

  Returns:
      Fitted XGBClassifier.
  """
  num_class = len(set(train_labels))
  logging.info(f"Training XGBoost: {len(train_labels)} samples, {num_class} classes")

  # XGBoost requires explicit num_class for multi:softprob.
  xgb_model = XGBClassifier(
      objective="multi:softprob",
      num_class=num_class,
      max_depth=args.xgb_max_depth,
      n_estimators=args.xgb_n_estimators,
      learning_rate=args.xgb_learning_rate,
      subsample=args.xgb_subsample,
      colsample_bytree=args.xgb_colsample_bytree,
      min_child_weight=args.xgb_min_child_weight,
      gamma=args.xgb_gamma,
      reg_alpha=args.xgb_reg_alpha,
      random_state=42,
      verbosity=0,
  )

  xgb_model.fit(train_features, train_labels)
  logging.info("XGBoost training complete")
  return xgb_model


def eval_xgboost(xgb_model, features, labels, id2label=None):
  """Evaluate an XGBoost classifier.

  Args:
      xgb_model: Fitted XGBClassifier.
      features: np.ndarray of shape [N, hidden_size].
      labels: np.ndarray of shape [N].
      id2label: Optional dict mapping class indices to names.

  Returns:
      {"accuracy": float, "per_class_accuracy": dict[str, float]}
  """
  # XGBoost 2.x predict() with multi:softprob returns probabilities (2D),
  # not class labels. Use predict_proba + argmax to get class predictions.
  proba = xgb_model.predict_proba(features)
  predictions = np.argmax(proba, axis=1)
  accuracy = np.mean(predictions == labels)

  per_class = {}
  for cls in sorted(set(labels)):
    mask = labels == cls
    cls_acc = np.mean(predictions[mask] == labels[mask])
    name = id2label[str(cls)] if id2label and str(cls) in id2label else f"CLASS_{cls}"
    per_class[name] = float(cls_acc)

  return {"accuracy": float(accuracy), "per_class_accuracy": per_class}
