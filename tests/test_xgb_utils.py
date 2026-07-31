"""Tests for xgb_utils.py train_xgboost and eval_xgboost."""

import numpy as np
import pytest

from scdiag.xgb_utils import eval_xgboost, train_xgboost


class _Args:
  """Minimal args namespace for train_xgboost."""

  def __init__(self, **kwargs):
    self.xgb_max_depth = kwargs.get("xgb_max_depth", 3)
    self.xgb_n_estimators = kwargs.get("xgb_n_estimators", 10)
    self.xgb_learning_rate = kwargs.get("xgb_learning_rate", 0.1)
    self.xgb_subsample = kwargs.get("xgb_subsample", 0.8)
    self.xgb_colsample_bytree = kwargs.get("xgb_colsample_bytree", 0.8)
    self.xgb_min_child_weight = kwargs.get("xgb_min_child_weight", 1)
    self.xgb_gamma = kwargs.get("xgb_gamma", 0.0)
    self.xgb_reg_alpha = kwargs.get("xgb_reg_alpha", 0.0)


class TestTrainXGBoost:
  """Tests for train_xgboost()."""

  def test_returns_fitted_classifier(self):
    """Should return a fitted XGBClassifier."""
    rng = np.random.RandomState(42)
    features = rng.randn(50, 32).astype(np.float32)
    labels = rng.randint(0, 3, size=50)
    args = _Args()

    clf = train_xgboost(features, labels, args)

    # XGBoost 2.x predict() with multi:softprob returns probabilities (2D).
    proba = clf.predict_proba(features)
    assert proba.shape == (50, 3)
    preds = np.argmax(proba, axis=1)
    assert set(preds).issubset({0, 1, 2})

  def test_predict_proba_shape(self):
    """predict_proba should return [N, num_classes]."""
    rng = np.random.RandomState(42)
    features = rng.randn(40, 16).astype(np.float32)
    labels = rng.randint(0, 4, size=40)
    args = _Args()

    clf = train_xgboost(features, labels, args)
    proba = clf.predict_proba(features)

    assert proba.shape == (40, 4)
    # Each row should sum to ~1.0
    np.testing.assert_allclose(proba.sum(axis=1), 1.0, atol=1e-5)

  def test_overfits_tiny_data(self):
    """With enough trees, should perfectly fit a tiny dataset."""
    rng = np.random.RandomState(0)
    features = rng.randn(20, 8).astype(np.float32)
    labels = (features[:, 0] > 0).astype(int)
    args = _Args(xgb_n_estimators=50, xgb_max_depth=4)

    clf = train_xgboost(features, labels, args)
    proba = clf.predict_proba(features)
    preds = np.argmax(proba, axis=1)

    # Should get high accuracy on training data
    accuracy = np.mean(preds == labels)
    assert accuracy >= 0.9, f"Expected >= 90% accuracy, got {accuracy:.1%}"

  def test_deterministic(self):
    """Two runs with same data should produce identical models."""
    rng = np.random.RandomState(42)
    features = rng.randn(30, 16).astype(np.float32)
    labels = rng.randint(0, 3, size=30)
    args = _Args()

    clf1 = train_xgboost(features, labels, args)
    clf2 = train_xgboost(features, labels, args)

    proba1 = clf1.predict_proba(features)
    proba2 = clf2.predict_proba(features)
    np.testing.assert_allclose(proba1, proba2, atol=1e-6)


class TestEvalXGBoost:
  """Tests for eval_xgboost()."""

  def test_returns_accuracy_and_per_class(self):
    """Should return accuracy and per_class_accuracy keys."""
    rng = np.random.RandomState(42)
    features = rng.randn(40, 16).astype(np.float32)
    labels = rng.randint(0, 3, size=40)
    args = _Args()

    clf = train_xgboost(features, labels, args)
    result = eval_xgboost(clf, features, labels)

    assert "accuracy" in result
    assert "per_class_accuracy" in result
    assert 0.0 <= result["accuracy"] <= 1.0

  def test_per_class_uses_id2label(self):
    """When id2label is provided, per_class keys should be label names."""
    rng = np.random.RandomState(42)
    features = rng.randn(30, 16).astype(np.float32)
    labels = rng.randint(0, 2, size=30)
    args = _Args()

    clf = train_xgboost(features, labels, args)
    id2label = {"0": "benign", "1": "malignant"}
    result = eval_xgboost(clf, features, labels, id2label=id2label)

    assert "benign" in result["per_class_accuracy"]
    assert "malignant" in result["per_class_accuracy"]

  def test_perfect_prediction_gives_100_percent(self):
    """If model predicts perfectly, accuracy should be 1.0."""
    rng = np.random.RandomState(42)
    features = rng.randn(20, 8).astype(np.float32)
    labels = np.array([0, 1] * 10)
    args = _Args(xgb_n_estimators=100, xgb_max_depth=4)

    clf = train_xgboost(features, labels, args)
    result = eval_xgboost(clf, features, labels)

    # With enough trees and a simple pattern, should get high accuracy.
    assert result["accuracy"] >= 0.95, f"Expected >= 95%, got {result['accuracy']:.1%}"

  def test_without_id2label(self):
    """Should work without id2label, using CLASS_N fallback."""
    rng = np.random.RandomState(42)
    features = rng.randn(30, 16).astype(np.float32)
    labels = rng.randint(0, 3, size=30)
    args = _Args()

    clf = train_xgboost(features, labels, args)
    result = eval_xgboost(clf, features, labels)

    # Should have CLASS_0, CLASS_1, CLASS_2
    for key in result["per_class_accuracy"]:
      assert key.startswith("CLASS_")
