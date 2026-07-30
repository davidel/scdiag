"""Tests for WeightedTrainer.log() override.

Since WeightedTrainer inherits from HuggingFace Trainer (which requires
accelerate), we test the log() method in isolation by extracting just
that method and running it on a fake trainer object with the necessary
attributes.
"""

import logging
from types import MethodType
from unittest.mock import MagicMock, patch

import numpy as np
import pytest


def _import_train():
    """Lazy-import train.py so module-level evaluate.load() is mocked."""
    with patch("evaluate.load", return_value=MagicMock()):
        from scdiag import train
    return train


class _FakeTrainer:
    """Bare minimum stand-in with the attributes log() needs."""

    _GPU_KEYS = {"gpu_mem_used_mb", "gpu_mem_reserved_mb", "gpu_util_pct"}
    _METRIC_FORMATS = {
        "loss": ".3f",
        "eval_loss": ".3f",
        "grad_norm": ".3f",
        "learning_rate": ".3e",
        "epoch": ".4f",
        "accuracy": ".2%",
        "eval_accuracy": ".2%",
        "macro_f1": ".2%",
        "eval_macro_f1": ".2%",
        "eval_runtime": ".1f",
        "eval_samples_per_second": ".1f",
        "eval_steps_per_second": ".1f",
    }

    def __init__(self):
        self.args = MagicMock()
        self.callback_handler = MagicMock()
        self.callback_handler.callbacks = []
        self.control = MagicMock()
        self.state = MagicMock()
        self.state.epoch = None
        self.state.global_step = 0
        self.state.log_history = []


class _FakeCallback:
    """Real class so type().__name__ checks work on it."""

    def on_log(self, args, state, control, logs=None, **kwargs):
        return control


# Named to match the production class names the log() method checks for.
ProgressCallback = type("ProgressCallback", (_FakeCallback,), {})
PrinterCallback = type("PrinterCallback", (_FakeCallback,), {})


def _make_weighted_trainer():
    """Create a FakeTrainer with WeightedTrainer.log bound to it."""
    train = _import_train()
    wt = _FakeTrainer()
    wt._weighted_log = MethodType(train.WeightedTrainer.log, wt)
    wt._format_value = MethodType(train.WeightedTrainer._format_value, wt)
    return wt


# ---------------------------------------------------------------------------
# Basic formatting tests
# ---------------------------------------------------------------------------

def test_log_calls_logging_info(caplog):
    wt = _make_weighted_trainer()
    logs = {"loss": 1.5, "learning_rate": 5e-5, "epoch": 0.1}
    wt.state.global_step = 10
    wt.state.epoch = 0.1

    with caplog.at_level(logging.INFO):
        wt._weighted_log(logs)

    assert len(caplog.records) == 1
    msg = caplog.records[0].message
    assert "loss=1.500" in msg
    assert "learning_rate=5.000e-05" in msg
    assert "epoch=0.1000" in msg


def test_loss_uses_fixed_point(caplog):
    wt = _make_weighted_trainer()
    logs = {"loss": 0.4377}
    wt.state.global_step = 1
    wt.state.epoch = None

    with caplog.at_level(logging.INFO):
        wt._weighted_log(logs)

    msg = caplog.records[0].message
    assert "loss=0.438" in msg


def test_grad_norm_uses_fixed_point(caplog):
    wt = _make_weighted_trainer()
    logs = {"loss": 1.0, "grad_norm": 28.86}
    wt.state.global_step = 1
    wt.state.epoch = None

    with caplog.at_level(logging.INFO):
        wt._weighted_log(logs)

    msg = caplog.records[0].message
    assert "grad_norm=28.860" in msg


def test_accuracy_uses_percentage(caplog):
    wt = _make_weighted_trainer()
    logs = {"eval_accuracy": 0.8178}
    wt.state.global_step = 1
    wt.state.epoch = None

    with caplog.at_level(logging.INFO):
        wt._weighted_log(logs)

    msg = caplog.records[0].message
    assert "eval_accuracy=81.78%" in msg


def test_eval_runtime_uses_one_decimal(caplog):
    wt = _make_weighted_trainer()
    logs = {"eval_runtime": 42.43}
    wt.state.global_step = 1
    wt.state.epoch = None

    with caplog.at_level(logging.INFO):
        wt._weighted_log(logs)

    msg = caplog.records[0].message
    assert "eval_runtime=42.4" in msg


def test_unknown_float_falls_back_to_scientific(caplog):
    wt = _make_weighted_trainer()
    logs = {"mystery_metric": 0.001234}
    wt.state.global_step = 1
    wt.state.epoch = None

    with caplog.at_level(logging.INFO):
        wt._weighted_log(logs)

    msg = caplog.records[0].message
    assert "mystery_metric=1.234e-03" in msg


# ---------------------------------------------------------------------------
# GPU keys
# ---------------------------------------------------------------------------

def test_log_no_gpu_keys_without_callback(caplog):
    """Without GPUStatsCallback, gpu_mem should not appear."""
    wt = _make_weighted_trainer()
    logs = {"loss": 1.5}
    wt.state.global_step = 10
    wt.state.epoch = 0.1

    with caplog.at_level(logging.INFO):
        wt._weighted_log(logs)

    msg = caplog.records[0].message
    assert "gpu_mem" not in msg


class _GPUInjectorCallback(_FakeCallback):
    """Mimics GPUStatsCallback: injects gpu_mem and gpu_util keys into logs."""

    def on_log(self, args, state, control, logs=None, **kwargs):
        if logs is not None:
            logs["gpu_mem_used_mb"] = 1060.0
            logs["gpu_mem_reserved_mb"] = 16200.0
            logs["gpu_util_pct"] = 85.0
        return control


def test_log_includes_gpu_when_callback_injects(caplog):
    """GPUStatsCallback injects gpu_mem and gpu_util; log() should pick them up."""
    wt = _make_weighted_trainer()
    wt.callback_handler.callbacks = [_GPUInjectorCallback()]

    logs = {"loss": 1.5}
    wt.state.global_step = 10
    wt.state.epoch = 0.1

    with caplog.at_level(logging.INFO):
        wt._weighted_log(logs)

    msg = caplog.records[0].message
    assert "gpu_mem=1060/16200 MB" in msg
    assert "gpu_util=85%" in msg


# ---------------------------------------------------------------------------
# Callback behavior
# ---------------------------------------------------------------------------

def test_progress_and_printer_callbacks_not_called():
    """ProgressCallback and PrinterCallback should NOT be invoked."""
    wt = _make_weighted_trainer()

    progress_cb = ProgressCallback()
    printer_cb = PrinterCallback()

    progress_cb.on_log = MagicMock()
    printer_cb.on_log = MagicMock()

    wt.callback_handler.callbacks = [progress_cb, printer_cb]

    logs = {"loss": 1.0}
    wt.state.global_step = 5
    wt.state.epoch = 0.05

    wt._weighted_log(logs)

    progress_cb.on_log.assert_not_called()
    printer_cb.on_log.assert_not_called()


def test_custom_callback_is_still_called():
    """Non-skip callbacks should still receive on_log."""
    wt = _make_weighted_trainer()

    custom_cb = MagicMock()
    custom_cb.on_log.return_value = wt.control
    wt.callback_handler.callbacks = [custom_cb]

    logs = {"loss": 1.0}
    wt.state.global_step = 5
    wt.state.epoch = 0.05

    wt._weighted_log(logs)

    custom_cb.on_log.assert_called_once()


# ---------------------------------------------------------------------------
# Log history
# ---------------------------------------------------------------------------

def test_log_records_to_log_history():
    wt = _make_weighted_trainer()
    logs = {"loss": 2.0, "learning_rate": 1e-4, "epoch": 0.2}
    wt.state.global_step = 20
    wt.state.epoch = 0.2

    wt._weighted_log(logs)

    assert len(wt.state.log_history) == 1
    entry = wt.state.log_history[0]
    assert entry["loss"] == 2.0
    assert entry["step"] == 20
    assert entry["epoch"] == 0.2


# ---------------------------------------------------------------------------
# Minimal / edge cases
# ---------------------------------------------------------------------------

def test_log_minimal_keys_no_crash(caplog):
    """log() should not crash if only loss is present."""
    wt = _make_weighted_trainer()
    logs = {"loss": 1.0}
    wt.state.global_step = 1
    wt.state.epoch = None

    with caplog.at_level(logging.INFO):
        wt._weighted_log(logs)

    msg = caplog.records[0].message
    assert "loss=1.000" in msg
    assert "gpu_mem=" not in msg


def test_log_output_format_is_clean(caplog):
    """The log line should be pipe-delimited, not a dict repr."""
    wt = _make_weighted_trainer()
    logs = {"loss": 1.234, "learning_rate": 3e-6, "epoch": 0.33}
    wt.state.global_step = 33
    wt.state.epoch = 0.33

    with caplog.at_level(logging.INFO):
        wt._weighted_log(logs)

    msg = caplog.records[0].message
    assert not msg.startswith("{")
    assert not msg.endswith("}")
    assert " | " in msg


def test_gpu_callback_controls_log_order(caplog):
    """GPU memory should appear in log AFTER callback injection."""
    wt = _make_weighted_trainer()
    wt.callback_handler.callbacks = [_GPUInjectorCallback()]

    logs = {"loss": 1.0}
    wt.state.global_step = 1
    wt.state.epoch = None

    with caplog.at_level(logging.INFO):
        wt._weighted_log(logs)

    msg = caplog.records[0].message
    assert "loss=1.000" in msg
    assert "gpu_mem=" in msg


# ---------------------------------------------------------------------------
# Regression tests
# ---------------------------------------------------------------------------

class _StrictCallback:
    """Callback that rejects positional logs arg — mirrors real TrainerCallback."""

    def on_log(self, args, state, control, **kwargs):
        logs = kwargs.get("logs")
        assert logs is not None, "logs must be passed as keyword argument"
        self.seen_logs = logs
        return control


def test_logs_passed_as_keyword_not_positional():
    """Regression: logs must be passed as keyword arg, not positional."""
    train = _import_train()
    wt = _FakeTrainer()
    wt._weighted_log = MethodType(train.WeightedTrainer.log, wt)
    wt._format_value = MethodType(train.WeightedTrainer._format_value, wt)

    strict_cb = _StrictCallback()
    wt.callback_handler.callbacks = [strict_cb]

    logs = {"loss": 1.0}
    wt.state.global_step = 1
    wt.state.epoch = None

    wt._weighted_log(logs)
    assert strict_cb.seen_logs is logs


class _ReturningNoneCallback:
    """Callback whose on_log returns None — some real callbacks do this."""

    def on_log(self, args, state, control, **kwargs):
        return None


def test_control_not_nulled_by_callback_returning_none():
    """Regression: a callback returning None must not overwrite self.control."""
    train = _import_train()
    wt = _FakeTrainer()
    wt._weighted_log = MethodType(train.WeightedTrainer.log, wt)
    wt._format_value = MethodType(train.WeightedTrainer._format_value, wt)

    original_control = wt.control
    wt.callback_handler.callbacks = [_ReturningNoneCallback()]

    logs = {"loss": 1.0}
    wt.state.global_step = 1
    wt.state.epoch = None

    wt._weighted_log(logs)
    assert wt.control is original_control


# ---------------------------------------------------------------------------
# Type-based fallback tests
# ---------------------------------------------------------------------------

def test_integer_values_emitted_as_is(caplog):
    wt = _make_weighted_trainer()
    logs = {"loss": 1.0, "global_step": 42}
    wt.state.global_step = 42
    wt.state.epoch = None

    with caplog.at_level(logging.INFO):
        wt._weighted_log(logs)

    msg = caplog.records[0].message
    assert "global_step=42" in msg


def test_numpy_integer_emitted_as_is(caplog):
    wt = _make_weighted_trainer()
    logs = {"loss": 1.0, "num_items": np.int64(128)}
    wt.state.global_step = 1
    wt.state.epoch = None

    with caplog.at_level(logging.INFO):
        wt._weighted_log(logs)

    msg = caplog.records[0].message
    assert "num_items=128" in msg


def test_string_values_emitted_as_is(caplog):
    wt = _make_weighted_trainer()
    logs = {"loss": 1.0, "status": "training"}
    wt.state.global_step = 1
    wt.state.epoch = None

    with caplog.at_level(logging.INFO):
        wt._weighted_log(logs)

    msg = caplog.records[0].message
    assert "status=training" in msg


def test_known_metrics_use_dict_formats(caplog):
    """Verify that known metrics use _METRIC_FORMATS, not the fallback."""
    wt = _make_weighted_trainer()

    logs = {
        "loss": 0.4377,
        "grad_norm": 28.86,
        "learning_rate": 3.6e-5,
        "epoch": 2.839,
        "eval_accuracy": 0.8178,
        "eval_macro_f1": 0.7998,
        "eval_runtime": 42.43,
    }
    wt.state.global_step = 100
    wt.state.epoch = 2.839

    with caplog.at_level(logging.INFO):
        wt._weighted_log(logs)

    msg = caplog.records[0].message
    # loss/grad_norm: :.3f
    assert "loss=0.438" in msg
    assert "grad_norm=28.860" in msg
    # learning_rate: :.3e
    assert "learning_rate=3.600e-05" in msg
    # epoch: :.4f
    assert "epoch=2.8390" in msg
    # accuracy/f1: :.2%
    assert "eval_accuracy=81.78%" in msg
    assert "eval_macro_f1=79.98%" in msg
    # runtime: :.1f
    assert "eval_runtime=42.4" in msg
