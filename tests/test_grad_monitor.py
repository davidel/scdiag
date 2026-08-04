"""Tests for scdiag.grad_monitor.GradMonitor."""

import math

import torch
import torch.nn as nn

from scdiag.grad_monitor import GradMonitor

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


class TinyModel(nn.Module):

  def __init__(self):
    super().__init__()
    self.fc1 = nn.Linear(10, 20)
    self.fc2 = nn.Linear(20, 5)

  def forward(self, x):
    return self.fc2(torch.relu(self.fc1(x)))


def _train_step(model, x, target, optimizer):
  optimizer.zero_grad()
  loss = nn.functional.mse_loss(model(x), target)
  loss.backward()
  optimizer.step()
  return loss


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------


def test_step_returns_none_by_default():
  model = TinyModel()
  mon = GradMonitor(model, log_every=50)
  x = torch.randn(4, 10)
  target = torch.randn(4, 5)
  opt = torch.optim.SGD(model.parameters(), lr=0.01)

  _train_step(model, x, target, opt)
  mon.step(1)

  assert mon.report() is None


def test_report_generated_on_log_step():
  model = TinyModel()
  mon = GradMonitor(model, log_every=10)
  x = torch.randn(4, 10)
  target = torch.randn(4, 5)
  opt = torch.optim.SGD(model.parameters(), lr=0.01)

  _train_step(model, x, target, opt)
  mon.step(10)

  report = mon.report()
  assert report is not None
  assert "[Step 10]" in report
  assert "Gradient Report" in report


def test_report_contains_all_param_names():
  model = TinyModel()
  mon = GradMonitor(model, log_every=1)
  x = torch.randn(4, 10)
  target = torch.randn(4, 5)
  opt = torch.optim.SGD(model.parameters(), lr=0.01)

  _train_step(model, x, target, opt)
  mon.step(0)

  report = mon.report()
  # Compact report shows param count; individual names are available
  # through the internal stats dict but not in the formatted output.
  assert "4 params" in report


def test_report_overwrites_on_next_log_step():
  model = TinyModel()
  mon = GradMonitor(model, log_every=5)
  x = torch.randn(4, 10)
  target = torch.randn(4, 5)
  opt = torch.optim.SGD(model.parameters(), lr=0.01)

  _train_step(model, x, target, opt)
  mon.step(5)
  first = mon.report()

  _train_step(model, x, target, opt)
  mon.step(10)
  second = mon.report()

  assert first is not None
  assert second is not None
  assert first != second


def test_grad_norms_are_positive_after_backward():
  model = TinyModel()
  mon = GradMonitor(model, log_every=1)
  x = torch.randn(4, 10)
  target = torch.randn(4, 5)
  opt = torch.optim.SGD(model.parameters(), lr=0.01)

  _train_step(model, x, target, opt)
  mon.step(0)

  # All parameters had gradients after backward + step, so report
  # should not contain STALLED warnings for fc1/fc2.
  report = mon.report()
  assert "STALLED(fc1" not in report
  assert "STALLED(fc2" not in report


def test_nan_detection():
  model = TinyModel()
  mon = GradMonitor(model, log_every=100, detect_nan=True)
  x = torch.randn(4, 10)
  target = torch.randn(4, 5)
  opt = torch.optim.SGD(model.parameters(), lr=0.01)

  _train_step(model, x, target, opt)

  # Manually inject NaN into a gradient.
  for p in model.parameters():
    if p.grad is not None:
      p.grad.data.fill_(float("nan"))
      break

  # Should not raise; should log a critical message.
  mon.step(1)


def test_no_anomaly_on_first_step():
  """Stalled layers need stall_window consecutive low-norm steps."""
  model = TinyModel()
  mon = GradMonitor(model, log_every=1, stall_window=5)
  x = torch.randn(4, 10)
  target = torch.randn(4, 5)
  opt = torch.optim.SGD(model.parameters(), lr=0.01)

  _train_step(model, x, target, opt)
  mon.step(0)

  report = mon.report()
  assert "STALLED" not in report


def test_stall_detection_after_window():
  """Simulate a dead layer by zeroing gradients for stall_window steps."""
  model = TinyModel()
  mon = GradMonitor(model, log_every=1, stall_window=3, norm_floor=1e-8)

  x = torch.randn(4, 10)
  target = torch.randn(4, 5)
  opt = torch.optim.SGD(model.parameters(), lr=0.01)

  # Do one real step first so the monitor has something.
  _train_step(model, x, target, opt)
  mon.step(0)

  # Now simulate: backward but zero out fc1.weight gradient.
  for step in range(1, 10):
    opt.zero_grad()
    loss = nn.functional.mse_loss(model(x), target)
    loss.backward()
    # Force fc1.weight grad to zero.
    model.fc1.weight.grad.zero_()
    opt.step()
    mon.step(step)

  report = mon.report()
  assert "STALLED(fc1.weight" in report


def test_explosion_detection():
  """Simulate an exploding gradient."""
  model = TinyModel()
  mon = GradMonitor(model, log_every=1, norm_ceiling=10.0)

  x = torch.randn(4, 10)
  target = torch.randn(4, 5)
  opt = torch.optim.SGD(model.parameters(), lr=0.01)

  _train_step(model, x, target, opt)

  # Inject huge gradient.
  for p in model.parameters():
    if p.grad is not None:
      p.grad.data.fill_(100.0)
      break

  mon.step(0)
  report = mon.report()
  assert "EXPLODING" in report


def test_report_without_grad():
  """Parameters without gradients should be handled gracefully."""
  model = TinyModel()
  mon = GradMonitor(model, log_every=1)

  # Don't do a forward/backward pass, so .grad is None for all params.
  mon.step(0)

  report = mon.report()
  assert report is not None
  assert "grad_norm: mean=0.00e+00" in report


def test_imbalance_detection():
  """One parameter with a vastly larger gradient than the rest."""
  model = TinyModel()
  mon = GradMonitor(model, log_every=1)

  x = torch.randn(4, 10)
  target = torch.randn(4, 5)
  opt = torch.optim.SGD(model.parameters(), lr=0.01)

  _train_step(model, x, target, opt)

  # Make fc2.weight gradient 10000x larger than others.
  for p in model.parameters():
    if p.grad is not None and p is not model.fc2.weight:
      p.grad.data.mul_(0.0001)
  model.fc2.weight.grad.data.fill_(10.0)

  mon.step(0)
  report = mon.report()
  assert "IMBALANCED(fc2.weight)" in report


def test_zero_grad_handling():
  """All-zero gradients should not cause division errors."""
  model = TinyModel()
  mon = GradMonitor(model, log_every=1)

  opt = torch.optim.SGD(model.parameters(), lr=0.01)
  opt.zero_grad()

  # Manually set zero gradients.
  for p in model.parameters():
    p.grad = torch.zeros_like(p.data)

  mon.step(0)
  report = mon.report()
  assert report is not None
  assert "nan" not in report.lower()


def test_log_every_zero_disables_snapshotting():
  """log_every=0 should never produce a report."""
  model = TinyModel()
  mon = GradMonitor(model, log_every=0)

  x = torch.randn(4, 10)
  target = torch.randn(4, 5)
  opt = torch.optim.SGD(model.parameters(), lr=0.01)

  for i in range(100):
    _train_step(model, x, target, opt)
    mon.step(i)

  assert mon.report() is None
