"""Tests for scdiag.optim_factory — optimizer and scheduler factories."""

import pytest
import torch.nn as nn
import torch.optim as optim

from scdiag.optim_factory import create_optimizer, create_scheduler


class TestCreateOptimizer:
  """Tests for the optimizer factory."""

  @pytest.fixture
  def params(self):
    return nn.Linear(10, 2).parameters()

  def test_default_is_adamw(self, params):
    opt = create_optimizer(params)
    assert isinstance(opt, optim.AdamW)

  def test_adam(self, params):
    opt = create_optimizer(params, name="Adam")
    assert isinstance(opt, optim.Adam)

  def test_sgd(self, params):
    opt = create_optimizer(params, name="SGD")
    assert isinstance(opt, optim.SGD)

  def test_exact_match(self, params):
    opt = create_optimizer(params, name="AdamW")
    assert isinstance(opt, optim.AdamW)

  def test_case_sensitive_rejects_lowercase(self, params):
    with pytest.raises(ValueError, match="Unknown optimizer"):
      create_optimizer(params, name="adamw")

  def test_lr_forwarded(self, params):
    opt = create_optimizer(params, lr=0.123)
    assert opt.param_groups[0]["lr"] == 0.123

  def test_weight_decay_forwarded(self, params):
    opt = create_optimizer(params, weight_decay=0.5)
    assert opt.param_groups[0]["weight_decay"] == 0.5

  def test_extra_kwargs_forwarded(self, params):
    opt = create_optimizer(params, name="SGD", momentum=0.9)
    assert opt.param_groups[0]["momentum"] == 0.9

  def test_extra_kwargs_adamw_betas(self, params):
    opt = create_optimizer(params, betas=(0.9, 0.999))
    assert opt.param_groups[0]["betas"] == (0.9, 0.999)

  def test_unknown_optimizer_raises(self, params):
    with pytest.raises(ValueError, match="Unknown optimizer"):
      create_optimizer(params, name="nonexistent")


class TestCreateScheduler:
  """Tests for the scheduler factory."""

  @pytest.fixture
  def optimizer(self):
    model = nn.Linear(10, 2)
    return optim.AdamW(model.parameters(), lr=1e-3)

  def test_default_returns_none(self, optimizer):
    sched = create_scheduler(optimizer)
    assert sched is None

  def test_none_returns_none(self, optimizer):
    sched = create_scheduler(optimizer, name=None)
    assert sched is None

  def test_cosine(self, optimizer):
    sched = create_scheduler(optimizer, name="CosineAnnealingLR", T_max=50,
                             eta_min=1e-5)
    assert isinstance(sched, optim.lr_scheduler.CosineAnnealingLR)
    assert sched.T_max == 50
    assert sched.eta_min == pytest.approx(1e-5)

  def test_cosine_requires_t_max(self, optimizer):
    with pytest.raises(TypeError):
      create_scheduler(optimizer, name="CosineAnnealingLR")

  def test_step(self, optimizer):
    sched = create_scheduler(optimizer, name="StepLR", step_size=10, gamma=0.5)
    assert isinstance(sched, optim.lr_scheduler.StepLR)
    assert sched.step_size == 10
    assert sched.gamma == 0.5

  def test_case_sensitive_rejects_lowercase(self, optimizer):
    with pytest.raises(ValueError, match="Unknown scheduler"):
      create_scheduler(optimizer, name="cosine")

  def test_unknown_scheduler_raises(self, optimizer):
    with pytest.raises(ValueError, match="Unknown scheduler"):
      create_scheduler(optimizer, name="nonexistent")

  def test_custom_scheduler_script(self, optimizer, tmp_path):
    script = tmp_path / "my_sched.py"
    script.write_text("import torch.optim.lr_scheduler as s\n"
                      "def create_scheduler(optimizer, **kw):\n"
                      "    return s.LambdaLR(optimizer, lr_lambda=lambda e: 1.0)\n")
    sched = create_scheduler(optimizer, name=str(script))
    assert isinstance(sched, optim.lr_scheduler.LambdaLR)

  def test_custom_scheduler_missing_fn(self, optimizer, tmp_path):
    script = tmp_path / "bad_sched.py"
    script.write_text("x = 42\n")
    with pytest.raises(ValueError, match="does not define"):
      create_scheduler(optimizer, name=str(script))
