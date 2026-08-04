"""Tests for scdiag.optim_factory — optimizer and scheduler factories."""

import pytest
import torch
import torch.nn as nn
import torch.optim as optim

from scdiag.optim_factory import create_optimizer, create_scheduler


# ---------------------------------------------------------------------------
# create_optimizer
# ---------------------------------------------------------------------------


class TestCreateOptimizer:
    """Tests for the optimizer factory."""

    @pytest.fixture
    def params(self):
        return nn.Linear(10, 2).parameters()

    def test_default_is_adamw(self, params):
        opt = create_optimizer(params)
        assert isinstance(opt, optim.AdamW)

    def test_adam(self, params):
        opt = create_optimizer(params, name="adam")
        assert isinstance(opt, optim.Adam)

    def test_sgd(self, params):
        opt = create_optimizer(params, name="sgd")
        assert isinstance(opt, optim.SGD)

    def test_case_insensitive(self, params):
        opt = create_optimizer(params, name="AdamW")
        assert isinstance(opt, optim.AdamW)

    def test_lr_forwarded(self, params):
        opt = create_optimizer(params, lr=0.123)
        assert opt.param_groups[0]["lr"] == 0.123

    def test_weight_decay_forwarded(self, params):
        opt = create_optimizer(params, weight_decay=0.5)
        assert opt.param_groups[0]["weight_decay"] == 0.5

    def test_extra_kwargs_forwarded(self, params):
        opt = create_optimizer(params, name="sgd", momentum=0.9)
        assert opt.param_groups[0]["momentum"] == 0.9

    def test_extra_kwargs_adamw_betas(self, params):
        opt = create_optimizer(params, betas=(0.9, 0.999))
        assert opt.param_groups[0]["betas"] == (0.9, 0.999)

    def test_unknown_optimizer_raises(self, params):
        with pytest.raises(ValueError, match="Unknown optimizer"):
            create_optimizer(params, name="nonexistent")


# ---------------------------------------------------------------------------
# create_scheduler
# ---------------------------------------------------------------------------


class TestCreateScheduler:
    """Tests for the scheduler factory."""

    @pytest.fixture
    def optimizer(self):
        model = nn.Linear(10, 2)
        return optim.AdamW(model.parameters(), lr=1e-3)

    def test_default_is_cosine(self, optimizer):
        sched = create_scheduler(optimizer)
        assert isinstance(sched, optim.lr_scheduler.CosineAnnealingLR)

    def test_cosine_warmup(self, optimizer):
        sched = create_scheduler(optimizer, name="cosine_warmup",
                                 warmup_epochs=5)
        # cosine_warmup + warmup_epochs > 0 produces SequentialLR
        assert isinstance(sched, optim.lr_scheduler.SequentialLR)

    def test_cosine_warmup_no_warmup_epochs(self, optimizer):
        # With warmup_epochs=0, cosine_warmup behaves like plain cosine
        sched = create_scheduler(optimizer, name="cosine_warmup",
                                 warmup_epochs=0)
        assert isinstance(sched, optim.lr_scheduler.CosineAnnealingLR)

    def test_constant(self, optimizer):
        sched = create_scheduler(optimizer, name="constant")
        assert isinstance(sched, optim.lr_scheduler.LambdaLR)

    def test_step(self, optimizer):
        sched = create_scheduler(optimizer, name="step", step_size=10)
        assert isinstance(sched, optim.lr_scheduler.StepLR)

    def test_warmup_epochs_creates_sequential(self, optimizer):
        sched = create_scheduler(optimizer, warmup_epochs=5)
        assert isinstance(sched, optim.lr_scheduler.SequentialLR)

    def test_no_warmup_no_sequential(self, optimizer):
        sched = create_scheduler(optimizer, warmup_epochs=0)
        assert isinstance(sched, optim.lr_scheduler.CosineAnnealingLR)

    def test_cosine_t_max_from_epochs(self, optimizer):
        sched = create_scheduler(optimizer, name="cosine", epochs=200)
        assert sched.T_max == 200

    def test_cosine_eta_min_override(self, optimizer):
        sched = create_scheduler(optimizer, name="cosine", epochs=100,
                                 base_lr=1e-3)
        # Default eta_min = base_lr * 0.01
        assert sched.eta_min == pytest.approx(1e-5)

    def test_unknown_scheduler_raises(self, optimizer):
        with pytest.raises(ValueError, match="Unknown scheduler"):
            create_scheduler(optimizer, name="nonexistent")

    def test_step_scheduler_params(self, optimizer):
        sched = create_scheduler(optimizer, name="step",
                                 step_size=5, gamma=0.5)
        assert sched.step_size == 5
        assert sched.gamma == 0.5
