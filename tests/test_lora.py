"""Tests for LoRA / PEFT support."""

import torch
import torch.nn as nn
from peft import LoraConfig, PeftModel, get_peft_model

from scdiag.checkpointing import (
    checkpoint_dict,
    deserialize_lora_state,
    serialize_lora_state,
)
from scdiag.model_utils import apply_lora, extract_lora_params, freeze_model


class _TinyModel(nn.Module):

  def __init__(self, in_features=16, out_features=4):
    super().__init__()
    self.fc = nn.Linear(in_features, out_features)

  def forward(self, x):
    return self.fc(x)


def _make_linear_model(in_features=16, out_features=4):
  """Return a tiny model wrapped with LoRA."""
  model = _TinyModel(in_features, out_features)
  config = LoraConfig(
      r=4,
      lora_alpha=8,
      target_modules=["fc"],
      bias="none",
  )
  return get_peft_model(model, config)


class TestSerializeDeserialize:

  def test_round_trip(self):
    model = _make_linear_model()
    blob = serialize_lora_state(model)
    assert isinstance(blob, bytes)
    assert len(blob) > 0

    new_model = _TinyModel()
    restored = deserialize_lora_state(new_model, blob)
    assert isinstance(restored, PeftModel)

    orig_sd = model.state_dict()
    restored_sd = restored.state_dict()
    for key in orig_sd:
      if "lora_" in key:
        assert key in restored_sd, f"Missing adapter key after restore: {key}"
        assert torch.equal(orig_sd[key],
                           restored_sd[key]), (f"Adapter mismatch for {key}")

  def test_serialize_non_peftmodel_raises(self):
    model = nn.Linear(10, 5)
    try:
      serialize_lora_state(model)
      assert False, "Expected TypeError"
    except TypeError as e:
      assert "Expected a PeftModel" in str(e)


class TestApplyLora:

  def test_returns_peft_model(self):
    model = _TinyModel()
    wrapped = apply_lora(model, r=4, alpha=8, target_modules=["fc"])
    assert isinstance(wrapped, PeftModel)

  def test_freezes_base_params(self):
    model = _TinyModel()
    wrapped = apply_lora(model, r=4, alpha=8, target_modules=["fc"])
    for name, p in wrapped.named_parameters():
      if "lora_" not in name:
        assert not p.requires_grad, (f"Non-LoRA param {name} should be frozen")

  def test_lora_params_are_trainable(self):
    model = _TinyModel()
    wrapped = apply_lora(model, r=4, alpha=8, target_modules=["fc"])
    lora_params = [p for n, p in wrapped.named_parameters() if "lora_" in n]
    assert len(lora_params) > 0
    assert all(p.requires_grad for p in lora_params)


class TestCheckpointDictLoraStripping:

  def test_lora_keys_stripped_when_blob_present(self):
    model = _make_linear_model()
    optimizer = torch.optim.Adam(model.parameters())
    blob = b"fake-blob"

    d = checkpoint_dict(
        model,
        optimizer,
        scheduler=None,
        epoch=0,
        states_to_save=set(),
        lora_state_blob=blob,
    )

    for key in d["model_state_dict"]:
      assert "lora_" not in key, (
          f"LoRA key {key} should be stripped from model_state_dict")

  def test_lora_keys_kept_when_no_blob(self):
    model = _make_linear_model()
    optimizer = torch.optim.Adam(model.parameters())

    d = checkpoint_dict(
        model,
        optimizer,
        scheduler=None,
        epoch=0,
        states_to_save=set(),
    )

    lora_keys = [k for k in d["model_state_dict"] if "lora_" in k]
    assert len(lora_keys) > 0, "LoRA keys should be present without blob"


class TestResumeCheckpointWithLoRA:
  """End-to-end: save checkpoint with LoRA blob, resume, verify weights."""

  def test_resume_restores_lora_weights(self, tmp_path):
    from scdiag.checkpointing import resume_checkpoint

    model = _make_linear_model()
    optimizer = torch.optim.Adam(model.parameters())
    blob = serialize_lora_state(model)

    ckpt = checkpoint_dict(
        model,
        optimizer,
        scheduler=None,
        epoch=5,
        states_to_save=set(),
        best_macro_f1=0.9,
        lora_state_blob=blob,
    )
    path = str(tmp_path / "ckpt.pt")
    torch.save(ckpt, path)

    # Fresh model + fresh LoRA
    fresh = _TinyModel()
    fresh = apply_lora(fresh, r=4, alpha=8, target_modules=["fc"])
    fresh_opt = torch.optim.Adam(fresh.parameters())

    restored, epoch, metric, _extra = resume_checkpoint(
        path,
        path,
        fresh,
        fresh_opt,
        scheduler=None,
        scaler=None,
        device=torch.device("cpu"),
        states_to_load=set(),
    )

    assert epoch == 6
    assert metric == 0.9
    assert isinstance(restored, PeftModel)

    # Adapter weights must match the original
    orig_sd = model.state_dict()
    restored_sd = restored.state_dict()
    for key in orig_sd:
      if "lora_" in key:
        assert key in restored_sd, f"Missing adapter key: {key}"
        assert torch.equal(orig_sd[key],
                           restored_sd[key]), (f"Adapter mismatch for {key}")


class TestExtractLoraParams:
  """Tests for extract_lora_params()."""

  def test_returns_correct_keys(self):
    model = _TinyModel()
    wrapped = apply_lora(model, r=4, alpha=8, target_modules=["fc"])
    params = extract_lora_params(wrapped)
    expected = {
        "base_model.model.fc.lora_A.default.weight",
        "base_model.model.fc.lora_B.default.weight",
    }
    assert params == expected

  def test_excludes_base_layer(self):
    model = _TinyModel()
    wrapped = apply_lora(model, r=4, alpha=8, target_modules=["fc"])
    params = extract_lora_params(wrapped)
    for p in params:
      assert "base_layer" not in p

  def test_excludes_non_adapter_params(self):
    model = _TinyModel()
    wrapped = apply_lora(model, r=4, alpha=8, target_modules=["fc"])
    params = extract_lora_params(wrapped)
    all_names = {n for n, _ in wrapped.named_parameters()}
    assert params.issubset(all_names)
    non_lora = {n for n in all_names if n not in params}
    assert non_lora.isdisjoint(params)

  def test_multiple_target_modules(self):

    class TwoLayer(nn.Module):

      def __init__(self):
        super().__init__()
        self.fc1 = nn.Linear(16, 16)
        self.fc2 = nn.Linear(16, 4)

      def forward(self, x):
        return self.fc2(self.fc1(x))

    wrapped = apply_lora(TwoLayer(), r=4, alpha=8, target_modules=["fc1", "fc2"])
    params = extract_lora_params(wrapped)
    assert len(params) == 4  # lora_A + lora_B for each of fc1, fc2
    for p in params:
      assert "fc1" in p or "fc2" in p


class TestFreezeWithLoRA:
  """Tests for the full freeze_model + extract_lora_params flow."""

  def test_lora_params_thawed_after_freeze(self):
    model = _TinyModel()
    wrapped = apply_lora(model, r=4, alpha=8, target_modules=["fc"])
    lora_names = extract_lora_params(wrapped)
    import re
    patterns = [re.escape(k) for k in lora_names]
    freeze_model(wrapped, tuple(patterns))

    for name, p in wrapped.named_parameters():
      if name in lora_names:
        assert p.requires_grad, f"LoRA param {name} should be trainable"
      else:
        assert not p.requires_grad, (f"Non-LoRA param {name} should be frozen")

  def test_freeze_with_user_pattern_and_lora(self):
    model = _TinyModel()
    wrapped = apply_lora(model, r=4, alpha=8, target_modules=["fc"])
    lora_names = extract_lora_params(wrapped)
    import re
    patterns = [re.escape(k) for k in lora_names]
    # User pattern to also thaw the classifier head.
    patterns.append("fc")
    freeze_model(wrapped, tuple(patterns))

    for name, p in wrapped.named_parameters():
      if name in lora_names or "fc" in name:
        assert p.requires_grad, (f"Expected {name} to be trainable")
      else:
        assert not p.requires_grad, (f"Expected {name} to be frozen")

  def test_only_lora_thawed_without_user_pattern(self):
    """When --lora is set but --freeze is not, only LoRA params are thawed."""
    model = _TinyModel()
    wrapped = apply_lora(model, r=4, alpha=8, target_modules=["fc"])
    lora_names = extract_lora_params(wrapped)
    import re
    patterns = [re.escape(k) for k in lora_names]
    freeze_model(wrapped, tuple(patterns))

    for name, p in wrapped.named_parameters():
      if "lora_" in name:
        assert p.requires_grad, f"LoRA param {name} should be trainable"
      else:
        assert not p.requires_grad, (f"Param {name} should be frozen")
