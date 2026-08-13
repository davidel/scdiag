"""Tests for the classifier registry and ClsModelWrapper."""

import torch
import torch.nn as nn

from scdiag.classifiers import _CLASSIFIERS, build_classifier, register_classifier


class _DummyBackbone(nn.Module):
  """Minimal fake backbone with a config-like attribute."""

  class _Config:
    hidden_size = 32

  def __init__(self):
    super().__init__()
    self.config = self._Config()
    self.linear = nn.Linear(32, 32)

  def forward(self, x):
    # Return an object with last_hidden_state for MLP-style classifiers.
    class _Out:
      pass

    out = _Out()
    out.last_hidden_state = self.linear(x[:, :1, :])  # (B, 1, D)
    return out


class TestRegisterClassifier:

  def test_register_and_retrieve(self):
    """A registered classifier appears in _CLASSIFIERS."""
    name = "_test_dummy_clf"

    @register_classifier(name)
    class DummyClassifier(nn.Module):

      def __init__(self, backbone, num_labels, **kwargs):
        super().__init__()
        self.head = nn.Linear(32, num_labels)

      def forward(self, x):
        return self.head(x[:, 0])

    assert name in _CLASSIFIERS
    assert _CLASSIFIERS[name] is DummyClassifier
    # Cleanup
    del _CLASSIFIERS[name]


class TestBuildClassifier:

  def test_builtin_by_name(self):
    """build_classifier resolves a registered name."""
    backbone = _DummyBackbone()
    model = build_classifier("mlp", backbone, num_labels=5, hidden=16, dropout=0.0)
    assert isinstance(model, nn.Module)
    x = torch.randn(2, 3, 32)
    out = model(x)
    assert out.shape == (2, 5)

  def test_unknown_name_raises(self):
    """Unknown classifier name raises ValueError (via fatal)."""
    backbone = _DummyBackbone()
    with __import__("pytest").raises(ValueError, match="Unknown classifier"):
      build_classifier("nonexistent_clf_xyz", backbone, num_labels=3)

  def test_from_py_file(self, tmp_path):
    """Load a classifier from a .py file."""
    script = tmp_path / "my_clf.py"
    script.write_text("import torch.nn as nn\n"
                      "\n"
                      "\n"
                      "class Classifier(nn.Module):\n"
                      "    def __init__(self, backbone, num_labels, scale=1.0, **kw):\n"
                      "        super().__init__()\n"
                      "        self.head = nn.Linear(32, num_labels)\n"
                      "        self.scale = scale\n"
                      "\n"
                      "    def forward(self, x):\n"
                      "        return self.head(x[:, 0]) * self.scale\n")
    backbone = _DummyBackbone()
    model = build_classifier(str(script), backbone, num_labels=3, scale=2.0)
    assert hasattr(model, "scale")
    assert model.scale == 2.0
    x = torch.randn(2, 3, 32)
    out = model(x)
    assert out.shape == (2, 3)


class TestClsAttention:
  """Tests for the cls_attention classifier slicing."""

  def _make_backbone(self, hidden_size=32, num_tokens=6):
    """Create a dummy backbone returning multi-token output."""

    class _Backbone(nn.Module):

      class _Config:
        pass

      def __init__(self):
        super().__init__()
        self.config = self._Config()
        self.config.hidden_size = hidden_size
        self.linear = nn.Linear(hidden_size, hidden_size)

      def forward(self, x):
        B = x.shape[0]
        out = self.linear(x[:, :1, :])  # (B, 1, D)
        # Repeat to create multiple tokens (simulates CLS + spatial)
        out = out.expand(B, num_tokens, -1).clone()

        class _Out:
          pass

        ret = _Out()
        ret.last_hidden_state = out
        return ret

    return _Backbone()

  def test_default_slices(self):
    """Default cls_slice=(0,1) and spc_slice=(1,) match old hardcoded behavior."""
    backbone = self._make_backbone(hidden_size=16, num_tokens=5)
    model = build_classifier("cls_attention", backbone, num_labels=3)
    assert model.cls_slice == slice(0, 1)
    assert model.spc_slice == slice(1, None)  # (1, None) default
    x = torch.randn(2, 3, 16)
    out = model(x)
    assert out.shape == (2, 3)

  def test_custom_slices(self):
    """Custom slices are stored and used correctly."""
    backbone = self._make_backbone(hidden_size=16, num_tokens=6)
    model = build_classifier(
        "cls_attention",
        backbone,
        num_labels=3,
        cls_slice=(0, 1),
        spc_slice=(1, 4),
    )
    assert model.cls_slice == slice(0, 1)
    assert model.spc_slice == slice(1, 4)
    x = torch.randn(2, 3, 16)
    out = model(x)
    assert out.shape == (2, 3)


class TestColonParsingInLoadModel:

  def test_colon_splits_backbone(self):
    """load_model splits 'name:extra' and passes backbone= kwarg."""
    from unittest.mock import patch

    called_with = {}

    def fake_loader(**kwargs):
      called_with.update(kwargs)
      return nn.Linear(10, 5)

    with patch("scdiag.models.registry._MODEL_REGISTRY", {"test_split": fake_loader}):
      from scdiag.models.registry import load_model
      load_model(
          "test_split:my/awesome-model",
          num_labels=5,
          device="cpu",
      )

    assert called_with["backbone"] == "my/awesome-model"

  def test_no_colon_no_backbone(self):
    """Without ':', no backbone kwarg is passed."""
    from unittest.mock import patch

    called_with = {}

    def fake_loader(**kwargs):
      called_with.update(kwargs)
      return nn.Linear(10, 5)

    with patch("scdiag.models.registry._MODEL_REGISTRY", {"test_nocolon": fake_loader}):
      from scdiag.models.registry import load_model
      load_model(
          "test_nocolon",
          num_labels=5,
          device="cpu",
      )

    assert "backbone" not in called_with
