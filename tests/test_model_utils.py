"""Tests for scdiag.model_utils."""

from scdiag.model_utils import filter_kwargs


class TestFilterKwargs:
  """Tests for filter_kwargs."""

  def test_filters_unknown_keys(self):

    def foo(a, b):
      pass

    result = filter_kwargs(foo, {"a": 1, "b": 2, "c": 3})
    assert result == {"a": 1, "b": 2}

  def test_passes_all_when_var_keyword(self):

    def foo(a, **kwargs):
      pass

    result = filter_kwargs(foo, {"a": 1, "b": 2, "c": 3})
    assert result == {"a": 1, "b": 2, "c": 3}

  def test_empty_kwargs(self):

    def foo(a, b):
      pass

    result = filter_kwargs(foo, {})
    assert result == {}

  def test_no_matching_keys(self):

    def foo(a, b):
      pass

    result = filter_kwargs(foo, {"x": 1, "y": 2})
    assert result == {}

  def test_preserves_values(self):

    def foo(lr, weight_decay):
      pass

    result = filter_kwargs(foo, {"lr": 1e-3, "weight_decay": 0.01, "extra": 42})
    assert result == {"lr": 1e-3, "weight_decay": 0.01}

  def test_does_not_mutate_input(self):

    def foo(a):
      pass

    original = {"a": 1, "b": 2}
    result = filter_kwargs(foo, original)
    assert result == {"a": 1}
    assert original == {"a": 1, "b": 2}
