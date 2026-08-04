"""Tests for scdiag.logging_utils."""

import logging
import re

from scdiag.logging_utils import GlogFormatter, setup_logging

# GlogFormatter


class TestGlogFormatter:

  def test_basic_format_contains_level_char(self):
    fmt = GlogFormatter()
    record = logging.LogRecord(
        name="test",
        level=logging.INFO,
        pathname="train.py",
        lineno=42,
        msg="hello %s",
        args=("world",),
        exc_info=None,
    )
    out = fmt.format(record)
    # Should start with the level char 'I' followed by month/day digits.
    assert re.match(r"I\d{4} ", out)
    assert "hello world" in out
    assert "train:42]" in out

  def test_multiline_message(self):
    fmt = GlogFormatter()
    record = logging.LogRecord(
        name="test",
        level=logging.WARNING,
        pathname="foo.py",
        lineno=1,
        msg="line1\nline2\nline3",
        args=(),
        exc_info=None,
    )
    out = fmt.format(record)
    lines = out.split("\n")
    assert len(lines) == 3
    # Every line should carry the glog header.
    for line in lines:
      assert re.match(r"W\d{4} ", line)

  def test_level_characters(self):
    fmt = GlogFormatter()
    expected = {
        logging.DEBUG: "D",
        logging.INFO: "I",
        logging.WARNING: "W",
        logging.ERROR: "E",
        logging.CRITICAL: "F",
    }
    for lvl, char in expected.items():
      record = logging.LogRecord(
          name="t",
          level=lvl,
          pathname="x.py",
          lineno=1,
          msg="m",
          args=(),
          exc_info=None,
      )
      assert fmt.format(record).startswith(char)

  def test_no_args_message(self):
    fmt = GlogFormatter()
    record = logging.LogRecord(
        name="t",
        level=logging.INFO,
        pathname="a.py",
        lineno=10,
        msg="plain",
        args=(),
        exc_info=None,
    )
    out = fmt.format(record)
    assert "plain" in out


# setup_logging


class TestSetupLogging:

  def _clear_root(self):
    root = logging.getLogger()
    root.handlers[:] = []
    return root

  def test_adds_glog_handler(self):
    root = self._clear_root()
    setup_logging()
    assert any(isinstance(h.formatter, GlogFormatter) for h in root.handlers)

  def test_idempotent(self):
    root = self._clear_root()
    setup_logging()
    setup_logging()
    glog_handlers = [h for h in root.handlers if isinstance(h.formatter, GlogFormatter)]
    assert len(glog_handlers) == 1

  def test_replaces_existing_handlers(self):
    root = self._clear_root()
    root.addHandler(logging.StreamHandler())  # plain handler
    setup_logging()
    # The plain handler should be gone.
    assert all(isinstance(h.formatter, GlogFormatter) for h in root.handlers)

  def test_respects_level_arg(self):
    root = self._clear_root()
    setup_logging(level="DEBUG")
    assert root.level == logging.DEBUG

  def test_second_call_updates_level(self):
    root = self._clear_root()
    setup_logging(level="DEBUG")
    setup_logging(level="WARNING")
    assert root.level == logging.WARNING

  def test_actual_log_output(self, capsys):
    root = self._clear_root()
    setup_logging()
    logging.info("test message")
    captured = capsys.readouterr()
    assert "test message" in captured.err
