"""Glog-style logging utilities (copied from conv_vit)."""

import datetime
import logging
import os


class GlogFormatter(logging.Formatter):
  """Glog-style formatter: ``E0924 22:19:15.123456 PID MODULE:LINE] MESSAGE``"""

  LEVEL_MAP = {
      logging.DEBUG: "D",
      logging.INFO: "I",
      logging.WARNING: "W",
      logging.ERROR: "E",
      logging.CRITICAL: "F",
  }

  def format(self, record):
    now = datetime.datetime.now()
    level_char = self.LEVEL_MAP.get(record.levelno, "I")
    month_day = now.strftime("%m%d")
    time_str = now.strftime("%H:%M:%S.%f")
    pid = os.getpid()
    # Glog header: "E0924 22:19:15.123456 12345 train:42]"
    stem = os.path.splitext(record.filename)[0]
    hdr = f"{level_char}{month_day} {time_str} {pid} {stem}:{record.lineno}]"
    msg = record.getMessage()
    lines = msg.split("\n")
    return "\n".join(f"{hdr} {line}" for line in lines)


def setup_logging(level="INFO"):
  """Configure root logger with glog-style formatting.

  Safe to call more than once — the handler is only added when not already
  present.
  """
  root = logging.getLogger()
  root.setLevel(getattr(logging, level))
  if not any(isinstance(h, logging.StreamHandler) for h in root.handlers):
    handler = logging.StreamHandler()
    handler.setFormatter(GlogFormatter())
    root.addHandler(handler)
