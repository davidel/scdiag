"""Pytest configuration — suppress known third-party warnings."""
import warnings


def pytest_configure(config):
  """Suppress the joblib shared-memory permission warning."""
  warnings.filterwarnings(
      "ignore",
      message=r"\[Errno 13\] Permission denied.*joblib",
      category=UserWarning,
  )
