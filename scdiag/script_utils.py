"""Utilities for loading and executing user-supplied Python scripts."""

import logging
import os
import tempfile
import urllib.request

from scdiag.logging_utils import fatal


def _load_script(path_or_url):
  """Fetch, compile and execute a Python script, returning its namespace.

    Handles both local files and HTTP/HTTPS URLs.  For URLs the code is
    written to a named temporary file so that tracebacks show a meaningful
    filename.
  """
  namespace = {}

  if path_or_url.startswith(("http://", "https://")):
    with urllib.request.urlopen(path_or_url) as resp:
      code = resp.read().decode("utf-8")
    with tempfile.NamedTemporaryFile(mode="w",
                                     suffix=".py",
                                     delete=False,
                                     prefix="ext_") as tmp:
      tmp.write(code)
      tmp_path = tmp.name
    try:
      exec(compile(code, path_or_url, "exec"), namespace)  # noqa: S102
    finally:
      os.unlink(tmp_path)
  else:
    with open(path_or_url) as f:
      code = f.read()
    exec(compile(code, path_or_url, "exec"), namespace)  # noqa: S102

  return namespace


def _extract_fn(namespace, path_or_url, fn_name):
  """Extract a callable *fn_name* from a script namespace."""
  fn = namespace.get(fn_name)
  if fn is None or not callable(fn):
    fatal(
        f"Script {path_or_url!r} does not define a callable '{fn_name}'.",
        ValueError)
  return fn


def load_extern(path_or_url, fn_name):
  """Load a Python script and return a callable extracted from it.

    The script is fetched from *path_or_url* (a local path or HTTP/HTTPS
    URL), compiled and executed, then the function *fn_name* is extracted
    from its namespace and returned.

    Args:
        path_or_url: Local file path or HTTP/HTTPS URL to the script.
        fn_name: Name of the callable to extract from the script.

    Returns:
        The extracted callable.

    Raises:
        FileNotFoundError: If a local path does not exist.
        ValueError: If the script does not define a callable *fn_name*.
  """
  namespace = _load_script(path_or_url)
  logging.info("Loading %s from %s", fn_name, path_or_url)
  return _extract_fn(namespace, path_or_url, fn_name)


def extern_call(path_or_url, fn_name, *args, **kwargs):
  """Load a Python script and call one of its top-level functions.

    The script is fetched from *path_or_url* (a local path or HTTP/HTTPS
    URL), compiled and executed, then the function *fn_name* is extracted
    from its namespace and called with the supplied positional and keyword
    arguments.

    Args:
        path_or_url: Local file path or HTTP/HTTPS URL to the script.
        fn_name: Name of the callable to extract from the script.
        *args: Positional arguments forwarded to the callable.
        **kwargs: Keyword arguments forwarded to the callable.

    Returns:
        Whatever *fn_name* returns.

    Raises:
        FileNotFoundError: If a local path does not exist.
        ValueError: If the script does not define a callable *fn_name*.
  """
  fn = load_extern(path_or_url, fn_name)
  return fn(*args, **kwargs)
