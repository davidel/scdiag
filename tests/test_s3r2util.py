"""Tests for scripts/s3r2util.py (CloudFlare R2 CLI utility)."""

import os
import sys
from argparse import Namespace

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "scripts"))

import s3r2util  # noqa: E402


class _RecordingS3Client:
  """Minimal stub capturing upload/download/copy/delete calls."""

  def __init__(self):
    self.uploads = []
    self.downloads = []
    self.copies = []
    self.deleted = []

  def upload_file(self, local_path, bucket, key):
    self.uploads.append((local_path, bucket, key))

  def download_file(self, bucket, key, local_path):
    self.downloads.append((bucket, key, local_path))

  def copy(self, copy_source, bucket, key):
    self.copies.append((copy_source["Bucket"], copy_source["Key"], bucket, key))

  def delete_object(self, Bucket, Key):
    self.deleted.append((Bucket, Key))


def _cp_args(sources, destination, quiet=True, progress=False):
  return Namespace(sources=sources,
                   destination=destination,
                   quiet=quiet,
                   progress=progress)


class TestJoinDirDestination:

  def test_trailing_slash(self):
    assert s3r2util.join_dir_destination("r2://destr2/", "bbbb") == ("r2://destr2/bbbb")

  def test_no_trailing_slash(self):
    assert s3r2util.join_dir_destination("r2://destr2", "bbbb") == ("r2://destr2/bbbb")

  def test_local_dir(self):
    assert s3r2util.join_dir_destination("/dest/dir", "bbbb") == ("/dest/dir/bbbb")


class TestCpDirDestination:
  """cp into a directory destination follows Unix cp: basename."""

  def test_local_to_r2_uses_basename(self, tmp_path):
    src = tmp_path / "bbbb"
    src.write_bytes(b"data")
    client = _RecordingS3Client()
    args = _cp_args([str(src)], "r2://destr2/")
    s3r2util.handle_cp(args, client)
    assert client.uploads == [(str(src), "destr2", "bbbb")]

  def test_local_nested_file_uses_basename(self, tmp_path, monkeypatch):
    src = tmp_path / "sub" / "bbbb"
    src.parent.mkdir()
    src.write_bytes(b"data")
    monkeypatch.chdir(tmp_path / "sub")
    client = _RecordingS3Client()
    args = _cp_args([str(src)], "r2://destr2/")
    s3r2util.handle_cp(args, client)
    # Unix cp places the basename in the destination directory.
    assert client.uploads == [(str(src), "destr2", "bbbb")]

  def test_r2_to_r2_uses_basename(self):
    client = _RecordingS3Client()
    args = _cp_args(["r2://srca/aaaa/bbbb"], "r2://destr2/")
    s3r2util.handle_cp(args, client)
    assert client.copies == [("srca", "aaaa/bbbb", "destr2", "bbbb")]

  def test_r2_to_local_uses_basename(self, tmp_path):
    client = _RecordingS3Client()
    dest = str(tmp_path / "destr2") + "/"
    args = _cp_args(["r2://srca/aaaa/bbbb"], dest)
    s3r2util.handle_cp(args, client)
    assert client.downloads == [("srca", "aaaa/bbbb", str(tmp_path / "destr2" / "bbbb"))
                               ]

  def test_local_file_to_r2_file_dest(self, tmp_path):
    src = tmp_path / "bbbb"
    src.write_bytes(b"data")
    client = _RecordingS3Client()
    args = _cp_args([str(src)], "r2://destr2/cccc")
    s3r2util.handle_cp(args, client)
    assert client.uploads == [(str(src), "destr2", "cccc")]

  def test_multiple_sources(self, tmp_path):
    s1 = tmp_path / "s1"
    s2 = tmp_path / "s2"
    s1.write_bytes(b"1")
    s2.write_bytes(b"2")
    client = _RecordingS3Client()
    args = _cp_args([str(s1), str(s2)], "r2://destr2/")
    s3r2util.handle_cp(args, client)
    assert client.uploads == [(str(s1), "destr2", "s1"), (str(s2), "destr2", "s2")]


class TestComputeDestKey:

  def test_appends_relative_to_prefix(self):
    assert s3r2util.compute_dest_key("aaaa/bbbb", "aaaa/", "dest/") == ("dest/bbbb")

  def test_bucket_root_dest_no_leading_slash(self):
    # Regression: used to produce "/bbbb", an invisible-to-prefix-ops key.
    assert s3r2util.compute_dest_key("aaaa/bbbb", "aaaa/", "") == "bbbb"

  def test_bucket_root_dest_nested_relative(self):
    assert s3r2util.compute_dest_key("aaaa/bbbb/cccc", "aaaa/", "") == ("bbbb/cccc")

  def test_root_dest_no_slash_in_relative(self):
    assert s3r2util.compute_dest_key("bbbb", "", "") == "bbbb"

  def test_prefix_dest_normalizes(self):
    assert s3r2util.compute_dest_key("aaaa/bbbb", "aaaa", "dest") == ("dest/bbbb")

  def test_empty_relative_returns_dst_prefix(self):
    assert s3r2util.compute_dest_key("aaaa", "aaaa", "dest/") == "dest/"


class TestMvDirDestination:
  """mv keeps its Unix semantics (regression guard against cp drift)."""

  def test_single_object_into_prefix(self):
    client = _RecordingS3Client()
    args = Namespace(sources=["r2://srca/aaaa/bbbb"],
                     destination="r2://destr2/",
                     quiet=True)
    s3r2util.handle_mv(args, client)
    assert client.copies == [("srca", "aaaa/bbbb", "destr2", "bbbb")]
    assert client.deleted == [("srca", "aaaa/bbbb")]

  def test_rename_to_explicit_key(self):
    client = _RecordingS3Client()
    args = Namespace(sources=["r2://srca/aaaa/bbbb"],
                     destination="r2://destr2/cccc",
                     quiet=True)
    s3r2util.handle_mv(args, client)
    assert client.copies == [("srca", "aaaa/bbbb", "destr2", "cccc")]
