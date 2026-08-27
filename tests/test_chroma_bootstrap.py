"""
Tests for app/api/chroma_bootstrap.py -- the Render Free startup snapshot
restore.

All I/O is against temporary directories and the network download is always
mocked (patched :func:`_download_snapshot`). No test makes a real HTTP request
and no test touches chromadb or Gemini -- the collection-presence check is
patched to a pure on-disk predicate so the whole verify/extract/install path
is exercised on the lightweight interpreter.
"""
from __future__ import annotations

import hashlib
import os
import shutil
import tempfile
import unittest
import zipfile
from pathlib import Path
from unittest.mock import patch

from app.api import chroma_bootstrap as cb

_SEGMENT_DIR = "0b461859-8514-4e71-8d14-9bd57243c330"


def _make_snapshot_zip(
    dest: Path,
    *,
    include_traversal_member: bool = False,
    include_non_allowlisted_member: bool = False,
    include_chroma_db: bool = True,
    sqlite_body: bytes = b"SQLITE-FORMAT-3-PLACEHOLDER",
) -> Path:
    """Build a snapshot-shaped zip mirroring the real archive's layout."""
    with zipfile.ZipFile(dest, "w") as archive:
        if include_chroma_db:
            archive.writestr("data/processed/chroma_db/chroma.sqlite3", sqlite_body)
            archive.writestr(
                f"data/processed/chroma_db/{_SEGMENT_DIR}/data_level0.bin", b"\x00\x01\x02"
            )
            archive.writestr(
                f"data/processed/chroma_db/{_SEGMENT_DIR}/header.bin", b"H" * 10
            )
            archive.writestr(
                f"data/processed/chroma_db/{_SEGMENT_DIR}/link_lists.bin", b""
            )
        if include_non_allowlisted_member:
            archive.writestr("data/processed/derived_embeddings.json", b'{"vectors": 1}')
        if include_traversal_member:
            archive.writestr("../evil.txt", b"pwned")
    return dest


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


class ChromaBootstrapTests(unittest.TestCase):
    def setUp(self) -> None:
        self._tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self._tmp.cleanup)
        self.root = Path(self._tmp.name)
        self.db_path = self.root / "data" / "processed" / "chroma_db"

        # Drive presence purely off the on-disk sqlite file: no chromadb,
        # no Gemini, fully deterministic on every interpreter.
        presence = patch.object(
            cb,
            "collection_is_present",
            side_effect=lambda path, name=cb.CANONICAL_COLLECTION_NAME: (
                Path(path) / "chroma.sqlite3"
            ).is_file(),
        )
        presence.start()
        self.addCleanup(presence.stop)

    def _mock_download_from(self, zip_source: Path) -> None:
        def _fake_download(url: str, dest_path) -> None:  # noqa: ANN001
            shutil.copyfile(zip_source, dest_path)

        dl = patch.object(cb, "_download_snapshot", side_effect=_fake_download)
        dl.start()
        self.addCleanup(dl.stop)

    # -- no-op / config guards ------------------------------------------------

    def test_present_collection_is_a_pure_noop(self) -> None:
        self.db_path.mkdir(parents=True)
        (self.db_path / "chroma.sqlite3").write_bytes(b"EXISTING-DB")

        with patch.object(
            cb, "_download_snapshot", side_effect=AssertionError("must not download")
        ):
            status = cb.ensure_chroma_snapshot(
                self.db_path, url="https://example.com/s.zip", sha256="deadbeef"
            )

        self.assertEqual(status, "present")
        self.assertEqual((self.db_path / "chroma.sqlite3").read_bytes(), b"EXISTING-DB")

    def test_missing_collection_without_config_returns_unconfigured(self) -> None:
        self.assertEqual(
            cb.ensure_chroma_snapshot(self.db_path, url=None, sha256=None),
            "unconfigured",
        )
        self.assertFalse(self.db_path.exists())

        self.assertEqual(
            cb.ensure_chroma_snapshot(self.db_path, url="https://x/s.zip", sha256=""),
            "unconfigured",
        )

    # -- happy path ---------------------------------------------------------

    def test_restores_snapshot_when_collection_missing(self) -> None:
        zip_src = _make_snapshot_zip(self.root / "snap.zip")
        digest = _sha256(zip_src)
        self._mock_download_from(zip_src)

        status = cb.ensure_chroma_snapshot(
            self.db_path,
            url="https://example.com/snap.zip",
            sha256=digest.upper(),  # case-insensitive match
        )

        self.assertEqual(status, "restored")
        self.assertTrue((self.db_path / "chroma.sqlite3").is_file())
        self.assertTrue(
            (self.db_path / _SEGMENT_DIR / "data_level0.bin").is_file()
        )
        self.assertTrue((self.db_path / _SEGMENT_DIR / "header.bin").is_file())

    def test_restore_replaces_stale_invalid_directory(self) -> None:
        self.db_path.mkdir(parents=True)
        (self.db_path / "leftover.txt").write_text("stale partial extraction")

        zip_src = _make_snapshot_zip(self.root / "snap.zip")
        self._mock_download_from(zip_src)

        status = cb.ensure_chroma_snapshot(
            self.db_path, url="https://example.com/snap.zip", sha256=_sha256(zip_src)
        )

        self.assertEqual(status, "restored")
        self.assertFalse((self.db_path / "leftover.txt").exists())
        self.assertTrue((self.db_path / "chroma.sqlite3").is_file())

    # -- checksum enforcement --------------------------------------------

    def test_checksum_mismatch_raises_and_installs_nothing(self) -> None:
        zip_src = _make_snapshot_zip(self.root / "snap.zip")
        self._mock_download_from(zip_src)

        with self.assertRaises(cb.SnapshotChecksumError):
            cb.ensure_chroma_snapshot(
                self.db_path,
                url="https://example.com/snap.zip",
                sha256="0" * 64,
            )

        self.assertFalse(self.db_path.exists())

    def test_checksum_mismatch_leaves_existing_invalid_dir_untouched(self) -> None:
        self.db_path.mkdir(parents=True)
        (self.db_path / "old.bin").write_bytes(b"do-not-delete-on-failed-checksum")

        zip_src = _make_snapshot_zip(self.root / "snap.zip")
        self._mock_download_from(zip_src)

        with self.assertRaises(cb.SnapshotChecksumError):
            cb.ensure_chroma_snapshot(
                self.db_path, url="https://example.com/snap.zip", sha256="1" * 64
            )

        self.assertEqual(
            (self.db_path / "old.bin").read_bytes(), b"do-not-delete-on-failed-checksum"
        )

    # -- extraction allow-list / path safety ----------------------------

    def test_only_allowlisted_chroma_db_members_are_extracted(self) -> None:
        zip_src = _make_snapshot_zip(
            self.root / "snap.zip",
            include_traversal_member=True,
            include_non_allowlisted_member=True,
        )
        self._mock_download_from(zip_src)

        status = cb.ensure_chroma_snapshot(
            self.db_path, url="https://example.com/snap.zip", sha256=_sha256(zip_src)
        )

        self.assertEqual(status, "restored")
        # derived_embeddings.json is a build artifact, not needed by the API.
        self.assertFalse((self.root / "data" / "processed" / "derived_embeddings.json").exists())
        # ../evil.txt must not escape anywhere.
        self.assertFalse((self.root / "evil.txt").exists())
        self.assertFalse((self.root.parent / "evil.txt").exists())

    def test_extract_helper_returns_only_allowlisted_members(self) -> None:
        zip_src = _make_snapshot_zip(
            self.root / "snap.zip",
            include_traversal_member=True,
            include_non_allowlisted_member=True,
        )
        staging = self.root / "staging"
        staging.mkdir()

        extracted = cb._extract_chroma_db(zip_src, staging)

        self.assertTrue(extracted)
        for rel in extracted:
            self.assertTrue(rel.startswith(cb.ALLOWED_MEMBER_PREFIX))
        self.assertNotIn("data/processed/derived_embeddings.json", extracted)
        self.assertFalse((staging / "evil.txt").exists())
        self.assertFalse((staging.parent / "evil.txt").exists())

    def test_is_safe_member_name(self) -> None:
        self.assertTrue(
            cb._is_safe_member_name("data/processed/chroma_db/chroma.sqlite3")
        )
        for bad in [
            "",
            "../evil",
            "/etc/passwd",
            "data/../../escape",
            "C:\\Windows\\system32",
            "\\\\server\\share\\x",
        ]:
            with self.subTest(bad=bad):
                self.assertFalse(cb._is_safe_member_name(bad))

    def test_snapshot_without_chroma_db_tree_raises_restore_error(self) -> None:
        zip_src = _make_snapshot_zip(
            self.root / "snap.zip",
            include_chroma_db=False,
            include_non_allowlisted_member=True,
        )
        self._mock_download_from(zip_src)

        with self.assertRaises(cb.SnapshotRestoreError):
            cb.ensure_chroma_snapshot(
                self.db_path, url="https://example.com/snap.zip", sha256=_sha256(zip_src)
            )
        self.assertFalse(self.db_path.exists())

    # -- download guardrails ------------------------------------------------

    def test_clear_chroma_client_cache_is_a_safe_noop(self) -> None:
        # Must never raise, regardless of whether chromadb is importable.
        cb._clear_chroma_client_cache()

    def test_download_rejects_non_http_scheme(self) -> None:
        with self.assertRaises(ValueError):
            cb._download_snapshot("file:///etc/passwd", self.root / "out.zip")
        with self.assertRaises(ValueError):
            cb._download_snapshot("ftp://host/x.zip", self.root / "out.zip")

    # -- startup wrapper never crashes the process ------------------------

    def test_startup_wrapper_swallows_checksum_failure(self) -> None:
        zip_src = _make_snapshot_zip(self.root / "snap.zip")
        self._mock_download_from(zip_src)

        status = cb.bootstrap_chroma_on_startup(
            self.db_path, url="https://example.com/snap.zip", sha256="f" * 64
        )

        self.assertEqual(status, "checksum_mismatch")
        self.assertFalse(self.db_path.exists())

    def test_startup_wrapper_swallows_download_error(self) -> None:
        def _boom(url, dest_path):  # noqa: ANN001
            raise OSError("network unreachable")

        with patch.object(cb, "_download_snapshot", side_effect=_boom):
            status = cb.bootstrap_chroma_on_startup(
                self.db_path, url="https://example.com/snap.zip", sha256="a" * 64
            )

        self.assertEqual(status, "error")
        self.assertFalse(self.db_path.exists())

    def test_startup_wrapper_reads_env_when_args_omitted(self) -> None:
        with patch.dict(
            os.environ,
            {cb.SNAPSHOT_URL_ENV: "", cb.SNAPSHOT_SHA256_ENV: ""},
            clear=False,
        ):
            self.assertEqual(
                cb.bootstrap_chroma_on_startup(self.db_path), "unconfigured"
            )

    def test_startup_wrapper_present_collection_is_noop(self) -> None:
        self.db_path.mkdir(parents=True)
        (self.db_path / "chroma.sqlite3").write_bytes(b"EXISTING")

        with patch.object(
            cb, "_download_snapshot", side_effect=AssertionError("must not download")
        ):
            status = cb.bootstrap_chroma_on_startup(
                self.db_path,
                url="https://example.com/s.zip",
                sha256="b" * 64,
            )
        self.assertEqual(status, "present")


if __name__ == "__main__":
    unittest.main()
