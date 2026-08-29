"""
Render Free Chroma bootstrap.

Render's Free tier has an ephemeral filesystem: anything written at build or
run time is discarded on every deploy and every idle restart. The validated
Chroma runtime (the ``adtu_knowledge`` collection, 957 vectors) is NOT in
Git -- it is published once as a checksummed GitHub Release asset. This
module restores it on process startup, and ONLY when it is actually missing.

Guarantees (kept deliberately narrow):

  * If the canonical collection is already present and non-empty, this is a
    pure no-op -- the existing database is never opened for writing, never
    overwritten, never touched. This is the normal case for local dev.
  * The downloaded archive's SHA-256 is verified against the expected digest
    BEFORE a single byte is extracted. A mismatch installs nothing.
  * Only archive members under ``data/processed/chroma_db/`` are extracted
    (the sole path the running API needs). Absolute paths, ``..`` traversal,
    and every other member are refused.
  * No embeddings are ever regenerated, ``scripts/chroma_ingest.py`` is never
    invoked, and Gemini is never called. This module moves bytes, nothing
    more.

The default on-disk location (``data/processed/chroma_db`` under the repo
root) is unchanged; ``app/api/main.py`` reads the same path.
"""
from __future__ import annotations

import hashlib
import logging
import os
import shutil
import tempfile
import urllib.request
import zipfile
from pathlib import Path
from urllib.parse import urlparse

logger = logging.getLogger(__name__)

# Canonical collection name -- must match app/api/main.py's get_collection().
CANONICAL_COLLECTION_NAME = "adtu_knowledge"

# Environment variables that carry the snapshot location + expected digest.
SNAPSHOT_URL_ENV = "ADTU_KB_SNAPSHOT_URL"
SNAPSHOT_SHA256_ENV = "ADTU_KB_SNAPSHOT_SHA256"

# The ONLY archive path prefix the running API depends on. derived_embeddings
# .json and every other snapshot member are intentionally not restored here --
# they belong to the offline embedding/ingestion tooling, not the request path.
ALLOWED_MEMBER_PREFIX = "data/processed/chroma_db/"

# Read/verify the download in fixed-size chunks so a large archive never has
# to be held in memory in full.
_CHUNK_BYTES = 1024 * 1024

# Startup download guard -- a snapshot that cannot be fetched in this long is
# treated as a failed bootstrap (the app still starts; /ready reports 503).
_DOWNLOAD_TIMEOUT_SECONDS = 120


class SnapshotChecksumError(RuntimeError):
    """Raised when the downloaded archive's SHA-256 does not match the
    expected digest. Nothing is extracted or installed when this is raised."""


class SnapshotRestoreError(RuntimeError):
    """Raised when the archive verified and extracted, but the resulting
    directory still does not yield a usable canonical collection."""


def collection_is_present(
    chroma_db_path: str | os.PathLike[str],
    collection_name: str = CANONICAL_COLLECTION_NAME,
) -> bool:
    """Return True iff *chroma_db_path* already holds a usable, non-empty
    copy of *collection_name*.

    Uses the real chromadb client when it is importable (production, and the
    project's own virtualenv). Falls back to a conservative on-disk check --
    a non-empty ``chroma.sqlite3`` exists -- only in environments where
    chromadb is not installed (e.g. the lightweight test interpreter), so
    that this module stays importable everywhere.
    """
    db_path = Path(chroma_db_path)
    sqlite_file = db_path / "chroma.sqlite3"
    if not sqlite_file.is_file():
        return False

    try:
        import chromadb
    except ImportError:
        logger.debug(
            "chromadb not importable; using on-disk heuristic for collection presence"
        )
        try:
            return sqlite_file.stat().st_size > 0
        except OSError:
            return False

    try:
        client = chromadb.PersistentClient(path=str(db_path))
        collection = client.get_collection(collection_name)
        return collection.count() > 0
    except Exception as exc:  # collection missing / unreadable / schema error
        logger.debug("collection %r not usable at %s: %s", collection_name, db_path, exc)
        return False


def _clear_chroma_client_cache() -> None:
    """Best-effort drop of chromadb's per-path client/system cache.

    chromadb caches an open SQLite-backed "system" keyed by path+settings.
    If :func:`collection_is_present` opened a client against a stale/broken
    directory that we then replace on disk, a later open of the same path
    would otherwise hand back the cached (now-stale) handle. Called right
    after the snapshot files are moved into place so the re-verify -- and the
    request path's own client -- open the freshly restored database. A no-op
    when chromadb is absent or the cache API differs by version.
    """
    try:
        from chromadb.api.client import SharedSystemClient

        SharedSystemClient.clear_system_cache()
    except Exception as exc:  # not installed / API moved / nothing cached
        logger.debug("chroma client cache clear skipped: %s", exc)


def _sha256_of_file(path: str | os.PathLike[str]) -> str:
    digest = hashlib.sha256()
    with open(path, "rb") as handle:
        for chunk in iter(lambda: handle.read(_CHUNK_BYTES), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _download_snapshot(url: str, dest_path: str | os.PathLike[str]) -> None:
    """Stream *url* to *dest_path*. Only http/https is accepted.

    Isolated into its own function so tests can replace it with a local-file
    copy -- the test suite makes no real network calls.
    """
    scheme = urlparse(url).scheme.lower()
    if scheme not in ("http", "https"):
        raise ValueError(f"Refusing to download snapshot from non-http(s) URL: {url!r}")

    request = urllib.request.Request(
        url, headers={"User-Agent": "adtu-campus-copilot-chroma-bootstrap"}
    )
    with urllib.request.urlopen(  # noqa: S310 - scheme is validated above
        request, timeout=_DOWNLOAD_TIMEOUT_SECONDS
    ) as response, open(dest_path, "wb") as out:
        shutil.copyfileobj(response, out, _CHUNK_BYTES)


def _is_safe_member_name(name: str) -> bool:
    """True iff *name* is a plain relative path with no traversal."""
    if not name or name.startswith("/") or name.startswith("\\"):
        return False
    normalized = os.path.normpath(name).replace("\\", "/")
    if normalized == ".." or normalized.startswith("../") or "/../" in normalized:
        return False
    if os.path.isabs(normalized) or (len(normalized) > 1 and normalized[1] == ":"):
        return False
    return True


def _extract_chroma_db(zip_path: str | os.PathLike[str], staging_root: Path) -> list[str]:
    """Extract only the allow-listed ``data/processed/chroma_db/`` members of
    *zip_path* into *staging_root*. Returns the list of extracted relative
    paths. Unsafe or non-allow-listed members are skipped with a log line.
    """
    extracted: list[str] = []
    with zipfile.ZipFile(zip_path) as archive:
        for member in archive.infolist():
            name = member.filename
            if name.endswith("/"):
                continue  # directory entry
            if not _is_safe_member_name(name):
                logger.warning("snapshot: refusing unsafe archive member %r", name)
                continue
            normalized = os.path.normpath(name).replace("\\", "/")
            if not normalized.startswith(ALLOWED_MEMBER_PREFIX):
                logger.info("snapshot: skipping non-allow-listed member %r", name)
                continue
            target = staging_root / normalized
            target.parent.mkdir(parents=True, exist_ok=True)
            with archive.open(member) as source, open(target, "wb") as out:
                shutil.copyfileobj(source, out, _CHUNK_BYTES)
            extracted.append(normalized)
    return extracted


def ensure_chroma_snapshot(
    chroma_db_path: str | os.PathLike[str],
    *,
    url: str | None,
    sha256: str | None,
    collection_name: str = CANONICAL_COLLECTION_NAME,
) -> str:
    """Ensure the canonical Chroma collection exists at *chroma_db_path*,
    restoring it from the published snapshot only if it is currently missing.

    Returns a short status string:
        "present"      -- collection already usable; nothing done
        "unconfigured" -- collection missing but no snapshot env vars set
        "restored"     -- snapshot verified and installed successfully

    Raises:
        SnapshotChecksumError -- downloaded archive failed SHA-256 verification
                                 (nothing installed)
        SnapshotRestoreError  -- archive installed but collection still unusable
        Exception             -- network / extraction errors propagate to the
                                 caller (startup wrapper logs and continues)
    """
    db_path = Path(chroma_db_path)

    # (1)(2)(10) Already valid -> do nothing. The existing DB is never opened
    # for writing and never overwritten.
    if collection_is_present(db_path, collection_name):
        logger.info(
            "Chroma bootstrap: collection %r already present at %s -- skipping",
            collection_name,
            db_path,
        )
        return "present"

    if not url or not sha256:
        logger.warning(
            "Chroma bootstrap: collection %r missing at %s and %s / %s are not set "
            "-- cannot restore snapshot; /ready will report not-ready",
            collection_name,
            db_path,
            SNAPSHOT_URL_ENV,
            SNAPSHOT_SHA256_ENV,
        )
        return "unconfigured"

    expected_digest = sha256.strip().lower()

    with tempfile.TemporaryDirectory(prefix="adtu-kb-snapshot-") as tmp_name:
        tmp_dir = Path(tmp_name)
        archive_path = tmp_dir / "snapshot.zip"

        logger.info("Chroma bootstrap: downloading snapshot from %s", url)
        _download_snapshot(url, archive_path)

        # (4)(5) Verify BEFORE extracting. A mismatch installs nothing.
        actual_digest = _sha256_of_file(archive_path)
        if actual_digest != expected_digest:
            raise SnapshotChecksumError(
                "Snapshot SHA-256 mismatch: expected "
                f"{expected_digest}, got {actual_digest}. Archive not installed."
            )
        logger.info("Chroma bootstrap: snapshot checksum verified (%s)", actual_digest)

        # (6) Extract only the allow-listed chroma_db subtree, into a staging
        # dir first so a partial extraction never lands on the live path.
        staging_root = tmp_dir / "staging"
        staging_root.mkdir()
        extracted = _extract_chroma_db(archive_path, staging_root)
        staged_db = staging_root / ALLOWED_MEMBER_PREFIX.rstrip("/")
        if not staged_db.is_dir() or not (staged_db / "chroma.sqlite3").is_file():
            raise SnapshotRestoreError(
                "Snapshot did not contain a usable data/processed/chroma_db/ tree "
                f"(extracted {len(extracted)} member(s))."
            )

        # Move into place. We only get here when the collection was NOT
        # already valid, so replacing whatever stale/partial dir exists is
        # safe. shutil.move within the same tree is effectively atomic.
        db_path.parent.mkdir(parents=True, exist_ok=True)
        if db_path.exists():
            logger.info(
                "Chroma bootstrap: replacing stale/invalid directory at %s", db_path
            )
            shutil.rmtree(db_path)
        shutil.move(str(staged_db), str(db_path))
        _clear_chroma_client_cache()
        logger.info(
            "Chroma bootstrap: installed %d snapshot file(s) to %s",
            len(extracted),
            db_path,
        )

    # Re-verify the restored collection is actually usable.
    if not collection_is_present(db_path, collection_name):
        raise SnapshotRestoreError(
            f"Snapshot extracted to {db_path} but collection {collection_name!r} "
            "is still not usable."
        )

    logger.info("Chroma bootstrap: collection %r restored and usable", collection_name)
    return "restored"


def bootstrap_chroma_on_startup(
    chroma_db_path: str | os.PathLike[str],
    *,
    url: str | None = None,
    sha256: str | None = None,
    collection_name: str = CANONICAL_COLLECTION_NAME,
) -> str:
    """Startup entry point: resolve the snapshot config from the environment
    (unless explicitly passed) and run :func:`ensure_chroma_snapshot`,
    swallowing every failure so a bootstrap problem can never stop the API
    process from starting. The failure is logged loudly and surfaces through
    the unchanged ``/ready`` check (which will report Chroma not-ready).

    Returns the status string from :func:`ensure_chroma_snapshot`, or one of
    "checksum_mismatch" / "restore_failed" / "error" when a failure was
    caught here.
    """
    resolved_url = url if url is not None else os.getenv(SNAPSHOT_URL_ENV)
    resolved_sha = sha256 if sha256 is not None else os.getenv(SNAPSHOT_SHA256_ENV)

    try:
        return ensure_chroma_snapshot(
            chroma_db_path,
            url=resolved_url,
            sha256=resolved_sha,
            collection_name=collection_name,
        )
    except SnapshotChecksumError as exc:
        logger.error("Chroma bootstrap FAILED (checksum): %s", exc)
        return "checksum_mismatch"
    except SnapshotRestoreError as exc:
        logger.error("Chroma bootstrap FAILED (restore): %s", exc)
        return "restore_failed"
    except Exception as exc:  # network, zip corruption, disk, ...
        logger.error("Chroma bootstrap FAILED: %s", exc)
        return "error"
