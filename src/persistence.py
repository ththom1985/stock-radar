"""Durable JSON persistence and explicit data-corruption errors."""
from __future__ import annotations

import json
import os
import tempfile
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


SCHEMA_VERSION = 2


class PersistenceError(RuntimeError):
    """Base class for durable-storage failures."""


class CorruptDataError(PersistenceError):
    """Raised when an existing JSON file cannot be decoded or validated."""


def effective_path(path: Path, *, for_write: bool = False) -> Path:
    """Redirect all JSON state during an explicit dry run.

    Reads prefer already-produced dry-run state and otherwise fall back to the
    production source, allowing smoke runs to seed from caches without mutating
    any tracked file.
    """
    path = Path(path)
    if os.environ.get("STOCK_RADAR_DRY_RUN") != "1":
        return path
    override = (
        os.environ.get("STOCK_RADAR_DRY_RUN_DIR")
        or os.environ.get("STOCK_RADAR_OUTPUT_DIR")
        or str(Path.cwd() / "data" / "dry-run")
    )
    root = Path(override).expanduser().resolve()
    resolved = path.resolve()
    try:
        resolved.relative_to(root)
        return resolved
    except ValueError:
        pass
    parts = list(resolved.parts)
    lower = [part.lower() for part in parts]
    relative = Path(path.name)
    if "data" in lower:
        index = len(lower) - 1 - lower[::-1].index("data")
        relative = Path(*parts[index + 1 :])
    redirected = root / relative
    if for_write or redirected.exists():
        return redirected
    return path


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds")


def schema_meta(name: str, **extra: Any) -> dict[str, Any]:
    return {
        "schema": name,
        "schema_version": SCHEMA_VERSION,
        "written_at": utc_now(),
        **extra,
    }


def load_json(
    path: Path,
    *,
    required: bool = False,
    expected_type: type | tuple[type, ...] | None = None,
    default: Any = None,
) -> Any:
    """Load JSON without hiding corruption.

    Missing optional files return ``default``. Existing malformed files always
    raise because treating corruption as an empty cache or portfolio destroys
    provenance and can silently reset user data.
    """
    path = effective_path(Path(path))
    if not path.exists():
        if required:
            raise PersistenceError(f"Required JSON file does not exist: {path}")
        return default
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, UnicodeError, json.JSONDecodeError) as exc:
        raise CorruptDataError(f"Cannot read valid JSON from {path}: {exc}") from exc
    if expected_type is not None and not isinstance(value, expected_type):
        name = (
            ", ".join(t.__name__ for t in expected_type)
            if isinstance(expected_type, tuple)
            else expected_type.__name__
        )
        raise CorruptDataError(f"JSON root in {path} must be {name}, got {type(value).__name__}")
    return value


def atomic_write_json(path: Path, value: Any, *, indent: int | None = 2) -> None:
    """Atomically replace a JSON file using a flushed sibling temporary file."""
    path = effective_path(Path(path), for_write=True)
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp_name: str | None = None
    try:
        with tempfile.NamedTemporaryFile(
            mode="w",
            encoding="utf-8",
            newline="\n",
            dir=path.parent,
            prefix=f".{path.name}.",
            suffix=".tmp",
            delete=False,
        ) as handle:
            tmp_name = handle.name
            json.dump(
                value,
                handle,
                ensure_ascii=False,
                indent=indent,
                allow_nan=False,
            )
            handle.write("\n")
            handle.flush()
            os.fsync(handle.fileno())
        for attempt in range(5):
            try:
                os.replace(tmp_name, path)
                break
            except PermissionError:
                if attempt == 4:
                    raise
                time.sleep(0.05 * (attempt + 1))
        tmp_name = None
        if os.name != "nt":
            directory_fd = os.open(path.parent, os.O_RDONLY)
            try:
                os.fsync(directory_fd)
            finally:
                os.close(directory_fd)
    except (OSError, TypeError, ValueError) as exc:
        raise PersistenceError(f"Atomic JSON write failed for {path}: {exc}") from exc
    finally:
        if tmp_name:
            try:
                Path(tmp_name).unlink(missing_ok=True)
            except OSError:
                pass


def atomic_write_bytes(path: Path, value: bytes) -> None:
    """Atomically write exact bytes without re-serialization or added newlines."""
    if not isinstance(value, bytes):
        raise TypeError("atomic_write_bytes requires bytes")
    path = effective_path(Path(path), for_write=True)
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp_name: str | None = None
    try:
        with tempfile.NamedTemporaryFile(
            mode="wb",
            dir=path.parent,
            prefix=f".{path.name}.",
            suffix=".tmp",
            delete=False,
        ) as handle:
            tmp_name = handle.name
            handle.write(value)
            handle.flush()
            os.fsync(handle.fileno())
        for attempt in range(5):
            try:
                os.replace(tmp_name, path)
                break
            except PermissionError:
                if attempt == 4:
                    raise
                time.sleep(0.05 * (attempt + 1))
        tmp_name = None
        if os.name != "nt":
            directory_fd = os.open(path.parent, os.O_RDONLY)
            try:
                os.fsync(directory_fd)
            finally:
                os.close(directory_fd)
    except OSError as exc:
        raise PersistenceError(f"Atomic byte write failed for {path}: {exc}") from exc
    finally:
        if tmp_name:
            try:
                Path(tmp_name).unlink(missing_ok=True)
            except OSError:
                pass


def cache_failure(previous: dict[str, Any] | None, error: Exception | str) -> dict[str, Any]:
    """Return a copy of stale-good data annotated with the latest failed refresh."""
    entry = dict(previous or {})
    entry["_refresh_failure"] = {
        "at": utc_now(),
        "error": str(error)[:300],
    }
    return entry


def clear_cache_failure(entry: dict[str, Any]) -> dict[str, Any]:
    entry = dict(entry)
    entry.pop("_refresh_failure", None)
    return entry
