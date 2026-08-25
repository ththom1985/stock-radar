"""Transactional local ledger and signed immutable files for forward validation."""
from __future__ import annotations

import gzip
import hashlib
import hmac
import json
import os
import secrets
import sqlite3
from contextlib import contextmanager
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Callable, Iterator

from .persistence import atomic_write_json

LEDGER_SCHEMA_VERSION = 3
GENESIS_CHAIN_HASH = "0" * 64
SIGNING_KEY_BYTES = 32


class ForwardIntegrityError(RuntimeError):
    pass


class ForwardStaleSnapshotError(ForwardIntegrityError):
    pass


def canonical_json_bytes(value: Any) -> bytes:
    return json.dumps(
        value,
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=True,
        allow_nan=False,
    ).encode("utf-8")


def sha256_bytes(value: bytes) -> str:
    return hashlib.sha256(value).hexdigest()


def signed_digest(digest: str, key: bytes) -> str:
    return hmac.new(key, digest.encode("ascii"), hashlib.sha256).hexdigest()


def signing_key_id(key: bytes) -> str:
    if not isinstance(key, bytes) or len(key) < SIGNING_KEY_BYTES:
        raise ValueError("forward manifest signing key is too short")
    return sha256_bytes(key)


def load_or_create_signing_key(path: Path) -> bytes:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    if path.exists():
        key = path.read_bytes()
        signing_key_id(key)
        return key
    key = secrets.token_bytes(SIGNING_KEY_BYTES)
    flags = os.O_WRONLY | os.O_CREAT | os.O_EXCL
    if hasattr(os, "O_BINARY"):
        flags |= os.O_BINARY
    descriptor = os.open(path, flags, 0o600)
    try:
        with os.fdopen(descriptor, "wb") as handle:
            handle.write(key)
            handle.flush()
            os.fsync(handle.fileno())
    except Exception:
        path.unlink(missing_ok=True)
        raise
    try:
        os.chmod(path, 0o600)
    except OSError:
        pass
    return key


def write_immutable_bytes(path: Path, value: bytes) -> bool:
    """Create exact bytes once; return False only for an identical existing file."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    if path.exists():
        existing = path.read_bytes()
        if existing != value:
            raise ForwardIntegrityError(
                f"immutable file mismatch; refusing rewrite: {path}"
            )
        return False
    pending = path.with_name(
        f".{path.name}.{os.getpid()}.{secrets.token_hex(8)}.pending"
    )
    flags = os.O_WRONLY | os.O_CREAT | os.O_EXCL
    if hasattr(os, "O_BINARY"):
        flags |= os.O_BINARY
    descriptor = os.open(pending, flags, 0o600)
    try:
        with os.fdopen(descriptor, "wb") as handle:
            handle.write(value)
            handle.flush()
            os.fsync(handle.fileno())
        try:
            os.link(pending, path)
            return True
        except FileExistsError:
            existing = path.read_bytes()
            if existing == value:
                return False
            raise ForwardIntegrityError(
                f"concurrent immutable file mismatch; refusing rewrite: {path}"
            )
    finally:
        pending.unlink(missing_ok=True)


def write_immutable_json(path: Path, value: dict[str, Any]) -> bool:
    encoded = json.dumps(
        value,
        sort_keys=True,
        indent=2,
        ensure_ascii=False,
        allow_nan=False,
    ).encode("utf-8") + b"\n"
    return write_immutable_bytes(path, encoded)


def deterministic_gzip_json(value: dict[str, Any]) -> bytes:
    return gzip.compress(canonical_json_bytes(value), compresslevel=9, mtime=0)


def read_gzip_json(path: Path) -> dict[str, Any]:
    try:
        value = json.loads(gzip.decompress(Path(path).read_bytes()).decode("utf-8"))
    except (OSError, UnicodeError, json.JSONDecodeError) as exc:
        raise RuntimeError(f"invalid immutable forward capture {path}: {exc}") from exc
    if not isinstance(value, dict):
        raise RuntimeError(f"forward capture root is not an object: {path}")
    return value


def make_capture_envelope(core: dict[str, Any], key: bytes) -> dict[str, Any]:
    required = (
        "anchor_date",
        "artifact_hash",
        "cohort_id",
        "previous_chain_hash",
        "predictions",
        "exclusions",
    )
    if any(name not in core for name in required):
        raise ValueError("capture core is incomplete")
    record_digest = sha256_bytes(canonical_json_bytes(core))
    chain_material = "|".join(
        (
            str(core["previous_chain_hash"]),
            record_digest,
            str(core["artifact_hash"]),
            str(core["anchor_date"]),
        )
    ).encode("ascii")
    chain_hash = sha256_bytes(chain_material)
    return {
        "schema": "stock-radar-probability-forward-week",
        "schema_version": 1,
        "classification": "REJECTED_SHADOW_NOT_FORECAST",
        "core": core,
        "record_digest": record_digest,
        "chain_hash": chain_hash,
        "signature": signed_digest(chain_hash, key),
        "signing_key_id": signing_key_id(key),
    }


def validate_capture_envelope(value: Any, key: bytes) -> dict[str, Any]:
    if (
        not isinstance(value, dict)
        or value.get("schema") != "stock-radar-probability-forward-week"
        or value.get("schema_version") != 1
        or value.get("classification") != "REJECTED_SHADOW_NOT_FORECAST"
        or value.get("signing_key_id") != signing_key_id(key)
        or not isinstance(value.get("core"), dict)
    ):
        raise RuntimeError("unsupported forward capture envelope")
    expected = make_capture_envelope(value["core"], key)
    for field in ("record_digest", "chain_hash", "signature", "signing_key_id"):
        if not hmac.compare_digest(str(value.get(field)), str(expected[field])):
            raise RuntimeError(f"forward capture {field} mismatch")
    return value


class ForwardLedger:
    def __init__(
        self,
        path: Path,
        *,
        cohort_id: str,
        artifact_hash: str,
        preregistration_hash: str,
        signing_key: bytes,
    ) -> None:
        self.path = Path(path)
        signing_key_id(signing_key)
        self.signing_key = signing_key
        self._migrated_from_v1 = False
        self._migrated_from_v2 = False
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self.connection = sqlite3.connect(
            self.path,
            timeout=30.0,
            isolation_level=None,
        )
        try:
            self.connection.row_factory = sqlite3.Row
            self.connection.execute("PRAGMA foreign_keys = ON")
            self.connection.execute("PRAGMA journal_mode = WAL")
            self.connection.execute("PRAGMA synchronous = FULL")
            self.connection.execute("PRAGMA busy_timeout = 30000")
            self._migrate()
            self._bind_metadata(
                cohort_id=cohort_id,
                artifact_hash=artifact_hash,
                preregistration_hash=preregistration_hash,
            )
        except Exception:
            self.connection.close()
            raise

    def close(self) -> None:
        self.connection.close()

    def __enter__(self) -> "ForwardLedger":
        return self

    def __exit__(self, *_args: Any) -> None:
        self.close()

    @contextmanager
    def transaction(self) -> Iterator[sqlite3.Connection]:
        self.connection.execute("BEGIN IMMEDIATE")
        try:
            yield self.connection
        except Exception:
            self.connection.execute("ROLLBACK")
            raise
        else:
            self.connection.execute("COMMIT")

    @contextmanager
    def read_snapshot(self) -> Iterator[sqlite3.Connection]:
        self.connection.execute("BEGIN")
        try:
            # Establish the WAL snapshot immediately, before expensive metrics.
            self.connection.execute("SELECT COUNT(*) FROM events").fetchone()
            yield self.connection
        except Exception:
            self.connection.execute("ROLLBACK")
            raise
        else:
            self.connection.execute("COMMIT")

    def _migrate(self) -> None:
        version = int(self.connection.execute("PRAGMA user_version").fetchone()[0])
        if version > LEDGER_SCHEMA_VERSION:
            raise RuntimeError(
                f"forward ledger schema {version} is newer than supported "
                f"{LEDGER_SCHEMA_VERSION}"
            )
        if version == 0:
            self.connection.executescript(
                f"""
                    BEGIN IMMEDIATE;
                    CREATE TABLE metadata (
                        key TEXT PRIMARY KEY,
                        value_json TEXT NOT NULL
                    );
                    CREATE TABLE anchors (
                        anchor_date TEXT PRIMARY KEY,
                        iso_year INTEGER NOT NULL,
                        iso_week INTEGER NOT NULL,
                        captured_at TEXT NOT NULL,
                        expected_us_session TEXT NOT NULL,
                        spy_asof TEXT NOT NULL,
                        requested_issuer_count INTEGER NOT NULL CHECK(
                            requested_issuer_count >= 0
                        ),
                        successful_issuer_count INTEGER NOT NULL CHECK(
                            successful_issuer_count >= 0
                        ),
                        provider_success_coverage REAL NOT NULL CHECK(
                            provider_success_coverage >= 0
                            AND provider_success_coverage <= 1
                        ),
                        provider_failures_json TEXT NOT NULL,
                        artifact_hash TEXT NOT NULL,
                        record_digest TEXT NOT NULL,
                        file_digest TEXT NOT NULL,
                        previous_chain_hash TEXT NOT NULL,
                        chain_hash TEXT NOT NULL,
                        signature TEXT NOT NULL,
                        prediction_count INTEGER NOT NULL CHECK(prediction_count >= 0),
                        eligible_count INTEGER NOT NULL CHECK(eligible_count >= 0),
                        withheld_count INTEGER NOT NULL CHECK(withheld_count >= 0),
                        exclusion_count INTEGER NOT NULL CHECK(exclusion_count >= 0),
                        UNIQUE(iso_year, iso_week)
                    );
                    CREATE TABLE predictions (
                        prediction_id TEXT PRIMARY KEY,
                        anchor_date TEXT NOT NULL REFERENCES anchors(anchor_date),
                        cohort_id TEXT NOT NULL,
                        artifact_hash TEXT NOT NULL,
                        model_key TEXT NOT NULL,
                        horizon_sessions INTEGER NOT NULL,
                        symbol TEXT NOT NULL,
                        issuer_key TEXT NOT NULL,
                        asset_type TEXT NOT NULL,
                        currency TEXT NOT NULL,
                        partition_name TEXT NOT NULL,
                        feature_date TEXT NOT NULL,
                        feature_timestamp TEXT NOT NULL,
                        spy_asof TEXT NOT NULL,
                        feature_hash TEXT NOT NULL,
                        source_bar_checksum TEXT NOT NULL,
                        raw_ordered_json TEXT NOT NULL,
                        derived_json TEXT NOT NULL,
                        baseline_json TEXT NOT NULL,
                        ood_json TEXT NOT NULL,
                        regime TEXT NOT NULL,
                        eligible_for_evaluation INTEGER NOT NULL CHECK(
                            eligible_for_evaluation IN (0, 1)
                        ),
                        exclusion_reason TEXT,
                        entry_status TEXT NOT NULL,
                        maturity_session_count INTEGER NOT NULL,
                        maturity_session_date TEXT NOT NULL,
                        maturity_schedule_version TEXT NOT NULL,
                        record_hash TEXT NOT NULL,
                        record_signature TEXT NOT NULL,
                        UNIQUE(anchor_date, symbol, horizon_sessions)
                    );
                    CREATE TABLE exclusions (
                        exclusion_id TEXT PRIMARY KEY,
                        anchor_date TEXT NOT NULL REFERENCES anchors(anchor_date),
                        symbol TEXT NOT NULL,
                        issuer_key TEXT,
                        reason TEXT NOT NULL,
                        record_hash TEXT NOT NULL,
                        record_signature TEXT NOT NULL,
                        UNIQUE(anchor_date, symbol, reason)
                    );
                    CREATE TABLE labels (
                        prediction_id TEXT PRIMARY KEY REFERENCES predictions(prediction_id),
                        entry_timestamp TEXT NOT NULL,
                        exit_timestamp TEXT NOT NULL,
                        entry_open_adjusted REAL NOT NULL,
                        entry_open_raw REAL NOT NULL,
                        exit_close_adjusted REAL NOT NULL,
                        exit_close_raw REAL NOT NULL,
                        gross_return REAL NOT NULL,
                        long_net_return REAL NOT NULL,
                        material_net_return REAL NOT NULL,
                        ordered_label INTEGER NOT NULL CHECK(ordered_label BETWEEN 0 AND 6),
                        threshold_labels_json TEXT NOT NULL,
                        source_checksum TEXT NOT NULL,
                        convention TEXT NOT NULL,
                        labeled_at TEXT NOT NULL,
                        label_hash TEXT NOT NULL,
                        label_signature TEXT NOT NULL
                    );
                    CREATE TABLE resolution_attempts (
                        attempt_id INTEGER PRIMARY KEY AUTOINCREMENT,
                        prediction_id TEXT NOT NULL REFERENCES predictions(prediction_id),
                        attempted_at TEXT NOT NULL,
                        observed_through TEXT,
                        reason TEXT NOT NULL,
                        UNIQUE(prediction_id, observed_through, reason)
                    );
                    CREATE TABLE events (
                        sequence INTEGER PRIMARY KEY,
                        event_type TEXT NOT NULL,
                        entity_key TEXT NOT NULL,
                        created_at TEXT NOT NULL,
                        payload_json TEXT NOT NULL,
                        payload_hash TEXT NOT NULL,
                        previous_event_hash TEXT NOT NULL,
                        event_hash TEXT NOT NULL UNIQUE,
                        event_signature TEXT NOT NULL
                    );
                    CREATE TABLE seal_state (
                        singleton INTEGER PRIMARY KEY CHECK(singleton = 1),
                        finalized_event_count INTEGER NOT NULL,
                        finalized_event_head_hash TEXT NOT NULL,
                        finalized_table_snapshot_root TEXT NOT NULL,
                        finalized_manifest_hash TEXT,
                        pending_event_count INTEGER,
                        pending_event_head_hash TEXT,
                        pending_table_snapshot_root TEXT,
                        pending_weekly_anchor_count INTEGER,
                        pending_weekly_head_chain_hash TEXT,
                        pending_candidate_report_hash TEXT,
                        pending_created_at TEXT
                    );
                    INSERT INTO seal_state(
                        singleton, finalized_event_count,
                        finalized_event_head_hash,
                        finalized_table_snapshot_root,
                        finalized_manifest_hash
                    ) VALUES (
                        1, 0,
                        '{GENESIS_CHAIN_HASH}',
                        '{GENESIS_CHAIN_HASH}',
                        NULL
                    );
                    CREATE INDEX predictions_pending_idx ON predictions(
                        eligible_for_evaluation, horizon_sessions, feature_date
                    );
                    CREATE INDEX predictions_anchor_idx ON predictions(anchor_date);
                    CREATE TRIGGER anchors_no_update BEFORE UPDATE ON anchors
                    BEGIN SELECT RAISE(ABORT, 'anchors are immutable'); END;
                    CREATE TRIGGER anchors_no_delete BEFORE DELETE ON anchors
                    BEGIN SELECT RAISE(ABORT, 'anchors are immutable'); END;
                    CREATE TRIGGER predictions_no_update BEFORE UPDATE ON predictions
                    BEGIN SELECT RAISE(ABORT, 'predictions are immutable'); END;
                    CREATE TRIGGER predictions_no_delete BEFORE DELETE ON predictions
                    BEGIN SELECT RAISE(ABORT, 'predictions are immutable'); END;
                    CREATE TRIGGER exclusions_no_update BEFORE UPDATE ON exclusions
                    BEGIN SELECT RAISE(ABORT, 'exclusions are immutable'); END;
                    CREATE TRIGGER exclusions_no_delete BEFORE DELETE ON exclusions
                    BEGIN SELECT RAISE(ABORT, 'exclusions are immutable'); END;
                    CREATE TRIGGER labels_no_update BEFORE UPDATE ON labels
                    BEGIN SELECT RAISE(ABORT, 'labels are immutable'); END;
                    CREATE TRIGGER labels_no_delete BEFORE DELETE ON labels
                    BEGIN SELECT RAISE(ABORT, 'labels are immutable'); END;
                    CREATE TRIGGER metadata_no_update BEFORE UPDATE ON metadata
                    BEGIN SELECT RAISE(ABORT, 'metadata is immutable'); END;
                    CREATE TRIGGER metadata_no_delete BEFORE DELETE ON metadata
                    BEGIN SELECT RAISE(ABORT, 'metadata is immutable'); END;
                    CREATE TRIGGER resolution_no_update BEFORE UPDATE ON resolution_attempts
                    BEGIN SELECT RAISE(ABORT, 'resolution attempts are immutable'); END;
                    CREATE TRIGGER resolution_no_delete BEFORE DELETE ON resolution_attempts
                    BEGIN SELECT RAISE(ABORT, 'resolution attempts are immutable'); END;
                    CREATE TRIGGER events_no_update BEFORE UPDATE ON events
                    BEGIN SELECT RAISE(ABORT, 'events are immutable'); END;
                    CREATE TRIGGER events_no_delete BEFORE DELETE ON events
                    BEGIN SELECT RAISE(ABORT, 'events are immutable'); END;
                    PRAGMA user_version = {LEDGER_SCHEMA_VERSION};
                    COMMIT;
                """
            )
            version = LEDGER_SCHEMA_VERSION
        if version == 1:
            anchor_count = int(
                self.connection.execute("SELECT COUNT(*) FROM anchors").fetchone()[0]
            )
            prediction_count = int(
                self.connection.execute("SELECT COUNT(*) FROM predictions").fetchone()[0]
            )
            label_count = int(
                self.connection.execute("SELECT COUNT(*) FROM labels").fetchone()[0]
            )
            if anchor_count or prediction_count or label_count:
                raise RuntimeError(
                    "nonempty unsealed schema-v1 ledger cannot be migrated safely"
                )
            self.connection.executescript(
                f"""
                    BEGIN IMMEDIATE;
                    ALTER TABLE anchors ADD COLUMN expected_us_session TEXT NOT NULL DEFAULT '';
                    ALTER TABLE anchors ADD COLUMN requested_issuer_count INTEGER NOT NULL DEFAULT 0;
                    ALTER TABLE anchors ADD COLUMN successful_issuer_count INTEGER NOT NULL DEFAULT 0;
                    ALTER TABLE anchors ADD COLUMN provider_success_coverage REAL NOT NULL DEFAULT 0;
                    ALTER TABLE anchors ADD COLUMN provider_failures_json TEXT NOT NULL DEFAULT '{{}}';
                    ALTER TABLE predictions ADD COLUMN maturity_session_date TEXT NOT NULL DEFAULT '';
                    ALTER TABLE predictions ADD COLUMN maturity_schedule_version TEXT NOT NULL DEFAULT '';
                    CREATE UNIQUE INDEX exclusions_unique_reason_idx
                    ON exclusions(anchor_date, symbol, reason);
                    CREATE TABLE events (
                        sequence INTEGER PRIMARY KEY,
                        event_type TEXT NOT NULL,
                        entity_key TEXT NOT NULL,
                        created_at TEXT NOT NULL,
                        payload_json TEXT NOT NULL,
                        payload_hash TEXT NOT NULL,
                        previous_event_hash TEXT NOT NULL,
                        event_hash TEXT NOT NULL UNIQUE,
                        event_signature TEXT NOT NULL
                    );
                    CREATE TABLE seal_state (
                        singleton INTEGER PRIMARY KEY CHECK(singleton = 1),
                        finalized_event_count INTEGER NOT NULL,
                        finalized_event_head_hash TEXT NOT NULL,
                        finalized_table_snapshot_root TEXT NOT NULL,
                        finalized_manifest_hash TEXT,
                        pending_event_count INTEGER,
                        pending_event_head_hash TEXT,
                        pending_table_snapshot_root TEXT,
                        pending_weekly_anchor_count INTEGER,
                        pending_weekly_head_chain_hash TEXT,
                        pending_candidate_report_hash TEXT,
                        pending_created_at TEXT
                    );
                    INSERT INTO seal_state(
                        singleton, finalized_event_count,
                        finalized_event_head_hash,
                        finalized_table_snapshot_root,
                        finalized_manifest_hash
                    ) VALUES (
                        1, 0,
                        '{GENESIS_CHAIN_HASH}',
                        '{GENESIS_CHAIN_HASH}',
                        NULL
                    );
                    UPDATE metadata SET value_json = '3'
                    WHERE key = 'schema_version';
                    INSERT INTO metadata(key, value_json)
                    VALUES ('signing_key_id', '"{signing_key_id(self.signing_key)}"');
                    CREATE TRIGGER metadata_no_update BEFORE UPDATE ON metadata
                    BEGIN SELECT RAISE(ABORT, 'metadata is immutable'); END;
                    CREATE TRIGGER metadata_no_delete BEFORE DELETE ON metadata
                    BEGIN SELECT RAISE(ABORT, 'metadata is immutable'); END;
                    CREATE TRIGGER resolution_no_update BEFORE UPDATE ON resolution_attempts
                    BEGIN SELECT RAISE(ABORT, 'resolution attempts are immutable'); END;
                    CREATE TRIGGER resolution_no_delete BEFORE DELETE ON resolution_attempts
                    BEGIN SELECT RAISE(ABORT, 'resolution attempts are immutable'); END;
                    CREATE TRIGGER events_no_update BEFORE UPDATE ON events
                    BEGIN SELECT RAISE(ABORT, 'events are immutable'); END;
                    CREATE TRIGGER events_no_delete BEFORE DELETE ON events
                    BEGIN SELECT RAISE(ABORT, 'events are immutable'); END;
                    PRAGMA user_version = {LEDGER_SCHEMA_VERSION};
                    COMMIT;
                """
            )
            version = LEDGER_SCHEMA_VERSION
            self._migrated_from_v1 = True
        if version == 2:
            self.connection.executescript(
                f"""
                    BEGIN IMMEDIATE;
                    DROP TRIGGER metadata_no_update;
                    UPDATE metadata SET value_json = '3'
                    WHERE key = 'schema_version';
                    CREATE TRIGGER metadata_no_update BEFORE UPDATE ON metadata
                    BEGIN SELECT RAISE(ABORT, 'metadata is immutable'); END;
                    CREATE TABLE seal_state (
                        singleton INTEGER PRIMARY KEY CHECK(singleton = 1),
                        finalized_event_count INTEGER NOT NULL,
                        finalized_event_head_hash TEXT NOT NULL,
                        finalized_table_snapshot_root TEXT NOT NULL,
                        finalized_manifest_hash TEXT,
                        pending_event_count INTEGER,
                        pending_event_head_hash TEXT,
                        pending_table_snapshot_root TEXT,
                        pending_weekly_anchor_count INTEGER,
                        pending_weekly_head_chain_hash TEXT,
                        pending_candidate_report_hash TEXT,
                        pending_created_at TEXT
                    );
                    INSERT INTO seal_state(
                        singleton, finalized_event_count,
                        finalized_event_head_hash,
                        finalized_table_snapshot_root,
                        finalized_manifest_hash
                    ) VALUES (
                        1, 0,
                        '{GENESIS_CHAIN_HASH}',
                        '{GENESIS_CHAIN_HASH}',
                        NULL
                    );
                    PRAGMA user_version = {LEDGER_SCHEMA_VERSION};
                    COMMIT;
                """
            )
            version = LEDGER_SCHEMA_VERSION
            self._migrated_from_v2 = True
        if version != LEDGER_SCHEMA_VERSION:
            raise RuntimeError(f"unsupported forward ledger schema {version}")

    @staticmethod
    def _row_value(row: sqlite3.Row) -> dict[str, Any]:
        return {key: row[key] for key in row.keys()}

    def _append_event(
        self,
        connection: sqlite3.Connection,
        *,
        event_type: str,
        entity_key: str,
        payload: dict[str, Any],
        created_at: str | None = None,
    ) -> sqlite3.Row:
        previous = connection.execute(
            "SELECT sequence, event_hash FROM events ORDER BY sequence DESC LIMIT 1"
        ).fetchone()
        sequence = int(previous["sequence"]) + 1 if previous is not None else 1
        previous_hash = (
            str(previous["event_hash"]) if previous is not None else GENESIS_CHAIN_HASH
        )
        payload_json = canonical_json_bytes(payload).decode("ascii")
        payload_hash = sha256_bytes(payload_json.encode("ascii"))
        event_created_at = created_at or datetime.now(timezone.utc).isoformat(
            timespec="seconds"
        )
        material = {
            "sequence": sequence,
            "event_type": str(event_type),
            "entity_key": str(entity_key),
            "created_at": event_created_at,
            "payload_hash": payload_hash,
            "previous_event_hash": previous_hash,
        }
        event_hash = sha256_bytes(canonical_json_bytes(material))
        signature = signed_digest(event_hash, self.signing_key)
        connection.execute(
            """
            INSERT INTO events(
                sequence, event_type, entity_key, created_at, payload_json,
                payload_hash, previous_event_hash, event_hash, event_signature
            ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
            """,
            (
                sequence,
                event_type,
                entity_key,
                event_created_at,
                payload_json,
                payload_hash,
                previous_hash,
                event_hash,
                signature,
            ),
        )
        return connection.execute(
            "SELECT * FROM events WHERE sequence = ?",
            (sequence,),
        ).fetchone()

    def _bind_metadata(
        self,
        *,
        cohort_id: str,
        artifact_hash: str,
        preregistration_hash: str,
    ) -> None:
        expected = {
            "cohort_id": cohort_id,
            "artifact_hash": artifact_hash,
            "preregistration_hash": preregistration_hash,
            "schema_version": LEDGER_SCHEMA_VERSION,
            "signing_key_id": signing_key_id(self.signing_key),
        }
        with self.transaction() as connection:
            rows = {
                row["key"]: json.loads(row["value_json"])
                for row in connection.execute("SELECT key, value_json FROM metadata")
            }
            if rows:
                if rows != expected:
                    raise RuntimeError("forward ledger binding does not match the cohort")
                event = connection.execute(
                    "SELECT * FROM events ORDER BY sequence LIMIT 1"
                ).fetchone()
                if event is None and self._migrated_from_v1:
                    metadata_rows = [
                        self._row_value(row)
                        for row in connection.execute(
                            "SELECT key, value_json FROM metadata ORDER BY key"
                        )
                    ]
                    self._append_event(
                        connection,
                        event_type="cohort_initialized",
                        entity_key=cohort_id,
                        payload={"metadata": metadata_rows},
                    )
                elif event is None or event["event_type"] != "cohort_initialized":
                    raise RuntimeError("forward ledger is missing its sealed genesis event")
                if self._migrated_from_v1 or self._migrated_from_v2:
                    self._stage_pending_seal(connection)
                return
            connection.executemany(
                "INSERT INTO metadata(key, value_json) VALUES (?, ?)",
                [
                    (
                        key,
                        json.dumps(value, sort_keys=True, separators=(",", ":")),
                    )
                    for key, value in expected.items()
                ],
            )
            metadata_rows = [
                self._row_value(row)
                for row in connection.execute(
                    "SELECT key, value_json FROM metadata ORDER BY key"
                )
            ]
            self._append_event(
                connection,
                event_type="cohort_initialized",
                entity_key=cohort_id,
                payload={"metadata": metadata_rows},
            )
            self._stage_pending_seal(connection)

    def latest_anchor(self) -> sqlite3.Row | None:
        return self.connection.execute(
            "SELECT * FROM anchors ORDER BY anchor_date DESC LIMIT 1"
        ).fetchone()

    def anchor_for_week(self, iso_year: int, iso_week: int) -> sqlite3.Row | None:
        return self.connection.execute(
            "SELECT * FROM anchors WHERE iso_year = ? AND iso_week = ?",
            (int(iso_year), int(iso_week)),
        ).fetchone()

    def insert_capture(
        self,
        envelope: dict[str, Any],
        *,
        file_digest: str,
    ) -> bool:
        core = envelope["core"]
        anchor_date = str(core["anchor_date"])
        iso_year = int(core["iso_year"])
        iso_week = int(core["iso_week"])
        predictions = list(core.get("predictions") or [])
        exclusions = list(core.get("exclusions") or [])
        eligible_count = sum(
            int(bool(item["eligible_for_evaluation"])) for item in predictions
        )
        with self.transaction() as connection:
            existing = connection.execute(
                "SELECT * FROM anchors WHERE iso_year = ? AND iso_week = ?",
                (iso_year, iso_week),
            ).fetchone()
            if existing is not None:
                matches = (
                    existing["anchor_date"] == anchor_date
                    and existing["record_digest"] == envelope["record_digest"]
                    and existing["chain_hash"] == envelope["chain_hash"]
                    and existing["file_digest"] == file_digest
                )
                if not matches:
                    raise RuntimeError(
                        "weekly anchor already exists with different content; "
                        "refusing rewrite"
                    )
                return False
            latest = connection.execute(
                "SELECT * FROM anchors ORDER BY anchor_date DESC LIMIT 1"
            ).fetchone()
            if latest is not None and anchor_date <= latest["anchor_date"]:
                raise RuntimeError(
                    "forward anchors must be appended in chronological order"
                )
            expected_previous = (
                latest["chain_hash"] if latest is not None else GENESIS_CHAIN_HASH
            )
            if core["previous_chain_hash"] != expected_previous:
                raise RuntimeError("capture would fork the weekly prediction chain")
            provider = core["provider"]
            connection.execute(
                """
                INSERT INTO anchors(
                    anchor_date, iso_year, iso_week, captured_at,
                    expected_us_session, spy_asof, requested_issuer_count,
                    successful_issuer_count, provider_success_coverage,
                    provider_failures_json, artifact_hash, record_digest,
                    file_digest, previous_chain_hash, chain_hash, signature,
                    prediction_count, eligible_count, withheld_count, exclusion_count
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                """,
                (
                    anchor_date,
                    iso_year,
                    iso_week,
                    core["captured_at"],
                    core["expected_us_session"],
                    core["spy_asof"],
                    int(provider["requested_issuer_count"]),
                    int(provider["successful_issuer_count"]),
                    float(provider["success_coverage"]),
                    canonical_json_bytes(provider["failures"]).decode("ascii"),
                    core["artifact_hash"],
                    envelope["record_digest"],
                    file_digest,
                    core["previous_chain_hash"],
                    envelope["chain_hash"],
                    envelope["signature"],
                    len(predictions),
                    eligible_count,
                    len(predictions) - eligible_count,
                    len(exclusions),
                ),
            )
            connection.executemany(
                """
                INSERT INTO predictions(
                    prediction_id, anchor_date, cohort_id, artifact_hash, model_key,
                    horizon_sessions, symbol, issuer_key, asset_type, currency,
                    partition_name, feature_date, feature_timestamp, spy_asof,
                    feature_hash, source_bar_checksum, raw_ordered_json, derived_json,
                    baseline_json, ood_json, regime, eligible_for_evaluation,
                    exclusion_reason, entry_status, maturity_session_count,
                    maturity_session_date, maturity_schedule_version,
                    record_hash, record_signature
                ) VALUES (
                    ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?,
                    ?, ?, ?, ?, ?, ?, ?, ?, ?
                )
                """,
                [
                    (
                        item["prediction_id"],
                        anchor_date,
                        item["cohort_id"],
                        item["artifact_hash"],
                        item["model_key"],
                        int(item["horizon_sessions"]),
                        item["symbol"],
                        item["issuer_key"],
                        item["asset_type"],
                        item["currency"],
                        item["partition"],
                        item["feature_date"],
                        item["feature_timestamp"],
                        item["spy_asof"],
                        item["feature_hash"],
                        item["source_bar_checksum"],
                        canonical_json_bytes(item["raw_ordered_probabilities"]).decode(
                            "ascii"
                        ),
                        canonical_json_bytes(item["derived_probabilities"]).decode(
                            "ascii"
                        ),
                        canonical_json_bytes(item["baseline_rates"]).decode("ascii"),
                        canonical_json_bytes(item["ood"]).decode("ascii"),
                        item["regime"],
                        int(bool(item["eligible_for_evaluation"])),
                        item.get("exclusion_reason"),
                        item["entry_status"],
                        int(item["maturity_target"]["sessions_after_feature"]),
                        item["maturity_target"]["scheduled_exit_session"],
                        item["maturity_target"]["schedule_version"],
                        item["record_hash"],
                        item["record_signature"],
                    )
                    for item in predictions
                ],
            )
            connection.executemany(
                """
                INSERT OR IGNORE INTO exclusions(
                    exclusion_id, anchor_date, symbol, issuer_key, reason,
                    record_hash, record_signature
                ) VALUES (?, ?, ?, ?, ?, ?, ?)
                """,
                [
                    (
                        item["exclusion_id"],
                        anchor_date,
                        item["symbol"],
                        item.get("issuer_key"),
                        item["reason"],
                        item["record_hash"],
                        item["record_signature"],
                    )
                    for item in exclusions
                ],
            )
            anchor_row = self._row_value(
                connection.execute(
                    "SELECT * FROM anchors WHERE anchor_date = ?",
                    (anchor_date,),
                ).fetchone()
            )
            prediction_rows = [
                self._row_value(row)
                for row in connection.execute(
                    "SELECT * FROM predictions WHERE anchor_date = ? "
                    "ORDER BY prediction_id",
                    (anchor_date,),
                )
            ]
            exclusion_rows = [
                self._row_value(row)
                for row in connection.execute(
                    "SELECT * FROM exclusions WHERE anchor_date = ? "
                    "ORDER BY exclusion_id",
                    (anchor_date,),
                )
            ]
            self._append_event(
                connection,
                event_type="capture",
                entity_key=anchor_date,
                payload={
                    "anchor": anchor_row,
                    "predictions": prediction_rows,
                    "exclusions": exclusion_rows,
                },
                created_at=core["captured_at"],
            )
            self._stage_pending_seal(connection)
        return True

    def pending_predictions(self) -> list[sqlite3.Row]:
        return list(
            self.connection.execute(
                """
                SELECT p.*
                FROM predictions AS p
                LEFT JOIN labels AS l ON l.prediction_id = p.prediction_id
                WHERE p.eligible_for_evaluation = 1
                  AND l.prediction_id IS NULL
                ORDER BY p.feature_date, p.symbol, p.horizon_sessions
                """
            )
        )

    def labels_for_metrics(self, horizon: int) -> list[sqlite3.Row]:
        return list(
            self.connection.execute(
                """
                SELECT p.*, l.*
                FROM predictions AS p
                JOIN labels AS l ON l.prediction_id = p.prediction_id
                WHERE p.eligible_for_evaluation = 1
                  AND p.horizon_sessions = ?
                ORDER BY p.feature_date, p.issuer_key
                """,
                (int(horizon),),
            )
        )

    def insert_label(self, value: dict[str, Any]) -> bool:
        prediction_id = str(value["prediction_id"])
        database_value = (
            value["entry_timestamp"],
            value["exit_timestamp"],
            float(value["entry_open_adjusted"]),
            float(value["entry_open_raw"]),
            float(value["exit_close_adjusted"]),
            float(value["exit_close_raw"]),
            float(value["gross_return"]),
            float(value["long_net_return"]),
            float(value["material_net_return"]),
            int(value["ordered_label"]),
            canonical_json_bytes(value["threshold_labels"]).decode("ascii"),
            value["source_checksum"],
            value["convention"],
            value["labeled_at"],
            value["label_hash"],
            value["label_signature"],
        )
        with self.transaction() as connection:
            existing = connection.execute(
                "SELECT * FROM labels WHERE prediction_id = ?",
                (prediction_id,),
            ).fetchone()
            if existing is not None:
                columns = (
                    "entry_timestamp",
                    "exit_timestamp",
                    "entry_open_adjusted",
                    "entry_open_raw",
                    "exit_close_adjusted",
                    "exit_close_raw",
                    "gross_return",
                    "long_net_return",
                    "material_net_return",
                    "ordered_label",
                    "threshold_labels_json",
                    "source_checksum",
                    "convention",
                    "labeled_at",
                    "label_hash",
                    "label_signature",
                )
                if tuple(existing[column] for column in columns) != database_value:
                    raise RuntimeError(
                        f"immutable outcome differs for prediction {prediction_id}"
                    )
                return False
            connection.execute(
                """
                INSERT INTO labels(
                    prediction_id, entry_timestamp, exit_timestamp,
                    entry_open_adjusted, entry_open_raw, exit_close_adjusted,
                    exit_close_raw, gross_return, long_net_return,
                    material_net_return, ordered_label, threshold_labels_json,
                    source_checksum, convention, labeled_at, label_hash,
                    label_signature
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                """,
                (prediction_id, *database_value),
            )
            row = self._row_value(
                connection.execute(
                    "SELECT * FROM labels WHERE prediction_id = ?",
                    (prediction_id,),
                ).fetchone()
            )
            self._append_event(
                connection,
                event_type="label",
                entity_key=prediction_id,
                payload={"label": row},
                created_at=value["labeled_at"],
            )
            self._stage_pending_seal(connection)
        return True

    def record_resolution_attempt(
        self,
        prediction_id: str,
        *,
        attempted_at: str,
        observed_through: str | None,
        reason: str,
    ) -> bool:
        normalized_observed = observed_through or ""
        with self.transaction() as connection:
            cursor = connection.execute(
                """
                INSERT OR IGNORE INTO resolution_attempts(
                    prediction_id, attempted_at, observed_through, reason
                ) VALUES (?, ?, ?, ?)
                """,
                (prediction_id, attempted_at, normalized_observed, reason),
            )
            if cursor.rowcount:
                row = self._row_value(
                    connection.execute(
                        """
                        SELECT * FROM resolution_attempts
                        WHERE prediction_id = ? AND observed_through = ?
                          AND reason = ?
                        """,
                        (prediction_id, normalized_observed, reason),
                    ).fetchone()
                )
                self._append_event(
                    connection,
                    event_type="resolution_attempt",
                    entity_key=f"{prediction_id}:{row['attempt_id']}",
                    payload={"resolution_attempt": row},
                    created_at=attempted_at,
                )
                self._stage_pending_seal(connection)
        return bool(cursor.rowcount)

    def record_candidate_report(
        self,
        report: dict[str, Any],
        *,
        expected_snapshot: dict[str, Any],
    ) -> None:
        report_hash = report.get("report_hash")
        if not isinstance(report_hash, str):
            raise ValueError("candidate report must be finalized before sealing")
        with self.transaction() as connection:
            current = self.snapshot_identity()
            if current != expected_snapshot:
                raise ForwardStaleSnapshotError(
                    "ledger changed while candidate metrics were being computed"
                )
            self._append_event(
                connection,
                event_type="candidate_report",
                entity_key=report_hash,
                payload={"report": report},
                created_at=str(report.get("generated_at")),
            )
            self._stage_pending_seal(connection)

    def aggregate_counts(self) -> dict[str, Any]:
        anchors = self.connection.execute(
            """
            SELECT COUNT(*) AS count, MIN(anchor_date) AS first_date,
                   MAX(anchor_date) AS last_date
            FROM anchors
            """
        ).fetchone()
        output: dict[str, Any] = {
            "weeks_captured": int(anchors["count"]),
            "first_anchor_date": anchors["first_date"],
            "latest_anchor_date": anchors["last_date"],
            "eligible_prediction_counts": {},
            "matured_outcomes": {},
            "unresolved_outcomes": {},
            "next_maturity_dates": {},
        }
        for horizon in (21, 63, 126, 252):
            eligible = self.connection.execute(
                """
                SELECT COUNT(*) FROM predictions
                WHERE eligible_for_evaluation = 1 AND horizon_sessions = ?
                """,
                (horizon,),
            ).fetchone()[0]
            matured = self.connection.execute(
                """
                SELECT COUNT(*)
                FROM labels AS l
                JOIN predictions AS p ON p.prediction_id = l.prediction_id
                WHERE p.horizon_sessions = ?
                """,
                (horizon,),
            ).fetchone()[0]
            unresolved = self.connection.execute(
                """
                SELECT COUNT(DISTINCT r.prediction_id)
                FROM resolution_attempts AS r
                JOIN predictions AS p ON p.prediction_id = r.prediction_id
                LEFT JOIN labels AS l ON l.prediction_id = p.prediction_id
                WHERE p.horizon_sessions = ? AND l.prediction_id IS NULL
                """,
                (horizon,),
            ).fetchone()[0]
            next_date = self.connection.execute(
                """
                SELECT MIN(p.maturity_session_date)
                FROM predictions AS p
                LEFT JOIN labels AS l ON l.prediction_id = p.prediction_id
                WHERE p.eligible_for_evaluation = 1
                  AND p.horizon_sessions = ?
                  AND l.prediction_id IS NULL
                """,
                (horizon,),
            ).fetchone()[0]
            key = str(horizon)
            output["eligible_prediction_counts"][key] = int(eligible)
            output["matured_outcomes"][key] = int(matured)
            output["unresolved_outcomes"][key] = int(unresolved)
            output["next_maturity_dates"][key] = next_date
        return output

    def provider_support(self) -> dict[str, Any]:
        row = self.connection.execute(
            """
            SELECT
                SUM(requested_issuer_count) AS requested,
                SUM(successful_issuer_count) AS successful,
                MIN(provider_success_coverage) AS minimum_coverage,
                MIN(successful_issuer_count) AS minimum_successful_issuers
            FROM anchors
            """
        ).fetchone()
        requested = int(row["requested"] or 0)
        successful = int(row["successful"] or 0)
        return {
            "requested_issuer_observations": requested,
            "successful_issuer_observations": successful,
            "aggregate_success_coverage": (
                successful / requested if requested else 0.0
            ),
            "minimum_anchor_success_coverage": float(
                row["minimum_coverage"] or 0.0
            ),
            "minimum_anchor_successful_issuers": int(
                row["minimum_successful_issuers"] or 0
            ),
        }

    def _stage_pending_seal(self, connection: sqlite3.Connection) -> None:
        state = self._event_state(self.signing_key)
        seal = connection.execute(
            "SELECT * FROM seal_state WHERE singleton = 1"
        ).fetchone()
        if seal is None:
            raise ForwardIntegrityError("forward seal state is missing")
        finalized_count = int(seal["finalized_event_count"])
        current_count = int(state["event_count"])
        if finalized_count > current_count:
            raise ForwardIntegrityError("finalized seal is ahead of the event ledger")
        if finalized_count:
            ancestor = connection.execute(
                "SELECT event_hash FROM events WHERE sequence = ?",
                (finalized_count,),
            ).fetchone()
            if (
                ancestor is None
                or ancestor["event_hash"] != seal["finalized_event_head_hash"]
            ):
                raise ForwardIntegrityError(
                    "finalized seal is not an event-chain ancestor"
                )
        if current_count == finalized_count:
            if (
                state["event_head_hash"] != seal["finalized_event_head_hash"]
                or state["table_snapshot_root"]
                != seal["finalized_table_snapshot_root"]
            ):
                raise ForwardIntegrityError(
                    "finalized seal does not match current ledger state"
                )
            if seal["pending_event_count"] is not None:
                raise ForwardIntegrityError(
                    "pending seal exists without an extending ledger head"
                )
            return
        weekly = connection.execute(
            "SELECT COUNT(*) AS count, MAX(anchor_date) AS latest FROM anchors"
        ).fetchone()
        latest_anchor = (
            connection.execute(
                "SELECT chain_hash FROM anchors ORDER BY anchor_date DESC LIMIT 1"
            ).fetchone()
            if int(weekly["count"])
            else None
        )
        pending_values = (
            current_count,
            state["event_head_hash"],
            state["table_snapshot_root"],
            int(weekly["count"]),
            (
                latest_anchor["chain_hash"]
                if latest_anchor is not None
                else GENESIS_CHAIN_HASH
            ),
            state["latest_candidate_report_hash"],
        )
        existing_pending = (
            seal["pending_event_count"],
            seal["pending_event_head_hash"],
            seal["pending_table_snapshot_root"],
            seal["pending_weekly_anchor_count"],
            seal["pending_weekly_head_chain_hash"],
            seal["pending_candidate_report_hash"],
        )
        if existing_pending == pending_values:
            return
        if seal["pending_event_count"] is not None:
            pending_count = int(seal["pending_event_count"])
            if pending_count > current_count:
                raise ForwardIntegrityError("pending seal is ahead of the event ledger")
            ancestor = connection.execute(
                "SELECT event_hash FROM events WHERE sequence = ?",
                (pending_count,),
            ).fetchone()
            if (
                ancestor is None
                or ancestor["event_hash"] != seal["pending_event_head_hash"]
            ):
                raise ForwardIntegrityError(
                    "pending seal is not an event-chain ancestor"
                )
        connection.execute(
            """
            UPDATE seal_state
            SET pending_event_count = ?,
                pending_event_head_hash = ?,
                pending_table_snapshot_root = ?,
                pending_weekly_anchor_count = ?,
                pending_weekly_head_chain_hash = ?,
                pending_candidate_report_hash = ?,
                pending_created_at = ?
            WHERE singleton = 1
            """,
            (
                *pending_values,
                datetime.now(timezone.utc).isoformat(timespec="seconds"),
            ),
        )

    def _table_snapshot(self) -> dict[str, list[dict[str, Any]]]:
        definitions = {
            "metadata": ("key",),
            "anchors": ("anchor_date",),
            "predictions": ("prediction_id",),
            "exclusions": ("exclusion_id",),
            "labels": ("prediction_id",),
            "resolution_attempts": ("attempt_id",),
        }
        output: dict[str, list[dict[str, Any]]] = {}
        for table, order_columns in definitions.items():
            order = ", ".join(order_columns)
            output[table] = [
                self._row_value(row)
                for row in self.connection.execute(
                    f"SELECT * FROM {table} ORDER BY {order}"
                )
            ]
        return output

    def _event_state(self, key: bytes) -> dict[str, Any]:
        if not hmac.compare_digest(signing_key_id(key), signing_key_id(self.signing_key)):
            raise RuntimeError("forward ledger signing key mismatch")
        expected: dict[str, list[dict[str, Any]]] = {
            "metadata": [],
            "anchors": [],
            "predictions": [],
            "exclusions": [],
            "labels": [],
            "resolution_attempts": [],
        }
        previous = GENESIS_CHAIN_HASH
        expected_sequence = 1
        latest_candidate_hash = None
        candidate_report_count = 0
        rows = list(self.connection.execute("SELECT * FROM events ORDER BY sequence"))
        if not rows:
            raise RuntimeError("forward ledger event chain is empty")
        for row in rows:
            if int(row["sequence"]) != expected_sequence:
                raise RuntimeError("forward ledger event sequence has a gap")
            if row["previous_event_hash"] != previous:
                raise RuntimeError("forward ledger event chain link mismatch")
            payload_bytes = str(row["payload_json"]).encode("ascii")
            if sha256_bytes(payload_bytes) != row["payload_hash"]:
                raise RuntimeError("forward ledger event payload hash mismatch")
            material = {
                "sequence": int(row["sequence"]),
                "event_type": row["event_type"],
                "entity_key": row["entity_key"],
                "created_at": row["created_at"],
                "payload_hash": row["payload_hash"],
                "previous_event_hash": previous,
            }
            event_hash = sha256_bytes(canonical_json_bytes(material))
            if (
                event_hash != row["event_hash"]
                or not hmac.compare_digest(
                    signed_digest(event_hash, key),
                    row["event_signature"],
                )
            ):
                raise RuntimeError("forward ledger event signature mismatch")
            payload = json.loads(row["payload_json"])
            event_type = row["event_type"]
            if expected_sequence == 1 and event_type != "cohort_initialized":
                raise RuntimeError("forward ledger genesis event is invalid")
            if event_type == "cohort_initialized":
                if expected["metadata"]:
                    raise RuntimeError("forward ledger has multiple genesis events")
                expected["metadata"] = list(payload["metadata"])
            elif event_type == "capture":
                expected["anchors"].append(payload["anchor"])
                expected["predictions"].extend(payload["predictions"])
                expected["exclusions"].extend(payload["exclusions"])
            elif event_type == "label":
                expected["labels"].append(payload["label"])
            elif event_type == "resolution_attempt":
                expected["resolution_attempts"].append(
                    payload["resolution_attempt"]
                )
            elif event_type == "candidate_report":
                report = payload.get("report")
                if not isinstance(report, dict):
                    raise RuntimeError("candidate report event payload is invalid")
                report_hash = report.get("report_hash")
                unhashed_report = {
                    name: item
                    for name, item in report.items()
                    if name != "report_hash"
                }
                if (
                    not isinstance(report_hash, str)
                    or sha256_bytes(canonical_json_bytes(unhashed_report))
                    != report_hash
                ):
                    raise RuntimeError("candidate report event hash mismatch")
                latest_candidate_hash = report_hash
                candidate_report_count += 1
            else:
                raise RuntimeError(f"unknown forward ledger event type: {event_type}")
            previous = event_hash
            expected_sequence += 1
        actual = self._table_snapshot()
        sort_keys = {
            "metadata": "key",
            "anchors": "anchor_date",
            "predictions": "prediction_id",
            "exclusions": "exclusion_id",
            "labels": "prediction_id",
            "resolution_attempts": "attempt_id",
        }
        for table, key_name in sort_keys.items():
            expected_rows = sorted(expected[table], key=lambda item: item[key_name])
            if expected_rows != actual[table]:
                raise RuntimeError(
                    f"forward ledger {table} rows do not reconcile to signed events"
                )
        return {
            "event_count": len(rows),
            "event_head_hash": previous,
            "latest_candidate_report_hash": latest_candidate_hash,
            "candidate_report_count": candidate_report_count,
            "table_snapshot": actual,
            "table_snapshot_root": sha256_bytes(canonical_json_bytes(actual)),
        }

    def verify_integrity(self, key: bytes) -> dict[str, Any]:
        event_state = self._event_state(key)
        previous = GENESIS_CHAIN_HASH
        anchor_count = 0
        for anchor in self.connection.execute(
            "SELECT * FROM anchors ORDER BY anchor_date"
        ):
            if anchor["previous_chain_hash"] != previous:
                raise RuntimeError("forward weekly prediction chain link mismatch")
            material = "|".join(
                (
                    previous,
                    anchor["record_digest"],
                    anchor["artifact_hash"],
                    anchor["anchor_date"],
                )
            ).encode("ascii")
            expected_chain = sha256_bytes(material)
            if (
                expected_chain != anchor["chain_hash"]
                or not hmac.compare_digest(
                    signed_digest(expected_chain, key),
                    anchor["signature"],
                )
            ):
                raise RuntimeError("forward weekly anchor signature mismatch")
            previous = expected_chain
            anchor_count += 1
        prediction_count = 0
        for row in self.connection.execute("SELECT * FROM predictions"):
            payload = {
                "prediction_id": row["prediction_id"],
                "classification": "REJECTED_SHADOW_NOT_FORECAST",
                "shadow_only": True,
                "actionable": False,
                "cohort_id": row["cohort_id"],
                "artifact_hash": row["artifact_hash"],
                "model_key": row["model_key"],
                "horizon_sessions": row["horizon_sessions"],
                "symbol": row["symbol"],
                "issuer_key": row["issuer_key"],
                "asset_type": row["asset_type"],
                "currency": row["currency"],
                "partition": row["partition_name"],
                "feature_date": row["feature_date"],
                "feature_timestamp": row["feature_timestamp"],
                "expected_us_session": row["feature_date"],
                "spy_asof": row["spy_asof"],
                "feature_hash": row["feature_hash"],
                "source_bar_checksum": row["source_bar_checksum"],
                "raw_ordered_probabilities": json.loads(row["raw_ordered_json"]),
                "derived_probabilities": json.loads(row["derived_json"]),
                "baseline_rates": json.loads(row["baseline_json"]),
                "ood": json.loads(row["ood_json"]),
                "regime": row["regime"],
                "eligible_for_evaluation": bool(row["eligible_for_evaluation"]),
                "exclusion_reason": row["exclusion_reason"],
                "entry_status": row["entry_status"],
                "maturity_target": {
                    "sessions_after_feature": row["maturity_session_count"],
                    "scheduled_exit_session": row["maturity_session_date"],
                    "schedule_version": row["maturity_schedule_version"],
                },
            }
            expected_hash = sha256_bytes(canonical_json_bytes(payload))
            if (
                expected_hash != row["record_hash"]
                or not hmac.compare_digest(
                    signed_digest(expected_hash, key),
                    row["record_signature"],
                )
            ):
                raise RuntimeError("forward ledger prediction signature mismatch")
            prediction_count += 1
        for row in self.connection.execute("SELECT * FROM exclusions"):
            payload = {
                "exclusion_id": row["exclusion_id"],
                "symbol": row["symbol"],
                "issuer_key": row["issuer_key"],
                "reason": row["reason"],
            }
            expected_hash = sha256_bytes(canonical_json_bytes(payload))
            if (
                expected_hash != row["record_hash"]
                or not hmac.compare_digest(
                    signed_digest(expected_hash, key),
                    row["record_signature"],
                )
            ):
                raise RuntimeError("forward ledger exclusion signature mismatch")
        label_count = 0
        for row in self.connection.execute("SELECT * FROM labels"):
            payload = {
                "prediction_id": row["prediction_id"],
                "entry_timestamp": row["entry_timestamp"],
                "exit_timestamp": row["exit_timestamp"],
                "entry_open_adjusted": row["entry_open_adjusted"],
                "entry_open_raw": row["entry_open_raw"],
                "exit_close_adjusted": row["exit_close_adjusted"],
                "exit_close_raw": row["exit_close_raw"],
                "gross_return": row["gross_return"],
                "long_net_return": row["long_net_return"],
                "material_net_return": row["material_net_return"],
                "ordered_label": row["ordered_label"],
                "threshold_labels": json.loads(row["threshold_labels_json"]),
                "source_checksum": row["source_checksum"],
                "convention": row["convention"],
                "labeled_at": row["labeled_at"],
            }
            expected_hash = sha256_bytes(canonical_json_bytes(payload))
            if (
                expected_hash != row["label_hash"]
                or not hmac.compare_digest(
                    signed_digest(expected_hash, key),
                    row["label_signature"],
                )
            ):
                raise RuntimeError("forward ledger label signature mismatch")
            label_count += 1
        return {
            "anchors": anchor_count,
            "predictions": prediction_count,
            "labels": label_count,
            "events": event_state["event_count"],
            "event_head_hash": event_state["event_head_hash"],
            "table_snapshot_root": event_state["table_snapshot_root"],
            "latest_candidate_report_hash": event_state[
                "latest_candidate_report_hash"
            ],
        }

    def snapshot_identity(self) -> dict[str, Any]:
        state = self._event_state(self.signing_key)
        return {
            "event_count": state["event_count"],
            "event_head_hash": state["event_head_hash"],
            "table_snapshot_root": state["table_snapshot_root"],
        }

    def latest_candidate_report(self) -> dict[str, Any] | None:
        row = self.connection.execute(
            """
            SELECT payload_json FROM events
            WHERE event_type = 'candidate_report'
            ORDER BY sequence DESC LIMIT 1
            """
        ).fetchone()
        if row is None:
            return None
        payload = json.loads(row["payload_json"])
        report = payload.get("report")
        if not isinstance(report, dict):
            raise ForwardIntegrityError("candidate report event payload is invalid")
        return report

    @staticmethod
    def _validate_manifest_value(value: Any, key: bytes) -> dict[str, Any]:
        if (
            not isinstance(value, dict)
            or value.get("schema")
            != "stock-radar-probability-forward-sealed-manifest"
            or value.get("schema_version") != 3
            or value.get("signing_key_id") != signing_key_id(key)
        ):
            raise RuntimeError("unsupported sealed forward manifest")
        manifest_hash = value.get("manifest_hash")
        core = {
            name: item
            for name, item in value.items()
            if name not in {"manifest_hash", "manifest_signature"}
        }
        if (
            not isinstance(manifest_hash, str)
            or sha256_bytes(canonical_json_bytes(core)) != manifest_hash
            or not hmac.compare_digest(
                signed_digest(manifest_hash, key),
                str(value.get("manifest_signature")),
            )
        ):
            raise RuntimeError("sealed forward manifest signature mismatch")
        return value

    @staticmethod
    def _manifest_matches_state(
        manifest: dict[str, Any] | None,
        *,
        event_count: int,
        event_head_hash: str,
        table_snapshot_root: str,
        manifest_hash: str | None,
    ) -> bool:
        if event_count == 0:
            return manifest is None and manifest_hash is None
        return bool(
            manifest is not None
            and int(manifest.get("event_count") or -1) == event_count
            and manifest.get("event_head_hash") == event_head_hash
            and manifest.get("table_snapshot_root") == table_snapshot_root
            and manifest.get("manifest_hash") == manifest_hash
        )

    def recover_seal(
        self,
        path: Path,
        key: bytes,
        *,
        candidate_report_path: Path | None = None,
        crash_injector: Callable[[str], None] | None = None,
    ) -> dict[str, Any]:
        path = Path(path)
        with self.transaction() as connection:
            integrity = self.verify_integrity(key)
            seal = connection.execute(
                "SELECT * FROM seal_state WHERE singleton = 1"
            ).fetchone()
            manifest = (
                self._validate_manifest_value(
                    json.loads(path.read_text(encoding="utf-8")),
                    key,
                )
                if path.exists()
                else None
            )
            if manifest is not None and int(manifest["event_count"]) > int(
                integrity["events"]
            ):
                raise ForwardIntegrityError(
                    "sealed manifest is ahead of the database event ledger"
                )
            finalized_matches = self._manifest_matches_state(
                manifest,
                event_count=int(seal["finalized_event_count"]),
                event_head_hash=seal["finalized_event_head_hash"],
                table_snapshot_root=seal["finalized_table_snapshot_root"],
                manifest_hash=seal["finalized_manifest_hash"],
            )
            pending_count = seal["pending_event_count"]
            if pending_count is None:
                if not finalized_matches:
                    raise ForwardIntegrityError(
                        "external manifest diverges from finalized database seal"
                    )
                if int(seal["finalized_event_count"]) != int(integrity["events"]):
                    raise ForwardIntegrityError(
                        "database head is unsealed without a pending seal intent"
                    )
                if integrity["latest_candidate_report_hash"] is not None:
                    if candidate_report_path is None:
                        raise ForwardIntegrityError(
                            "candidate report path is required to verify the seal"
                        )
                    report = self.latest_candidate_report()
                    if report is None:
                        raise ForwardIntegrityError(
                            "finalized candidate report event is missing"
                        )
                    if (
                        not Path(candidate_report_path).exists()
                        or json.loads(
                            Path(candidate_report_path).read_text(encoding="utf-8")
                        )
                        != report
                    ):
                        atomic_write_json(Path(candidate_report_path), report)
                return manifest
            pending_values = {
                "event_count": int(pending_count),
                "event_head_hash": seal["pending_event_head_hash"],
                "table_snapshot_root": seal["pending_table_snapshot_root"],
                "weekly_anchor_count": int(
                    seal["pending_weekly_anchor_count"] or 0
                ),
                "weekly_head_chain_hash": seal[
                    "pending_weekly_head_chain_hash"
                ],
                "latest_candidate_report_hash": seal[
                    "pending_candidate_report_hash"
                ],
            }
            if (
                pending_values["event_count"] != int(integrity["events"])
                or pending_values["event_head_hash"]
                != integrity["event_head_hash"]
                or pending_values["table_snapshot_root"]
                != integrity["table_snapshot_root"]
            ):
                raise ForwardIntegrityError(
                    "pending database seal does not match reconciled ledger state"
                )
            if pending_values["latest_candidate_report_hash"] is not None:
                if candidate_report_path is None:
                    raise ForwardIntegrityError(
                        "candidate report path is required to recover the seal"
                    )
                report = self.latest_candidate_report()
                if (
                    report is None
                    or report.get("report_hash")
                    != pending_values["latest_candidate_report_hash"]
                ):
                    raise ForwardIntegrityError(
                        "pending candidate report event is inconsistent"
                    )
                atomic_write_json(Path(candidate_report_path), report)
            pending_matches = bool(
                manifest is not None
                and int(manifest["event_count"])
                == pending_values["event_count"]
                and manifest["event_head_hash"]
                == pending_values["event_head_hash"]
                and manifest["table_snapshot_root"]
                == pending_values["table_snapshot_root"]
                and int(manifest["weekly_anchor_count"])
                == pending_values["weekly_anchor_count"]
                and manifest["weekly_head_chain_hash"]
                == pending_values["weekly_head_chain_hash"]
                and manifest.get("latest_candidate_report_hash")
                == pending_values["latest_candidate_report_hash"]
                and manifest.get("previous_seal_hash")
                == seal["finalized_manifest_hash"]
                and manifest.get("sealed_at") == seal["pending_created_at"]
            )
            if not pending_matches:
                if not finalized_matches:
                    raise ForwardIntegrityError(
                        "external manifest is neither finalized nor the exact "
                        "pending database head"
                    )
                if crash_injector is not None:
                    crash_injector("after_db_commit_before_manifest")
                core = {
                    "schema": "stock-radar-probability-forward-sealed-manifest",
                    "schema_version": 3,
                    "signing_key_id": signing_key_id(key),
                    "genesis_event_hash": connection.execute(
                        "SELECT event_hash FROM events WHERE sequence = 1"
                    ).fetchone()["event_hash"],
                    **pending_values,
                    "sealed_at": seal["pending_created_at"],
                    "previous_seal_hash": seal["finalized_manifest_hash"],
                    "rollback_protection": (
                        "monotonic only relative to this independently retained "
                        "seal; no external cryptographic timestamp"
                    ),
                }
                manifest_hash = sha256_bytes(canonical_json_bytes(core))
                manifest = {
                    **core,
                    "manifest_hash": manifest_hash,
                    "manifest_signature": signed_digest(manifest_hash, key),
                }
                atomic_write_json(path, manifest)
            else:
                manifest_hash = manifest["manifest_hash"]
            if crash_injector is not None:
                crash_injector("after_manifest_before_finalize")
            unchanged = connection.execute(
                """
                SELECT pending_event_count, pending_event_head_hash,
                       pending_table_snapshot_root
                FROM seal_state WHERE singleton = 1
                """
            ).fetchone()
            if (
                int(unchanged["pending_event_count"] or -1)
                != pending_values["event_count"]
                or unchanged["pending_event_head_hash"]
                != pending_values["event_head_hash"]
                or unchanged["pending_table_snapshot_root"]
                != pending_values["table_snapshot_root"]
            ):
                raise ForwardIntegrityError(
                    "pending seal changed during manifest finalization"
                )
            connection.execute(
                """
                UPDATE seal_state
                SET finalized_event_count = pending_event_count,
                    finalized_event_head_hash = pending_event_head_hash,
                    finalized_table_snapshot_root = pending_table_snapshot_root,
                    finalized_manifest_hash = ?,
                    pending_event_count = NULL,
                    pending_event_head_hash = NULL,
                    pending_table_snapshot_root = NULL,
                    pending_weekly_anchor_count = NULL,
                    pending_weekly_head_chain_hash = NULL,
                    pending_candidate_report_hash = NULL,
                    pending_created_at = NULL
                WHERE singleton = 1
                """,
                (manifest_hash,),
            )
            return manifest

    def verify_sealed_manifest(
        self,
        path: Path,
        key: bytes,
        *,
        candidate_report_path: Path | None = None,
    ) -> dict[str, Any]:
        value = self.recover_seal(
            path,
            key,
            candidate_report_path=candidate_report_path,
        )
        if value is None:
            raise ForwardIntegrityError("sealed forward manifest is missing")
        return value

    def export_manifest(
        self,
        path: Path,
        key: bytes,
        *,
        candidate_report_path: Path | None = None,
    ) -> dict[str, Any]:
        return self.verify_sealed_manifest(
            path,
            key,
            candidate_report_path=candidate_report_path,
        )

    def backup(
        self,
        backup_dir: Path,
        *,
        manifest_path: Path,
        candidate_report_path: Path | None = None,
        keep: int = 8,
    ) -> Path:
        backup_dir = Path(backup_dir)
        backup_dir.mkdir(parents=True, exist_ok=True)
        before = self.verify_integrity(self.signing_key)
        manifest = self.verify_sealed_manifest(
            manifest_path,
            self.signing_key,
            candidate_report_path=candidate_report_path,
        )
        timestamp = datetime.now(timezone.utc).strftime("%Y%m%d%H%M%S%f")
        final_path = backup_dir / f"f-{timestamp}.db"
        pending_path = backup_dir / f".{final_path.name}.new"
        manifest_backup = backup_dir / f"f-{timestamp}.manifest.json"
        candidate_backup = backup_dir / f"f-{timestamp}.candidate.json"
        destination = sqlite3.connect(pending_path)
        try:
            self.connection.backup(destination)
            integrity_result = destination.execute(
                "PRAGMA integrity_check"
            ).fetchone()[0]
            if integrity_result != "ok":
                raise RuntimeError(
                    f"SQLite backup integrity_check failed: {integrity_result}"
                )
        finally:
            destination.close()
        os.replace(pending_path, final_path)
        metadata = {
            row["key"]: json.loads(row["value_json"])
            for row in self.connection.execute(
                "SELECT key, value_json FROM metadata"
            )
        }
        write_immutable_bytes(
            manifest_backup,
            canonical_json_bytes(manifest) + b"\n",
        )
        latest_candidate = self.latest_candidate_report()
        if latest_candidate is not None:
            write_immutable_bytes(
                candidate_backup,
                canonical_json_bytes(latest_candidate) + b"\n",
            )
        with ForwardLedger(
            final_path,
            cohort_id=metadata["cohort_id"],
            artifact_hash=metadata["artifact_hash"],
            preregistration_hash=metadata["preregistration_hash"],
            signing_key=self.signing_key,
        ) as backed_up:
            after_backup = backed_up.verify_integrity(self.signing_key)
            if after_backup != before:
                raise RuntimeError("SQLite backup differs from the sealed source ledger")
            backed_up.verify_sealed_manifest(
                manifest_backup,
                self.signing_key,
                candidate_report_path=(
                    candidate_backup if latest_candidate is not None else None
                ),
            )
        after_source = self.verify_integrity(self.signing_key)
        if after_source != before:
            raise RuntimeError("source ledger changed during backup")
        backups = sorted(backup_dir.glob("f-*.db"))
        for stale in backups[:-max(1, int(keep))]:
            stale_manifest = stale.with_name(
                stale.name.replace(".db", ".manifest.json")
            )
            stale_candidate = stale.with_name(
                stale.name.replace(".db", ".candidate.json")
            )
            stale.unlink(missing_ok=True)
            stale_manifest.unlink(missing_ok=True)
            stale_candidate.unlink(missing_ok=True)
        return final_path


__all__ = [
    "ForwardLedger",
    "ForwardIntegrityError",
    "ForwardStaleSnapshotError",
    "GENESIS_CHAIN_HASH",
    "LEDGER_SCHEMA_VERSION",
    "canonical_json_bytes",
    "deterministic_gzip_json",
    "load_or_create_signing_key",
    "make_capture_envelope",
    "read_gzip_json",
    "sha256_bytes",
    "signed_digest",
    "signing_key_id",
    "validate_capture_envelope",
    "write_immutable_bytes",
    "write_immutable_json",
]
