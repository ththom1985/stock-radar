"""Versioned Yahoo panel, weekly anchors, labels, and purged temporal folds."""
from __future__ import annotations

import csv
import gzip
import hashlib
import io
import json
import math
import pickle
import re
import time
from dataclasses import dataclass
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Any, Callable, Iterable

import numpy as np
import pandas as pd
import yfinance as yf

from .config import DATA, PROBABILITY_HISTORY_START, ROOT
from .fetch import _symbol_frame, completed_daily_bars
from .persistence import atomic_write_bytes, atomic_write_json, load_json
from .probability_contract import (
    CLASS_NAMES,
    HORIZONS,
    ROUND_TRIP_COST,
    THRESHOLD_GRIDS,
    label_column,
    model_key,
    ordered_label_column,
)
from .probability_features import (
    FEATURE_NAMES,
    FEATURE_VERSION,
    MIN_HISTORY_BARS,
    build_probability_features,
    feature_schema_hash,
    probability_code_hash,
)

PANEL_SCHEMA_VERSION = 2
DATASET_SCHEMA_VERSION = 3
DEFAULT_START = PROBABILITY_HISTORY_START
DEFAULT_CACHE = DATA / "probability_cache"
PANEL_DIRNAME = "panel_v2"
DATASET_FILENAME = "weekly_dataset_v3.pkl.gz"
LEGACY_DATASET_FILENAME = "weekly_dataset_v2.pkl.gz"
MANIFEST_FILENAME = "manifest.json"
EMBARGO_DAYS = 7


@dataclass(frozen=True)
class EligibilityResult:
    symbols: tuple[str, ...]
    issuer_keys: dict[str, str]
    excluded: dict[str, str]


def classify_material_move(
    gross_return: float | np.ndarray | pd.Series,
    threshold: float,
    *,
    cost: float = ROUND_TRIP_COST,
) -> np.ndarray:
    """Classify a symmetric material move after a conservative friction dead-band.

    DOWN is gross return <= -(threshold + cost), UP is gross return >=
    +(threshold + cost), and every other finite observation is MIDDLE.  This
    direction-symmetric material-move convention is deliberately distinct from
    account P&L.  Long-only net P&L is stored separately as gross return - cost.
    """
    values = np.asarray(gross_return, dtype=float)
    result = np.full(values.shape, 1, dtype=np.int8)
    result[values <= -(float(threshold) + float(cost))] = 0
    result[values >= float(threshold) + float(cost)] = 2
    result[~np.isfinite(values)] = -1
    return result


def classify_ordered_move(
    gross_return: float | np.ndarray | pd.Series,
    thresholds: Iterable[float],
    *,
    cost: float = ROUND_TRIP_COST,
) -> np.ndarray:
    """Assign the preregistered seven mutually exclusive ordered return bins."""
    values = np.asarray(gross_return, dtype=float)
    selected = tuple(float(value) for value in thresholds)
    if (
        len(selected) != 3
        or not all(math.isfinite(value) and value > 0 for value in selected)
        or not selected[0] < selected[1] < selected[2]
    ):
        raise ValueError("ordered thresholds must be three finite increasing positives")
    a, b, c = (value + float(cost) for value in selected)
    result = np.full(values.shape, 3, dtype=np.int8)
    result[values <= -c] = 0
    result[(values > -c) & (values <= -b)] = 1
    result[(values > -b) & (values <= -a)] = 2
    result[(values >= a) & (values < b)] = 4
    result[(values >= b) & (values < c)] = 5
    result[values >= c] = 6
    result[~np.isfinite(values)] = -1
    return result


def cost_adjusted_material_return(
    gross_return: float | np.ndarray,
    *,
    cost: float = ROUND_TRIP_COST,
) -> np.ndarray:
    values = np.asarray(gross_return, dtype=float)
    return np.sign(values) * np.maximum(np.abs(values) - float(cost), 0.0)


def dataset_content_hash(dataset: pd.DataFrame) -> str:
    """Stable content hash independent of row/column order and host platform."""
    if not isinstance(dataset, pd.DataFrame):
        raise TypeError("dataset content hash requires a DataFrame")
    columns = sorted(str(column) for column in dataset.columns)
    frame = dataset.loc[:, columns].copy()
    sort_columns = [
        column
        for column in ("feature_date", "issuer_key", "symbol")
        if column in frame
    ]
    if sort_columns:
        frame = frame.sort_values(sort_columns, kind="mergesort")
    frame = frame.reset_index(drop=True)
    for column in columns:
        if pd.api.types.is_datetime64_any_dtype(frame[column]):
            frame[column] = (
                pd.to_datetime(frame[column], utc=True)
                .dt.strftime("%Y-%m-%dT%H:%M:%S.%fZ")
                .fillna("<NA>")
            )
    digest = hashlib.sha256()
    schema = [
        (column, str(frame[column].dtype))
        for column in columns
    ]
    digest.update(
        json.dumps(schema, separators=(",", ":"), ensure_ascii=True).encode("utf-8")
    )
    row_hashes = pd.util.hash_pandas_object(
        frame,
        index=False,
        categorize=False,
    ).to_numpy(dtype="<u8", copy=False)
    digest.update(row_hashes.tobytes())
    return digest.hexdigest()


def dataset_content_summary(dataset: pd.DataFrame) -> dict[str, Any]:
    dates = (
        pd.to_datetime(dataset["feature_date"], errors="coerce", utc=True)
        if "feature_date" in dataset
        else pd.Series([], dtype="datetime64[ns, UTC]")
    )
    symbols = sorted(
        set(dataset.get("symbol", pd.Series(dtype=str)).dropna().astype(str))
    )
    issuers = set(
        dataset.get("issuer_key", pd.Series(dtype=str)).dropna().astype(str)
    )
    return {
        "row_count": int(len(dataset)),
        "column_count": int(len(dataset.columns)),
        "symbol_count": len(symbols),
        "symbols_sha256": hashlib.sha256(
            "\n".join(symbols).encode("utf-8")
        ).hexdigest(),
        "issuer_count": len(issuers),
        "forecast_date_count": int(dates.nunique()),
        "first_feature_date": (
            dates.min().date().isoformat() if len(dates) and dates.notna().any() else None
        ),
        "last_feature_date": (
            dates.max().date().isoformat() if len(dates) and dates.notna().any() else None
        ),
    }


def _normalized_issuer_name(value: Any) -> str | None:
    text = re.sub(r"[^a-z0-9]+", "", str(value or "").casefold())
    for suffix in ("incorporated", "corporation", "company", "limited", "plc", "inc"):
        if text.endswith(suffix) and len(text) > len(suffix) + 3:
            text = text[: -len(suffix)]
            break
    return text or None


def _primary_listing_score(symbol: str, exchange: str) -> tuple[int, int, str]:
    exchange_rank = {
        "NYSE": 0,
        "NASDAQ": 1,
        "NYSE ARCA": 2,
        "AMEX": 3,
    }.get(str(exchange or "").upper(), 9)
    suffix_penalty = int("." in symbol or "-" in symbol or "=" in symbol)
    return exchange_rank, suffix_penalty, symbol


def select_eligible_universe(
    universe: Iterable[dict[str, Any]],
    metadata: dict[str, Any],
) -> EligibilityResult:
    """Keep one USD Yahoo EQUITY listing per available issuer key."""
    candidates: list[tuple[str, str, str]] = []
    excluded: dict[str, str] = {}
    for item in universe:
        symbol = str(item.get("symbol") or "").strip().upper()
        if not symbol:
            continue
        info = metadata.get(symbol)
        if not isinstance(info, dict):
            excluded[symbol] = "missing provider security metadata"
            continue
        if str(info.get("quote_type") or "").upper() != "EQUITY":
            excluded[symbol] = "not a provider-classified company equity"
            continue
        if str(info.get("reported_currency") or "").upper() != "USD":
            excluded[symbol] = "listing currency is not USD"
            continue
        issuer_key = str(info.get("issuer_uuid") or "").strip()
        if not issuer_key:
            normalized = _normalized_issuer_name(info.get("provider_long_name"))
            issuer_key = f"name:{normalized}" if normalized else f"symbol:{symbol}"
        candidates.append((symbol, issuer_key, str(item.get("exchange") or "")))

    selected: dict[str, tuple[str, str]] = {}
    for symbol, issuer_key, exchange in candidates:
        existing = selected.get(issuer_key)
        if existing is None or _primary_listing_score(
            symbol, exchange
        ) < _primary_listing_score(existing[0], existing[1]):
            if existing is not None:
                excluded[existing[0]] = f"duplicate issuer listing; retained {symbol}"
            selected[issuer_key] = (symbol, exchange)
        else:
            excluded[symbol] = f"duplicate issuer listing; retained {existing[0]}"
    symbols = tuple(sorted(value[0] for value in selected.values()))
    issuer_keys = {
        symbol: issuer_key
        for issuer_key, (symbol, _exchange) in selected.items()
    }
    return EligibilityResult(symbols=symbols, issuer_keys=issuer_keys, excluded=excluded)


def load_training_universe(
    tickers_path: Path = DATA / "tickers.csv",
    metadata_path: Path = DATA / "fundamentals.json",
) -> EligibilityResult:
    with tickers_path.open("r", encoding="utf-8-sig", newline="") as handle:
        universe = list(csv.DictReader(handle))
    metadata = load_json(metadata_path, required=True, expected_type=dict)
    return select_eligible_universe(universe, metadata)


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _frame_bytes(frame: pd.DataFrame) -> bytes:
    text = frame.to_csv(
        index=True,
        index_label="Date",
        date_format="%Y-%m-%dT%H:%M:%S%z",
        float_format="%.17g",
        lineterminator="\n",
    ).encode("utf-8")
    return gzip.compress(text, compresslevel=6, mtime=0)


def _write_frame(path: Path, frame: pd.DataFrame) -> None:
    atomic_write_bytes(path, _frame_bytes(frame))


def _read_frame(path: Path) -> pd.DataFrame:
    frame = pd.read_csv(path, compression="gzip", index_col="Date", parse_dates=["Date"])
    frame.index = pd.to_datetime(frame.index, utc=True)
    return frame


def write_exact_dataset_cache(path: Path, dataset: pd.DataFrame) -> None:
    """Atomically persist an exact trusted-local DataFrame checkpoint.

    Pickle can execute code while loading. This format is only for files created
    locally by this repository under ignored ``data/probability_cache``; never
    copy/download a pickle into that directory and load it as trusted input.
    """
    if not isinstance(dataset, pd.DataFrame):
        raise TypeError("exact dataset cache requires a pandas DataFrame")
    payload = pickle.dumps(dataset, protocol=5)
    atomic_write_bytes(Path(path), gzip.compress(payload, compresslevel=6, mtime=0))


def read_exact_dataset_cache(path: Path) -> pd.DataFrame:
    """Load a trusted-local exact checkpoint; caller must verify file SHA first."""
    try:
        payload = gzip.decompress(Path(path).read_bytes())
        dataset = pickle.loads(payload)
    except Exception as exc:
        raise ValueError(f"cannot load exact local dataset checkpoint: {exc}") from exc
    if not isinstance(dataset, pd.DataFrame):
        raise ValueError("exact local dataset checkpoint root is not a DataFrame")
    return dataset


def _default_history_download(
    symbols: list[str],
    start: str,
    end: str,
) -> pd.DataFrame:
    return yf.download(
        tickers=symbols,
        start=start,
        end=end,
        interval="1d",
        group_by="ticker",
        auto_adjust=False,
        actions=True,
        repair=False,
        threads=True,
        progress=False,
        timeout=30,
    )


def download_probability_panel(
    symbols: Iterable[str],
    *,
    start: str = DEFAULT_START,
    end: str | None = None,
    cache_dir: Path = DEFAULT_CACHE,
    batch_size: int = 100,
    retries: int = 2,
    now: datetime | None = None,
    resume: bool = True,
    downloader: Callable[[list[str], str, str], pd.DataFrame] | None = None,
) -> dict[str, Any]:
    """Build/restart a checksummed per-symbol Yahoo panel with split fallback."""
    now = now or datetime.now(timezone.utc)
    end = end or (now.date() + timedelta(days=1)).isoformat()
    panel_dir = Path(cache_dir) / PANEL_DIRNAME
    panel_dir.mkdir(parents=True, exist_ok=True)
    manifest_path = panel_dir / MANIFEST_FILENAME
    requested = tuple(sorted(set(str(symbol).upper() for symbol in symbols)))
    panel_cache_key = hashlib.sha256(
        json.dumps(
            {
                "requested_symbols": requested,
                "start": start,
                "end": end,
                "feature_version": FEATURE_VERSION,
                "feature_schema_hash": feature_schema_hash(),
                "code_hash": probability_code_hash(),
                "batch_size": batch_size,
                "retries": retries,
                "source": "yahoo-yfinance-daily-actions-auto_adjust_false",
            },
            sort_keys=True,
            separators=(",", ":"),
        ).encode("utf-8")
    ).hexdigest()
    existing = load_json(manifest_path, expected_type=dict, default={}) if resume else {}
    compatible = (
        existing.get("schema_version") == PANEL_SCHEMA_VERSION
        and existing.get("panel_cache_key") == panel_cache_key
    )
    files: dict[str, Any] = dict(existing.get("files") or {}) if compatible else {}
    failures: dict[str, str] = dict(existing.get("failures") or {}) if compatible else {}
    completed: set[str] = set()
    if compatible:
        for symbol, record in files.items():
            path = panel_dir / str(record.get("file") or "")
            if path.exists() and record.get("sha256") == _sha256(path):
                completed.add(symbol)

    retrieve = downloader or _default_history_download

    def checkpoint() -> None:
        manifest = {
            "schema": "stock-radar-probability-panel",
            "schema_version": PANEL_SCHEMA_VERSION,
            "panel_cache_key": panel_cache_key,
            "source": "Yahoo Finance via yfinance; adjusted OHLC reconstructed from Adj Close",
            "retrieved_at": now.isoformat(),
            "universe_membership_as_of": now.isoformat(),
            "universe_membership_mode": "current configured/provider-filtered universe",
            "start": start,
            "end": end,
            "requested_symbols": list(requested),
            "completed_symbols": sorted(completed),
            "requested_symbol_count": len(requested),
            "successful_symbol_count": len(completed),
            "provider_success_coverage": (
                len(completed) / len(requested) if requested else 0.0
            ),
            "failures": dict(sorted(failures.items())),
            "files": {key: files[key] for key in sorted(files)},
            "feature_version": FEATURE_VERSION,
            "feature_schema_hash": feature_schema_hash(),
            "code_hash": probability_code_hash(),
            "pit_action_reconstruction": False,
            "pit_limitation": (
                "Yahoo's current adjusted history is used. Later split/dividend factors "
                "may uniformly rescale pre-event adjusted OHLC; dimensionless features are "
                "invariant to that scaling. Contemporaneous RawClose is used for liquidity. "
                "A fully point-in-time action tape is not available from this source."
            ),
        }
        atomic_write_json(manifest_path, manifest)

    def ingest(chunk: list[str]) -> None:
        pending = [symbol for symbol in chunk if symbol not in completed]
        if not pending:
            return
        data: pd.DataFrame | None = None
        error: Exception | None = None
        for attempt in range(retries + 1):
            try:
                data = retrieve(pending, start, end)
                error = None
                break
            except Exception as exc:  # provider failure types vary
                error = exc
                if attempt < retries:
                    time.sleep(min(2**attempt, 4))
        if error is not None:
            if len(pending) > 1:
                midpoint = len(pending) // 2
                ingest(pending[:midpoint])
                ingest(pending[midpoint:])
            else:
                failures[pending[0]] = f"download failed: {str(error)[:240]}"
            checkpoint()
            return

        missing: list[str] = []
        for symbol in pending:
            try:
                raw = _symbol_frame(data, symbol, len(pending))
                frame, info = completed_daily_bars(raw, now=now, symbol=symbol)
                if frame.empty or len(frame) < MIN_HISTORY_BARS + 2:
                    raise ValueError(
                        f"insufficient completed history ({len(frame)} bars)"
                    )
                keep = [
                    column
                    for column in (
                        "Open",
                        "High",
                        "Low",
                        "Close",
                        "Volume",
                        "RawOpen",
                        "RawHigh",
                        "RawLow",
                        "RawClose",
                        "Dividends",
                        "Stock Splits",
                    )
                    if column in frame
                ]
                frame = frame.loc[:, keep]
                filename = f"{re.sub(r'[^A-Za-z0-9._-]+', '_', symbol)}.csv.gz"
                path = panel_dir / filename
                _write_frame(path, frame)
                files[symbol] = {
                    "file": filename,
                    "sha256": _sha256(path),
                    "bytes": path.stat().st_size,
                    "rows": len(frame),
                    "first_session": frame.index[0].date().isoformat(),
                    "last_session": frame.index[-1].date().isoformat(),
                    "last_bar_timestamp": info.get("bar_timestamp"),
                    "action_rows": int(
                        (
                            frame.get("Dividends", pd.Series(0.0, index=frame.index))
                            .fillna(0)
                            .ne(0)
                            | frame.get(
                                "Stock Splits", pd.Series(0.0, index=frame.index)
                            )
                            .fillna(0)
                            .ne(0)
                        ).sum()
                    ),
                }
                completed.add(symbol)
                failures.pop(symbol, None)
            except Exception as exc:
                missing.append(symbol)
                failures[symbol] = f"invalid/insufficient response: {str(exc)[:240]}"
        checkpoint()
        if len(pending) > 1:
            for symbol in missing:
                completed.discard(symbol)
                ingest([symbol])

    for offset in range(0, len(requested), max(1, batch_size)):
        ingest(list(requested[offset : offset + batch_size]))
    checkpoint()
    return load_json(manifest_path, required=True, expected_type=dict)


def load_probability_panel(
    cache_dir: Path = DEFAULT_CACHE,
    *,
    verify_checksums: bool = True,
) -> tuple[dict[str, pd.DataFrame], dict[str, Any]]:
    panel_dir = Path(cache_dir) / PANEL_DIRNAME
    manifest = load_json(
        panel_dir / MANIFEST_FILENAME, required=True, expected_type=dict
    )
    if (
        manifest.get("schema_version") != PANEL_SCHEMA_VERSION
        or manifest.get("feature_version") != FEATURE_VERSION
        or manifest.get("feature_schema_hash") != feature_schema_hash()
    ):
        raise ValueError("probability panel manifest is incompatible")
    histories: dict[str, pd.DataFrame] = {}
    for symbol, record in sorted((manifest.get("files") or {}).items()):
        path = panel_dir / record["file"]
        if verify_checksums and _sha256(path) != record.get("sha256"):
            raise ValueError(f"panel checksum mismatch for {symbol}")
        histories[symbol] = _read_frame(path)
    return histories, manifest


def _weekly_anchor_positions(index: pd.DatetimeIndex) -> list[int]:
    calendar = index.isocalendar()
    keys = list(zip(calendar["year"].astype(int), calendar["week"].astype(int)))
    positions: dict[tuple[int, int], int] = {}
    for position, key in enumerate(keys):
        positions[key] = position
    return [positions[key] for key in sorted(positions)]


def build_symbol_dataset(
    symbol: str,
    history: pd.DataFrame,
    spy_history: pd.DataFrame,
    *,
    issuer_key: str | None = None,
    cost: float = ROUND_TRIP_COST,
) -> pd.DataFrame:
    """Create deterministic weekly feature/label rows for one eligible listing."""
    frame = history.sort_index().copy()
    frame.index = pd.to_datetime(frame.index, utc=True).tz_convert(None).normalize()
    frame = frame[~frame.index.duplicated(keep="last")]
    features = build_probability_features(frame, spy_history)
    records: list[dict[str, Any]] = []
    for position in _weekly_anchor_positions(frame.index):
        if position < MIN_HISTORY_BARS:
            continue
        if position + min(HORIZONS) >= len(frame):
            continue
        feature_date = frame.index[position]
        feature_values = features.reindex([feature_date]).iloc[0]
        record: dict[str, Any] = {
            "symbol": symbol,
            "issuer_key": issuer_key or symbol,
            "feature_date": feature_date,
            "feature_timestamp": feature_date,
            "entry_timestamp": frame.index[position + 1],
            "history_start": frame.index[0],
            "history_bars_before": position,
        }
        for name in FEATURE_NAMES:
            record[name] = float(feature_values[name])
        entry_open = float(frame["Open"].iloc[position + 1])
        if not math.isfinite(entry_open) or entry_open <= 0:
            continue
        for horizon in HORIZONS:
            exit_position = position + horizon
            if exit_position >= len(frame):
                record[f"exit_timestamp_h{horizon}"] = pd.NaT
                record[f"gross_return_h{horizon}"] = np.nan
                record[f"long_net_return_h{horizon}"] = np.nan
                record[f"material_net_return_h{horizon}"] = np.nan
                record[ordered_label_column(horizon)] = np.nan
                for threshold_pct in THRESHOLD_GRIDS[horizon]:
                    record[label_column(horizon, threshold_pct)] = np.nan
                continue
            exit_close = float(frame["Close"].iloc[exit_position])
            if not math.isfinite(exit_close) or exit_close <= 0:
                record[f"exit_timestamp_h{horizon}"] = pd.NaT
                record[f"gross_return_h{horizon}"] = np.nan
                record[f"long_net_return_h{horizon}"] = np.nan
                record[f"material_net_return_h{horizon}"] = np.nan
                record[ordered_label_column(horizon)] = np.nan
                for threshold_pct in THRESHOLD_GRIDS[horizon]:
                    record[label_column(horizon, threshold_pct)] = np.nan
                continue
            gross = exit_close / entry_open - 1.0
            record[f"exit_timestamp_h{horizon}"] = frame.index[exit_position]
            record[f"gross_return_h{horizon}"] = gross
            record[f"long_net_return_h{horizon}"] = gross - cost
            record[f"material_net_return_h{horizon}"] = float(
                cost_adjusted_material_return(gross, cost=cost)
            )
            record[ordered_label_column(horizon)] = int(
                classify_ordered_move(
                    gross,
                    (
                        threshold_pct / 100.0
                        for threshold_pct in THRESHOLD_GRIDS[horizon]
                    ),
                    cost=cost,
                )
            )
            for threshold_pct in THRESHOLD_GRIDS[horizon]:
                record[label_column(horizon, threshold_pct)] = int(
                    classify_material_move(
                        gross, threshold_pct / 100.0, cost=cost
                    )
                )
        available_exits = [
            record[f"exit_timestamp_h{horizon}"]
            for horizon in HORIZONS
            if not pd.isna(record[f"exit_timestamp_h{horizon}"])
        ]
        record["max_exit_date"] = max(available_exits)
        records.append(record)
    result = pd.DataFrame.from_records(records)
    if not result.empty:
        result = result.sort_values(["feature_date", "issuer_key", "symbol"]).reset_index(
            drop=True
        )
    return result


def build_weekly_dataset(
    histories: dict[str, pd.DataFrame],
    *,
    spy_symbol: str = "SPY",
    issuer_keys: dict[str, str] | None = None,
    cache_dir: Path | None = None,
    panel_manifest: dict[str, Any] | None = None,
) -> tuple[pd.DataFrame, dict[str, Any]]:
    if spy_symbol not in histories:
        raise ValueError("SPY history is required for market features")
    issuer_keys = issuer_keys or {}
    pieces = []
    failures: dict[str, str] = {}
    for symbol in sorted(histories):
        if symbol == spy_symbol:
            continue
        try:
            piece = build_symbol_dataset(
                symbol,
                histories[symbol],
                histories[spy_symbol],
                issuer_key=issuer_keys.get(symbol),
            )
            if piece.empty:
                failures[symbol] = "no complete weekly anchors and future labels"
            else:
                pieces.append(piece)
        except Exception as exc:
            failures[symbol] = f"dataset build failed: {str(exc)[:240]}"
    dataset = (
        pd.concat(pieces, ignore_index=True)
        if pieces
        else pd.DataFrame(
            columns=[
                "symbol",
                "issuer_key",
                "feature_date",
                *FEATURE_NAMES,
            ]
        )
    )
    dataset = dataset.sort_values(
        ["feature_date", "issuer_key", "symbol"]
    ).reset_index(drop=True)
    metadata = {
        "schema": "stock-radar-probability-dataset",
        "schema_version": DATASET_SCHEMA_VERSION,
        "feature_version": FEATURE_VERSION,
        "feature_schema_hash": feature_schema_hash(),
        "code_hash": probability_code_hash(),
        "row_count": len(dataset),
        "symbol_count": int(dataset["symbol"].nunique()) if not dataset.empty else 0,
        "issuer_count": int(dataset["issuer_key"].nunique()) if not dataset.empty else 0,
        "forecast_date_count": (
            int(dataset["feature_date"].nunique()) if not dataset.empty else 0
        ),
        "failures": failures,
        "weekly_anchor": "final completed session in each ISO week",
        "entry": "first adjusted open strictly after feature session t",
        "exit": "adjusted close at t + H sessions",
        "label_definition": (
            "DOWN gross<=-(X+0.003), MIDDLE otherwise, UP gross>=X+0.003; "
            "long_net_return=gross-0.003 is stored but no positive-return "
            "probability is inferred from material-threshold classes"
        ),
        "ordered_label_definition": (
            "seven disjoint bins at symmetric a<b<c thresholds plus 0.003 cost; "
            "negative equalities enter the lower tail and positive equalities "
            "enter the upper tail"
        ),
        "panel_manifest_sha256": None,
    }
    if panel_manifest:
        encoded = json.dumps(
            panel_manifest, sort_keys=True, separators=(",", ":")
        ).encode("utf-8")
        metadata["panel_manifest_sha256"] = hashlib.sha256(encoded).hexdigest()
    if cache_dir is not None:
        output = Path(cache_dir) / DATASET_FILENAME
        write_exact_dataset_cache(output, dataset)
        metadata["file"] = output.name
        metadata["storage_format"] = "trusted-local-pandas-pickle-protocol5-gzip"
        metadata["trust_boundary"] = (
            "Load only repository-generated files under ignored "
            "data/probability_cache; pickle is unsafe for untrusted input."
        )
        metadata["legacy_dataset_ignored"] = LEGACY_DATASET_FILENAME
        metadata["sha256"] = _sha256(output)
        atomic_write_json(Path(cache_dir) / "dataset_manifest.json", metadata)
    return dataset, metadata


def load_weekly_dataset(
    cache_dir: Path = DEFAULT_CACHE,
) -> tuple[pd.DataFrame, dict[str, Any]]:
    manifest = load_json(
        Path(cache_dir) / "dataset_manifest.json",
        required=True,
        expected_type=dict,
    )
    if (
        manifest.get("schema_version") != DATASET_SCHEMA_VERSION
        or manifest.get("feature_version") != FEATURE_VERSION
        or manifest.get("feature_schema_hash") != feature_schema_hash()
        or manifest.get("code_hash") != probability_code_hash()
    ):
        raise ValueError("probability dataset manifest is incompatible with code")
    for required in (
        "dataset_content_hash",
        "dataset_cache_key",
        "storage_format",
        "trust_boundary",
        "provider_requested_issuer_count",
        "provider_successful_issuer_count",
        "provider_success_coverage",
        "provider_unavailable_symbols",
        "panel_manifest_sha256",
    ):
        if required not in manifest:
            raise ValueError(
                f"probability dataset manifest is incomplete: missing {required}"
            )
    path = Path(cache_dir) / manifest["file"]
    if _sha256(path) != manifest.get("sha256"):
        raise ValueError("probability dataset checksum mismatch")
    if manifest.get("storage_format") != "trusted-local-pandas-pickle-protocol5-gzip":
        raise ValueError("legacy/lossy probability dataset cache is not loadable; rebuild it")
    dataset = read_exact_dataset_cache(path)
    if (
        manifest.get("dataset_content_hash")
        and dataset_content_hash(dataset) != manifest["dataset_content_hash"]
    ):
        raise ValueError("probability dataset content hash mismatch")
    return dataset, manifest


def make_purged_expanding_folds(
    dataset: pd.DataFrame,
    *,
    minimum_train_years: int = 5,
    calibration_months: int = 12,
    test_months: int = 12,
    embargo_days: int = EMBARGO_DAYS,
    minimum_folds: int = 0,
) -> list[dict[str, Any]]:
    """Build annual expanding folds with an explicit longest-horizon purge gap."""
    if dataset.empty:
        return []
    dates = pd.to_datetime(dataset["feature_date"]).dt.tz_localize(None)
    if (
        "history_bars_before" in dataset
        and (pd.to_numeric(dataset["history_bars_before"], errors="coerce") < 252).any()
    ):
        raise ValueError("fold input contains a feature row before 252-bar warm-up")
    max_exit = pd.to_datetime(dataset["max_exit_date"]).dt.tz_localize(None)
    first_calibration = pd.Timestamp(dates.min()) + pd.DateOffset(
        years=minimum_train_years
    )
    first_calibration = pd.Timestamp(
        year=first_calibration.year,
        month=first_calibration.month,
        day=1,
    )
    latest = dates.max()
    folds: list[dict[str, Any]] = []
    fold_number = 0
    calibration_start = first_calibration
    while True:
        calibration_end = calibration_start + pd.DateOffset(
            months=calibration_months
        )
        calibration_window = (dates >= calibration_start) & (
            dates < calibration_end
        )
        calibration_max_exit = max_exit.loc[calibration_window].max()
        if pd.isna(calibration_max_exit):
            calibration_start += pd.DateOffset(years=1)
            continue
        test_start = max(
            calibration_end,
            pd.Timestamp(calibration_max_exit)
            + pd.Timedelta(days=embargo_days + 1),
        )
        test_end = test_start + pd.DateOffset(months=test_months)
        if test_end > latest:
            break
        train_cutoff = calibration_start - pd.Timedelta(days=embargo_days)
        calibration_cutoff = test_start - pd.Timedelta(days=embargo_days)
        train_mask = (dates < calibration_start) & (max_exit < train_cutoff)
        calibration_mask = (
            (dates >= calibration_start)
            & (dates < calibration_end)
            & (max_exit < calibration_cutoff)
        )
        test_mask = (dates >= test_start) & (dates < test_end)
        train_dates = dates.loc[train_mask]
        test_dates = dates.loc[test_mask]
        usable_train_years = (
            (train_dates.max() - train_dates.min()).days / 365.2425
            if len(train_dates) > 1
            else 0.0
        )
        full_test_window = bool(
            len(test_dates)
            and test_dates.min() <= test_start + pd.Timedelta(days=7)
            and test_dates.max() >= test_end - pd.Timedelta(days=7)
        )
        fold = {
            "fold": fold_number,
            "train_indices": np.flatnonzero(train_mask.to_numpy()),
            "calibration_indices": np.flatnonzero(calibration_mask.to_numpy()),
            "test_indices": np.flatnonzero(test_mask.to_numpy()),
            "train_start": dates.loc[train_mask].min(),
            "train_end": dates.loc[train_mask].max(),
            "calibration_start": calibration_start,
            "calibration_end": calibration_end,
            "test_start": test_start,
            "test_end": test_end,
            "embargo_days": embargo_days,
            "purge_horizon_sessions": max(HORIZONS),
            "usable_train_years": usable_train_years,
            "full_test_window": full_test_window,
        }
        if all(
            len(fold[key])
            for key in (
                "train_indices",
                "calibration_indices",
                "test_indices",
            )
        ) and usable_train_years >= minimum_train_years and full_test_window:
            train_exit = max_exit.iloc[fold["train_indices"]]
            calibration_exit = max_exit.iloc[fold["calibration_indices"]]
            if not (train_exit < train_cutoff).all():
                raise AssertionError("training label interval crosses calibration")
            if not (calibration_exit < calibration_cutoff).all():
                raise AssertionError("calibration label interval crosses test")
            folds.append(fold)
            fold_number += 1
        calibration_start += pd.DateOffset(years=1)
    if len(folds) < minimum_folds:
        raise ValueError(
            f"only {len(folds)} purged outer folds; {minimum_folds} required"
        )
    return folds


def summarize_fold(fold: dict[str, Any]) -> dict[str, Any]:
    return {
        key: (
            value.isoformat()
            if isinstance(value, (pd.Timestamp, datetime))
            else len(value)
            if isinstance(value, np.ndarray)
            else value
        )
        for key, value in fold.items()
    }


__all__ = [
    "CLASS_NAMES",
    "DATASET_FILENAME",
    "DEFAULT_CACHE",
    "DEFAULT_START",
    "EMBARGO_DAYS",
    "HORIZONS",
    "ROUND_TRIP_COST",
    "THRESHOLD_GRIDS",
    "EligibilityResult",
    "build_symbol_dataset",
    "build_weekly_dataset",
    "classify_material_move",
    "classify_ordered_move",
    "cost_adjusted_material_return",
    "dataset_content_hash",
    "dataset_content_summary",
    "download_probability_panel",
    "label_column",
    "load_probability_panel",
    "load_training_universe",
    "load_weekly_dataset",
    "make_purged_expanding_folds",
    "model_key",
    "ordered_label_column",
    "select_eligible_universe",
    "summarize_fold",
]
