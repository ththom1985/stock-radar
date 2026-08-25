"""Append-only expert observation journal and horizon outcome evaluation."""
from __future__ import annotations

import hashlib
import json
import os
from datetime import datetime, timezone
from pathlib import Path

from .config import DATA
from .persistence import effective_path

LOG_PATH = DATA / "recommendation_log.jsonl"
OUTCOME_PATH = DATA / "recommendation_outcomes.jsonl"
HORIZONS = {"1m": 21, "3m": 63, "6m": 126, "12m": 252}
MIN_CALIBRATION_WINDOWS = 100


def _read_jsonl(path):
    resolved = effective_path(path, for_write=False)
    if not resolved.exists():
        return []
    rows = []
    for line_number, line in enumerate(
        resolved.read_text(encoding="utf-8").splitlines(), 1
    ):
        if not line.strip():
            continue
        try:
            value = json.loads(line)
        except json.JSONDecodeError as exc:
            raise ValueError(f"corrupt JSONL {resolved}:{line_number}: {exc}") from exc
        if not isinstance(value, dict):
            raise ValueError(f"invalid JSONL object {resolved}:{line_number}")
        rows.append(value)
    return rows


def _write_jsonl(path, rows):
    resolved = effective_path(path, for_write=True)
    resolved.parent.mkdir(parents=True, exist_ok=True)
    temporary = resolved.with_name(f".{resolved.name}.{os.getpid()}.tmp")
    text = "".join(
        json.dumps(row, ensure_ascii=False, sort_keys=True) + "\n" for row in rows
    )
    with temporary.open("w", encoding="utf-8", newline="\n") as handle:
        handle.write(text)
        handle.flush()
        os.fsync(handle.fileno())
    os.replace(temporary, resolved)


def _observation_id(observed_date, symbol, horizon):
    value = f"{observed_date}|{symbol}|{horizon}".encode("utf-8")
    return hashlib.sha256(value).hexdigest()[:24]


def record_top_observations(rows, rankings, generated_at):
    observed_at = datetime.fromisoformat(str(generated_at).replace("Z", "+00:00"))
    if observed_at.tzinfo is None:
        observed_at = observed_at.replace(tzinfo=timezone.utc)
    observed_date = observed_at.date().isoformat()
    existing = _read_jsonl(LOG_PATH)
    existing_ids = {row.get("id") for row in existing}
    by_symbol = {row.get("symbol"): row for row in rows}
    additions = []
    for horizon in ("long_term", "short_term"):
        for rank, item in enumerate(rankings.get(horizon) or [], 1):
            symbol = item.get("symbol")
            source = by_symbol.get(symbol) or {}
            observation_id = _observation_id(observed_date, symbol, horizon)
            if observation_id in existing_ids:
                continue
            analysis = source.get("expert_analysis") or {}
            detail = analysis.get(horizon) or {}
            additions.append(
                {
                    "id": observation_id,
                    "observed_at": observed_at.astimezone(timezone.utc).isoformat(),
                    "bar_date": source.get("bar_date"),
                    "symbol": symbol,
                    "name": source.get("display_name_full") or source.get("name"),
                    "currency": source.get("currency"),
                    "price_local": source.get("price_local"),
                    "horizon": horizon,
                    "rank": rank,
                    "score": detail.get("score"),
                    "coverage_pct": detail.get("coverage_pct"),
                    "signal": analysis.get("signal"),
                    "evidence_quality": analysis.get("evidence_quality"),
                    "components": detail.get("components"),
                    "alternative_signals": source.get("alternative_signals") or {},
                    "confluence_tier": (
                        source.get("alternative_signals") or {}
                    ).get("confluence_tier"),
                    "contributing_groups": (
                        source.get("alternative_signals") or {}
                    ).get("contributing_groups") or [],
                    "actionable": False,
                }
            )
    if additions:
        _write_jsonl(LOG_PATH, [*existing, *additions])
    return additions


def record_confluence_observations(rows, generated_at):
    observed_at = datetime.fromisoformat(str(generated_at).replace("Z", "+00:00"))
    if observed_at.tzinfo is None:
        observed_at = observed_at.replace(tzinfo=timezone.utc)
    observed_date = observed_at.date().isoformat()
    existing = _read_jsonl(LOG_PATH)
    existing_ids = {row.get("id") for row in existing}
    additions = []
    for row in rows:
        alternative = row.get("alternative_signals") or {}
        if alternative.get("activation_status") != "active":
            continue
        observation_id = _observation_id(
            observed_date, row.get("symbol"), "confluence"
        )
        if observation_id in existing_ids:
            continue
        additions.append(
            {
                "id": observation_id,
                "observed_at": observed_at.astimezone(timezone.utc).isoformat(),
                "bar_date": row.get("bar_date"),
                "symbol": row.get("symbol"),
                "name": row.get("display_name_full") or row.get("name"),
                "currency": row.get("currency"),
                "price_local": row.get("price_local"),
                "horizon": "confluence",
                "score": alternative.get("confluence_score"),
                "signal": "confluence_active",
                "confluence_tier": alternative.get("confluence_tier"),
                "contributing_groups": alternative.get("contributing_groups") or [],
                "alternative_signals": alternative,
                "actionable": False,
            }
        )
    if additions:
        _write_jsonl(LOG_PATH, [*existing, *additions])
    return additions


def evaluate_mature_observations(price_histories):
    observations = _read_jsonl(LOG_PATH)
    outcomes = _read_jsonl(OUTCOME_PATH)
    existing = {(row.get("observation_id"), row.get("horizon")) for row in outcomes}
    additions = []
    for observation in observations:
        history = price_histories.get(observation.get("symbol"))
        if history is None or getattr(history, "empty", True):
            continue
        bar_date = observation.get("bar_date")
        if not bar_date:
            continue
        dates = [index.date().isoformat() for index in history.index]
        try:
            start_index = dates.index(bar_date)
        except ValueError:
            continue
        entry_price = observation.get("price_local")
        if not isinstance(entry_price, (int, float)) or entry_price <= 0:
            continue
        for horizon, sessions in HORIZONS.items():
            key = (observation["id"], horizon)
            end_index = start_index + sessions
            if key in existing or end_index >= len(history):
                continue
            end_price = float(history.iloc[end_index]["RawClose"])
            return_pct = (end_price / entry_price - 1.0) * 100.0
            additions.append(
                {
                    "observation_id": observation["id"],
                    "symbol": observation["symbol"],
                    "horizon": horizon,
                    "sessions": sessions,
                    "start_bar_date": bar_date,
                    "end_bar_date": dates[end_index],
                    "start_price_local": entry_price,
                    "end_price_local": end_price,
                    "return_pct": round(return_pct, 4),
                    "positive": return_pct > 0,
                    "evaluated_at": datetime.now(timezone.utc).isoformat(),
                }
            )
    if additions:
        _write_jsonl(OUTCOME_PATH, [*outcomes, *additions])
    return additions


def journal_summary():
    observations = _read_jsonl(LOG_PATH)
    outcomes = _read_jsonl(OUTCOME_PATH)
    by_horizon = {}
    for horizon in HORIZONS:
        subset = [row for row in outcomes if row.get("horizon") == horizon]
        by_horizon[horizon] = {
            "evaluated": len(subset),
            "positive": sum(bool(row.get("positive")) for row in subset),
            "hit_rate_pct": (
                round(
                    sum(bool(row.get("positive")) for row in subset)
                    / len(subset)
                    * 100.0,
                    1,
                )
                if subset
                else None
            ),
            "calibration_status": (
                "calibrated"
                if len(subset) >= MIN_CALIBRATION_WINDOWS * 3
                else "coarse"
                if len(subset) >= MIN_CALIBRATION_WINDOWS
                else "uncalibrated"
            ),
            "minimum_windows": MIN_CALIBRATION_WINDOWS,
            "windows_remaining": max(0, MIN_CALIBRATION_WINDOWS - len(subset)),
            "probability_band": (
                [
                    max(0, round(
                        sum(bool(row.get("positive")) for row in subset)
                        / len(subset) * 100 - 10
                    )),
                    min(100, round(
                        sum(bool(row.get("positive")) for row in subset)
                        / len(subset) * 100 + 10
                    )),
                ]
                if len(subset) >= MIN_CALIBRATION_WINDOWS
                else None
            ),
        }
    return {
        "schema": "stock-radar-recommendation-journal-summary",
        "schema_version": 1,
        "model_status": "unvalidated",
        "actionable": False,
        "observation_count": len(observations),
        "outcome_count": len(outcomes),
        "by_horizon": by_horizon,
    }
