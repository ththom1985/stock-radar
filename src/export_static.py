"""Export a compact, login-free GitHub Pages insight payload."""
from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from .data_quality import validate_insight_contract, validate_output_contract
from .persistence import atomic_write_bytes, load_json, schema_meta

ROOT = Path(__file__).resolve().parent.parent
DEFAULT_INPUT = ROOT / "data" / "output" / "latest.json"
DEFAULT_OUTPUT = ROOT / "docs" / "data.json"
STATIC_SCHEMA_VERSION = 2
MAX_STATIC_BYTES = 10 * 1024 * 1024

ROW_FIELDS = (
    "symbol",
    "name",
    "asset_type",
    "currency",
    "country",
    "cc",
    "sector",
    "industry",
    "price",
    "price_local",
    "fx_usd",
    "bar_date",
    "bar_age_days",
    "bar_timestamp",
    "source_interval",
    "radar_score",
    "longterm_score",
    "longterm_reasons",
    "daily_signal_score",
    "daily_signal_direction",
    "daily_signal_reasons",
    "entry_timing_score",
    "entry_timing_label",
    "entry_timing_reason",
    "entry_timing",
    "falling_knife",
    "bottoming",
    "downside_structure",
    "risk_warnings",
    "risk_context",
    "bull_thesis",
    "priced_in_note",
    "thesis_context",
    "trend_phase",
    "research_summary",
    "research_actions",
    "research_context",
    "analyst_context",
    "valuation_context",
    "potential_context",
    "technical_observation_zone",
    "insight_provenance",
    "scenario_long",
    "next_earnings",
    "earnings_in_days",
    "news",
    "rsi",
    "macd",
    "macd_signal",
    "macd_hist",
    "macd_hist_prev",
    "ret_20d",
    "ret_60d",
    "pct_from_high52",
    "atr_pct",
    "vol_annual_pct",
    "rvol",
    "minervini_score",
    "weinstein_stage",
    "weinstein_label",
    "piotroski",
    "altman_z",
)


def _contains_actionable_true(value: Any) -> bool:
    if isinstance(value, dict):
        return value.get("actionable") is True or any(
            _contains_actionable_true(item) for item in value.values()
        )
    if isinstance(value, list):
        return any(_contains_actionable_true(item) for item in value)
    return False


def _compact_group(
    value: Any,
    keys: tuple[str, ...],
    provenance_ref: str,
) -> Any:
    if not isinstance(value, dict):
        return value
    base = {
        key: value.get(key)
        for key in ("model_status", "actionable", "missing_inputs")
    }
    base["inputs_used"] = [f"catalog:{provenance_ref}"]
    base.update({key: value.get(key) for key in keys})
    return base


def _compact_row(row: dict[str, Any]) -> dict[str, Any]:
    compact = {key: row.get(key) for key in ROW_FIELDS}
    compact["news"] = (compact.get("news") or [])[:3]
    compact["scenario_long"] = [
        {
            key: scenario.get(key)
            for key in (
                "label",
                "reference_change_pct",
                "reference_price",
                "range_low_pct",
                "range_high_pct",
                "range_low_price",
                "range_high_price",
                "model_status",
                "interpretation",
            )
        }
        for scenario in (compact.get("scenario_long") or [])[:4]
    ]
    compact["longterm_reasons"] = (compact.get("longterm_reasons") or [])[:5]
    compact["daily_signal_reasons"] = (compact.get("daily_signal_reasons") or [])[:5]
    compact["risk_warnings"] = (compact.get("risk_warnings") or [])[:6]
    compact["research_actions"] = (compact.get("research_actions") or [])[:4]
    compact["entry_timing"] = _compact_group(
        compact.get("entry_timing"),
        ("available", "score", "label", "reason"),
        "entry",
    )
    compact["analyst_context"] = _compact_group(
        compact.get("analyst_context"),
        ("available", "analyst_count", "consensus", "target_price", "upside_pct", "note"),
        "analyst",
    )
    compact["valuation_context"] = _compact_group(
        compact.get("valuation_context"),
        (
            "available",
            "ranking_eligible",
            "unavailable_reason",
            "value_score",
            "quality_score",
            "growth_score",
            "fundamental_score",
            "reasons",
            "why_undervalued",
            "comparison_note",
        ),
        "valuation",
    )
    compact["potential_context"] = _compact_group(
        compact.get("potential_context"),
        ("note",),
        "potential",
    )
    compact["risk_context"] = _compact_group(
        compact.get("risk_context"),
        ("critical",),
        "risk",
    )
    compact["thesis_context"] = _compact_group(
        compact.get("thesis_context"),
        (),
        "thesis",
    )
    compact["research_context"] = _compact_group(
        compact.get("research_context"),
        (),
        "research",
    )
    compact["falling_knife"] = _compact_group(
        compact.get("falling_knife"),
        ("warning", "severity"),
        "knife",
    )
    compact["bottoming"] = _compact_group(
        compact.get("bottoming"),
        ("strength", "n", "signals", "speculative", "note"),
        "bottom",
    )
    compact["downside_structure"] = _compact_group(
        compact.get("downside_structure"),
        ("support1", "support1_pct", "support2", "support2_pct", "risk", "verdict"),
        "downside",
    )
    compact["trend_phase"] = _compact_group(
        compact.get("trend_phase"),
        ("phase", "risk_observation", "tone"),
        "trend",
    )
    compact["technical_observation_zone"] = _compact_group(
        compact.get("technical_observation_zone"),
        ("label", "lower", "upper", "currency_display", "note"),
        "zone",
    )
    compact["insight_provenance"] = _compact_group(
        compact.get("insight_provenance"),
        (
            "technical_missing",
            "fundamental_complete_current",
            "analyst_coverage_sufficient",
        ),
        "overall",
    )
    return compact


def validate_static_payload(payload: Any) -> dict[str, Any]:
    if not isinstance(payload, dict):
        raise ValueError("Static payload root must be an object")
    if payload.get("schema") != "stock-radar-static":
        raise ValueError("Unsupported static payload schema")
    if payload.get("schema_version") != STATIC_SCHEMA_VERSION:
        raise ValueError("Unsupported static payload version")
    if not isinstance(payload.get("instruments"), list):
        raise ValueError("Static payload instruments must be a list")
    if not isinstance(payload.get("rankings"), dict):
        raise ValueError("Static payload core rankings must be an object")
    try:
        validate_insight_contract(payload.get("insight_rankings"), payload["instruments"])
    except Exception as exc:
        raise ValueError(f"Static payload insight contract is invalid: {exc}") from exc
    metadata = payload.get("insight_metadata")
    if (
        not isinstance(metadata, dict)
        or metadata.get("model_status") != "heuristic_unvalidated"
        or metadata.get("actionable") is not False
        or _contains_actionable_true(metadata)
    ):
        raise ValueError("Static payload insight metadata is invalid")
    return payload


def export_static(
    input_path: Path = DEFAULT_INPUT,
    output_path: Path = DEFAULT_OUTPUT,
) -> dict[str, Any]:
    snapshot = load_json(input_path, required=True, expected_type=dict)
    validate_output_contract(snapshot)

    rankings = {}
    for currency, by_asset in snapshot["rankings_by_currency_asset"].items():
        rankings[currency] = {
            asset_type: [
                row.get("symbol")
                for row in rows
                if isinstance(row, dict) and row.get("symbol")
            ]
            for asset_type, rows in by_asset.items()
            if isinstance(rows, list)
        }

    payload = {
        "_meta": schema_meta("stock-radar-static-export", insight_contract=1),
        "schema": "stock-radar-static",
        "schema_version": STATIC_SCHEMA_VERSION,
        "generated_at": snapshot["generated_at"],
        "data_status": snapshot["data_status"],
        "model_status": snapshot["model_status"],
        "insight_metadata": snapshot["insight_metadata"],
        "market_data_contract": snapshot.get("market_data_contract") or {},
        "rankings": rankings,
        "insight_rankings": snapshot["insight_rankings"],
        "instruments": [
            _compact_row(row)
            for row in snapshot["all"]
            if isinstance(row, dict) and row.get("symbol")
        ],
    }
    validate_static_payload(payload)
    encoded = json.dumps(
        payload,
        ensure_ascii=False,
        separators=(",", ":"),
        allow_nan=False,
    ).encode("utf-8")
    encoded_size = len(encoded)
    if encoded_size >= MAX_STATIC_BYTES:
        raise ValueError(
            f"Static payload is {encoded_size / 1024 / 1024:.2f} MiB; "
            "must remain below 10 MiB"
        )
    atomic_write_bytes(output_path, encoded)
    return payload


def main() -> None:
    payload = export_static()
    size_mb = DEFAULT_OUTPUT.stat().st_size / 1024 / 1024
    print(
        f"Static dashboard export: {len(payload['instruments'])} instruments, "
        f"{size_mb:.2f} MiB -> {DEFAULT_OUTPUT}"
    )


if __name__ == "__main__":
    main()
