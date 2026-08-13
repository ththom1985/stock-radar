"""Export a compact, login-free GitHub Pages insight payload."""
from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from .data_quality import validate_insight_contract, validate_output_contract
from .insights import INSIGHT_CONTRACT_VERSION
from .persistence import atomic_write_bytes, load_json, schema_meta

ROOT = Path(__file__).resolve().parent.parent
DEFAULT_INPUT = ROOT / "data" / "output" / "latest.json"
DEFAULT_OUTPUT = ROOT / "docs" / "data.json"
STATIC_SCHEMA_VERSION = 3
MAX_STATIC_BYTES = 10 * 1024 * 1024
TARGET_STATIC_BYTES = int(8.5 * 1024 * 1024)

ROW_FIELDS = (
    "symbol",
    "short_name",
    "display_name_full",
    "headquarters_country",
    "legal_domicile",
    "legal_domicile_verified",
    "legal_domicile_source",
    "asset_type",
    "currency",
    "economic_exposure_country",
    "economic_exposure_region",
    "listing_market",
    "listing_country",
    "sector_display",
    "industry_display",
    "jurisdiction_risk",
    "price",
    "bar_date",
    "radar_score",
    "longterm_score",
    "daily_signal_direction",
    "entry_timing_score",
    "entry_timing_label",
    "entry_timing_reason",
    "falling_knife",
    "bottoming",
    "downside_structure",
    "risk_warnings",
    "bull_thesis",
    "priced_in_note",
    "trend_phase",
    "research_summary",
    "analyst_context",
    "valuation_context",
    "valuation_thesis",
    "entry_thesis",
    "technical_observation_zone",
    "scenario_long",
    "next_earnings",
    "earnings_in_days",
    "news",
    "rsi",
    "macd",
    "macd_signal",
    "ret_20d",
    "ret_60d",
    "pct_from_high52",
    "atr_pct",
    "vol_annual_pct",
    "rvol",
    "minervini_score",
    "weinstein_label",
)
STATIC_INSTRUMENT_CONTRACT = {
    "model_status": "heuristic_unvalidated",
    "actionable": False,
    "group_provenance": "insight_metadata.provenance_catalog",
    "omitted_redundant_fields": [
        "per-row model_status/actionable/inputs_used/missing_inputs",
        "identity compatibility aliases and ISO/source metadata",
        "non-rendered context groups and feature internals",
    ],
}


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
) -> Any:
    if not isinstance(value, dict):
        return value
    return {key: value.get(key) for key in keys}


def _compact_row(row: dict[str, Any]) -> dict[str, Any]:
    compact = {key: row.get(key) for key in ROW_FIELDS}
    compact["news"] = (compact.get("news") or [])[:3]
    compact["scenario_long"] = [
        {
            key: scenario.get(key)
            for key in (
                "label",
                "reference_change_pct",
                "range_low_price",
                "range_high_price",
            )
        }
        for scenario in (compact.get("scenario_long") or [])[:4]
    ]
    compact["risk_warnings"] = (compact.get("risk_warnings") or [])[:6]
    compact["analyst_context"] = _compact_group(
        compact.get("analyst_context"),
        ("available", "analyst_count", "consensus", "target_price", "upside_pct"),
    )
    compact["valuation_context"] = _compact_group(
        compact.get("valuation_context"),
        (
            "available",
            "unavailable_reason",
            "value_score",
            "quality_score",
            "growth_score",
            "fundamental_score",
            "reasons",
            "comparison_note",
        ),
    )
    compact["valuation_thesis"] = _compact_group(
        compact.get("valuation_thesis"),
        (
            "available",
            "why_it_looks_cheap",
            "why_discount_may_be_justified",
            "strongest_positive_evidence",
            "strongest_counterarguments",
            "raw_score",
            "risk_penalty",
            "risk_adjusted_score",
            "value_trap_risk",
            "penalty_components",
            "penalty_evidence_ids",
            "penalty_reasons",
            "formula",
        ),
    )
    compact["entry_thesis"] = _compact_group(
        compact.get("entry_thesis"),
        (
            "available",
            "why_timing_may_be_good",
            "what_confirms",
            "what_invalidates",
            "strongest_supporting_evidence",
            "strongest_counterarguments",
            "timing_score",
            "trend",
            "regime",
            "falling_knife_bottoming_status",
        ),
    )
    compact["jurisdiction_risk"] = _compact_group(
        compact.get("jurisdiction_risk"),
        (
            "level",
            "penalty_points",
            "reasons",
            "heuristic_note",
        ),
    )
    compact["falling_knife"] = _compact_group(
        compact.get("falling_knife"),
        ("warning", "severity"),
    )
    compact["bottoming"] = _compact_group(
        compact.get("bottoming"),
        ("strength", "n", "signals", "speculative", "note"),
    )
    compact["downside_structure"] = _compact_group(
        compact.get("downside_structure"),
        ("support1", "support1_pct", "risk", "verdict"),
    )
    compact["trend_phase"] = _compact_group(
        compact.get("trend_phase"),
        ("phase",),
    )
    compact["technical_observation_zone"] = _compact_group(
        compact.get("technical_observation_zone"),
        ("label", "lower", "upper"),
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
    contract = payload.get("instrument_contract")
    if (
        not isinstance(contract, dict)
        or contract.get("model_status") != "heuristic_unvalidated"
        or contract.get("actionable") is not False
        or contract.get("group_provenance")
        != "insight_metadata.provenance_catalog"
        or _contains_actionable_true(contract)
    ):
        raise ValueError("Static instrument contract is invalid")

    def require_text_lists(group: dict[str, Any], keys: tuple[str, ...]) -> None:
        for key in keys:
            values = group.get(key)
            if not isinstance(values, list) or not all(
                isinstance(value, str) for value in values
            ):
                raise ValueError(f"Static group field {key!r} must be a text list")

    for row in payload["instruments"]:
        if not isinstance(row, dict) or any(key not in row for key in ROW_FIELDS):
            raise ValueError("Static payload instrument cockpit contract is invalid")
        if (
            not isinstance(row.get("display_name_full"), str)
            or not row["display_name_full"].strip()
            or not isinstance(row.get("sector_display"), str)
            or not isinstance(row.get("industry_display"), str)
        ):
            raise ValueError("Static payload instrument identity is invalid")
        if _contains_actionable_true(row):
            raise ValueError("Static payload instrument contains actionable=true")
        jurisdiction = row.get("jurisdiction_risk")
        if (
            not isinstance(jurisdiction, dict)
            or jurisdiction.get("level") not in {"low", "medium", "high", "unknown"}
            or not isinstance(jurisdiction.get("penalty_points"), (int, float))
            or not 0 <= jurisdiction["penalty_points"] <= 20
            or not isinstance(jurisdiction.get("reasons"), list)
            or not all(
                isinstance(reason, str) for reason in jurisdiction["reasons"]
            )
        ):
            raise ValueError("Static jurisdiction context is invalid")
        valuation = row.get("valuation_context")
        if not isinstance(valuation, dict) or not isinstance(
            valuation.get("available"), bool
        ):
            raise ValueError("Static valuation context is invalid")
        valuation_thesis = row.get("valuation_thesis")
        if not isinstance(valuation_thesis, dict) or not isinstance(
            valuation_thesis.get("available"), bool
        ):
            raise ValueError("Static valuation thesis is invalid")
        require_text_lists(
            valuation_thesis,
            (
                "why_it_looks_cheap",
                "why_discount_may_be_justified",
                "strongest_positive_evidence",
                "strongest_counterarguments",
                "penalty_reasons",
            ),
        )
        evidence = valuation_thesis.get("penalty_evidence_ids")
        if not isinstance(evidence, dict) or any(
            not isinstance(values, list)
            or not all(isinstance(value, str) for value in values)
            for values in evidence.values()
        ):
            raise ValueError("Static valuation evidence is invalid")
        entry = row.get("entry_thesis")
        if not isinstance(entry, dict) or not isinstance(entry.get("available"), bool):
            raise ValueError("Static entry thesis is invalid")
        require_text_lists(
            entry,
            (
                "why_timing_may_be_good",
                "what_confirms",
                "what_invalidates",
                "strongest_supporting_evidence",
                "strongest_counterarguments",
            ),
        )
        analyst = row.get("analyst_context")
        if not isinstance(analyst, dict) or not isinstance(
            analyst.get("available"), bool
        ):
            raise ValueError("Static analyst context is invalid")
        if not isinstance(row.get("scenario_long"), list) or len(
            row["scenario_long"]
        ) > 4:
            raise ValueError("Static scenarios are invalid")
        if not isinstance(row.get("news"), list) or len(row["news"]) > 3:
            raise ValueError("Static news context is invalid")
    instrument_symbols = {
        row.get("symbol") for row in payload["instruments"] if isinstance(row, dict)
    }
    for by_asset in payload["rankings"].values():
        if not isinstance(by_asset, dict):
            raise ValueError("Static payload ranking partition is invalid")
        for symbols in by_asset.values():
            if (
                not isinstance(symbols, list)
                or any(not isinstance(symbol, str) for symbol in symbols)
                or any(symbol not in instrument_symbols for symbol in symbols)
            ):
                raise ValueError(
                    "Static ranking references missing/inconsistent instrument context"
                )
    try:
        validate_insight_contract(payload.get("insight_rankings"), None)
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
    for category in payload["insight_rankings"]["categories"].values():
        for items in category["items_by_currency"].values():
            if any(item.get("symbol") not in instrument_symbols for item in items):
                raise ValueError(
                    "Static insight ranking references missing instrument"
                )
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
        "_meta": schema_meta(
            "stock-radar-static-export",
            insight_contract=INSIGHT_CONTRACT_VERSION,
        ),
        "schema": "stock-radar-static",
        "schema_version": STATIC_SCHEMA_VERSION,
        "generated_at": snapshot["generated_at"],
        "data_status": snapshot["data_status"],
        "model_status": snapshot["model_status"],
        "insight_metadata": snapshot["insight_metadata"],
        "instrument_contract": STATIC_INSTRUMENT_CONTRACT,
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
