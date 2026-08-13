"""Export a compact, login-free GitHub Pages payload from the v2 snapshot."""
from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from .data_quality import validate_output_contract
from .persistence import atomic_write_json, load_json, schema_meta

ROOT = Path(__file__).resolve().parent.parent
DEFAULT_INPUT = ROOT / "data" / "output" / "latest.json"
DEFAULT_OUTPUT = ROOT / "docs" / "data.json"

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
    "daily_signal_direction",
    "fundamental_score",
    "value_score",
    "quality_score",
    "growth_score",
    "heuristic_summary",
    "scenario_long",
    "next_earnings",
    "earnings_in_days",
    "news",
    "feature_coverage",
    "paper_eligibility",
)


def _compact_row(row: dict[str, Any]) -> dict[str, Any]:
    compact = {key: row.get(key) for key in ROW_FIELDS}
    compact["news"] = (compact.get("news") or [])[:5]
    return compact


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
        "_meta": schema_meta("stock-radar-static-export"),
        "schema": "stock-radar-static",
        "schema_version": 1,
        "generated_at": snapshot["generated_at"],
        "data_status": snapshot["data_status"],
        "model_status": snapshot["model_status"],
        "market_data_contract": snapshot.get("market_data_contract") or {},
        "rankings": rankings,
        "instruments": [
            _compact_row(row)
            for row in snapshot["all"]
            if isinstance(row, dict) and row.get("symbol")
        ],
    }
    atomic_write_json(output_path, payload, indent=None)
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
