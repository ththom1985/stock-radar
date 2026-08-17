"""Provider-free, atomic v2 -> v3 insight enrichment utility."""
from __future__ import annotations

import argparse
from pathlib import Path

from .config import OUTPUT
from .insights import enrich_snapshot
from .persistence import atomic_write_json, load_json
from .probability_inference import (
    attach_probability_forecasts,
    load_probability_baselines,
    load_probability_validation_summary,
)
from .universe import load_universe

DEFAULT_INPUT = OUTPUT / "latest.json"
DEFAULT_PREVIEW = OUTPUT / "latest.enriched.json"


def _merge_provider_free_context(snapshot: dict) -> None:
    """Merge existing local caches/config only; never contact a provider."""
    fundamentals = load_json(
        OUTPUT.parent / "fundamentals.json",
        expected_type=dict,
        default={},
    )
    configured = {item["symbol"]: item for item in load_universe()}
    for row in snapshot.get("all") or []:
        symbol = row.get("symbol")
        if not symbol:
            continue
        cached = fundamentals.get(symbol) or {}
        universe_row = configured.get(symbol) or {}
        row["short_name"] = row.get("short_name") or row.get("name")
        row["exchange"] = row.get("exchange") or universe_row.get("exchange")
        row["listing_market"] = row.get("listing_market") or row.get("exchange")
        for key in (
            "provider_long_name",
            "provider_country",
            "reported_currency",
            "sector",
            "industry",
        ):
            if not row.get(key) and cached.get(key):
                row[key] = cached[key]
        market_cap = cached.get("market_cap")
        fx_usd = row.get("fx_usd")
        reported = cached.get("reported_currency")
        if (
            isinstance(market_cap, (int, float))
            and market_cap > 0
            and isinstance(fx_usd, (int, float))
            and fx_usd > 0
            and (not reported or reported == row.get("currency"))
        ):
            row["market_cap_local"] = market_cap
            row["market_cap_usd"] = market_cap * fx_usd


def recompute(
    input_path: Path = DEFAULT_INPUT,
    output_path: Path = DEFAULT_PREVIEW,
) -> dict:
    source = load_json(input_path, required=True, expected_type=dict)
    _merge_provider_free_context(source)
    enriched = enrich_snapshot(source)
    attach_probability_forecasts(
        enriched["all"], {}, embed_baselines=False
    )
    enriched["probability_baselines"] = load_probability_baselines()
    enriched["probability_validation"] = (
        load_probability_validation_summary()
    )
    atomic_write_json(output_path, enriched)
    return enriched


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Recompute Stock Radar insights without provider downloads."
    )
    parser.add_argument("--input", type=Path, default=DEFAULT_INPUT)
    parser.add_argument("--output", type=Path, default=DEFAULT_PREVIEW)
    parser.add_argument(
        "--in-place",
        action="store_true",
        help="Atomically replace the input snapshot instead of writing a preview.",
    )
    parser.add_argument(
        "--export-static",
        action="store_true",
        help="Also regenerate docs/data.json from the enriched snapshot.",
    )
    args = parser.parse_args()
    target = args.input if args.in_place else args.output
    result = recompute(args.input, target)
    if args.export_static:
        from .export_static import DEFAULT_OUTPUT, export_static

        export_static(target, DEFAULT_OUTPUT)
        print(f"Static export refreshed: {DEFAULT_OUTPUT}")
    print(
        f"Insight enrichment v{result['schema_version']}: "
        f"{len(result['all'])} instruments -> {target}"
    )


if __name__ == "__main__":
    main()
