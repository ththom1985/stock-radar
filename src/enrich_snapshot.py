"""Provider-free, atomic v2 -> v3 insight enrichment utility."""
from __future__ import annotations

import argparse
from pathlib import Path

from .config import OUTPUT
from .insights import enrich_snapshot
from .persistence import atomic_write_json, load_json

DEFAULT_INPUT = OUTPUT / "latest.json"
DEFAULT_PREVIEW = OUTPUT / "latest.enriched.json"


def recompute(
    input_path: Path = DEFAULT_INPUT,
    output_path: Path = DEFAULT_PREVIEW,
) -> dict:
    source = load_json(input_path, required=True, expected_type=dict)
    enriched = enrich_snapshot(source)
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
