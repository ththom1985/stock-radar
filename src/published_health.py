"""Non-blocking diagnostics for the newly generated published-data snapshot."""
from __future__ import annotations

import json
import os
from collections import Counter
from pathlib import Path

from .config import OUTPUT
from .data_quality import validate_output_contract
from .sweet_spot import format_price


def _finite_positive(value):
    return isinstance(value, (int, float)) and value > 0


def classify_unavailable(row: dict) -> str:
    if not _finite_positive(row.get("price")):
        return "missing_or_nonpositive_price"
    if not _finite_positive(row.get("atr")):
        return "missing_or_nonpositive_atr"
    if not (row.get("feature_coverage") or {}).get("technical_complete"):
        return "incomplete_technical_features"
    return "no_eligible_reference_anchor"


def inspect_snapshot(path: Path | None = None) -> dict:
    path = path or OUTPUT / "latest.json"
    snapshot = json.loads(path.read_text(encoding="utf-8"))
    validate_output_contract(snapshot)
    rows = snapshot.get("all") or []
    by_symbol = {row.get("symbol"): row for row in rows}
    findings = []

    doge = by_symbol.get("DOGE-USD")
    doge_format = {"status": "not_present"}
    if doge:
        sweet = doge.get("sweet_spot") or {}
        zone = [sweet.get(key) for key in ("lower", "ideal", "upper")]
        if all(_finite_positive(value) for value in zone):
            rendered = [format_price(value, zone) for value in zone]
            valid = (
                zone[0] < zone[1] < zone[2]
                and len(set(rendered)) == 3
                and all("." in value for value in rendered)
            )
            doge_format = {"status": "ok" if valid else "invalid", "marks": rendered}
            if not valid:
                findings.append("DOGE zone marks are not distinct, ascending, and formatted")
        else:
            doge_format = {"status": "unavailable"}

    confirmed_overlap = [
        row["symbol"]
        for row in rows
        if (row.get("sweet_spot") or {}).get("combined_status")
        == "in_zone_confirmed"
        and (row.get("falling_knife") or row.get("bottoming"))
    ]
    if confirmed_overlap:
        findings.append(
            f"confirmed Sweet Spot overlaps knife/bottoming: {confirmed_overlap}"
        )

    categories = (snapshot.get("insight_rankings") or {}).get("categories") or {}
    ranked = {
        item.get("symbol")
        for key in ("in_sweet_spot", "approaching_sweet_spot")
        for items in ((categories.get(key) or {}).get("items_by_currency") or {}).values()
        for item in items
    }
    reference_rows = [
        row
        for row in rows
        if (row.get("sweet_spot") or {}).get("zone_tier") == "reference_only"
    ]
    invalid_reference = [
        row["symbol"]
        for row in reference_rows
        if not (
            (row.get("sweet_spot") or {}).get("available")
            and 0
            < (row.get("sweet_spot") or {}).get("lower", 0)
            < (row.get("sweet_spot") or {}).get("ideal", 0)
            < (row.get("sweet_spot") or {}).get("upper", 0)
            and (row.get("sweet_spot") or {}).get("reliability_score", 100) <= 49
            and row["symbol"] not in ranked
        )
    ]
    if invalid_reference:
        findings.append(f"invalid/ranked reference-only rows: {invalid_reference}")

    unavailable = [
        row for row in rows if not (row.get("sweet_spot") or {}).get("available")
    ]
    unavailable_causes = Counter()
    unavailable_symbols = {}
    for row in unavailable:
        cause = classify_unavailable(row)
        unavailable_causes[cause] += 1
        unavailable_symbols[row["symbol"]] = cause

    return {
        "status": "warning" if findings or unavailable else "ok",
        "generated_at": snapshot.get("generated_at"),
        "instrument_count": len(rows),
        "doge_format": doge_format,
        "confirmed_overlap": confirmed_overlap,
        "reference_only_count": len(reference_rows),
        "invalid_reference_only": invalid_reference,
        "unavailable_count": len(unavailable),
        "unavailable_causes": dict(sorted(unavailable_causes.items())),
        "unavailable_symbols": unavailable_symbols,
        "findings": findings,
    }


def _markdown(report: dict) -> str:
    lines = [
        "## Published data health",
        "",
        f"- Status: **{report['status']}**",
        f"- Generated: `{report.get('generated_at')}`",
        f"- Instruments: {report['instrument_count']}",
        f"- Reference-only zones: {report['reference_only_count']}",
        f"- Unavailable zones: {report['unavailable_count']}",
        f"- Unavailable causes: `{json.dumps(report['unavailable_causes'], sort_keys=True)}`",
        f"- DOGE formatting: `{json.dumps(report['doge_format'], sort_keys=True)}`",
    ]
    if report["findings"]:
        lines.extend(["", "### Findings", *[f"- {item}" for item in report["findings"]]])
    return "\n".join(lines) + "\n"


def main() -> None:
    report = inspect_snapshot()
    print(json.dumps(report, ensure_ascii=False, indent=2, sort_keys=True))
    summary = os.environ.get("GITHUB_STEP_SUMMARY")
    if summary:
        with open(summary, "a", encoding="utf-8") as handle:
            handle.write(_markdown(report))


if __name__ == "__main__":
    main()
