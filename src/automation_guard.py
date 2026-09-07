"""Decide whether the scheduled recovery run needs to rebuild Stock Radar."""
from __future__ import annotations

import argparse
import gzip
import json
from datetime import datetime, timezone
from pathlib import Path
from urllib.request import Request, urlopen

from .freshness import build_session_freshness, evaluate_session_freshness, timestamp_failures
from .verify_live import live_matches


def _generated_at(payload: dict) -> datetime:
    value = payload.get("generated_at")
    if not isinstance(value, str):
        raise ValueError("generated_at is missing")
    parsed = datetime.fromisoformat(value.replace("Z", "+00:00"))
    if parsed.tzinfo is None:
        raise ValueError("generated_at has no timezone")
    return parsed.astimezone(timezone.utc)


def _load_snapshot(path: Path) -> dict:
    with gzip.open(path, "rt", encoding="utf-8") as handle:
        return json.load(handle)


def _load_live(url: str, timeout: int = 30) -> dict:
    request = Request(url, headers={"Cache-Control": "no-cache"})
    with urlopen(request, timeout=timeout) as response:
        return json.load(response)


def recovery_needed(
    snapshot: dict,
    exported: dict,
    live: dict,
    *,
    now: datetime,
) -> tuple[bool, str]:
    for label, payload in (
        ("snapshot", snapshot),
        ("export", exported),
        ("live payload", live),
    ):
        if not isinstance(payload, dict):
            raise ValueError(f"{label} root is not an object")
    timestamps = {
        "snapshot": _generated_at(snapshot),
        "export": _generated_at(exported),
        "live payload": _generated_at(live),
    }
    if len(set(timestamps.values())) != 1:
        rendered = {key: value.isoformat() for key, value in timestamps.items()}
        return True, f"generation timestamps differ: {rendered}"
    matched, reason = live_matches(exported, live)
    if not matched:
        return True, reason
    snapshot_contract = build_session_freshness(snapshot.get("all") or [])
    exported_contract = build_session_freshness(exported.get("instruments") or [])
    if snapshot_contract != exported_contract:
        return True, "snapshot/export completed-bar evidence differs"
    for label, payload in (("snapshot", snapshot), ("export", exported), ("live", live)):
        failures = timestamp_failures(payload, now)
        rows = payload.get("all") if label == "snapshot" else payload.get("instruments")
        if not isinstance(rows, list) or not rows:
            return True, f"{label}: no completed-bar evidence"
        current = evaluate_session_freshness(build_session_freshness(rows), now=now)
        failures.extend(current["blocking_reasons"])
        # Retry even a small missing minority which still meets the dashboard SLA.
        if current["stale_symbols"]:
            failures.append(f"missing completed sessions for {len(current['stale_symbols'])} instruments")
        if failures:
            return True, f"{label}: " + "; ".join(failures)
    if (exported.get("data_status") or {}).get("session_freshness") != exported_contract:
        return True, "export completed-session freshness contract is missing or differs"
    return False, "all completed sessions are present and deployed content matches"


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--snapshot", type=Path, required=True)
    parser.add_argument("--export", type=Path, required=True)
    parser.add_argument("--live-url", required=True)
    args = parser.parse_args()
    try:
        snapshot = _load_snapshot(args.snapshot)
        exported = json.loads(args.export.read_text(encoding="utf-8"))
        live = _load_live(args.live_url)
        rebuild, reason = recovery_needed(
            snapshot,
            exported,
            live,
            now=datetime.now(timezone.utc),
        )
    except (OSError, ValueError, TypeError, KeyError) as exc:
        rebuild, reason = True, f"freshness check failed: {exc}"
    print(f"rebuild={'true' if rebuild else 'false'}")
    print(f"reason={reason}")


if __name__ == "__main__":
    main()
