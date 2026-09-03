"""Decide whether the scheduled recovery run needs to rebuild Stock Radar."""
from __future__ import annotations

import argparse
import gzip
import json
from datetime import datetime, timezone
from pathlib import Path
from urllib.request import Request, urlopen


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
    max_age_hours: float,
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
    deployed_time = timestamps["live payload"]
    age_hours = (
        now.astimezone(timezone.utc) - deployed_time
    ).total_seconds() / 3600
    if age_hours < 0 or age_hours > max_age_hours:
        return True, f"deployed payload age is {age_hours:.1f} hours"
    return False, f"deployed payload is {age_hours:.1f} hours old and matches"


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--snapshot", type=Path, required=True)
    parser.add_argument("--export", type=Path, required=True)
    parser.add_argument("--live-url", required=True)
    parser.add_argument("--max-age-hours", type=float, default=12)
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
            max_age_hours=args.max_age_hours,
        )
    except (OSError, ValueError, json.JSONDecodeError) as exc:
        rebuild, reason = True, f"freshness check failed: {exc}"
    print(f"rebuild={'true' if rebuild else 'false'}")
    print(f"reason={reason}")


if __name__ == "__main__":
    main()
