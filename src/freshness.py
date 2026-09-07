"""Bar-derived freshness deadlines shared by recovery and both dashboards.

Deadlines are the next scheduled session's conservative close plus 90 minutes,
not the time an analysis happened to finish. Non-US holidays are not inferred
from the US calendar: their weekday fallback can block conservatively.
"""
from __future__ import annotations

from datetime import date, datetime, timedelta, timezone
from functools import lru_cache
from zoneinfo import ZoneInfo

from .market_calendar import _is_projected_us_session
from .markets import market_profile, session_bounds

POLICY = "completed-session-v1"
CLOSE_BUFFER_MINUTES = 90
MIN_FRESH_PCT = 97.0
US_INDICES = {"^GSPC", "^DJI", "^IXIC", "^NDX", "^RUT", "^VIX"}


def parse_timestamp(value: str) -> datetime:
    if not isinstance(value, str):
        raise ValueError("timestamp is missing")
    parsed = datetime.fromisoformat(value.replace("Z", "+00:00"))
    if parsed.tzinfo is None:
        raise ValueError("timestamp has no timezone")
    return parsed.astimezone(timezone.utc)


def timestamp_failures(payload: dict, now: datetime) -> list[str]:
    try:
        generated = parse_timestamp(payload.get("generated_at"))
        return ["output timestamp is in the future"] if generated > now else []
    except (TypeError, ValueError):
        return ["generated_at is not a valid timezone-aware ISO timestamp"]


@lru_cache(maxsize=4096)
def _schedule(symbol: str) -> tuple[str, str]:
    symbol = symbol.upper()
    profile = market_profile(symbol)
    if (
        not symbol
        or profile.mapping_status != "verified_conservative"
        or (symbol.startswith("^") and symbol not in US_INDICES)
    ):
        return "unknown", "unsupported exchange calendar; blocked"
    if symbol.endswith("-USD"):
        return "UTC-24x7", "UTC daily, including weekends and holidays"
    if symbol.endswith("=X"):
        return "UTC/00:00:00/Mon-Fri", "Mon-Fri FX daily; conservative UTC day-end"
    if "." not in symbol:
        return "US", "projected NYSE holidays; conservative regular close"
    week = "Sun-Thu" if symbol.endswith(".SR") else "Mon-Fri"
    return (
        f"{profile.timezone_name}/{profile.close_time.isoformat()}/{week}",
        f"{week} fallback; local holidays NOT modeled",
    )


def _scheduled(day: date, market: str) -> bool:
    if market == "US":
        return _is_projected_us_session(day)
    if market == "UTC-24x7":
        return True
    if market.endswith("/Sun-Thu"):
        return day.weekday() not in (4, 5)
    return day.weekday() < 5


def latest_completed_session(symbol: str, now: datetime) -> date:
    market, _ = _schedule(symbol)
    if market == "unknown":
        raise ValueError("unsupported exchange calendar")
    profile = market_profile(symbol)
    day = now.astimezone(ZoneInfo(profile.timezone_name)).date()
    while True:
        _, close = session_bounds(day, profile)
        if _scheduled(day, market) and now >= close + timedelta(minutes=CLOSE_BUFFER_MINUTES):
            return day
        day -= timedelta(days=1)


@lru_cache(maxsize=16384)
def _bar_deadlines(symbol: str, bar_date: str) -> dict:
    market, calendar = _schedule(symbol)
    result = {"market": market, "calendar": calendar, "bar_date": bar_date}
    try:
        day = date.fromisoformat(bar_date)
        if market == "unknown" or not _scheduled(day, market):
            raise ValueError("unsupported calendar or non-session bar date")
        profile = market_profile(symbol)
        _, close = session_bounds(day, profile)
        next_day = day + timedelta(days=1)
        while not _scheduled(next_day, market):
            next_day += timedelta(days=1)
        _, next_close = session_bounds(next_day, profile)
        buffer = timedelta(minutes=CLOSE_BUFFER_MINUTES)
        result.update(
            completed_after=(close + buffer).astimezone(timezone.utc).isoformat(),
            next_completed_after=(next_close + buffer).astimezone(timezone.utc).isoformat(),
            next_session=next_day.isoformat(),
        )
    except (TypeError, ValueError, OverflowError) as exc:
        result["error"] = f"invalid completed-bar date/calendar: {exc}"
    return result


def build_session_freshness(rows: list[dict]) -> dict:
    if not isinstance(rows, list) or any(not isinstance(row, dict) for row in rows):
        raise ValueError("completed-bar rows must be a list of objects")
    groups = {}
    for row in rows:
        symbol = str(row.get("symbol") or "")
        deadline = _bar_deadlines(symbol, str(row.get("bar_date") or ""))
        key = tuple(deadline.items())
        if key not in groups:
            groups[key] = {**deadline, "symbols": []}
        groups[key]["symbols"].append(symbol)
    return {
        "policy": POLICY,
        "close_buffer_minutes": CLOSE_BUFFER_MINUTES,
        "groups": list(groups.values()),
    }


def evaluate_session_freshness(
    contract: dict, *, now: datetime, min_coverage_pct: float = MIN_FRESH_PCT
) -> dict:
    reasons, stale, invalid, future = [], [], [], []
    market_symbols = {}
    markets = {}
    groups = contract.get("groups") if isinstance(contract, dict) else None
    if (
        not isinstance(contract, dict) or contract.get("policy") != POLICY
        or not isinstance(groups, list) or not groups
    ):
        return {
            "blocking_reasons": ["completed-session freshness contract is missing"],
            "stale_symbols": [], "invalid_symbols": [], "future_symbols": [],
            "fresh_bar_coverage_pct": 0.0, "markets": {},
            "fresh_symbols": [], "research_blocking_reasons": ["completed-session freshness contract is missing"],
        }
    for group in groups:
        symbols = group["symbols"]
        market = group["market"]
        market_symbols.setdefault(market, []).extend(symbols)
        scope = markets.setdefault(market, {
            "total": 0, "fresh": 0, "calendar": group["calendar"],
        })
        scope["total"] += len(symbols)
        if group.get("error"):
            invalid.extend(symbols)
        elif now < parse_timestamp(group["completed_after"]):
            future.extend(symbols)
        elif now >= parse_timestamp(group["next_completed_after"]):
            stale.extend(symbols)
        else:
            scope["fresh"] += len(symbols)
    for market, scope in markets.items():
        scope["fresh_pct"] = scope["fresh"] / scope["total"] * 100 if scope["total"] else 0
        if scope["fresh_pct"] < min_coverage_pct:
            reasons.append(
                f"{market}: missing completed sessions or invalid bars; fresh coverage "
                f"{scope['fresh_pct']:.2f}% is below {min_coverage_pct:.2f}% "
                f"({scope['calendar']})"
            )
    if invalid:
        reasons.append(f"{len(invalid)} rows have invalid completed-bar dates/calendars")
    if future:
        reasons.append(f"{len(future)} rows have future/uncompleted bar dates")
    total = sum(scope["total"] for scope in markets.values())
    fresh = sum(scope["fresh"] for scope in markets.values())
    excluded = set(stale + invalid + future)
    for market, scope in markets.items():
        if scope["fresh_pct"] < min_coverage_pct:
            excluded.update(market_symbols[market])
    usable = [symbol for symbols in market_symbols.values() for symbol in symbols if symbol not in excluded]
    return {
        "blocking_reasons": reasons,
        "stale_symbols": stale,
        "invalid_symbols": invalid,
        "future_symbols": future,
        "fresh_bar_coverage_pct": fresh / total * 100 if total else 0.0,
        "markets": markets,
        "fresh_symbols": usable,
        "research_blocking_reasons": [] if usable else reasons or ["no fresh research instruments"],
    }


def filter_research_symbols(value, allowed: set[str]):
    """Filter symbol-bearing research lists, retaining non-list metadata."""
    if isinstance(value, list):
        return [filter_research_symbols(item, allowed) for item in value
                if not isinstance(item, dict) or not item.get("symbol") or item["symbol"] in allowed]
    if isinstance(value, dict):
        return {key: filter_research_symbols(item, allowed) for key, item in value.items()}
    return value
