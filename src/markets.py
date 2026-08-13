"""Conservative exchange-session profiles for completed daily bars."""
from __future__ import annotations

from dataclasses import dataclass
from datetime import date, datetime, time, timedelta, timezone
from zoneinfo import ZoneInfo, ZoneInfoNotFoundError


@dataclass(frozen=True)
class MarketProfile:
    timezone_name: str
    open_time: time
    close_time: time
    close_day_offset: int = 0
    source: str = "symbol_mapping"
    mapping_status: str = "verified_conservative"


US = MarketProfile("America/New_York", time(9, 30), time(16, 0))
UTC_DAILY = MarketProfile("UTC", time(0, 0), time(0, 0), close_day_offset=1)

_PROFILES = {
    ".DE": MarketProfile("Europe/Berlin", time(9, 0), time(17, 30)),
    ".F": MarketProfile("Europe/Berlin", time(8, 0), time(22, 0)),
    ".PA": MarketProfile("Europe/Paris", time(9, 0), time(17, 30)),
    ".AS": MarketProfile("Europe/Amsterdam", time(9, 0), time(17, 30)),
    ".BR": MarketProfile("Europe/Brussels", time(9, 0), time(17, 30)),
    ".MI": MarketProfile("Europe/Rome", time(9, 0), time(17, 30)),
    ".MC": MarketProfile("Europe/Madrid", time(9, 0), time(17, 30)),
    ".VI": MarketProfile("Europe/Vienna", time(9, 0), time(17, 30)),
    ".HE": MarketProfile("Europe/Helsinki", time(10, 0), time(18, 30)),
    ".LS": MarketProfile("Europe/Lisbon", time(8, 0), time(16, 30)),
    ".IR": MarketProfile("Europe/Dublin", time(8, 0), time(16, 30)),
    ".L": MarketProfile("Europe/London", time(8, 0), time(16, 30)),
    ".SW": MarketProfile("Europe/Zurich", time(9, 0), time(17, 30)),
    ".ST": MarketProfile("Europe/Stockholm", time(9, 0), time(17, 30)),
    ".OL": MarketProfile("Europe/Oslo", time(9, 0), time(16, 20)),
    ".CO": MarketProfile("Europe/Copenhagen", time(9, 0), time(17, 0)),
    ".WA": MarketProfile("Europe/Warsaw", time(9, 0), time(17, 0)),
    ".AT": MarketProfile("Europe/Athens", time(10, 15), time(17, 20)),
    ".T": MarketProfile("Asia/Tokyo", time(9, 0), time(15, 0)),
    ".HK": MarketProfile("Asia/Hong_Kong", time(9, 30), time(16, 0)),
    ".KS": MarketProfile("Asia/Seoul", time(9, 0), time(15, 30)),
    ".KQ": MarketProfile("Asia/Seoul", time(9, 0), time(15, 30)),
    ".TW": MarketProfile("Asia/Taipei", time(9, 0), time(13, 30)),
    ".TWO": MarketProfile("Asia/Taipei", time(9, 0), time(13, 30)),
    ".NS": MarketProfile("Asia/Kolkata", time(9, 15), time(15, 30)),
    ".BO": MarketProfile("Asia/Kolkata", time(9, 15), time(15, 30)),
    ".JK": MarketProfile("Asia/Jakarta", time(9, 0), time(16, 0)),
    ".KL": MarketProfile("Asia/Kuala_Lumpur", time(9, 0), time(17, 0)),
    ".BK": MarketProfile("Asia/Bangkok", time(10, 0), time(16, 30)),
    ".SI": MarketProfile("Asia/Singapore", time(9, 0), time(17, 0)),
    ".SR": MarketProfile("Asia/Riyadh", time(10, 0), time(15, 20)),
    ".JO": MarketProfile("Africa/Johannesburg", time(9, 0), time(17, 0)),
    ".SA": MarketProfile("America/Sao_Paulo", time(10, 0), time(18, 0)),
    ".MX": MarketProfile("America/Mexico_City", time(8, 30), time(15, 0)),
    ".AX": MarketProfile("Australia/Sydney", time(10, 0), time(16, 0)),
    ".NZ": MarketProfile("Pacific/Auckland", time(10, 0), time(16, 45)),
    ".TO": MarketProfile("America/Toronto", time(9, 30), time(16, 0)),
    ".V": MarketProfile("America/Vancouver", time(6, 30), time(13, 0)),
    ".NE": MarketProfile("America/Toronto", time(9, 30), time(16, 0)),
    ".CN": MarketProfile("America/Toronto", time(9, 30), time(16, 0)),
}


def market_profile(symbol: str | None, index_timezone=None) -> MarketProfile:
    symbol = (symbol or "").upper()
    if symbol.endswith("-USD") or symbol.startswith("^") or symbol.endswith("=F"):
        if symbol.endswith("-USD"):
            base = UTC_DAILY
        elif symbol.startswith("^"):
            base = US
        else:
            base = MarketProfile(
                "UTC",
                time(0, 0),
                time(0, 0),
                close_day_offset=1,
                source="unverified_derivative_session",
                mapping_status="unknown_blocked",
            )
    elif "." in symbol:
        suffix = symbol[symbol.rindex(".") :]
        base = _PROFILES.get(
            suffix,
            MarketProfile(
                "UTC",
                time(0, 0),
                time(0, 0),
                close_day_offset=1,
                source=f"unknown_suffix:{suffix}",
                mapping_status="unknown_blocked",
            ),
        )
    else:
        base = US
    if index_timezone is None:
        return base
    timezone_name = str(index_timezone)
    try:
        ZoneInfo(timezone_name)
    except ZoneInfoNotFoundError:
        return base
    return MarketProfile(
        timezone_name,
        base.open_time,
        base.close_time,
        base.close_day_offset,
        source="yfinance_index_timezone",
        mapping_status=base.mapping_status,
    )


def session_bounds(
    session_date: date,
    profile: MarketProfile,
) -> tuple[datetime, datetime]:
    zone = ZoneInfo(profile.timezone_name)
    opened = datetime.combine(session_date, profile.open_time, zone)
    closed = datetime.combine(
        session_date + timedelta(days=profile.close_day_offset),
        profile.close_time,
        zone,
    )
    return opened, closed


def session_metadata(
    session_date: date,
    profile: MarketProfile,
    *,
    close_buffer_minutes: int = 90,
) -> dict[str, str | int | bool]:
    opened, closed = session_bounds(session_date, profile)
    completed_after = closed + timedelta(minutes=close_buffer_minutes)
    return {
        "exchange_timezone": profile.timezone_name,
        "timezone_source": profile.source,
        "session_open_timestamp": opened.astimezone(timezone.utc).isoformat(),
        "session_close_timestamp": closed.astimezone(timezone.utc).isoformat(),
        "completed_after_timestamp": completed_after.astimezone(timezone.utc).isoformat(),
        "close_buffer_minutes": close_buffer_minutes,
        "session_mapping_status": profile.mapping_status,
        "session_mapping_source": profile.source,
    }


def session_is_complete(
    session_date: date,
    profile: MarketProfile,
    now: datetime,
    *,
    close_buffer_minutes: int = 90,
) -> bool:
    if now.tzinfo is None:
        now = now.replace(tzinfo=timezone.utc)
    _, closed = session_bounds(session_date, profile)
    completed_after = closed + timedelta(minutes=close_buffer_minutes)
    return now.astimezone(timezone.utc) >= completed_after.astimezone(timezone.utc)
