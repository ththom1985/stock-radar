"""Shared projected NYSE full-session calendar (no early-close adjustment)."""
from datetime import date, timedelta
from functools import lru_cache


def _nth_weekday(year: int, month: int, weekday: int, occurrence: int) -> date:
    first = date(year, month, 1)
    return first + timedelta(days=(weekday - first.weekday()) % 7 + 7 * (occurrence - 1))


def _last_weekday(year: int, month: int, weekday: int) -> date:
    next_month = date(year + 1, 1, 1) if month == 12 else date(year, month + 1, 1)
    result = next_month - timedelta(days=1)
    return result - timedelta(days=(result.weekday() - weekday) % 7)


def _observed_fixed(value: date) -> date:
    if value.weekday() == 5:
        return value - timedelta(days=1)
    if value.weekday() == 6:
        return value + timedelta(days=1)
    return value


def _easter_sunday(year: int) -> date:
    a = year % 19
    b, c = divmod(year, 100)
    d, e = divmod(b, 4)
    f = (b + 8) // 25
    g = (b - f + 1) // 3
    h = (19 * a + b - d - g + 15) % 30
    i, k = divmod(c, 4)
    l = (32 + 2 * e + 2 * i - h - k) % 7
    m = (a + 11 * h + 22 * l) // 451
    month = (h + l - 7 * m + 114) // 31
    day = (h + l - 7 * m + 114) % 31 + 1
    return date(year, month, day)


@lru_cache(maxsize=128)
def _nyse_holidays(year: int) -> frozenset[date]:
    # NYSE does not observe a Saturday New Year's Day on December 31.
    new_year = date(year, 1, 1)
    holidays = {
        new_year + timedelta(days=1) if new_year.weekday() == 6 else new_year,
        _nth_weekday(year, 1, 0, 3),
        _nth_weekday(year, 2, 0, 3),
        _easter_sunday(year) - timedelta(days=2),
        _last_weekday(year, 5, 0),
        _observed_fixed(date(year, 7, 4)),
        _nth_weekday(year, 9, 0, 1),
        _nth_weekday(year, 11, 3, 4),
        _observed_fixed(date(year, 12, 25)),
    }
    if year >= 2022:
        holidays.add(_observed_fixed(date(year, 6, 19)))
    if year == 2025:
        holidays.add(date(2025, 1, 9))  # National day of mourning.
    return frozenset(holidays)


def _is_projected_us_session(value: date) -> bool:
    return value.weekday() < 5 and value not in (
        _nyse_holidays(value.year - 1)
        | _nyse_holidays(value.year)
        | _nyse_holidays(value.year + 1)
    )
