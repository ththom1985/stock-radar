"""Conservative asset/security classification."""
from __future__ import annotations

from typing import Any


COMPANY_EQUITY = "company_equity"
ETF_FUND = "etf_fund"
CRYPTO = "crypto"
INDEX_OTHER = "index_other"
UNKNOWN = "unknown"

_FUND_WORDS = (
    " etf",
    "fund",
    "ucits",
    "index trust",
    "ishares",
    "vanguard",
    "invesco",
    "wisdomtree",
    "spdr",
    "xtrackers",
    "amundi",
    "lyxor",
)


def classify_asset(
    symbol: str,
    name: str = "",
    fundamentals: dict[str, Any] | None = None,
    themes: list[str] | None = None,
) -> str:
    symbol = (symbol or "").upper()
    quote_type = str((fundamentals or {}).get("quote_type") or "").upper()
    labels = {str(t).upper() for t in (themes or [])}
    lower_name = f" {(name or '').lower()}"

    if symbol.endswith("-USD") or quote_type == "CRYPTOCURRENCY":
        return CRYPTO
    if quote_type == "EQUITY":
        return COMPANY_EQUITY
    if quote_type in {"ETF", "MUTUALFUND", "MONEYMARKET"}:
        return ETF_FUND
    if quote_type in {
        "INDEX",
        "FUTURE",
        "CURRENCY",
    }:
        return INDEX_OTHER
    if symbol.startswith("^") or symbol.endswith("=F"):
        return INDEX_OTHER
    if "ETF" in labels or any(word in lower_name for word in _FUND_WORDS):
        return ETF_FUND
    if not quote_type:
        return COMPANY_EQUITY
    return INDEX_OTHER


def is_company(asset_type: str) -> bool:
    return asset_type == COMPANY_EQUITY


def classify_configured_asset(
    symbol: str,
    name: str = "",
    exchange: str = "",
    themes: list[str] | None = None,
) -> str:
    """Best conservative class available before provider metadata is fetched."""
    exchange_lower = (exchange or "").strip().lower()
    if exchange_lower == "krypto":
        return CRYPTO
    if exchange_lower == "etf":
        return ETF_FUND
    if not symbol:
        return UNKNOWN
    return classify_asset(symbol, name, fundamentals={}, themes=themes)
