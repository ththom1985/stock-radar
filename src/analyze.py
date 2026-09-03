"""Conservative completed-daily-bar research pipeline.

The output is explicitly unvalidated and non-actionable. Rankings never use
projections or sparsely covered optional features.
"""
from __future__ import annotations

import math
import os
import re
from datetime import date, datetime, timezone
from numbers import Integral, Real

import numpy as np

from .alternative_signals import attach_alternative_signals
from .aschenbrenner import load_aschenbrenner, stance_for
from .assets import (
    COMPANY_EQUITY,
    classify_asset,
    classify_configured_asset,
    is_company,
)
from .config import DATA, OUTPUT, TOP_N
from .congress_trades import fetch_congress_signals
from .data_quality import OUTPUT_SCHEMA, OUTPUT_SCHEMA_VERSION, build_data_status
from .deep_fundamentals import fetch_deep
from .earnings import days_until, fetch_earnings
from .earnings_tone import fetch_earnings_tone_signals
from .expert_layer import (
    SOURCE_CATALOG as EXPERT_SOURCE_CATALOG,
    attach_expert_analysis,
    build_expert_rankings,
    load_score_weights,
)
from .expert_signals import (
    minervini,
    tech_momentum_score,
    tech_trend_score,
    tech_volume_score,
    weinstein_stage,
)
from .fetch import fetch_prices_with_status
from .fundamental_score import (
    combine_fundamental,
    magic_formula_ranks,
    score_growth,
    score_quality,
    score_value,
)
from .fundamentals import fetch_fundamentals
from .fx import currency_for, get_fx_rates_with_status
from .fred_regime import fetch_fred_regime
from .filing_diffs import fetch_filing_diff_signals
from .finra_short_interest import fetch_finra_signals
from .geo import country_flag
from .indicators import compute_features
from .job_postings import fetch_job_signals
from .insights import (
    INSIGHT_CONTRACT_VERSION,
    INSIGHT_STATUS,
    PROVENANCE_CATALOG,
    SWEET_SPOT_CONTRACT,
    enrich_rows_and_rankings,
    rehydrate_rankings,
)
from .macro import fetch_macro, macro_adjust
from .market_positioning import fetch_market_positioning
from .news_engine import fetch_all_ticker_news, fetch_market_news, news_signal
from .options_gex import fetch_gex_signals
from .paper_trader import update_portfolio
from .persistence import (
    SCHEMA_VERSION,
    atomic_write_json,
    effective_path,
    load_json,
    schema_meta,
    utc_now,
)
from .projection import project
from .probability_inference import (
    attach_probability_forecasts,
    load_probability_baselines,
    load_probability_validation_summary,
)
from .recommendation_journal import (
    evaluate_mature_observations,
    journal_summary,
    record_top_observations,
    record_confluence_observations,
)
from .rating import radar_elo, radar_score, stars
from .score import score_daily_signal, score_longterm
from .financial_sector_history import fetch_financial_sector_history
from .sec_companyfacts import fetch_sec_companyfacts, merge_official_fundamentals
from .sec_13f import fetch_13f_signals
from .sec_insiders import fetch_insider_signals
from .universe import load_universe
from .valuation_history import update_valuation_history
from .today_view import build_today_view
from .question_views import build_question_views, decision_overlay
from .opportunity_history import update_opportunity_history
from .wikipedia_attention import fetch_wikipedia_signals

FAILED_MANIFEST = DATA / "failed_symbols.json"
VALUATION_ANOMALIES = DATA / "valuation_anomalies.json"
COMPARABLE_FUNDAMENTAL_FIELDS = (
    "pe",
    "pb",
    "roe",
    "profit_margin",
    "debt_to_equity",
    "revenue_growth",
    "earnings_growth",
)
COMPARABLE_TECHNICAL_FIELDS = (
    "price",
    "sma50",
    "sma200",
    "rsi",
    "macd",
    "macd_signal",
    "macd_hist",
    "macd_hist_prev",
    "ret_60d",
    "atr_pct",
    "rvol",
    "vol",
    "vol_avg20",
    "high20",
    "low20",
    "daily_return",
)


def _load_json_map(filename: str) -> dict:
    return load_json(DATA / filename, expected_type=dict, default={})


def _json_value(value):
    if isinstance(value, np.generic):
        value = value.item()
    if isinstance(value, bool):
        return value
    if isinstance(value, Integral):
        return int(value)
    if isinstance(value, Real):
        value = float(value)
        return value if math.isfinite(value) else None
    if isinstance(value, str):
        return value
    return None


def _percent(value):
    return value * 100 if isinstance(value, (int, float)) and math.isfinite(value) else None


def _investment_score(longterm, fundamental):
    if fundamental is None:
        return None
    return 0.45 * longterm + 0.55 * fundamental


def _technical_complete(row: dict) -> bool:
    if not all(
        isinstance(row.get(key), (int, float)) and math.isfinite(row[key])
        for key in COMPARABLE_TECHNICAL_FIELDS
    ):
        return False
    return (
        row["price"] > 0
        and row["vol"] >= 0
        and row["vol_avg20"] > 0
        and row["rvol"] >= 0
        and row["atr_pct"] >= 0
    )


def _copy_fundamental_context(row: dict, fundamental: dict) -> None:
    row["sector"] = fundamental.get("sector")
    row["industry"] = fundamental.get("industry")
    row["provider_long_name"] = fundamental.get("provider_long_name")
    row["analyst_rating"] = fundamental.get("rec_key")
    row["analyst_mean"] = _json_value(fundamental.get("rec_mean"))
    row["analyst_n"] = _json_value(fundamental.get("analyst_n"))
    row["target_price_local"] = _json_value(fundamental.get("target_price"))
    row["pe"] = _json_value(fundamental.get("pe"))
    row["forward_pe"] = _json_value(fundamental.get("forward_pe"))
    row["pb"] = _json_value(fundamental.get("pb"))
    row["bvps"] = _json_value(fundamental.get("bvps"))
    row["peg"] = _json_value(fundamental.get("peg"))
    row["ev_ebitda"] = _json_value(fundamental.get("ev_ebitda"))
    row["price_to_sales"] = _json_value(fundamental.get("ps"))
    row["price_to_fcf"] = _json_value(fundamental.get("price_to_fcf"))
    row["profit_margin_pct"] = _percent(fundamental.get("profit_margin"))
    row["debt_to_equity_pct"] = _json_value(fundamental.get("debt_to_equity"))
    row["current_ratio"] = _json_value(fundamental.get("current_ratio"))
    row["free_cashflow"] = _json_value(fundamental.get("free_cashflow"))
    row["eps"] = _json_value(fundamental.get("eps"))
    row["revenue"] = _json_value(fundamental.get("revenue"))
    row["ebitda"] = _json_value(fundamental.get("ebitda"))
    row["shares_outstanding"] = _json_value(
        fundamental.get("shares_outstanding")
    )
    row["total_debt"] = _json_value(fundamental.get("total_debt"))
    row["total_cash"] = _json_value(fundamental.get("total_cash"))
    row["earnings_growth"] = _json_value(fundamental.get("earnings_growth"))
    row["roe_pct"] = _percent(fundamental.get("roe"))
    row["revenue_growth_pct"] = _percent(fundamental.get("revenue_growth"))
    row["beta"] = _json_value(fundamental.get("beta"))
    row["provider_country"] = fundamental.get("provider_country")
    row["reported_currency"] = fundamental.get("reported_currency")
    row["market_cap_local"] = _json_value(fundamental.get("market_cap"))
    row["issuer_uuid"] = fundamental.get("issuer_uuid")
    row["fundamental_field_sources"] = fundamental.get("field_sources") or {}
    row["sec_companyfacts"] = fundamental.get("sec_companyfacts")
    fetched_at = fundamental.get("last_success_at") or fundamental.get("fetched_at")
    age_days = None
    try:
        fetched = datetime.fromisoformat(str(fetched_at))
        if fetched.tzinfo is None:
            fetched = fetched.replace(tzinfo=timezone.utc)
        age_days = (
            datetime.now(timezone.utc) - fetched.astimezone(timezone.utc)
        ).total_seconds() / 86400
    except (TypeError, ValueError):
        pass
    row["fundamental_source_status"] = {
        "last_success_at": fetched_at,
        "age_days": age_days,
        "refresh_failure": fundamental.get("_refresh_failure"),
        "status": (
            "current"
            if age_days is not None
            and age_days <= 7
            and not fundamental.get("_refresh_failure")
            else "stale_or_unavailable"
        ),
    }


def _apply_company_fundamentals(row: dict, fundamental: dict, magic: dict) -> None:
    if not is_company(row["asset_type"]):
        row.update(
            {
                "value_score": None,
                "quality_score": None,
                "growth_score": None,
                "fundamental_score": None,
                "investment_score": None,
                "fundamental_reasons": [],
                "magic_score": None,
            }
        )
        return
    complete = all(
        isinstance(fundamental.get(field), (int, float))
        and math.isfinite(fundamental[field])
        for field in COMPARABLE_FUNDAMENTAL_FIELDS
    )
    comparable = (
        {field: fundamental[field] for field in COMPARABLE_FUNDAMENTAL_FIELDS}
        if complete
        else {}
    )
    value, value_reasons = score_value(comparable)
    quality, quality_reasons = score_quality(comparable)
    growth, growth_reasons = score_growth(comparable)
    complete = complete and all(score is not None for score in (value, quality, growth))
    combined = combine_fundamental(value, quality, growth) if complete else None
    row.update(
        {
            "value_score": value,
            "quality_score": quality,
            "growth_score": growth,
            "fundamental_score": combined,
            "investment_score": _investment_score(row["longterm_score"], combined),
            "fundamental_reasons": (
                quality_reasons + value_reasons + growth_reasons
            )[:5],
            "magic_score": magic.get(row["symbol"]),
            "fundamental_rank_fields": list(COMPARABLE_FUNDAMENTAL_FIELDS),
        }
    )


def _convert_to_usd(row: dict, rate: float) -> None:
    row["fx_usd"] = rate
    price_keys = (
        "price",
        "prev_close",
        "sma20",
        "sma50",
        "sma150",
        "sma200",
        "sma150_1m_ago",
        "sma200_1m_ago",
        "ema9",
        "ema21",
        "high52",
        "low52",
        "high20",
        "low20",
        "atr",
        "pivot",
        "pivot_r1",
        "pivot_s1",
    )
    for key in price_keys:
        value = row.get(key)
        if isinstance(value, (int, float)):
            row[key] = value * rate
    target = row.get("target_price_local")
    row["target_price"] = target * rate if isinstance(target, (int, float)) else None
    row["raw_open_usd"] = (
        row["raw_open_local"] * rate
        if isinstance(row.get("raw_open_local"), (int, float))
        else None
    )
    row["raw_close_usd"] = (
        row["raw_close_local"] * rate
        if isinstance(row.get("raw_close_local"), (int, float))
        else None
    )
    row["dividend_usd"] = (
        row["dividend_local"] * rate
        if isinstance(row.get("dividend_local"), (int, float))
        else None
    )
    row["avg_dollar_volume_20_usd"] = (
        row["avg_dollar_volume_20_local"] * rate
        if isinstance(row.get("avg_dollar_volume_20_local"), (int, float))
        else None
    )
    row["market_cap_usd"] = (
        row["market_cap_local"] * rate
        if isinstance(row.get("market_cap_local"), (int, float))
        and (
            not row.get("reported_currency")
            or row.get("reported_currency") == row.get("currency")
        )
        else None
    )
    row["corporate_actions"] = [
        {
            **action,
            "dividend_usd": (
                action.get("dividend_local", 0.0) * rate
                if isinstance(action.get("dividend_local"), (int, float))
                else 0.0
            ),
        }
        for action in (row.get("corporate_actions") or [])
    ]


def _issuer_key(row: dict) -> str:
    if row.get("issuer_uuid"):
        return f"uuid:{row['issuer_uuid']}"
    name = re.sub(
        r"\b(class\s+[a-z]|ordinary shares?|common stock|adr|ads)\b",
        "",
        str(row.get("name") or row["symbol"]),
        flags=re.IGNORECASE,
    )
    name = re.sub(r"[^a-z0-9]+", " ", name.lower()).strip()
    return f"name:{name or row['symbol'].lower()}"


def _paper_eligibility(row: dict) -> dict:
    reasons = []
    min_liquidity = float(
        os.environ.get("STOCK_RADAR_MIN_DOLLAR_VOLUME", "20000000")
    )
    max_atr = float(os.environ.get("STOCK_RADAR_MAX_PAPER_ATR_PCT", "5"))
    max_annual_vol = float(
        os.environ.get("STOCK_RADAR_MAX_PAPER_ANNUAL_VOL_PCT", "60")
    )
    if row.get("asset_type") != COMPANY_EQUITY:
        reasons.append("paper simulation accepts company equities only")
    if row.get("currency") != "USD":
        reasons.append("non-USD paper fills disabled until point-in-time FX exists")
    if not (row.get("feature_coverage") or {}).get("rank_eligible"):
        reasons.append("instrument is not rank-eligible")
    liquidity = row.get("avg_dollar_volume_20_usd")
    if not isinstance(liquidity, (int, float)) or liquidity < min_liquidity:
        reasons.append(
            f"20-day average dollar volume below USD {min_liquidity:,.0f}"
        )
    atr_pct = row.get("atr_pct")
    if not isinstance(atr_pct, (int, float)) or atr_pct > max_atr:
        reasons.append(f"daily ATR is missing or above {max_atr:g}%")
    annual_vol = row.get("vol_annual_pct")
    if not isinstance(annual_vol, (int, float)) or annual_vol > max_annual_vol:
        reasons.append(
            f"annualized volatility is missing or above {max_annual_vol:g}%"
        )
    return {"eligible": not reasons, "reasons": reasons}


def _fetch_benchmarks(now: datetime) -> tuple[dict, dict]:
    symbols = {"sp500": "^GSPC", "ndx": "^NDX", "world": "URTH"}
    fetched = fetch_prices_with_status(
        list(symbols.values()),
        period="3mo",
        now=now,
        verbose=False,
    )
    values = {}
    for name, symbol in symbols.items():
        frame = fetched.prices.get(symbol)
        info = fetched.bar_info.get(symbol)
        if frame is not None and info:
            values[name] = {
                "value": float(frame["Close"].iloc[-1]),
                "bar_date": info["bar_date"],
                "bar_timestamp": info["bar_timestamp"],
                "source_interval": "1d",
                "completed_bars_only": True,
            }
    return values, fetched.failed_symbols


def _partition_rankings(rows: list[dict], top_n: int = TOP_N) -> dict:
    """Rank only within a shared trading currency and asset class."""
    partitions: dict[str, dict[str, list[dict]]] = {}
    for row in rows:
        currency = row.get("currency") or "UNKNOWN"
        asset_type = row.get("asset_type") or "unknown"
        partitions.setdefault(currency, {}).setdefault(asset_type, []).append(row)
    for by_asset in partitions.values():
        for asset_type, members in by_asset.items():
            by_asset[asset_type] = sorted(
                members,
                key=lambda member: member["radar_score"],
                reverse=True,
            )[:top_n]
    return partitions


def _research_summary(row: dict) -> str:
    parts = [
        f"Completed-daily-bar trend score {row.get('longterm_score')}/100",
        f"daily momentum context {row.get('daily_signal_direction', 'NEUTRAL').lower()}",
    ]
    if row.get("fundamental_score") is not None:
        parts.append(f"complete company fundamental score {row['fundamental_score']}/100")
    else:
        parts.append("company fundamental score not applicable or incomplete")
    return "; ".join(parts) + ". Unvalidated heuristic research signal, not a recommendation."


def run(with_news=True, with_fundamentals=True):
    if os.environ.get("STOCK_RADAR_INTRADAY") == "1":
        raise RuntimeError("Intraday mode was removed: this pipeline uses completed daily bars only")

    now = datetime.now(timezone.utc)
    previous_snapshot = load_json(OUTPUT / "latest.json", expected_type=dict, default={})
    market_data_only = os.environ.get("STOCK_RADAR_MARKET_DATA_ONLY") == "1"
    if market_data_only:
        with_news = False
        with_fundamentals = False
    full_universe = load_universe()
    themes = _load_json_map("themes.json")
    configured_types_full = {
        item["symbol"]: classify_configured_asset(
            item["symbol"],
            item["name"],
            item.get("exchange", ""),
            themes.get(item["symbol"]) or [],
        )
        for item in full_universe
    }
    configured_asset_counts_full = {}
    for asset_type in configured_types_full.values():
        configured_asset_counts_full[asset_type] = (
            configured_asset_counts_full.get(asset_type, 0) + 1
        )
    universe = full_universe
    max_symbols_raw = os.environ.get("STOCK_RADAR_MAX_SYMBOLS")
    if max_symbols_raw:
        max_symbols = max(1, int(max_symbols_raw))
        universe = universe[:max_symbols]
    dry_run = os.environ.get("STOCK_RADAR_DRY_RUN") == "1"
    symbols = [item["symbol"] for item in universe]
    names = {item["symbol"]: item["name"] for item in universe}
    exchanges = {item["symbol"]: item.get("exchange", "") for item in universe}
    configured_types = {
        symbol: configured_types_full[symbol] for symbol in symbols
    }
    expected_asset_counts = {}
    for asset_type in configured_types.values():
        expected_asset_counts[asset_type] = expected_asset_counts.get(asset_type, 0) + 1
    expert_sources = _load_json_map("expert_sources.json")
    print(f"Universe: {len(symbols)} | completed daily bars only")

    fetched = fetch_prices_with_status(symbols, now=now)
    failed_symbols = dict(fetched.failed_symbols)
    rows = []
    for symbol, frame in fetched.prices.items():
        features = compute_features(frame)
        if not features:
            failed_symbols[symbol] = "insufficient features after completed-bar filtering"
            continue
        daily_score, daily_direction, daily_reasons = score_daily_signal(features)
        longterm_score, longterm_reasons = score_longterm(features)
        row = {key: _json_value(value) for key, value in features.items()}
        row.update(fetched.bar_info[symbol])
        row.update(
            {
                "symbol": symbol,
                "name": names.get(symbol, ""),
                "short_name": names.get(symbol, ""),
                "exchange": exchanges.get(symbol, ""),
                "listing_market": exchanges.get(symbol, ""),
                "configured_asset_type": configured_types.get(symbol, "unknown"),
                "price_local": _json_value(features["price"]),
                "price": _json_value(features["price"]),
                "daily_return_pct": _percent(features.get("daily_return")),
                "vol_daily_pct": _percent(features.get("vol_daily")),
                "daily_signal_score": daily_score,
                "daily_signal_direction": daily_direction,
                "daily_signal_reasons": daily_reasons,
                "longterm_score": longterm_score,
                "longterm_reasons": longterm_reasons,
                "raw_open_local": _json_value(features.get("raw_open")),
                "raw_close_local": _json_value(features.get("raw_close")),
                "dividend_local": _json_value(features.get("dividend")),
                "stock_split": _json_value(features.get("stock_split")),
                "avg_dollar_volume_20_local": _json_value(
                    features.get("avg_dollar_volume_20")
                ),
                # Deprecated v1 keys intentionally carry no trade assertion.
                "daytrade_score": None,
                "daytrade_direction": "DEPRECATED_DAILY_ONLY",
                "daytrade_reasons": [],
            }
        )
        try:
            row["bar_age_days"] = (now.date() - date.fromisoformat(row["bar_date"])).days
        except ValueError:
            row["bar_age_days"] = None
        rows.append(row)

    fundamental_by_symbol = (
        fetch_fundamentals([row["symbol"] for row in rows])
        if with_fundamentals
        else {}
    )
    if with_fundamentals:
        sec_by_symbol, sec_source_status = fetch_sec_companyfacts(
            [row["symbol"] for row in rows]
        )
        fundamental_by_symbol = {
            symbol: merge_official_fundamentals(
                fundamental_by_symbol.get(symbol, {}),
                sec_by_symbol.get(symbol, {}),
            )
            for symbol in [row["symbol"] for row in rows]
        }
    else:
        sec_by_symbol = {}
        sec_source_status = {
            "status": "skipped",
            "reason": "fundamental enrichment disabled",
            "refreshed": 0,
        }
    if not market_data_only:
        insider_by_symbol, insider_source_status = fetch_insider_signals(
            [row["symbol"] for row in rows]
        )
    else:
        insider_by_symbol = {}
        insider_source_status = {
            "status": "skipped",
            "reason": "market-data-only pipeline",
            "refreshed": 0,
        }
    magic = magic_formula_ranks(fundamental_by_symbol)
    for row in rows:
        fundamental = fundamental_by_symbol.get(row["symbol"], {})
        _copy_fundamental_context(row, fundamental)
        row["themes"] = themes.get(row["symbol"]) or []
        row["asset_type"] = classify_asset(
            row["symbol"],
            row["name"],
            fundamental,
            row["themes"],
        )
        _apply_company_fundamentals(row, fundamental, magic)

    currencies = {currency_for(row["symbol"]) for row in rows}
    fx_result = get_fx_rates_with_status(currencies)
    converted_rows = []
    for row in rows:
        currency = currency_for(row["symbol"])
        row["currency"] = currency
        rate = fx_result.rates.get(currency)
        if rate is None:
            failed_symbols[row["symbol"]] = (
                f"missing non-USD FX rate for {currency}: "
                f"{fx_result.missing.get(currency, 'unavailable')}"
            )
            continue
        _convert_to_usd(row, rate)
        target_local = row.get("target_price_local")
        if (
            isinstance(target_local, (int, float))
            and isinstance(row.get("price_local"), (int, float))
            and row["price_local"] > 0
        ):
            row["analyst_upside_pct"] = (
                target_local / row["price_local"] - 1
            ) * 100
        else:
            row["analyst_upside_pct"] = None
        converted_rows.append(row)
    rows = converted_rows

    news_status = {}
    market_news = {
        "headlines": [],
        "market_sentiment": None,
        "market_label": None,
        "model_status": "unavailable",
    }
    if with_news:
        by_symbol, news_status = fetch_all_ticker_news(
            [row["symbol"] for row in rows],
            now=now,
            return_status=True,
        )
        market_news = fetch_market_news(now=now)
    else:
        by_symbol = {}
    earnings = (
        fetch_earnings([row["symbol"] for row in rows])
        if not market_data_only
        else {}
    )
    for row in rows:
        row.update(news_signal(by_symbol.get(row["symbol"], [])))
        earning = earnings.get(row["symbol"], {})
        row["next_earnings"] = earning.get("next_earnings")
        row["previous_earnings"] = earning.get("previous_earnings")
        row["earnings_in_days"] = days_until(row["next_earnings"])
        if (
            isinstance(row["earnings_in_days"], int)
            and row["earnings_in_days"] < 0
        ):
            row["previous_earnings"] = (
                row["previous_earnings"] or row["next_earnings"]
            )
            row["next_earnings"] = None
            row["earnings_in_days"] = None

    # Optional features are context only and cannot alter cross-sectional ranks.
    relative = sorted(
        [row for row in rows if isinstance(row.get("ret_60d"), (int, float))],
        key=lambda row: row["ret_60d"],
    )
    for index, row in enumerate(relative):
        row["rs_rating"] = (index + 1) / len(relative) * 100 if relative else None
    company_symbols = [
        row["symbol"]
        for row in sorted(
            rows,
            key=lambda row: row.get("investment_score")
            if isinstance(row.get("investment_score"), (int, float))
            else -1,
            reverse=True,
        )
        if row["asset_type"] == COMPANY_EQUITY
    ]
    market_caps = {
        symbol: (fundamental_by_symbol.get(symbol) or {}).get("market_cap")
        for symbol in company_symbols
    }
    deep = fetch_deep(company_symbols, market_caps) if with_fundamentals else {}
    macro = (
        fetch_macro()
        if not market_data_only
        else {"status": "skipped", "context_only": True}
    )
    positioning, positioning_source_status = (
        fetch_market_positioning()
        if not market_data_only
        else (
            {"status": "skipped", "model_status": "heuristic_context_only"},
            {"status": "skipped", "reason": "market-data-only pipeline"},
        )
    )
    aschenbrenner = (
        load_aschenbrenner()
        if not market_data_only
        else {"holdings": {}, "report_quarter": None}
    )

    for row in rows:
        deep_row = deep.get(row["symbol"], {})
        row["piotroski"] = deep_row.get("piotroski")
        row["altman_z"] = deep_row.get("altman_z")
        score, met, failed = minervini(row, row.get("rs_rating"))
        row["minervini_score"] = score
        row["minervini_met"] = met
        row["minervini_failed"] = failed
        stage, label = weinstein_stage(row)
        row["weinstein_stage"] = stage
        row["weinstein_label"] = label
        row["tech_trend"] = tech_trend_score(row)
        row["tech_momentum"] = tech_momentum_score(row)
        row["tech_volume"] = tech_volume_score(row)
        row["expert_sources"] = expert_sources.get(row["symbol"]) or []
        row["aschenbrenner"] = stance_for(row["symbol"], aschenbrenner)
        row["country"], row["cc"] = country_flag(row["symbol"])
        row["issuer_key"] = _issuer_key(row)
        _macro_points, macro_notes = macro_adjust(row, macro)
        row["macro_notes"] = macro_notes
        row["macro_model_status"] = "heuristic_context_only"
        row["positioning_context"] = positioning
        fundamental_complete = all(
            row.get(key) is not None
            for key in ("value_score", "quality_score", "growth_score")
        )
        fundamental_current = (
            row.get("fundamental_source_status", {}).get("status") == "current"
        )
        technical_complete = _technical_complete(row)
        row["feature_coverage"] = {
            "technical_complete": technical_complete,
            "technical_rank_fields": list(COMPARABLE_TECHNICAL_FIELDS),
            "fundamental_applicable": is_company(row["asset_type"]),
            "fundamental_complete": fundamental_complete,
            "fundamental_current": fundamental_current,
            "fundamental_rank_fields": (
                list(COMPARABLE_FUNDAMENTAL_FIELDS)
                if is_company(row["asset_type"])
                else []
            ),
            "deep_fundamental_available": row.get("piotroski") is not None,
            "news_current_count": row.get("news_n") or 0,
            "rank_eligible": (
                technical_complete
            ),
        }
        elo, label, color = radar_elo(row)
        row["radar_elo"] = elo
        row["radar_score"] = radar_score(elo)
        row["radar_rating"] = label
        row["radar_color"] = color
        row["stars"] = stars(row["radar_score"])
        row["heuristic_summary"] = _research_summary(row)
        row["plain_summary"] = row["heuristic_summary"]
        row["scenario_long"] = project(row, "long")
        row["projection_long"] = row["scenario_long"]
        row["projection_short"] = []
        row["exp_return_12m"] = None
        row["conviction"] = None
        row["upside_pct"] = None
        row["urgency"] = None
        row["actions"] = []
        row["trade_plan_long"] = None
        row["trade_plan_short"] = None
        row["intraday_note"] = None
        row["paper_eligibility"] = _paper_eligibility(row)

    # Calibrated probabilities are a separate, fail-closed output.  This call
    # only appends probability_forecast and cannot alter any score or list input.
    attach_probability_forecasts(
        rows,
        fetched.prices,
        spy_history=fetched.prices.get("SPY"),
        now=now,
        embed_baselines=False,
    )
    probability_baselines = load_probability_baselines()
    probability_validation = load_probability_validation_summary()

    peer_counts = {}
    for row in rows:
        if (
            row["asset_type"] == COMPANY_EQUITY
            and row["feature_coverage"]["fundamental_complete"]
            and row["feature_coverage"]["fundamental_current"]
        ):
            peer_key = row.get("sector") or "unclassified"
            peer_counts[peer_key] = peer_counts.get(peer_key, 0) + 1
    for row in rows:
        peer_key = row.get("sector") or "unclassified"
        row["fundamental_comparability"] = {
            "used_in_overall_ranking": False,
            "peer_group": peer_key,
            "complete_current_peer_count": peer_counts.get(peer_key, 0),
            "reason": (
                "generic fundamental bands are descriptive only; robust sector-neutral "
                "point-in-time peer ranking is not implemented"
            ),
        }

    stale_fx = sorted(
        currency
        for currency, source in fx_result.status.items()
        if source.get("status") in {"stale", "legacy_stale"}
    )
    fx_blockers = []
    if fx_result.missing:
        fx_blockers.append(f"missing FX currencies: {sorted(fx_result.missing)}")
    if stale_fx:
        fx_blockers.append(f"stale FX currencies: {stale_fx}")
    unknown_sessions = sorted(
        row["symbol"]
        for row in rows
        if row.get("session_mapping_status") != "verified_conservative"
    )
    if unknown_sessions:
        fx_blockers.append(
            f"unknown/degraded exchange-session mappings: {unknown_sessions}"
        )
    feature_counts = {
        asset_type: {
            "total": total,
            "analyzed_successfully": 0,
            "technical_complete": 0,
            "rank_eligible": 0,
            "fundamental_complete_current": 0,
        }
        for asset_type, total in expected_asset_counts.items()
    }
    for row in rows:
        coverage_asset = row.get("configured_asset_type") or "unknown"
        counts = feature_counts.setdefault(
            coverage_asset,
            {
                "total": 0,
                "analyzed_successfully": 0,
                "technical_complete": 0,
                "rank_eligible": 0,
                "fundamental_complete_current": 0,
            },
        )
        counts["analyzed_successfully"] += 1
        counts["technical_complete"] += bool(
            row["feature_coverage"]["technical_complete"]
        )
        counts["rank_eligible"] += bool(row["feature_coverage"]["rank_eligible"])
        counts["fundamental_complete_current"] += bool(
            row["feature_coverage"]["fundamental_complete"]
            and row["feature_coverage"]["fundamental_current"]
        )
    data_status = build_data_status(
        universe_size=len(symbols),
        rows=rows,
        failed_symbols=failed_symbols,
        now=now,
        extra_blockers=fx_blockers,
        feature_coverage=feature_counts,
    )
    stale_or_invalid = set(data_status.get("stale_symbols") or [])
    stale_or_invalid.update(data_status.get("missing_bar_date_symbols") or [])
    stale_or_invalid.update(data_status.get("future_bar_symbols") or [])
    fresh_rows = [row for row in rows if row["symbol"] not in stale_or_invalid]
    rankable = [
        row
        for row in fresh_rows
        if row["feature_coverage"]["rank_eligible"]
    ]
    rankings_by_currency_asset = _partition_rankings(rankable)
    # Generic absolute fundamental bands remain descriptive only; no universal
    # cross-sector "top fundamental" rank is published.
    top_fundamental = []
    if not data_status["data_actionable"]:
        rankings_by_currency_asset = {}
        top_fundamental = []

    rows, insight_rankings = enrich_rows_and_rankings(
        rows,
        rankings_enabled=data_status["data_actionable"],
        blockers=data_status["blocking_reasons"],
    )
    if not market_data_only:
        gex_by_symbol, gex_source_status = fetch_gex_signals(rows)
        short_interest_by_symbol, finra_source_status = fetch_finra_signals(rows)
        institutional_by_symbol, sec_13f_status = fetch_13f_signals(rows)
        congress_by_symbol, congress_source_status = fetch_congress_signals(rows)
        wikipedia_by_symbol, wikipedia_source_status = fetch_wikipedia_signals(rows)
        jobs_by_symbol, jobs_coverage, jobs_source_status = fetch_job_signals(rows)
        filing_diff_by_symbol, filing_diff_status = fetch_filing_diff_signals(rows)
        earnings_tone_by_symbol, earnings_tone_status = fetch_earnings_tone_signals(rows)
        fred_regime, fred_source_status = fetch_fred_regime()
    else:
        gex_by_symbol = {}
        short_interest_by_symbol = {}
        institutional_by_symbol = {}
        congress_by_symbol = {}
        wikipedia_by_symbol = {}
        jobs_by_symbol = {}
        filing_diff_by_symbol = {}
        earnings_tone_by_symbol = {}
        jobs_coverage = {
            row["symbol"]: {"status": "skipped_market_data_only"} for row in rows
        }
        gex_source_status = {"status": "skipped", "reason": "market-data-only pipeline"}
        finra_source_status = {
            "status": "skipped",
            "reason": "market-data-only pipeline",
        }
        sec_13f_status = {"status": "skipped", "reason": "market-data-only pipeline"}
        congress_source_status = {
            "status": "skipped",
            "reason": "market-data-only pipeline",
        }
        wikipedia_source_status = {
            "status": "skipped",
            "reason": "market-data-only pipeline",
        }
        jobs_source_status = {
            "status": "skipped",
            "reason": "market-data-only pipeline",
        }
        filing_diff_status = {"status":"skipped","reason":"market-data-only pipeline"}
        earnings_tone_status = {"status":"skipped","reason":"market-data-only pipeline"}
        fred_regime = {}
        fred_source_status = {"status": "skipped", "reason": "market-data-only pipeline"}
    for row in rows:
        row["jobs_signal"] = jobs_coverage.get(row["symbol"])
        catalyst_values = [
            value
            for value in (
                row.get("news_score"),
                fred_regime.get("score"),
                positioning.get("score"),
            )
            if isinstance(value, (int, float)) and math.isfinite(value)
        ]
        row["market_regime"] = fred_regime or None
        row["catalyst_context"] = {
            "score": (
                round(sum(catalyst_values) / len(catalyst_values), 1)
                if catalyst_values
                else None
            ),
            "news_score": row.get("news_score"),
            "macro_regime_score": fred_regime.get("score"),
            "macro_regime": fred_regime.get("regime"),
            "positioning_score": positioning.get("score"),
            "positioning_regime": positioning.get("regime"),
            "earnings_event": row.get("next_earnings"),
            "earnings_in_days": row.get("earnings_in_days"),
            "note": (
                "Earnings proximity is event context only and receives no directional score."
            ),
        }
    attach_alternative_signals(
        rows,
        insider_by_symbol,
        gex_by_symbol,
        short_interest_by_symbol,
        institutional_by_symbol,
        congress_by_symbol,
        wikipedia_by_symbol,
        jobs_by_symbol,
        filing_diff_by_symbol,
        earnings_tone_by_symbol,
        now=now,
    )
    expert_weights = load_score_weights()
    financial_history_records, financial_history_status = (
        fetch_financial_sector_history(rows)
        if not market_data_only
        else (
            {},
            {
                "status": "skipped",
                "reason": "market-data-only pipeline",
                "candidate_count": 0,
                "complete_count": 0,
            },
        )
    )
    valuation_history = (
        update_valuation_history(
            rows,
            observed_at=now,
            sec_by_symbol=sec_by_symbol,
            price_histories=fetched.prices,
        )
        if not market_data_only
        else {"symbols": {}, "status": {"skipped": True}}
    )
    attach_expert_analysis(
        rows,
        expert_weights,
        valuation_history=valuation_history,
        financial_history_records=financial_history_records,
        price_histories=fetched.prices,
    )
    expert_rankings = build_expert_rankings(rows, top_n=TOP_N)
    if not market_data_only:
        evaluate_mature_observations(fetched.prices)
        record_top_observations(rows, expert_rankings, now.isoformat())
        record_confluence_observations(rows, now.isoformat())
    recommendation_journal = journal_summary()
    rankings_by_currency_asset = rehydrate_rankings(
        rankings_by_currency_asset,
        rows,
    )

    benchmarks, benchmark_failures = (
        _fetch_benchmarks(now)
        if not market_data_only
        else ({}, {})
    )
    price_action_rows = [
        row
        for row in fresh_rows
        if row.get("currency") == "USD"
        and row.get("corporate_actions") is not None
    ]
    paper = (
        update_portfolio(
            price_action_rows,
            benchmarks=benchmarks,
            action_data_allowed=bool(price_action_rows),
            allow_orders=data_status["data_actionable"],
            observed_at=now,
        )
        if not market_data_only
        else {
            "simulation_status": "skipped_market_data_contract",
            "performance_actionable": False,
        }
    )
    skipped_layers = (
        [
            "fundamentals",
            "deep_fundamentals",
            "earnings",
            "news",
            "macro",
            "benchmarks",
            "paper_simulation",
        ]
        if market_data_only
        else []
    )
    market_contract_blockers = [
        reason
        for reason in data_status["blocking_reasons"]
        if "fundamental descriptive coverage" not in reason
    ]
    model_status = {
        "validation": "unvalidated",
        "actionable": False,
        "ranking_inputs": [
            "completed-daily-bar technical score within currency and asset class",
        ],
        "excluded_from_ranking": [
            "scenario ranges",
            "news",
            "analyst targets",
            "social data",
            "deep fundamentals",
            "expert holdings",
            "macro context",
            "generic company fundamental bands",
            "calibrated probability forecasts",
            "expert long-term and short-term composites",
            "alternative-data confluence",
        ],
        "scenario_status": "unvalidated heuristic range; not a probability or expected return",
        "validation_gate": {
            "status": "blocked",
            "required_before_any_actionability": {
                "point_in_time_composite_coverage_pct": 95,
                "out_of_sample_avg_spearman_ic_min": 0.03,
                "net_top_minus_bottom_positive_horizons_min": 2,
                "net_hit_rate_pct_min": 55,
                "independent_holdout_required": True,
            },
            "reason": (
                "historical point-in-time fundamentals/news/analyst inputs are unavailable; "
                "the deployed composite cannot currently satisfy this gate"
            ),
        },
        "legacy_keys": {
            "top_daytrade": "deprecated and empty",
            "daytrade_*": "deprecated; daily-only replacement is daily_signal_*",
            "top_longterm": (
                "deprecated and empty; use rankings_by_currency_asset"
            ),
            "rankings_by_asset": (
                "deprecated and empty; cross-currency global ordering is disabled"
            ),
        },
        "fundamental_status": (
            "descriptive only; excluded from overall cross-sectional ranking because "
            "robust sector-neutral point-in-time peer ranks are not implemented"
        ),
        "cross_currency_status": (
            "disabled: no point-in-time historical FX; no shared global ordering"
        ),
        "skipped_layers": skipped_layers,
    }
    result = {
        "schema": OUTPUT_SCHEMA,
        "schema_version": OUTPUT_SCHEMA_VERSION,
        "generated_at": utc_now(),
        "run_mode": "dry_run" if dry_run else "production",
        "pipeline_scope": (
            "market_data_contract" if market_data_only else "full_pipeline"
        ),
        "market_data_contract": {
            "status": "blocked" if market_contract_blockers else "ok",
            "blocking_reasons": market_contract_blockers,
            "skipped_layers": skipped_layers,
            "full_model_ready": False,
        },
        "data_status": data_status,
        "model_status": model_status,
        "insight_rankings": insight_rankings,
        "insight_metadata": {
            "contract_version": INSIGHT_CONTRACT_VERSION,
            "model_status": INSIGHT_STATUS,
            "actionable": False,
            "core_ranking_unchanged": True,
            "scenario_ranges_used_in_core_ranking": False,
            "provenance_catalog": PROVENANCE_CATALOG,
            "sweet_spot_contract": SWEET_SPOT_CONTRACT,
        },
        "expert_layer": {
            "model_status": "heuristic_unvalidated",
            "actionable": False,
            "core_ranking_unchanged": True,
            "source_catalog": EXPERT_SOURCE_CATALOG,
            "rankings": expert_rankings,
            "recommendation_journal": recommendation_journal,
        },
        "probability_validation": probability_validation,
        "probability_baselines": probability_baselines,
        "universe_size": len(symbols),
        "configured_universe_size": len(full_universe),
        "expected_asset_counts": expected_asset_counts,
        "configured_asset_counts_full": configured_asset_counts_full,
        "analyzed": len(rows),
        "market_news": market_news,
        "news_source_status": news_status,
        "sec_companyfacts_status": sec_source_status,
        "financial_sector_history_status": financial_history_status,
        "sec_insider_status": insider_source_status,
        "options_gex_status": gex_source_status,
        "finra_short_interest_status": finra_source_status,
        "sec_13f_status": sec_13f_status,
        "congress_trades_status": congress_source_status,
        "wikipedia_attention_status": wikipedia_source_status,
        "job_postings_status": jobs_source_status,
        "filing_diff_status": filing_diff_status,
        "earnings_tone_status": earnings_tone_status,
        "fred_status": fred_source_status,
        "market_positioning_status": positioning_source_status,
        "fx_status": fx_result.status,
        "macro": macro,
        "benchmarks": benchmarks,
        "benchmark_failures": benchmark_failures,
        "paper": paper,
        "rankings_by_currency_asset": rankings_by_currency_asset,
        "rankings_by_asset": {},
        "top_longterm": [],
        "top_fundamental": top_fundamental,
        "top_daytrade": [],
        "top_hype": [],
        "aschenbrenner_holdings": [
            row for row in rows if row.get("aschenbrenner")
        ],
        "all": rows,
        "_meta": schema_meta(
            "stock-radar-output",
            schema_version=OUTPUT_SCHEMA_VERSION,
            insight_contract=INSIGHT_CONTRACT_VERSION,
        ),
    }
    preliminary_question_views = build_question_views(
        rows,
        previous_snapshot=previous_snapshot,
    )
    deal_history = update_opportunity_history(
        preliminary_question_views,
        observed_at=now,
    )
    result["question_views"] = build_question_views(
        rows,
        previous_snapshot=previous_snapshot,
        historical_deal_scores=deal_history,
    )
    result["today"] = build_today_view(
        rows,
        previous_snapshot=previous_snapshot,
        price_histories=fetched.prices,
        question_views=result["question_views"],
    )
    result["today"]["market_summary"] = (
        result["question_views"].get("market_state") or {}
    ).get("sentence")
    result["today"]["triggered_today"] = result["question_views"].get(
        "triggered_today"
    ) or []
    result["today"]["near_triggers"] = result["question_views"].get(
        "near_triggers"
    ) or []
    rows_by_symbol = {row.get("symbol"): row for row in rows}
    for candidate in result["today"].get("candidates") or []:
        row = rows_by_symbol.get(candidate.get("symbol"))
        if row:
            candidate.update(
                decision_overlay(
                    row,
                    historical_deal_scores=deal_history,
                )
            )
    valuation_anomalies = [
        {
            "symbol": row.get("symbol"),
            "name": row.get("display_name_full") or row.get("name"),
            "price_local": row.get("price_local"),
            "currency": row.get("currency"),
            "gate": ((row.get("expert_analysis") or {}).get("valuation") or {}).get(
                "plausibility_gate"
            ),
            "raw_fair_value_range": (
                (row.get("expert_analysis") or {}).get("valuation") or {}
            ).get("raw_fair_value_range"),
            "metrics": ((row.get("expert_analysis") or {}).get("valuation") or {}).get(
                "metrics"
            ),
        }
        for row in rows
        if (
            (((row.get("expert_analysis") or {}).get("valuation") or {}).get(
                "plausibility_gate"
            ) or {}).get("status")
            != "pass"
        )
    ]
    result["valuation_anomaly_status"] = {
        "status": "warning" if valuation_anomalies else "ok",
        "count": len(valuation_anomalies),
        "threshold_pct": 80,
    }
    atomic_write_json(
        FAILED_MANIFEST,
        {
            "schema": "stock-radar-failed-symbols",
            "schema_version": SCHEMA_VERSION,
            "generated_at": result["generated_at"],
            "failed_symbols": failed_symbols,
        },
    )
    # Preserve the full expanded contract below GitHub's single-file limit.
    atomic_write_json(OUTPUT / "latest.json", result, indent=None)
    atomic_write_json(
        VALUATION_ANOMALIES,
        {
            "schema": "stock-radar-valuation-anomalies",
            "schema_version": 1,
            "generated_at": result["generated_at"],
            "anomalies": valuation_anomalies,
        },
        indent=1,
    )
    print(
        f"Saved: {len(rows)}/{len(symbols)} ({data_status['coverage_pct']:.2f}%) | "
        f"data gate {data_status['status']} | model UNVALIDATED | "
        f"{effective_path(OUTPUT / 'latest.json', for_write=True)}"
    )
    return result


if __name__ == "__main__":
    run()
