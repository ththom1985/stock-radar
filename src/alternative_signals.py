"""Recency-aware confluence across independent free alternative-data signals."""
from __future__ import annotations

import math
from datetime import datetime, timezone

SIGNAL_SPECS = {
    "insider": {"half_life_days": 30, "source_group": "sec_insiders"},
    "congress": {"half_life_days": 60, "source_group": "congress"},
    "institutional": {"half_life_days": 120, "source_group": "sec_13f"},
    "short_interest": {"half_life_days": 30, "source_group": "finra"},
    "wikipedia": {"half_life_days": 14, "source_group": "wikimedia"},
    "jobs": {"half_life_days": 30, "source_group": "company_careers"},
    "earnings_tone": {"half_life_days": 120, "source_group": "filings_calls"},
    "filing_diff": {"half_life_days": 180, "source_group": "sec_filings"},
    "gex": {"half_life_days": 3, "source_group": "options_chain"},
}


def _number(value):
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        return None
    value = float(value)
    return value if math.isfinite(value) else None


def _age_days(timestamp, now):
    try:
        parsed = datetime.fromisoformat(str(timestamp).replace("Z", "+00:00"))
        if parsed.tzinfo is None:
            parsed = parsed.replace(tzinfo=timezone.utc)
    except (TypeError, ValueError):
        return None
    return max(0.0, (now - parsed.astimezone(timezone.utc)).total_seconds() / 86400)


def build_alternative_signals(
    row,
    *,
    insider=None,
    gex=None,
    short_interest=None,
    institutional=None,
    congress=None,
    wikipedia=None,
    jobs=None,
    filing_diff=None,
    earnings_tone=None,
    now=None,
):
    now = now or datetime.now(timezone.utc)
    raw = {}
    if isinstance(insider, dict) and _number(insider.get("score")) is not None:
        raw["insider"] = {
            "score": insider["score"],
            "observed_at": insider.get("last_success_at") or insider.get("fetched_at"),
            "source": insider.get("source"),
            "expected_delay": insider.get("expected_delay"),
            "evidence": {
                "cluster_purchase": insider.get("cluster_purchase"),
                "cluster_owner_count": insider.get("cluster_owner_count"),
                "purchase_count_90d": insider.get("purchase_count_90d"),
                "sale_count_90d": insider.get("sale_count_90d"),
            },
        }
    if isinstance(gex, dict) and _number(gex.get("score")) is not None:
        raw["gex"] = {
            "score": gex["score"],
            "observed_at": gex.get("last_success_at") or gex.get("fetched_at"),
            "source": gex.get("source"),
            "expected_delay": gex.get("expected_delay"),
            "evidence": {
                "direction": gex.get("direction"),
                "net_gex_usd_per_1pct": gex.get("net_gex_usd_per_1pct"),
                "gex_to_market_cap": gex.get("gex_to_market_cap"),
                "gamma_walls": gex.get("gamma_walls"),
                "limitations": gex.get("limitations"),
            },
        }
    if (
        isinstance(short_interest, dict)
        and _number(short_interest.get("score")) is not None
    ):
        raw["short_interest"] = {
            "score": short_interest["score"],
            "observed_at": short_interest.get("settlement_date"),
            "source": short_interest.get("source"),
            "expected_delay": short_interest.get("expected_delay"),
            "evidence": {
                "settlement_date": short_interest.get("settlement_date"),
                "fetched_at": short_interest.get("fetched_at"),
                "short_position": short_interest.get("short_position"),
                "days_to_cover": short_interest.get("days_to_cover"),
                "change_percent": short_interest.get("change_percent"),
                "period_trend_pct": short_interest.get("period_trend_pct"),
                "limitations": short_interest.get("limitations"),
            },
        }
    if isinstance(institutional, dict) and _number(institutional.get("score")) is not None:
        raw["institutional"] = {
            "score": institutional["score"],
            "observed_at": institutional.get("report_period"),
            "source": institutional.get("source"),
            "expected_delay": institutional.get("expected_delay"),
            "evidence": {
                "report_period": institutional.get("report_period"),
                "manager_count": institutional.get("manager_count"),
                "changes": institutional.get("changes"),
                "fetched_at": institutional.get("fetched_at"),
                "limitations": institutional.get("limitations"),
            },
        }
    if isinstance(congress, dict) and _number(congress.get("score")) is not None:
        raw["congress"] = {
            "score": congress["score"],
            "observed_at": congress.get("latest_transaction_date"),
            "source": congress.get("source"),
            "expected_delay": congress.get("expected_delay"),
            "evidence": {
                "trade_count": congress.get("trade_count"),
                "purchase_count": congress.get("purchase_count"),
                "sale_count": congress.get("sale_count"),
                "trades": congress.get("trades"),
                "fetched_at": congress.get("fetched_at"),
                "limitations": congress.get("limitations"),
            },
        }
    if isinstance(wikipedia, dict) and _number(wikipedia.get("score")) is not None:
        raw["wikipedia"] = {
            "score": wikipedia["score"],
            "observed_at": wikipedia.get("last_success_at"),
            "source": wikipedia.get("source"),
            "expected_delay": wikipedia.get("expected_delay"),
            "evidence": {
                "article": wikipedia.get("article"),
                "trend_change_pct": wikipedia.get("trend_change_pct"),
                "recent_7d_average": wikipedia.get("recent_7d_average"),
                "prior_28d_average": wikipedia.get("prior_28d_average"),
                "source_url": wikipedia.get("source_url"),
            },
        }
    if isinstance(jobs, dict) and _number(jobs.get("score")) is not None:
        raw["jobs"] = {
            "score": jobs["score"],
            "observed_at": jobs.get("last_success_at"),
            "source": jobs.get("source"),
            "expected_delay": jobs.get("expected_delay"),
            "evidence": {
                "open_jobs": jobs.get("open_jobs"),
                "baseline_jobs": jobs.get("baseline_jobs"),
                "baseline_date": jobs.get("baseline_date"),
                "change_count": jobs.get("change_count"),
                "change_pct": jobs.get("change_pct"),
                "source_url": jobs.get("source_url"),
            },
        }
    if isinstance(filing_diff, dict) and _number(filing_diff.get("score")) is not None:
        raw["filing_diff"] = {"score": filing_diff["score"], "observed_at": filing_diff.get("current_filing_date"), "source": filing_diff.get("source"), "expected_delay": filing_diff.get("expected_delay"), "evidence": {"current_form": filing_diff.get("current_form"), "current_filing_date": filing_diff.get("current_filing_date"), "previous_form": filing_diff.get("previous_form"), "new_paragraph_count": filing_diff.get("new_paragraph_count"), "intensified_count": filing_diff.get("intensified_count"), "new_risk_excerpts": filing_diff.get("new_risk_excerpts"), "source_url": filing_diff.get("source_url")}}
    if isinstance(earnings_tone, dict) and _number(earnings_tone.get("score")) is not None:
        raw["earnings_tone"]={"score":earnings_tone["score"],"observed_at":(earnings_tone.get("current_period") or {}).get("conference_date"),"source":earnings_tone.get("source"),"expected_delay":earnings_tone.get("expected_delay"),"evidence":{k:earnings_tone.get(k) for k in ["current_tone","previous_tone","tone_shift","hedging_shift","qa_evasiveness_shift","cfo_tone_shift","reason","cfo_weight"]}}
    for name in SIGNAL_SPECS:
        supplied = (row.get("raw_alternative_signals") or {}).get(name)
        if name not in raw and isinstance(supplied, dict):
            raw[name] = supplied

    weighted = []
    rendered = {}
    source_groups = set()
    for name, signal in raw.items():
        score = _number(signal.get("score"))
        if score is None:
            continue
        spec = SIGNAL_SPECS[name]
        age = _age_days(signal.get("observed_at"), now)
        recency = (
            0.5 ** (age / spec["half_life_days"])
            if age is not None
            else 0.25
        )
        centered = (max(0.0, min(100.0, score)) - 50.0) / 50.0
        weighted.append((centered, recency))
        source_groups.add(spec["source_group"])
        rendered[name] = {
            **signal,
            "age_days": round(age, 2) if age is not None else None,
            "recency_weight": round(recency, 4),
            "source_group": spec["source_group"],
        }
    activation_groups = {
        "insider": "insider" in rendered,
        "congress": "congress" in rendered,
        "attention": any(name in rendered for name in ("wikipedia", "jobs")),
        "earnings_tone": "earnings_tone" in rendered,
    }
    activated = all(activation_groups.values())
    if weighted and activated:
        denominator = sum(weight for _, weight in weighted)
        centered = sum(value * weight for value, weight in weighted) / denominator
        independence_bonus = min(10.0, max(0, len(source_groups) - 1) * 2.5)
        confluence = 50.0 + centered * 40.0
        if centered > 0:
            confluence += independence_bonus
        elif centered < 0:
            confluence -= independence_bonus
        confluence = round(max(0.0, min(100.0, confluence)), 1)
    else:
        confluence = None
    return {
        "model_status": "heuristic_unvalidated",
        "actionable": False,
        "confluence_score": confluence,
        "activation_status": "active" if activated else "building",
        "activation_requirements": activation_groups,
        "missing_activation_groups": [
            name for name, available in activation_groups.items() if not available
        ],
        "independent_source_count": len(source_groups),
        "coverage_count": len(rendered),
        "missing_signals": [name for name in SIGNAL_SPECS if name not in rendered],
        "signals": rendered,
    }


def attach_alternative_signals(
    rows,
    insider_by_symbol=None,
    gex_by_symbol=None,
    short_interest_by_symbol=None,
    institutional_by_symbol=None,
    congress_by_symbol=None,
    wikipedia_by_symbol=None,
    jobs_by_symbol=None,
    filing_diff_by_symbol=None,
    earnings_tone_by_symbol=None,
    now=None,
):
    insider_by_symbol = insider_by_symbol or {}
    gex_by_symbol = gex_by_symbol or {}
    short_interest_by_symbol = short_interest_by_symbol or {}
    institutional_by_symbol = institutional_by_symbol or {}
    congress_by_symbol = congress_by_symbol or {}
    wikipedia_by_symbol = wikipedia_by_symbol or {}
    jobs_by_symbol = jobs_by_symbol or {}
    filing_diff_by_symbol = filing_diff_by_symbol or {}
    earnings_tone_by_symbol = earnings_tone_by_symbol or {}
    for row in rows:
        row["alternative_signals"] = build_alternative_signals(
            row,
            insider=insider_by_symbol.get(row.get("symbol")),
            gex=gex_by_symbol.get(row.get("symbol")),
            short_interest=short_interest_by_symbol.get(row.get("symbol")),
            institutional=institutional_by_symbol.get(row.get("symbol")),
            congress=congress_by_symbol.get(row.get("symbol")),
            wikipedia=wikipedia_by_symbol.get(row.get("symbol")),
            jobs=jobs_by_symbol.get(row.get("symbol")),
            filing_diff=filing_diff_by_symbol.get(row.get("symbol")),
            earnings_tone=earnings_tone_by_symbol.get(row.get("symbol")),
            now=now,
        )
    return rows
