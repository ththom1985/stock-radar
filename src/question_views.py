"""Deterministic question-first rankings derived from the existing analysis."""
from __future__ import annotations

import math
from collections import Counter

from .today_view import STATUS_LABELS, local_zone, traffic_light

MIN_DISCOUNT_PCT = 15.0
MIN_VALUE_POTENTIAL_SCORE = 65.0
VALUATION_STATUS = "evidence_qualified_unbacktested"
VALUATION_STATUS_LABEL = "Bewertung belastbar; Modell noch nicht rückgeprüft"
VALUATION_LISTS_ENABLED = True
VALUATION_REPAIR_MESSAGE = (
    "Bewertungsmodell wird überarbeitet — Aussagen derzeit nicht belastbar."
)


def _number(value):
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        return None
    value = float(value)
    return value if math.isfinite(value) else None


def _german_number(value):
    return f"{value:,.2f}".replace(",", "X").replace(".", ",").replace("X", ".")


def _german_date(value):
    try:
        year, month, day = str(value).split("-")
        return f"{day}.{month}.{year}"
    except ValueError:
        return str(value or "heute")


def _active(value):
    return bool(value.get("active", True)) if isinstance(value, dict) else bool(value)


def _basis(valuation):
    return {
        "broad": "breit",
        "narrow": "schmal",
    }.get((valuation.get("basis_quality") or {}).get("status"), "schmal")


def _badge(row):
    status = (row.get("sweet_spot") or {}).get("combined_status")
    light = traffic_light(row)
    if light == "green":
        return {"code": "buy_zone", "label": "Kaufzone ✓", "tone": "green"}
    if light == "yellow":
        return {
            "code": "watch",
            "label": STATUS_LABELS.get(status, "Beobachten"),
            "tone": "yellow",
        }
    if status in {"safety_blocked", "broken_below"} or _active(row.get("falling_knife")):
        return {"code": "avoid", "label": "Meiden", "tone": "muted"}
    return {"code": "no_setup", "label": "Kein Setup", "tone": "neutral"}


def _potential_driver(row, analysis):
    revenue_growth = _number(row.get("revenue_growth_pct"))
    earnings_growth = _number(row.get("earnings_growth"))
    roe = _number(row.get("roe_pct"))
    if earnings_growth is not None:
        earnings_growth *= 100.0
    if revenue_growth is not None and revenue_growth >= 5:
        return f"Umsatzwachstum von {revenue_growth:+.1f}%"
    if earnings_growth is not None and earnings_growth >= 5:
        return f"Ergebniswachstum von {earnings_growth:+.1f}%"
    if roe is not None and roe >= 15:
        return f"eine Eigenkapitalrendite von {roe:.1f}%"
    groups = (row.get("alternative_signals") or {}).get("contributing_groups") or []
    if groups:
        return f"{len(groups)} bestätigende unabhängige Signalgruppen"
    return None


def _band(value, bands):
    for threshold, score in bands:
        if value <= threshold:
            return float(score)
    return float(bands[-1][1])


def _annual_fundamentals(row):
    annual = (row.get("sec_companyfacts") or {}).get("annual") or []
    return [point for point in annual if isinstance(point, dict)]


def _margin_trend_score(annual):
    margins = [
        point["net_income"] / point["revenue"]
        for point in annual
        if _number(point.get("net_income")) is not None
        and (_number(point.get("revenue")) or 0) > 0
    ]
    if len(margins) < 4:
        return None
    change_pct_points = (
        sum(margins[-2:]) / 2.0 - sum(margins[:2]) / 2.0
    ) * 100.0
    return _band(
        change_pct_points,
        [(-5, 15), (-2, 35), (0, 55), (2, 75), (5, 90), (math.inf, 100)],
    )


def _cagr_score(annual, field):
    values = [
        point[field]
        for point in annual
        if (_number(point.get(field)) or 0) > 0
    ]
    if len(values) < 4:
        return None
    cagr_pct = ((values[-1] / values[0]) ** (1.0 / (len(values) - 1)) - 1.0) * 100.0
    return _band(
        cagr_pct,
        [(0, 20), (5, 50), (10, 70), (20, 90), (math.inf, 100)],
    )


def _value_potential(row):
    quality = _number(row.get("quality_score"))
    growth = _number(row.get("growth_score"))
    if quality is None or growth is None:
        return None
    annual = _annual_fundamentals(row)
    quality_components = [("Aktuelle Qualität und Bilanz", quality)]
    growth_components = [("Aktueller Umsatz- und Gewinntrend", growth)]
    margin_trend = _margin_trend_score(annual)
    if margin_trend is not None:
        quality_components.append(("Mehrjähriger Nettomargentrend", margin_trend))
    fcf_conversion = _number(
        ((row.get("sec_companyfacts") or {}).get("derived") or {}).get("fcf_conversion")
    )
    if fcf_conversion is not None:
        quality_components.append(
            (
                "Free-Cashflow-Konversion",
                _band(
                    fcf_conversion,
                    [(0, 10), (0.5, 40), (0.8, 65), (1, 85), (1.5, 95), (math.inf, 100)],
                ),
            )
        )
    for field, label in (
        ("revenue", "Mehrjähriger Umsatztrend"),
        ("net_income", "Mehrjähriger Gewinntrend"),
    ):
        trend = _cagr_score(annual, field)
        if trend is not None:
            growth_components.append((label, trend))
    quality_potential = sum(score for _, score in quality_components) / len(
        quality_components
    )
    growth_potential = sum(score for _, score in growth_components) / len(
        growth_components
    )
    return {
        "score": 0.6 * quality_potential + 0.4 * growth_potential,
        "quality_score": quality_potential,
        "growth_score": growth_potential,
        "components": [label for label, _ in quality_components + growth_components],
        "missing_components": [
            "Ein einheitlicher ROIC ist aus den verfügbaren XBRL-Taxonomien nicht "
            "belastbar standardisiert.",
            "Historische Analystenrevisionen sind in den kostenlosen Quelldaten "
            "derzeit nicht verfügbar.",
        ],
    }


def _price_support(row):
    signals = ((row.get("alternative_signals") or {}).get("signals") or {})
    attention = signals.get("attention") or {}
    if _number(attention.get("score")) is not None and attention["score"] > 55:
        return "steigende Aufmerksamkeit"
    if (_number(row.get("ret_20d")) or 0) > 5 or (_number(row.get("longterm_score")) or 0) >= 70:
        return "positives Kursmomentum"
    if (_number(row.get("growth_score")) or 0) >= 70:
        return "hohe Wachstumserwartungen"
    return None


def _material_risk(analysis):
    risks = (analysis.get("risks") or {}).get("top_risks") or []
    for risk in risks:
        if not str(risk).startswith("Das Modell erkennt aktuell kein"):
            return str(risk)
    return None


def _value_trap_risk(row):
    return (
        (row.get("valuation_context") or {}).get("value_trap_risk")
        or (row.get("valuation_thesis") or {}).get("value_trap_risk")
    )


def _fundamental_risk_reasons(row):
    reasons = []
    trap = _value_trap_risk(row)
    if trap == "high":
        reasons.append("Das Value-Trap-Risiko ist hoch.")
    warning_tokens = (
        "altman",
        "bilanz",
        "verschuld",
        "debt/equity",
        "negative eigenkapital",
        "unternehmenszahlen in",
    )
    for warning in row.get("risk_warnings") or []:
        text = str(warning)
        if any(token in text.casefold() for token in warning_tokens):
            reasons.append(text)
    return list(dict.fromkeys(reasons))


def _cheap_action(row, previous_row=None):
    falling = _active(row.get("falling_knife"))
    bottoming = _active(row.get("bottoming"))
    zone = local_zone(row)
    target = _number((zone or {}).get("upper"))
    price = _number(row.get("price_local"))
    previous_price = _number((previous_row or {}).get("price_local"))
    price_triggered = (
        target is not None
        and price is not None
        and price <= target
    )
    triggered_today = (
        price_triggered
        and previous_price is not None
        and previous_price > target
    )
    if falling or bottoming:
        return {
            "code": "watch_falling",
            "label": "Beobachten, Kurs fällt noch",
            "sentence": "Beobachten, Kurs fällt noch.",
            "badge": {
                "code": "watch_falling",
                "label": "⚠️ Kurs fällt noch — beobachten, nicht greifen",
                "tone": "yellow",
            },
            "target_price": round(target, 4) if target is not None else None,
            "price_triggered": price_triggered,
            "triggered_today": triggered_today,
            "currency": row.get("currency"),
        }
    if traffic_light(row) == "green":
        return {
            "code": "entry_now",
            "label": "Einstieg jetzt möglich",
            "sentence": "Einstieg jetzt möglich.",
            "badge": {"code": "buy_zone", "label": "Kaufzone ✓", "tone": "green"},
            "target_price": None,
            "price_triggered": True,
            "triggered_today": triggered_today,
            "currency": row.get("currency"),
        }
    if target is not None and price_triggered:
        sentence = "Preiszone erreicht; für den Idealfall fehlt die technische Bestätigung."
    elif target is not None:
        sentence = (
            f"Beobachten, Einstieg bei etwa {_german_number(target)} "
            f"{row.get('currency') or ''}."
        )
    else:
        sentence = "Beobachten; eine belastbare Einstiegsmarke fehlt derzeit."
    return {
        "code": "watch_entry",
        "label": "Beobachten, Einstieg bei X",
        "sentence": sentence,
        "badge": {"code": "watch", "label": "Beobachten", "tone": "yellow"},
        "target_price": round(target, 4) if target is not None else None,
        "price_triggered": price_triggered,
        "triggered_today": triggered_today,
        "currency": row.get("currency"),
    }


def _situation(row, action):
    timing = _number(row.get("entry_timing_score")) or 0
    if timing >= 55 and action["code"] == "entry_now":
        return {
            "code": "ideal",
            "label": "Idealfall: günstig UND am Einstiegspunkt",
        }
    if action["code"] == "watch_falling":
        return {
            "code": "cheap_falling",
            "label": (
                "Günstiges Unternehmen, Kurs fällt noch — "
                "auf Stabilisierung warten"
            ),
        }
    return {
        "code": "cheap_wait",
        "label": (
            "Günstiges Unternehmen, technisches Einstiegssignal fehlt — "
            "auf Rücksetzer oder Bestätigung warten"
        ),
    }


def _deal_quality(row, discount, historical_scores):
    timing = _number(row.get("entry_timing_score")) or 0
    groups = (row.get("alternative_signals") or {}).get("contributing_groups") or []
    signal_score = min(100.0, len(groups) / 4.0 * 100.0)
    trap = _value_trap_risk(row)
    risk_score = 100.0 if trap == "low" else 60.0 if trap == "medium" else 20.0
    score = (
        0.40 * min(100.0, discount / 50.0 * 100.0)
        + 0.25 * timing
        + 0.20 * signal_score
        + 0.15 * risk_score
    )
    stars = max(1, min(5, math.ceil(score / 20.0)))
    history = (
        historical_scores
        if isinstance(historical_scores, dict)
        else {"scores": historical_scores or []}
    )
    reference = [
        value
        for value in history.get("scores") or []
        if _number(value) is not None
    ]
    percentile = None
    comparison = (
        "Noch keine Vergleichshistorie; Einordnung nach festen Schwellen."
    )
    if reference:
        percentile = sum(value <= score for value in reference) / len(reference) * 100.0
        relative = (
            f"Besser als {percentile:.0f}% der gespeicherten Gelegenheiten."
            if percentile >= 50
            else f"Schwächer als {100.0 - percentile:.0f}% der gespeicherten Gelegenheiten."
        )
        comparison = (
            relative
            if history.get("reliable")
            else f"Vorläufig: {relative}"
        )
    observation_count = len(reference)
    calendar_days = int(history.get("calendar_days") or 0)
    requirement = history.get("reliability_requirement") or (
        "mindestens 100 Gelegenheiten über mindestens 30 Kalendertage"
    )
    return {
        "score": round(score, 1),
        "stars": stars,
        "label": f"Deal-Qualität {stars} von 5",
        "historical_percentile": (
            round(percentile, 1) if percentile is not None else None
        ),
        "comparison": comparison,
        "comparison_basis": (
            f"Vergleich über {observation_count} Gelegenheiten seit "
            f"{_german_date(history.get('from_date'))} "
            f"({calendar_days or 1} Snapshot-Tag). Aufbauend; belastbar ab {requirement}"
            if reference and not history.get("reliable")
            else (
                f"Vergleich über {observation_count} Gelegenheiten seit "
                f"{_german_date(history.get('from_date'))} "
                f"({calendar_days} Kalendertage)"
                if reference
                else "feste Schwellen; Historie baut sich ab jetzt auf"
            )
        ),
        "history_reliable": bool(history.get("reliable")),
        "history_observation_count": observation_count,
        "history_calendar_days": calendar_days,
        "history_requirement": requirement,
        "components": {
            "valuation_discount": round(min(100.0, discount / 50.0 * 100.0), 1),
            "timing": round(timing, 1),
            "signals": round(signal_score, 1),
            "risk": round(risk_score, 1),
        },
    }


def decision_overlay(row, historical_deal_scores=None):
    valuation = ((row.get("expert_analysis") or {}).get("valuation") or {})
    fair = valuation.get("fair_value_range") or {}
    price = _number(row.get("price_local"))
    lower = _number(fair.get("lower"))
    timing = _number(row.get("entry_timing_score")) or 0
    verdict = valuation.get("verdict")
    discount = (
        max(0.0, (lower / price - 1.0) * 100.0)
        if price is not None and price > 0 and lower is not None
        else 0.0
    )
    action = _cheap_action(row)
    if verdict == "clearly_undervalued":
        situation = _situation(row, action)
    elif verdict in {"expensive", "overpriced"} and timing >= 55:
        situation = {
            "code": "momentum_only",
            "label": "Nur Momentum, fundamental nicht günstig",
        }
    elif verdict in {"expensive", "overpriced"}:
        situation = {
            "code": "avoid",
            "label": "Fundamental teuer und technisch kein Einstieg",
        }
    elif timing >= 55:
        situation = {
            "code": "fair_entry",
            "label": "Fair bewertet und technisch am Einstiegspunkt",
        }
    else:
        situation = {
            "code": "wait",
            "label": "Bewertung akzeptabel, technisches Einstiegssignal fehlt",
        }
    return {
        "situation": situation,
        "deal_quality": _deal_quality(
            row,
            discount,
            historical_deal_scores,
        ),
    }


def build_question_views(
    rows,
    cheap_limit=None,
    expensive_limit=None,
    *,
    previous_snapshot=None,
    historical_deal_scores=None,
):
    if not VALUATION_LISTS_ENABLED:
        return {
            "valuation_status": "under_repair",
            "valuation_status_label": VALUATION_REPAIR_MESSAGE,
            "enabled": False,
            "cheap_with_potential": [],
            "excluded_cheap": [],
            "expensive_now": [],
            "selection_counts": {
                "gate_ready": 0,
                "materially_cheap": 0,
                "potential_pass": 0,
                "risk_excluded": 0,
                "eligible": 0,
                "visible": 0,
            },
            "sector_concentration": {
                "dominant_sector": None,
                "dominant_pct": 0.0,
                "warning": None,
            },
            "empty_state": VALUATION_REPAIR_MESSAGE,
            "rules": {
                "cheap": VALUATION_REPAIR_MESSAGE,
                "expensive": VALUATION_REPAIR_MESSAGE,
            },
        }
    cheap = []
    excluded_cheap = []
    expensive = []
    gate_ready_count = 0
    materially_cheap_count = 0
    potential_pass_count = 0
    risk_excluded_count = 0
    previous_by_symbol = {
        row.get("symbol"): row
        for row in (previous_snapshot or {}).get("all", [])
        if isinstance(row, dict)
    }
    fair_midpoint_gaps = []
    for row in rows:
        if row.get("asset_type") != "company_equity":
            continue
        analysis = row.get("expert_analysis") or {}
        valuation = analysis.get("valuation") or {}
        fair = valuation.get("fair_value_range") or {}
        price = _number(row.get("price_local"))
        lower, upper = _number(fair.get("lower")), _number(fair.get("upper"))
        gate_passed = (valuation.get("plausibility_gate") or {}).get("status") == "pass"
        if not gate_passed or not price or not lower or not upper:
            continue
        gate_ready_count += 1
        fair_midpoint = (lower + upper) / 2.0
        fair_midpoint_gaps.append((price / fair_midpoint - 1.0) * 100.0)

        common = {
            "symbol": row.get("symbol"),
            "name": row.get("display_name_full") or row.get("name") or row.get("symbol"),
            "currency": row.get("currency"),
            "price": round(price, 4),
            "fair_lower": round(lower, 4),
            "fair_upper": round(upper, 4),
            "basis": _basis(valuation),
            "basis_definition": (valuation.get("basis_quality") or {}).get("definition"),
            "geographic_coverage": (analysis.get("coverage_basis") or {}).get("status"),
            "geographic_note": (
                "Für diesen Titel fehlt ein vergleichbares kostenloses "
                "Fundamentaldatenarchiv; die Bewertung stützt sich auf weniger Quellen."
                if (analysis.get("coverage_basis") or {}).get("status") == "narrower_non_us"
                else None
            ),
            "valuation_status": VALUATION_STATUS,
            "valuation_status_label": VALUATION_STATUS_LABEL,
            "badge": _badge(row),
        }
        if valuation.get("verdict") == "clearly_undervalued" and price < lower:
            discount = (lower / price - 1) * 100
            potential = _value_potential(row)
            fundamental_risks = _fundamental_risk_reasons(row)
            if discount >= MIN_DISCOUNT_PCT and common["basis"] == "breit":
                materially_cheap_count += 1
            if (
                discount >= MIN_DISCOUNT_PCT
                and potential is not None
                and potential["score"] >= MIN_VALUE_POTENTIAL_SCORE
                and common["basis"] == "breit"
            ):
                potential_pass_count += 1
                if fundamental_risks:
                    risk_excluded_count += 1
                    excluded_cheap.append(
                        {
                            "symbol": row.get("symbol"),
                            "name": (
                                row.get("display_name_full")
                                or row.get("name")
                                or row.get("symbol")
                            ),
                            "discount_pct": round(discount, 1),
                            "potential_score": round(potential["score"], 1),
                            "value_trap_risk": _value_trap_risk(row),
                            "risk_penalty": _number(
                                (row.get("valuation_context") or {}).get("risk_penalty")
                            ),
                            "reasons": fundamental_risks,
                        }
                    )
                    continue
                quality = _number(row.get("quality_score")) or 50.0
                risk_floor = max(20.0, 100.0 - quality)
                rank_value = discount * potential["score"] / risk_floor
                attractiveness = (
                    0.65 * potential["score"]
                    + 0.35 * min(100.0, discount / 80.0 * 100.0)
                )
                driver = _potential_driver(row, analysis)
                residual_risk = _material_risk(analysis)
                action = _cheap_action(
                    row,
                    previous_by_symbol.get(row.get("symbol")),
                )
                situation = _situation(row, action)
                deal_quality = _deal_quality(
                    row,
                    discount,
                    historical_deal_scores,
                )
                trap_risk = _value_trap_risk(row)
                risk_note = (
                    "Erhöhtes Rückschlagrisiko — kleiner positionieren."
                    if trap_risk == "medium"
                    else None
                )
                sentences = [
                    f"Der Kurs liegt rund {discount:.0f}% unter der unteren plausiblen fairen Grenze."
                ]
                if driver:
                    sentences.append(f"Dafür spricht {driver}.")
                if residual_risk and residual_risk not in fundamental_risks:
                    sentences.append(residual_risk)
                if risk_note:
                    sentences.append(risk_note)
                sentences.append(action["sentence"])
                cheap.append(
                    {
                        **common,
                        "badge": action["badge"],
                        "discount_pct": round(discount, 1),
                        "attractiveness_score": round(min(100.0, attractiveness), 1),
                        "_rank_value": rank_value,
                        "potential_score": round(potential["score"], 1),
                        "potential_components": potential["components"],
                        "potential_missing_components": potential["missing_components"],
                        "quality_score": round(potential["quality_score"], 1),
                        "growth_score": round(potential["growth_score"], 1),
                        "potential_driver": driver,
                        "residual_risk": residual_risk,
                        "value_trap_risk": trap_risk,
                        "risk_note": risk_note,
                        "course_state": (
                            "falling"
                            if _active(row.get("falling_knife"))
                            else "bottoming"
                            if _active(row.get("bottoming"))
                            else "stable"
                        ),
                        "entry_guidance": {
                            key: value
                            for key, value in action.items()
                            if key != "badge"
                        },
                        "situation": situation,
                        "deal_quality": deal_quality,
                        "sentences": sentences,
                    }
                )
        if valuation.get("verdict") == "overpriced" and price > upper:
            premium = (price / upper - 1) * 100
            timing_score = _number(row.get("entry_timing_score")) or 0
            if (
                premium >= MIN_DISCOUNT_PCT
                and common["basis"] == "breit"
                and timing_score >= 55
            ):
                support = _price_support(row)
                support_clause = f"; getragen durch {support}" if support else ""
                expensive.append(
                    {
                        **common,
                        "badge": {"code": "no_setup", "label": "Kein Setup", "tone": "neutral"},
                        "premium_pct": round(premium, 1),
                        "price_support": support,
                        "reentry_price": round(upper, 4),
                        "sentence": (
                            f"Rund {premium:.0f}% über dem plausiblen fairen Bereich"
                            f"{support_clause}. Wieder fair unter etwa {_german_number(upper)} "
                            f"{row.get('currency') or ''}."
                        ),
                        "situation": {
                            "code": "momentum_only",
                            "label": "Nur Momentum, fundamental nicht günstig",
                        },
                    }
                )
    cheap.sort(key=lambda item: (-item["_rank_value"], -item["discount_pct"], item["symbol"]))
    excluded_cheap.sort(key=lambda item: (-item["discount_pct"], item["symbol"]))
    expensive.sort(key=lambda item: (-item["premium_pct"], item["symbol"]))
    for item in cheap:
        item.pop("_rank_value", None)
    waiting = []
    for item in cheap:
        guidance = item.get("entry_guidance") or {}
        if (item.get("situation") or {}).get("code") == "ideal":
            continue
        target = _number(guidance.get("target_price"))
        price = _number(item.get("price"))
        distance_pct = (
            (price / target - 1.0) * 100.0
            if target is not None and target > 0 and price is not None
            else None
        )
        waiting.append(
            {
                "symbol": item["symbol"],
                "name": item["name"],
                "currency": item["currency"],
                "price": item["price"],
                "trigger_price": target,
                "distance_pct": (
                    round(distance_pct, 1)
                    if distance_pct is not None
                    else None
                ),
                "price_triggered": bool(guidance.get("price_triggered")),
                "triggered_today": bool(guidance.get("triggered_today")),
                "course_state": item.get("course_state"),
                "situation": item.get("situation"),
                "action": guidance.get("sentence"),
            }
        )
    waiting.sort(
        key=lambda item: (
            not item["triggered_today"],
            abs(item["distance_pct"])
            if item["distance_pct"] is not None
            else math.inf,
            item["symbol"],
        )
    )
    near_triggers = [
        item
        for item in waiting
        if item["distance_pct"] is not None
        and abs(item["distance_pct"]) <= 0.15
        and not item["triggered_today"]
    ]
    sectors = Counter(
        str(
            next(
                (
                    row.get("sector_display") or row.get("sector") or "Unbekannt"
                    for row in rows
                    if row.get("symbol") == item["symbol"]
                ),
                "Unbekannt",
            )
        )
        for item in cheap
    )
    dominant_sector, dominant_count = sectors.most_common(1)[0] if sectors else (None, 0)
    dominant_pct = dominant_count / len(cheap) * 100.0 if cheap else 0.0
    concentration_warning = (
        f"Verdachtsfall: {dominant_pct:.0f}% der Liste stammen aus dem Sektor "
        f"{dominant_sector}."
        if dominant_pct > 50
        else None
    )
    median_gap = (
        sorted(fair_midpoint_gaps)[len(fair_midpoint_gaps) // 2]
        if fair_midpoint_gaps
        else None
    )
    top_deals = sum(
        (item.get("deal_quality") or {}).get("stars", 0) >= 4
        and (item.get("situation") or {}).get("code") == "ideal"
        for item in cheap
    )
    if top_deals:
        market_sentence = f"Mehrere erstklassige Gelegenheiten heute: {top_deals} Idealfälle."
    elif median_gap is not None and median_gap > 0:
        market_sentence = (
            "Der Markt liegt derzeit im Median über den fairen Werten. "
            "Es gibt aktuell keine erstklassigen Gelegenheiten — "
            "die besten verfügbaren Kandidaten sind unten aufgeführt."
        )
    else:
        market_sentence = (
            "Der Markt liegt derzeit im Median nicht über den fairen Werten, "
            "aber aktuell fehlt ein erstklassiger Idealfall."
        )
    return {
        "enabled": True,
        "valuation_status": VALUATION_STATUS,
        "valuation_status_label": VALUATION_STATUS_LABEL,
        "cheap_with_potential": (
            cheap if cheap_limit is None else cheap[:cheap_limit]
        ),
        "excluded_cheap": excluded_cheap,
        "expensive_now": (
            expensive if expensive_limit is None else expensive[:expensive_limit]
        ),
        "waiting_for_entry": waiting,
        "triggered_today": [
            item for item in waiting if item["triggered_today"]
        ],
        "near_triggers": near_triggers,
        "market_state": {
            "median_vs_fair_midpoint_pct": (
                round(median_gap, 1) if median_gap is not None else None
            ),
            "top_deal_count": top_deals,
            "sentence": market_sentence,
        },
        "selection_counts": {
            "gate_ready": gate_ready_count,
            "materially_cheap": materially_cheap_count,
            "potential_pass": potential_pass_count,
            "risk_excluded": risk_excluded_count,
            "eligible": len(cheap),
            "visible": (
                len(cheap)
                if cheap_limit is None
                else min(len(cheap), cheap_limit)
            ),
        },
        "sector_concentration": {
            "dominant_sector": dominant_sector,
            "dominant_pct": round(dominant_pct, 1),
            "warning": concentration_warning,
        },
        "empty_state": (
            f"{gate_ready_count} Bewertungen bestehen die Daten- und "
            "Plausibilitätsprüfung. Aktuell erfüllt davon kein Titel zugleich den "
            "Mindestabschlag, das momentumfreie Qualitäts-/Wachstumskriterium und "
            "die Risikoprüfung. Das Bewertungsmodell ist noch nicht rückgeprüft."
        ),
        "rules": {
            "cheap": "Breite Bewertungsbasis aus zwei unabhängigen Referenzfamilien; Bereich höchstens Faktor 1,5; Abweichung höchstens 50%; mindestens 15% unter fair; momentumfreies Potenzial mindestens 65 aus 60% Qualität und 40% Wachstum. Nur hohes Value-Trap-, Bilanz- oder akutes Ereignisrisiko schließt aus; medium wird als erhöhtes Rückschlagrisiko markiert. Messer-, Boden- und Timingstatus werden nur angezeigt.",
            "expensive": "Breite Bewertungsbasis aus zwei unabhängigen Referenzfamilien; Bereich höchstens Faktor 1,5; Abweichung höchstens 50%; mindestens 15% über fair.",
        },
    }
