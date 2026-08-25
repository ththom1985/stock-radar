"""Deterministic question-first rankings derived from the existing analysis."""
from __future__ import annotations

import math

from .today_view import STATUS_LABELS, traffic_light

MIN_DISCOUNT_PCT = 15.0
MIN_LONG_TERM_SCORE = 65.0
MIN_LONG_TERM_COVERAGE_PCT = 60.0


def _number(value):
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        return None
    value = float(value)
    return value if math.isfinite(value) else None


def _german_number(value):
    return f"{value:,.2f}".replace(",", "X").replace(".", ",").replace("X", ".")


def _active(value):
    return bool(value.get("active", True)) if isinstance(value, dict) else bool(value)


def _basis(analysis, valuation):
    if (analysis.get("coverage_basis") or {}).get("status") == "narrower_non_us":
        return "EU-renormalisiert"
    if valuation.get("own_5y_complete") and (valuation.get("fair_value_range") or {}).get("input_count", 0) >= 4:
        return "breit"
    return "schmal"


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
    candidates = [
        ("langfristige Systemqualität", _number((analysis.get("long_term") or {}).get("score"))),
        ("Wachstum", _number(row.get("growth_score"))),
        ("Qualität", _number(row.get("quality_score"))),
        ("positive unabhängige Signale", _number((row.get("alternative_signals") or {}).get("confluence_score"))),
    ]
    available = [(label, value) for label, value in candidates if value is not None]
    available.sort(key=lambda item: (-item[1], item[0]))
    return available[0][0] if available else "die verfügbaren Langfrist-Signale"


def _price_support(row):
    signals = ((row.get("alternative_signals") or {}).get("signals") or {})
    attention = signals.get("attention") or {}
    if _number(attention.get("score")) is not None and attention["score"] > 55:
        return "steigende Aufmerksamkeit"
    if (_number(row.get("ret_20d")) or 0) > 5 or (_number(row.get("longterm_score")) or 0) >= 70:
        return "positives Kursmomentum"
    if (_number(row.get("growth_score")) or 0) >= 70:
        return "hohe Wachstumserwartungen"
    return "die aktuell eingepreisten Erwartungen"


def build_question_views(rows, cheap_limit=5, expensive_limit=12):
    cheap = []
    expensive = []
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

        common = {
            "symbol": row.get("symbol"),
            "name": row.get("display_name_full") or row.get("name") or row.get("symbol"),
            "currency": row.get("currency"),
            "price": round(price, 4),
            "fair_lower": round(lower, 4),
            "fair_upper": round(upper, 4),
            "basis": _basis(analysis, valuation),
            "badge": _badge(row),
        }
        if valuation.get("verdict") == "clearly_undervalued" and price < lower:
            long_term = analysis.get("long_term") or {}
            long_score = _number(long_term.get("score"))
            coverage = _number(long_term.get("coverage_pct")) or 0
            discount = (lower / price - 1) * 100
            excluded_risk = (
                _active(row.get("falling_knife"))
                or _active(row.get("bottoming"))
                or bool(row.get("risk_warnings"))
                or analysis.get("signal") == "risk_too_high"
                or (row.get("sweet_spot") or {}).get("combined_status") in {"safety_blocked", "broken_below"}
            )
            if (
                discount >= MIN_DISCOUNT_PCT
                and long_score is not None
                and long_score >= MIN_LONG_TERM_SCORE
                and coverage >= MIN_LONG_TERM_COVERAGE_PCT
                and not excluded_risk
            ):
                quality = _number(row.get("quality_score")) or 50.0
                risk_floor = max(20.0, 100.0 - quality)
                rank_value = discount * long_score / risk_floor
                attractiveness = (
                    0.5 * long_score
                    + 0.3 * min(100.0, discount / 80.0 * 100.0)
                    + 0.2 * quality
                )
                driver = _potential_driver(row, analysis)
                cheap.append(
                    {
                        **common,
                        "discount_pct": round(discount, 1),
                        "attractiveness_score": round(min(100.0, attractiveness), 1),
                        "_rank_value": rank_value,
                        "long_term_score": round(long_score, 1),
                        "potential_driver": driver,
                        "residual_risk": (
                            (analysis.get("risks") or {}).get("top_risks") or
                            ["Kein einzelnes dominantes Risiko in den verfügbaren Daten."]
                        )[0],
                        "sentences": [
                            f"Der Kurs liegt rund {discount:.0f}% unter der unteren plausiblen fairen Grenze.",
                            f"Das Potenzial stützt sich vor allem auf {driver}.",
                            (
                                (analysis.get("risks") or {}).get("top_risks") or
                                ["Kein einzelnes dominantes Risiko in den verfügbaren Daten."]
                            )[0],
                        ],
                    }
                )
        if valuation.get("verdict") == "overpriced" and price > upper:
            premium = (price / upper - 1) * 100
            if premium >= MIN_DISCOUNT_PCT:
                support = _price_support(row)
                expensive.append(
                    {
                        **common,
                        "badge": {"code": "no_setup", "label": "Kein Setup", "tone": "neutral"},
                        "premium_pct": round(premium, 1),
                        "price_support": support,
                        "reentry_price": round(upper, 4),
                        "sentence": (
                            f"Rund {premium:.0f}% über dem plausiblen fairen Bereich; "
                            f"getragen durch {support}. Wieder fair unter etwa {_german_number(upper)} "
                            f"{row.get('currency') or ''}."
                        ),
                    }
                )
    cheap.sort(key=lambda item: (-item["_rank_value"], -item["discount_pct"], item["symbol"]))
    expensive.sort(key=lambda item: (-item["premium_pct"], item["symbol"]))
    for item in cheap:
        item.pop("_rank_value", None)
    return {
        "model_status": "heuristic_unvalidated",
        "actionable": False,
        "cheap_with_potential": cheap[:cheap_limit],
        "expensive_now": expensive[:expensive_limit],
        "rules": {
            "cheap": "Plausibilitäts-Gate bestanden; >=15% unter fairem Bereich; Long-Term >=65; keine Messer-, Boden- oder akute Risikowarnung.",
            "expensive": "Plausibilitäts-Gate bestanden; >=15% über fairem Bereich.",
        },
    }
