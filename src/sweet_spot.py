"""Pure completed-daily sweet-spot observation-zone heuristic.

The model describes technical reference geometry and explicit safety filters.  It
does not emit orders, recommendations, forecasts, or calibrated likelihoods.
"""
from __future__ import annotations

import math
from typing import Any


MODEL_STATUS = "heuristic_unvalidated"
MODEL_VERSION = 3
MIN_LONGTERM_SCORE = 60.0
MIN_ENTRY_TIMING_SCORE = 55.0
MAX_ATR_PCT = 5.0
MAX_ANNUAL_VOL_PCT = 60.0
MAX_BAR_AGE_DAYS = 4
MIN_RELIABILITY_SCORE = 65.0
CLUSTER_DISTANCE_ATR = 0.90
CANDIDATE_MIN_DISTANCE_ATR = -4.0
CANDIDATE_MAX_DISTANCE_ATR = 2.0
STRATEGIC_MIN_DISTANCE_ATR = -10.0
APPROACHING_DISTANCE_ATR = 1.0

FORMULA = (
    "Positive completed-daily SMA20/SMA50/SMA150/SMA200/EMA21/prior-pivot/"
    "Pivot-S1/20T-low references are retained only from -4.0 to +2.0 ATR versus "
    "the current USD close, deduplicated within max(0.02 ATR, 0.02% of price), "
    "and clustered when their envelope is at most 0.90 ATR. Correlated references "
    "share conservative source families. Selection score = 34*independent-family "
    "count + 6*level count + 12*family-weight sum + 18*proximity factor + "
    "8*support-family share. IDEAL is the relevance/role-weighted mean. "
    "Raw bounds are min(IDEAL-0.35 ATR, cluster-low-0.10 ATR) and "
    "max(IDEAL+0.35 ATR, cluster-high+0.10 ATR), proportionally capped to "
    "1.20 ATR total width; a single-family reference uses +/-0.35 ATR. Evidence "
    "quality = min(32,12*families) + 22*(1-exp(-family-weight/2.5)) + "
    "18*tightness + 16*proximity + 12*data-completeness."
    " When confluence is unavailable, a non-current anchor is selected "
    "deterministically with structural MA/EMA/20T-low preference over pivots; "
    "strong structural references may extend down to -10 ATR. Such reference-only "
    "geometry is capped below the green evidence threshold."
)

THRESHOLDS = {
    "candidate_distance_atr": [
        CANDIDATE_MIN_DISTANCE_ATR,
        CANDIDATE_MAX_DISTANCE_ATR,
    ],
    "strategic_reference_min_distance_atr": STRATEGIC_MIN_DISTANCE_ATR,
    "duplicate_tolerance_atr": 0.02,
    "cluster_envelope_atr": CLUSTER_DISTANCE_ATR,
    "minimum_zone_width_atr": 0.70,
    "maximum_zone_width_atr": 1.20,
    "approaching_above_zone_atr": APPROACHING_DISTANCE_ATR,
    "minimum_independent_family_count": 2,
    "minimum_reliability_score": MIN_RELIABILITY_SCORE,
    "minimum_longterm_score": MIN_LONGTERM_SCORE,
    "minimum_entry_timing_score": MIN_ENTRY_TIMING_SCORE,
    "maximum_atr_pct": MAX_ATR_PCT,
    "maximum_annual_volatility_pct": MAX_ANNUAL_VOL_PCT,
    "maximum_completed_bar_age_days": MAX_BAR_AGE_DAYS,
    "earnings_exclusion_calendar_days": 7,
    "rsi_allowed_range": [32.0, 70.0],
}
SOURCE_FAMILY_DEFINITION = {
    "pivot": ["pivot", "pivot_s1"],
    "moving_average_fast": ["sma20", "ema21"],
    "moving_average_medium": ["sma50"],
    "moving_average_long": ["sma150", "sma200"],
    "price_structure": ["low20"],
}

_SOURCES = (
    ("sma20", "SMA20", "moving_average_fast", 1.00),
    ("sma50", "SMA50", "moving_average_medium", 1.20),
    ("sma150", "SMA150", "moving_average_long", 0.85),
    ("sma200", "SMA200", "moving_average_long", 1.15),
    ("ema21", "EMA21", "moving_average_fast", 1.10),
    ("pivot", "Prior Pivot", "pivot", 1.00),
    ("pivot_s1", "Pivot S1", "pivot", 1.15),
    ("low20", "20T-Tief", "price_structure", 1.05),
)
_STRONG_SINGLE_LEVELS = {"SMA50", "SMA200", "EMA21", "Pivot S1", "20T-Tief"}
_EXTENDED_ANCHOR_LEVELS = {
    "SMA50",
    "SMA150",
    "SMA200",
    "EMA21",
    "20T-Tief",
}
_ANCHOR_FAMILY_PRIORITY = {
    "price_structure": 5,
    "moving_average_medium": 4,
    "moving_average_long": 4,
    "moving_average_fast": 3,
    "pivot": 1,
}
_CRITICAL_BLOCK_CODES = {
    "falling_knife",
    "bottoming",
    "negative_regime",
    "below_sma200",
    "weak_longterm",
    "negative_daily",
    "severe_macd",
    "extreme_rsi",
    "high_downside",
    "technical_incomplete",
    "completed_daily",
    "bar_stale",
    "data_gate",
}


def _finite(value: Any) -> bool:
    return (
        isinstance(value, (int, float))
        and not isinstance(value, bool)
        and math.isfinite(value)
    )


def _round(value: float | None, digits: int = 4) -> float | None:
    return round(value, digits) if _finite(value) else None


def price_display_decimals(values: list[Any]) -> int:
    """Choose enough decimal places to preserve distinct positive price marks."""
    finite = [float(value) for value in values if _finite(value)]
    if not finite:
        return 2
    magnitude = max(abs(value) for value in finite)
    if magnitude >= 10:
        base = 2
    elif magnitude >= 1:
        base = 3
    elif magnitude >= 0.01:
        base = 4
    else:
        positive = [abs(value) for value in finite if value != 0]
        smallest = min(positive) if positive else 0.0
        base = (
            max(6, -math.floor(math.log10(smallest)) + 4)
            if smallest > 0
            else 8
        )
    distinct_values = len(set(finite))
    for digits in range(base, 15):
        if len({f"{value:.{digits}f}" for value in finite}) == distinct_values:
            return digits
    return 14


def format_price(value: Any, context: list[Any] | None = None) -> str:
    if not _finite(value):
        return "—"
    values = list(context or []) + [value]
    digits = price_display_decimals(values)
    return f"{value:,.{digits}f}"


def _unique_text(values: list[str], limit: int = 8) -> list[str]:
    return list(dict.fromkeys(value for value in values if value))[:limit]


def _provenance(inputs: list[str], missing: list[str] | None = None) -> dict[str, Any]:
    return {
        "model_status": MODEL_STATUS,
        "model_version": MODEL_VERSION,
        "actionable": False,
        "inputs_used": inputs,
        "missing_inputs": list(missing or []),
    }


def _base_result(missing: list[str]) -> dict[str, Any]:
    return {
        **_provenance([], missing),
        "available": False,
        "label": "Sweet-Spot-Beobachtungszone",
        "technical_label": "technische Einstiegsbeobachtung",
        "currency": "USD",
        "currency_status": "upstream_absolute_levels_usd",
        "lower": None,
        "ideal": None,
        "upper": None,
        "current_price": None,
        "current_distance_pct": None,
        "distance_to_zone_pct": None,
        "current_position": "unavailable",
        "zone_tier": "unavailable",
        "anchor_scope": None,
        "anchor_distance_class": None,
        "confluence_count": 0,
        "independent_family_count": 0,
        "components": [],
        "excluded_components": [],
        "zone_width_pct": None,
        "zone_width_atr": None,
        "nearest_support": None,
        "invalidation_reference": None,
        "technical_status": "unavailable",
        "combined_status": "unavailable",
        "tone": "neutral",
        "reliability_score": 0.0,
        "reliability_label": "heuristic evidence quality, not likelihood",
        "why_zone_here": [],
        "why_green_or_not": ["Keine belastbare technische Referenzzone verfügbar."],
        "confirmation_needed": [],
        "invalidation_signals": [],
        "investor_overlay_status": "unavailable",
        "investor_overlay_reasons": [],
        "valuation_alignment": {
            "status": "unavailable",
            "note": "Ohne technische Zone ist kein Abgleich möglich.",
        },
        "formula": FORMULA,
        "thresholds": THRESHOLDS,
        "note": "Beobachtungszone, keine Ordermarke; keine Empfehlung oder Garantie.",
    }


def _candidate_levels(
    row: dict[str, Any],
    price: float,
    atr: float,
) -> tuple[
    list[dict[str, Any]],
    list[dict[str, Any]],
    list[dict[str, Any]],
    list[str],
]:
    stage = row.get("weinstein_stage")
    tactical: list[dict[str, Any]] = []
    strategic: list[dict[str, Any]] = []
    excluded: list[dict[str, Any]] = []
    missing: list[str] = []
    tolerance = max(0.02 * atr, 0.0002 * price)
    for field, label, source_family, relevance in _SOURCES:
        value = row.get(field)
        if not (_finite(value) and value > 0):
            missing.append(field)
            continue
        distance_atr = (value - price) / atr
        if source_family == "pivot" and abs(value - price) <= tolerance:
            excluded.append(
                {
                    "label": label,
                    "source_family": source_family,
                    "value": value,
                    "reason": "degenerate pivot equal to current close",
                }
            )
            continue
        role = 1.0 if value <= price else 0.85 if stage == 2 else 0.70
        candidate = {
            "field": field,
            "label": label,
            "source_family": source_family,
            "value": float(value),
            "distance_atr": distance_atr,
            "weight": relevance * role,
            "relevance": relevance,
        }
        if CANDIDATE_MIN_DISTANCE_ATR <= distance_atr <= CANDIDATE_MAX_DISTANCE_ATR:
            tactical.append(
                {
                    **candidate,
                    "anchor_scope": "tactical_reference",
                    "anchor_distance_class": "tactical",
                }
            )
        elif (
            STRATEGIC_MIN_DISTANCE_ATR <= distance_atr < CANDIDATE_MIN_DISTANCE_ATR
            and label in _EXTENDED_ANCHOR_LEVELS
        ):
            strategic.append(
                {
                    **candidate,
                    "anchor_scope": "strategic_reference",
                    "anchor_distance_class": "far_below",
                }
            )
        else:
            excluded.append(
                {
                    "label": label,
                    "source_family": source_family,
                    "value": value,
                    "reason": "outside tactical/strategic reference range",
                }
            )

    def deduplicate(values: list[dict[str, Any]]) -> list[dict[str, Any]]:
        kept_values: list[dict[str, Any]] = []
        for candidate in sorted(
            values,
            key=lambda item: (item["value"], -item["weight"], item["label"]),
        ):
            duplicate_index = next(
                (
                    index
                    for index, kept in enumerate(kept_values)
                    if abs(kept["value"] - candidate["value"]) <= tolerance
                ),
                None,
            )
            if duplicate_index is None:
                kept_values.append(candidate)
                continue
            kept = kept_values[duplicate_index]
            winner, duplicate = (
                (candidate, kept)
                if (candidate["weight"], candidate["label"])
                > (kept["weight"], kept["label"])
                else (kept, candidate)
            )
            kept_values[duplicate_index] = winner
            excluded.append(
                {
                    "label": duplicate["label"],
                    "source_family": duplicate["source_family"],
                    "value": duplicate["value"],
                    "reason": f"duplicate of {winner['label']}",
                }
            )
        kept_values.sort(key=lambda item: (item["value"], item["label"]))
        return kept_values

    return deduplicate(tactical), deduplicate(strategic), excluded, missing


def _best_single_anchor(candidates: list[dict[str, Any]]) -> dict[str, Any] | None:
    if not candidates:
        return None
    return max(
        candidates,
        key=lambda item: (
            (
                3
                if item["anchor_scope"] == "tactical_reference"
                and item["source_family"] != "pivot"
                else 2
                if item["anchor_scope"] == "strategic_reference"
                else 1
            ),
            _ANCHOR_FAMILY_PRIORITY[item["source_family"]],
            item["weight"],
            -abs(item["distance_atr"]),
            item["value"],
            item["label"],
        ),
    )


def _weighted_center(cluster: list[dict[str, Any]]) -> float:
    total = sum(item["weight"] for item in cluster)
    return sum(item["value"] * item["weight"] for item in cluster) / total


def _family_weights(cluster: list[dict[str, Any]]) -> dict[str, float]:
    weights: dict[str, float] = {}
    for item in cluster:
        family = item["source_family"]
        weights[family] = max(weights.get(family, 0.0), item["weight"])
    return weights


def _best_cluster(
    candidates: list[dict[str, Any]],
    price: float,
    atr: float,
) -> tuple[list[dict[str, Any]], float] | None:
    choices: list[tuple[tuple[Any, ...], list[dict[str, Any]], float]] = []
    for start in range(len(candidates)):
        for end in range(start + 2, len(candidates) + 1):
            cluster = candidates[start:end]
            if cluster[-1]["value"] - cluster[0]["value"] > CLUSTER_DISTANCE_ATR * atr:
                break
            center = _weighted_center(cluster)
            family_weights = _family_weights(cluster)
            family_weight_sum = sum(family_weights.values())
            independent_family_count = len(family_weights)
            proximity = max(0.0, 1.0 - abs(center - price) / (4.0 * atr))
            support_families = {
                item["source_family"]
                for item in cluster
                if item["value"] <= price
            }
            support_family_weight = sum(
                weight
                for family, weight in family_weights.items()
                if family in support_families
            )
            support_share = support_family_weight / family_weight_sum
            score = (
                34.0 * independent_family_count
                + 6.0 * len(cluster)
                + 12.0 * family_weight_sum
                + 18.0 * proximity
                + 8.0 * support_share
            )
            signature = tuple(item["label"] for item in cluster)
            key = (
                round(score, 10),
                independent_family_count,
                len(cluster),
                round(family_weight_sum, 10),
                -round(abs(center - price), 10),
                tuple(reversed(signature)),
            )
            choices.append((key, cluster, center))
    if not choices:
        return None
    _, cluster, center = max(choices, key=lambda choice: choice[0])
    return cluster, center


def _single_reference_allowed(row: dict[str, Any], candidate: dict[str, Any]) -> bool:
    price = row.get("price")
    sma200 = row.get("sma200")
    return bool(
        candidate["label"] in _STRONG_SINGLE_LEVELS
        and candidate.get("anchor_scope") == "tactical_reference"
        and row.get("weinstein_stage") == 2
        and _finite(row.get("longterm_score"))
        and row["longterm_score"] >= MIN_LONGTERM_SCORE
        and (
            not _finite(sma200)
            or (_finite(price) and price >= sma200)
        )
        and (row.get("feature_coverage") or {}).get("technical_complete") is True
    )


def _zone_bounds(
    cluster: list[dict[str, Any]],
    center: float,
    atr: float,
) -> tuple[float, float]:
    raw_lower = min(center - 0.35 * atr, cluster[0]["value"] - 0.10 * atr)
    raw_upper = max(center + 0.35 * atr, cluster[-1]["value"] + 0.10 * atr)
    raw_width = raw_upper - raw_lower
    cap = 1.20 * atr
    if raw_width > cap:
        scale = cap / raw_width
        lower = center - (center - raw_lower) * scale
        upper = center + (raw_upper - center) * scale
    else:
        lower, upper = raw_lower, raw_upper
    if lower <= 0:
        width = min(cap, max(0.70 * atr, upper - lower))
        lower = max(1e-12, center * 1e-9)
        upper = lower + width
    return lower, upper


def _single_anchor_bounds(center: float, atr: float) -> tuple[float, float]:
    width = 0.70 * atr
    lower = center - 0.35 * atr
    if lower <= 0:
        lower = max(1e-12, center * 1e-9)
    upper = lower + width
    if not lower < center < upper:
        upper = center + max(1e-12, 0.35 * atr)
        lower = max(1e-12, upper - width)
    return lower, upper


def _reliability_score(
    row: dict[str, Any],
    cluster: list[dict[str, Any]],
    center: float,
    *,
    price: float,
    atr: float,
    data_ready: bool,
    reference_only: bool,
) -> float:
    family_weight_sum = sum(_family_weights(cluster).values())
    family_count = len(_family_weights(cluster))
    envelope_atr = (cluster[-1]["value"] - cluster[0]["value"]) / atr
    tightness = max(0.0, 1.0 - envelope_atr / CLUSTER_DISTANCE_ATR)
    proximity = max(0.0, 1.0 - abs(center - price) / (4.0 * atr))
    score = (
        min(32.0, 12.0 * family_count)
        + 22.0 * (1.0 - math.exp(-family_weight_sum / 2.5))
        + 18.0 * tightness
        + 16.0 * proximity
        + 12.0 * _data_completeness(row, data_ready=data_ready)
    )
    if reference_only:
        score = min(score, 49.0)
    return max(0.0, min(100.0, score))


def _technical_complete(row: dict[str, Any]) -> bool:
    explicit = row.get("technical_complete")
    if isinstance(explicit, bool):
        return explicit
    return (row.get("feature_coverage") or {}).get("technical_complete") is True


def _fundamentals_complete_current(row: dict[str, Any]) -> bool:
    explicit = row.get("fundamental_complete_current")
    if isinstance(explicit, bool):
        return explicit
    coverage = row.get("feature_coverage") or {}
    return (
        coverage.get("fundamental_complete") is True
        and coverage.get("fundamental_current") is True
    )


def _fundamental_source_current(row: dict[str, Any]) -> bool:
    explicit = row.get("fundamental_source_current")
    if isinstance(explicit, bool):
        return explicit
    return (row.get("fundamental_source_status") or {}).get("status") in {
        None,
        "current",
    }


def _data_completeness(row: dict[str, Any], *, data_ready: bool) -> float:
    age = row.get("bar_age_days")
    checks = (
        _technical_complete(row),
        row.get("completed_bars_only") is True,
        _finite(age) and 0 <= age <= MAX_BAR_AGE_DAYS,
        data_ready,
    )
    return sum(bool(check) for check in checks) / len(checks)


def severe_macd_deterioration(row: dict[str, Any]) -> bool | None:
    macd_hist = row.get("macd_hist")
    macd_prev = row.get("macd_hist_prev")
    atr = row.get("atr")
    if not all(_finite(value) for value in (macd_hist, macd_prev, atr)):
        return None
    return bool(
        macd_hist < 0
        and macd_hist < macd_prev
        and (
            macd_prev - macd_hist >= 0.05 * atr
            or macd_hist <= -0.35 * atr
        )
    )


def technical_green_blockers(
    row: dict[str, Any],
    sweet: dict[str, Any],
    *,
    data_ready: bool,
) -> list[tuple[str, str]]:
    """Return every failed technical green gate in stable contract order."""
    blockers: list[tuple[str, str]] = []
    age = row.get("bar_age_days")
    if not _technical_complete(row):
        blockers.append(("technical_incomplete", "Technische Merkmale sind unvollständig."))
    if row.get("completed_bars_only") is not True:
        blockers.append(("completed_daily", "Completed-daily-Kontext ist nicht bestätigt."))
    if not (_finite(age) and 0 <= age <= MAX_BAR_AGE_DAYS):
        blockers.append(("bar_stale", "Tagesbar ist fehlend, zukünftig oder zu alt."))
    if not data_ready:
        blockers.append(("data_gate", "Übergeordnetes Daten-Gate ist nicht freigegeben."))

    price = row.get("price")
    sma200 = row.get("sma200")
    stage = row.get("weinstein_stage")
    phase = row.get("trend_phase") or {}
    longterm = row.get("longterm_score")
    timing = row.get("entry_timing_score")
    direction = row.get("daily_signal_direction")
    atr_pct = row.get("atr_pct")
    annual_vol = row.get("vol_annual_pct")
    rsi = row.get("rsi")
    downside_risk = str((row.get("downside_structure") or {}).get("risk") or "").casefold()

    if row.get("falling_knife"):
        blockers.append(("falling_knife", "Falling-Knife-Warnung ist aktiv."))
    if row.get("bottoming"):
        blockers.append(("bottoming", "Bodenbildung bleibt eine separate spekulative Beobachtung."))
    if stage in {3, 4} or phase.get("tone") == "down" or "top" in str(
        phase.get("phase") or ""
    ).casefold():
        blockers.append(("negative_regime", "Abwärts-/Top-Regime ist nicht zulässig."))
    if _finite(sma200) and _finite(price) and price < sma200:
        blockers.append(("below_sma200", "Kurs liegt unter dem SMA200."))
    if not (_finite(longterm) and longterm >= MIN_LONGTERM_SCORE):
        blockers.append(
            ("weak_longterm", f"Langfrist-Score liegt unter {MIN_LONGTERM_SCORE:.0f}.")
        )
    if not (_finite(timing) and timing >= MIN_ENTRY_TIMING_SCORE):
        blockers.append(
            ("timing", f"Timing-Score liegt unter {MIN_ENTRY_TIMING_SCORE:.0f}.")
        )
    if direction == "NEGATIVE":
        blockers.append(("negative_daily", "Completed-daily-Kontext ist NEGATIVE."))
    if not _finite(rsi) or not 32.0 <= rsi <= 70.0:
        blockers.append(("extreme_rsi", "RSI liegt außerhalb des konservativen Bereichs 32–70."))
    severe_macd = severe_macd_deterioration(row)
    if severe_macd is True:
        blockers.append(("severe_macd", "MACD-Histogramm verschlechtert sich stark."))
    elif severe_macd is None:
        blockers.append(("severe_macd", "MACD-Verschlechterung ist nicht prüfbar."))
    if downside_risk in {"hoch", "high"}:
        blockers.append(("high_downside", "Abwärtsstruktur-Risiko ist hoch."))
    if not (_finite(atr_pct) and atr_pct < MAX_ATR_PCT):
        blockers.append(("atr_pct", f"ATR liegt nicht unter {MAX_ATR_PCT:.0f}%."))
    if not (_finite(annual_vol) and annual_vol < MAX_ANNUAL_VOL_PCT):
        blockers.append(
            (
                "annual_volatility",
                f"Annualisierte Volatilität liegt nicht unter {MAX_ANNUAL_VOL_PCT:.0f}%.",
            )
        )
    earnings = row.get("earnings_in_days")
    if isinstance(earnings, int) and 0 <= earnings <= 7:
        blockers.append(("earnings", f"Unternehmenszahlen liegen in {earnings} Tagen."))
    zone_tier = sweet.get("zone_tier")
    if zone_tier == "reference_only":
        blockers.append(
            (
                "reference_only",
                (
                    "In mathematischer Referenzzone, Bestätigung fehlt: "
                    "nur Einzelanker, keine Confluence."
                    if sweet.get("current_position") == "in"
                    else "Nur Einzelanker, keine Confluence; mathematische Referenzzone ohne Bestätigung."
                ),
            )
        )
    elif zone_tier == "single_anchor":
        blockers.append(
            (
                "single_anchor",
                "Nur ein bestätigungsbedürftiger Einzelanker; keine unabhängige Confluence.",
            )
        )
    if sweet.get("independent_family_count", 0) < 2:
        blockers.append(
            (
                "independent_families",
                "Weniger als zwei unabhängige Quellenfamilien bestätigen die Zone.",
            )
        )
    reliability = sweet.get("reliability_score")
    if not (_finite(reliability) and reliability >= MIN_RELIABILITY_SCORE):
        blockers.append(
            (
                "reliability",
                f"Evidenzqualität liegt unter {MIN_RELIABILITY_SCORE:.0f}/100.",
            )
        )
    return list(dict.fromkeys(blockers))


def green_invariant_blockers(
    row: dict[str, Any],
    sweet: dict[str, Any],
    *,
    data_ready: bool,
) -> list[tuple[str, str]]:
    """Recompute all technical, location, and investor blockers for green."""
    blockers: list[tuple[str, str]] = []
    if not sweet.get("available"):
        blockers.append(("zone_unavailable", "Keine belastbare technische Referenzzone verfügbar."))
        return blockers
    price = row.get("price")
    lower, upper = sweet.get("lower"), sweet.get("upper")
    calculated_position = (
        "below"
        if all(_finite(value) for value in (price, lower)) and price < lower
        else "above"
        if all(_finite(value) for value in (price, upper)) and price > upper
        else "in"
        if all(_finite(value) for value in (price, lower, upper))
        else "unavailable"
    )
    position = sweet.get("current_position")
    if position != calculated_position:
        blockers.append(
            (
                "position_inconsistent",
                "Gespeicherte Kursposition stimmt nicht mit Kurs und Zonengrenzen überein.",
            )
        )
    if calculated_position != "in":
        blockers.append(
            (
                "current_outside_zone",
                "Aktueller Kurs liegt nicht in der mathematischen Beobachtungszone.",
            )
        )
    invalidation = sweet.get("invalidation_reference") or {}
    invalidation_value = invalidation.get("value")
    if _finite(price) and _finite(invalidation_value) and price < invalidation_value:
        blockers.append(
            (
                "below_invalidation",
                "Kurs liegt unter der technischen Invalidation Reference.",
            )
        )
    blockers.extend(
        technical_green_blockers(row, sweet, data_ready=data_ready)
    )
    overlay_status, overlay_reasons, _ = _investor_overlay(
        row,
        current_position=calculated_position,
    )
    if row.get("asset_type") == "company_equity" and overlay_status != "passed":
        blockers.extend(
            (f"investor_overlay_{index + 1}", reason)
            for index, reason in enumerate(overlay_reasons)
        )
    return list(dict.fromkeys(blockers))


def confirmed_status_violations(
    row: dict[str, Any],
    sweet: dict[str, Any],
    *,
    data_ready: bool,
) -> list[str]:
    """Return violations when a row/category claims combined confirmed green."""
    violations: list[str] = []
    if sweet.get("combined_status") != "in_zone_confirmed":
        violations.append("combined_status is not in_zone_confirmed")
    if sweet.get("tone") != "green":
        violations.append("tone is not green")
    if sweet.get("technical_status") != "in_zone_confirmed":
        violations.append("technical_status is not in_zone_confirmed")
    violations.extend(
        reason
        for _, reason in green_invariant_blockers(
            row,
            sweet,
            data_ready=data_ready,
        )
    )
    return _unique_text(violations, limit=100)


def _investor_overlay(
    row: dict[str, Any],
    *,
    current_position: str,
) -> tuple[str, list[str], dict[str, str]]:
    if row.get("asset_type") != "company_equity":
        return (
            "technical_only",
            ["ETF/Krypto: kein Unternehmens-Fundamental-Overlay anwendbar."],
            {
                "status": "technical_only",
                "note": "Technische Beobachtung ohne Unternehmensbewertungs-Overlay.",
            },
        )

    reasons: list[str] = []
    complete = (
        _fundamentals_complete_current(row)
        and _fundamental_source_current(row)
    )
    if not complete:
        reasons.append("Unternehmensfundamentaldaten sind unvollständig oder nicht aktuell.")
    jurisdiction = row.get("jurisdiction_risk") or {}
    if jurisdiction.get("level") == "high":
        reasons.append("Hohes Jurisdiktionsrisiko passiert den Investor-Filter nicht.")
    valuation = row.get("valuation_thesis") or {}
    if valuation.get("value_trap_risk") == "high":
        reasons.append("Value-Trap-Risiko ist hoch.")
    altman = row.get("altman_z")
    if complete and _finite(altman) and altman < 1.81:
        reasons.append(f"Altman-Z {altman:.2f} zeigt schweren Bilanzrisiko-Kontext.")
    penalties = valuation.get("penalty_components") or {}
    if any(
        _finite(penalties.get(key)) and penalties[key] >= threshold
        for key, threshold in (
            ("cyclical_peak_penalty", 6.0),
            ("shrinking_fundamentals_penalty", 8.0),
            ("weak_trend_downside_penalty", 8.0),
        )
    ) or row.get("major_counterargument") is True:
        reasons.append("Ein wesentlicher Gegenargument-/Strukturrisiko-Filter ist aktiv.")

    value = (row.get("valuation_context") or {}).get("value_score")
    if reasons:
        alignment_status = "risk_filtered"
        alignment_note = (
            "Technische Zone und Bewertung werden getrennt gezeigt; der Investor-Filter "
            "ist wegen sichtbarer Risiken nicht passiert."
        )
    elif current_position == "in" and _finite(value) and value >= 65:
        alignment_status = "aligned"
        alignment_note = (
            "Mathematische technische Zone und günstiger deskriptiver Bewertungskontext "
            "fallen zusammen; dies bleibt unvalidierte Evidenz."
        )
    elif not _finite(value):
        alignment_status = "valuation_unavailable"
        alignment_note = "Kein belastbarer günstiger Bewertungskontext für einen Abgleich."
    elif current_position != "in":
        alignment_status = "technical_not_in_zone"
        alignment_note = "Aktueller Kurs liegt nicht in der mathematischen technischen Zone."
    else:
        alignment_status = "valuation_not_cheap"
        alignment_note = "Technischer Zonenstatus ist nicht mit einem Value-Score ab 65 gekoppelt."
    return (
        "risk_filtered" if reasons else "passed",
        _unique_text(reasons, 6),
        {"status": alignment_status, "note": alignment_note},
    )


def build_sweet_spot(
    row: dict[str, Any],
    *,
    data_ready: bool | None = None,
) -> dict[str, Any]:
    """Build one deterministic, non-actionable USD observation-zone contract."""
    price, atr = row.get("price"), row.get("atr")
    ready = True if data_ready is None else data_ready
    missing: list[str] = []
    if not (_finite(price) and price > 0):
        missing.append("finite positive USD price")
    if not (_finite(atr) and atr > 0):
        missing.append("finite positive ATR in USD")
    if missing:
        return _base_result(missing)

    candidates, strategic_candidates, excluded, missing_fields = _candidate_levels(
        row,
        price,
        atr,
    )
    selection = _best_cluster(candidates, price, atr)
    reference_only = False
    zone_tier = "confirmed_confluence"
    anchor_scope = "cluster"
    anchor_distance_class = "tactical"
    if selection is not None:
        selected_cluster, selected_center = selection
        if len(_family_weights(selected_cluster)) >= 2:
            cluster, center = selected_cluster, selected_center
        else:
            selection = None
    if selection is None:
        reference = _best_single_anchor([*candidates, *strategic_candidates])
        if reference is not None:
            cluster = [reference]
            center = reference["value"]
            reference_only = True
            zone_tier = (
                "single_anchor"
                if _single_reference_allowed(row, reference)
                else "reference_only"
            )
            anchor_scope = reference["anchor_scope"]
            anchor_distance_class = reference["anchor_distance_class"]
        else:
            result = _base_result(
                [
                    "no non-current valid technical reference within tactical or strategic range",
                ]
            )
            result["current_price"] = price
            result["excluded_components"] = excluded
            result["inputs_used"] = [
                item["field"] for item in [*candidates, *strategic_candidates]
            ]
            result["missing_inputs"].extend(missing_fields)
            return result

    lower, upper = (
        _single_anchor_bounds(center, atr)
        if reference_only
        else _zone_bounds(cluster, center, atr)
    )
    if not (0 < lower < center < upper and upper - lower > 0):
        return _base_result(["positive ordered zone geometry unavailable"])

    current_position = "below" if price < lower else "above" if price > upper else "in"
    current_distance_pct = (price / center - 1.0) * 100.0
    distance_to_zone_pct = (
        (price / upper - 1.0) * 100.0
        if price > upper
        else (price / lower - 1.0) * 100.0
        if price < lower
        else 0.0
    )
    family_weights = _family_weights(cluster)
    independent_family_count = len(family_weights)
    reliability = _reliability_score(
        row,
        cluster,
        center,
        price=price,
        atr=atr,
        data_ready=ready,
        reference_only=reference_only,
    )

    support_candidates = [
        item
        for item in [*candidates, *strategic_candidates]
        if item["value"] <= price
    ]
    nearest = (
        max(support_candidates, key=lambda item: (item["value"], item["weight"]))
        if support_candidates
        else None
    )
    invalidation_value = max(0.0001, lower - 0.35 * atr)
    technical_evidence = {
        "available": True,
        "current_position": current_position,
        "zone_tier": zone_tier,
        "independent_family_count": independent_family_count,
        "reliability_score": reliability,
        "invalidation_reference": {"value": invalidation_value},
    }
    technical_blockers = technical_green_blockers(
        row,
        technical_evidence,
        data_ready=ready,
    )
    blocker_codes = {code for code, _ in technical_blockers}
    critical = bool(blocker_codes & _CRITICAL_BLOCK_CODES)
    below_invalidation = price < invalidation_value

    if zone_tier == "reference_only" and (critical or below_invalidation):
        technical_status, tone = "safety_blocked", "red"
    elif zone_tier == "reference_only":
        technical_status, tone = "reference_only_far", "neutral"
    elif below_invalidation:
        technical_status, tone = "broken_below", "red"
    elif critical:
        technical_status, tone = "safety_blocked", "red"
    elif current_position == "in" and not technical_blockers:
        technical_status, tone = "in_zone_confirmed", "green"
    elif current_position == "above" and (price - upper) / atr <= APPROACHING_DISTANCE_ATR:
        technical_status, tone = "approaching", "amber"
    elif current_position == "above":
        technical_status, tone = "far_above", "neutral"
    else:
        technical_status, tone = "setup_waiting_confirmation", "amber"

    overlay_status, overlay_reasons, valuation_alignment = _investor_overlay(
        row,
        current_position=current_position,
    )
    if technical_status == "in_zone_confirmed" and overlay_status == "risk_filtered":
        combined_status, tone = "in_zone_risk_filtered", "amber"
    else:
        combined_status = technical_status

    why_zone = [
        (
            f"{len(cluster)} Referenzlevel aus {independent_family_count} unabhängigen "
            f"Quellenfamilien liegen in einem "
            f"{(cluster[-1]['value'] - cluster[0]['value']) / atr:.2f}-ATR-Cluster."
            if not reference_only
            else (
                "Eine starke Stage-2-Referenz liefert eine bestätigungsbedürftige Einzelanker-Zone."
                if zone_tier == "single_anchor"
                else (
                    f"Deterministischer Einzelanker ohne Confluence "
                    f"({anchor_scope}/{anchor_distance_class}); nur mathematische Referenzzone."
                )
            )
        ),
        "IDEAL ist das relevanz- und Trendrollen-gewichtete Zentrum der Referenzen.",
    ]

    confirmation_needed = [
        "Abgeschlossene Tagesbars müssen Zone und IDEAL stabilisieren.",
        (
            "Alle expliziten Trend-, Momentum-, Volatilitäts-, Ereignis- und "
            "Daten-Gates müssen gleichzeitig erfüllt sein."
        ),
    ]
    if overlay_status == "risk_filtered":
        confirmation_needed.append(
            "Für den kombinierten Status müssen die separaten Investor-Risikofilter passieren."
        )
    invalidation_signals = [
        (
            f"Abgeschlossener Tagesschluss unter der Invalidation Reference "
            f"{invalidation_value:.8g} USD schwächt den Zonen-Kontext."
        ),
        "Neue Falling-Knife-, Stage-4- oder hohe Abwärtsstruktur-Warnung invalidiert Grün.",
    ]
    components = [
        {
            "label": item["label"],
            "source_family": item["source_family"],
            "value": item["value"],
            "distance_atr": item["distance_atr"],
            "weight": item["weight"],
        }
        for item in sorted(cluster, key=lambda item: (-item["weight"], item["label"]))
    ]
    result = {
        **_provenance(
            [item["field"] for item in cluster]
            + [
                "price",
                "atr",
                "completed-daily safety gates",
                "company investor overlay when applicable",
            ],
            missing_fields,
        ),
        "available": True,
        "label": "Sweet-Spot-Beobachtungszone",
        "technical_label": "technische Einstiegsbeobachtung",
        "currency": "USD",
        "currency_status": "upstream_absolute_levels_usd",
        "lower": lower,
        "ideal": center,
        "upper": upper,
        "current_price": price,
        "current_distance_pct": current_distance_pct,
        "distance_to_zone_pct": distance_to_zone_pct,
        "current_position": current_position,
        "zone_tier": zone_tier,
        "anchor_scope": anchor_scope,
        "anchor_distance_class": anchor_distance_class,
        "confluence_count": len(cluster),
        "independent_family_count": independent_family_count,
        "components": components,
        "excluded_components": excluded,
        "zone_width_pct": (upper - lower) / center * 100.0,
        "zone_width_atr": (upper - lower) / atr,
        "nearest_support": (
            {
                "label": nearest["label"],
                "source_family": nearest["source_family"],
                "value": nearest["value"],
                "distance_atr": nearest["distance_atr"],
            }
            if nearest
            else None
        ),
        "invalidation_reference": {
            "value": invalidation_value,
            "basis": "zone lower minus 0.35 ATR",
            "technical_observation_only": True,
        },
        "technical_status": technical_status,
        "combined_status": combined_status,
        "tone": tone,
        "reliability_score": reliability,
        "reliability_label": "heuristic evidence quality, not likelihood",
        "why_zone_here": _unique_text(why_zone, 4),
        "why_green_or_not": [],
        "confirmation_needed": _unique_text(confirmation_needed, 4),
        "invalidation_signals": _unique_text(invalidation_signals, 4),
        "investor_overlay_status": overlay_status,
        "investor_overlay_reasons": overlay_reasons,
        "valuation_alignment": valuation_alignment,
        "formula": FORMULA,
        "thresholds": THRESHOLDS,
        "note": "Beobachtungszone, keine Ordermarke; keine Empfehlung oder Garantie.",
    }
    all_blockers = green_invariant_blockers(row, result, data_ready=ready)
    result["why_green_or_not"] = (
        ["Kurs liegt in der Zone und alle anwendbaren Sicherheits-Gates sind passiert."]
        if not all_blockers
        else _unique_text([reason for _, reason in all_blockers], limit=100)
    )
    return result
