"""Read-only dashboard for conservative, unvalidated daily research output."""
from __future__ import annotations

import sys
from datetime import datetime, timezone
from pathlib import Path

import pandas as pd
import streamlit as st

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src import config as _config  # loads optional project .env before gate constants
from src.data_quality import (
    DataContractError,
    dashboard_gate,
    validate_output_contract,
    validate_portfolio_contract,
)
from src.persistence import PersistenceError, load_json
from src.probability_forward_public import (
    load_forward_validation_status,
    validate_forward_validation_status,
)
from src.sweet_spot import format_price

LATEST = ROOT / "data" / "output" / "latest.json"
PORTFOLIO = ROOT / "data" / "portfolio.json"
BACKTEST = ROOT / "data" / "backtest.json"

st.set_page_config(page_title="Stock Radar Research", page_icon="📊", layout="wide")
st.title("📊 Stock Radar — completed-daily-bar research")
st.caption(
    "Heuristic radar sections remain unvalidated. Calibrated material-move "
    "probabilities appear only after every strict release gate passes."
)


def _stop_with_error(message: str) -> None:
    st.error(message)
    st.stop()


try:
    data = load_json(LATEST, required=True, expected_type=dict)
    validate_output_contract(data)
except (PersistenceError, DataContractError) as exc:
    _stop_with_error(
        f"Output is missing, corrupt, or from the unsupported legacy schema: {exc}. "
        "Run `python -m src.analyze` or provider-free `python -m src.enrich_snapshot "
        "--in-place` to create a validated v3 snapshot."
    )

status = data["data_status"]
model = data["model_status"]
try:
    forward_validation = validate_forward_validation_status(
        data.get("forward_validation_status")
        or load_forward_validation_status()
    )
except ValueError as exc:
    _stop_with_error(f"Forward-validation aggregate is invalid: {exc}")
allowed, blockers = dashboard_gate(data)

if not allowed:
    st.error("Research cards are blocked because the freshness/completeness contract failed.")
    for reason in blockers:
        st.write(f"- {reason}")
    st.json(
        {
            "data_status": status,
            "model_status": model,
            "probability_validation": data.get("probability_validation"),
            "forward_validation_status": forward_validation,
        }
    )
    failed = status.get("failed_symbols")
    if isinstance(failed, dict) and failed:
        st.dataframe(
            pd.DataFrame(
                [{"symbol": symbol, "failure": reason} for symbol, reason in failed.items()]
            ),
            hide_index=True,
            width="stretch",
        )
    st.stop()

metrics = st.columns(5)
metrics[0].metric("Snapshot", data["generated_at"].replace("T", " ")[:19] + " UTC")
metrics[1].metric("Price coverage", f"{status.get('coverage_pct', 0):.2f}%")
metrics[2].metric("Fresh bars", f"{status.get('fresh_bar_coverage_pct', 0):.2f}%")
metrics[3].metric("Failed symbols", status.get("failed_symbol_count", 0))
metrics[4].metric("Model", str(model.get("validation", "unknown")).upper())

st.warning(
    "The deployed composite is UNVALIDATED and non-actionable. Scenario ranges are "
    "heuristic volatility illustrations, not expected returns, medians, calibrated "
    "intervals, probabilities, targets, or recommendations."
)
st.info(
    "Technical research lists are partitioned by trading currency and asset class. "
    "There is no global cross-currency ordering because point-in-time historical FX "
    "is unavailable."
)

with st.expander("Data and model contract", expanded=not allowed):
    st.json(
        {
            "schema": data.get("schema"),
            "schema_version": data.get("schema_version"),
            "data_status": status,
            "model_status": model,
            "probability_validation": data.get("probability_validation"),
            "forward_validation_status": forward_validation,
        }
    )
    feature_gate = status.get("feature_coverage") or {}
    if feature_gate:
        st.write("Feature/rank coverage gate by asset class")
        st.dataframe(
            pd.DataFrame.from_dict(feature_gate, orient="index"),
            width="stretch",
        )

ASSET_LABELS = {
    "company_equity": "Company equities",
    "etf_fund": "ETFs / funds",
    "crypto": "Crypto",
    "index_other": "Indices / other",
}


def _number(value, digits=2, suffix=""):
    return f"{value:,.{digits}f}{suffix}" if isinstance(value, (int, float)) else "—"


def _scenario_table(row):
    records = []
    for item in row.get("scenario_long") or []:
        records.append(
            {
                "Horizont": item.get("label"),
                "Referenzpfad (heuristisch)": _number(item.get("reference_change_pct"), 1, "%"),
                "Spanne unten USD": _number(item.get("range_low_price"), 2),
                "Spanne oben USD": _number(item.get("range_high_price"), 2),
                "Status": item.get("model_status"),
            }
        )
    return pd.DataFrame(records)


def _render_points(title, points, *, kind="info"):
    st.markdown(f"**{title}**")
    if not points:
        st.caption("Keine belastbaren Angaben aus den verfügbaren Eingaben.")
        return
    for point in points:
        getattr(st, kind)(point)


def _render_sweet_spot(row):
    sweet = row.get("sweet_spot") or {}
    st.markdown("#### Sweet-Spot-Beobachtungszone")
    st.caption("Technische Einstiegsbeobachtung · Beobachtungszone, keine Ordermarke")
    status_labels = {
        "in_zone_confirmed": "Im Sweet Spot",
        "in_zone_risk_filtered": "In Zone, Investor-Risiko gefiltert",
        "approaching": "Nähert sich der Zone",
        "setup_waiting_confirmation": "Bestätigung ausstehend",
        "safety_blocked": "Sicherheitsfilter blockiert",
        "broken_below": "Unter Invalidation Reference",
        "far_above": "Weit oberhalb der Zone",
        "reference_only_far": "Nur Referenzzone",
        "unavailable": "Zone nicht verfügbar",
    }
    label = status_labels.get(
        sweet.get("combined_status"),
        sweet.get("combined_status") or "Zone nicht verfügbar",
    )
    tone = sweet.get("tone")
    message = (
        f"{label} · combined_status={sweet.get('combined_status') or 'unavailable'} · "
        "heuristic_unvalidated / actionable false"
    )
    if tone == "green":
        st.success(message)
    elif tone == "amber":
        st.warning(message)
    elif tone == "red":
        st.error(message)
    else:
        st.info(message)
    if not sweet.get("available"):
        st.caption("Keine belastbare Referenzzone; es werden keine Marken erzeugt.")
        return
    if sweet.get("zone_tier") == "reference_only":
        st.info(
            "Nur Einzelanker, keine Confluence · "
            f"{sweet.get('anchor_scope') or 'reference'} / "
            f"{sweet.get('anchor_distance_class') or 'unclassified'} · "
            "mathematische Referenzzone, Bestätigung fehlt."
        )
    zone_values = [
        sweet.get("lower"),
        sweet.get("ideal"),
        sweet.get("upper"),
    ]
    zone_metrics = st.columns(4)
    zone_metrics[0].metric(
        "Untergrenze USD",
        format_price(sweet.get("lower"), zone_values),
    )
    zone_metrics[1].metric(
        "IDEAL USD",
        format_price(sweet.get("ideal"), zone_values),
    )
    zone_metrics[2].metric(
        "Obergrenze USD",
        format_price(sweet.get("upper"), zone_values),
    )
    zone_metrics[3].metric(
        "Aktueller Kurs USD",
        format_price(sweet.get("current_price"), [*zone_values, sweet.get("current_price")]),
    )
    st.caption(
        f"{format_price(sweet.get('lower'), zone_values)} – IDEAL "
        f"{format_price(sweet.get('ideal'), zone_values)} – "
        f"{format_price(sweet.get('upper'), zone_values)} USD · Abstand zu IDEAL "
        f"{_number(sweet.get('current_distance_pct'), 2, '%')} · Abstand zur Zone "
        f"{_number(sweet.get('distance_to_zone_pct'), 2, '%')} · Reliabilität "
        f"{_number(sweet.get('reliability_score'), 0, '/100')} "
        f"· {sweet.get('independent_family_count', 0)} unabhängige "
        f"{'Quellenfamilie' if sweet.get('independent_family_count') == 1 else 'Quellenfamilien'} "
        "(heuristische Evidenzqualität, keine Wahrscheinlichkeit)"
    )
    components = sweet.get("components") or []
    if components:
        st.markdown(
            "**Konfluenzquellen:** "
            + " ".join(
                f"`{item.get('label')} "
                f"{format_price(item.get('value'), [*zone_values, item.get('value')])} USD`"
                for item in components
            )
        )
    _render_points("Warum diese Zone", sweet.get("why_zone_here"))
    _render_points(
        "Warum Grün oder nicht",
        sweet.get("why_green_or_not"),
        kind="success" if tone == "green" else "error" if tone == "red" else "warning",
    )
    _render_points("Was bestätigt", sweet.get("confirmation_needed"))
    _render_points("Was invalidiert", sweet.get("invalidation_signals"), kind="error")
    _render_points(
        "Investor-Overlay",
        sweet.get("investor_overlay_reasons"),
        kind="warning",
    )
    alignment = sweet.get("valuation_alignment") or {}
    st.caption(
        f"Bewertungsabgleich: {alignment.get('status') or 'unavailable'} · "
        f"{alignment.get('note') or ''}"
    )


def _render_probability_forecast(row, shared_baselines=None):
    forecast = row.get("probability_forecast") or {}
    if (
        not forecast.get("forecasts")
        and not forecast.get("baselines")
        and row.get("asset_type") == "company_equity"
        and row.get("currency") == "USD"
        and shared_baselines
    ):
        forecast = {**forecast, "baselines": shared_baselines}
    st.markdown("#### Kalibrierte Wahrscheinlichkeiten")
    st.caption(
        "Separates Modell · kein Bestandteil von Radar-Score, Insight-Rang, "
        "Sweet Spot oder Farbe · keine Anlageempfehlung"
    )
    if forecast.get("status") == "withheld":
        st.warning(
            f"{forecast.get('message') or 'No validated stock-specific probability edge'}."
        )
        st.caption(
            "Status: Forward Validation erforderlich. Historische Basisraten sind "
            "deskriptive Häufigkeiten des heutigen Survivor-Universums, keine "
            "Aktienprognose und kein Konfidenzintervall."
        )
        for reason in forecast.get("reasons") or []:
            st.write(f"- {reason}")
    elif forecast.get("status") == "partial":
        st.info(
            "Nur die unten aufgeführten Horizont-/Schwellenmodelle haben alle "
            "aktuellen Freigabekriterien erfüllt."
        )
    elif forecast.get("status") == "accepted":
        st.info("Alle unten aufgeführten Modelle haben die strikten Freigabekriterien erfüllt.")
    else:
        st.warning("No validated stock-specific probability edge. Output contract unavailable.")

    for item in forecast.get("forecasts") or []:
        probabilities = item.get("probabilities_pct") or {}
        intervals = item.get("model_interval_95_pct") or {}
        bootstrap = item.get("fixed_oos_bootstrap") or {}
        threshold = item.get("threshold_pct")
        gross_boundary = (
            threshold + 0.30 if isinstance(threshold, (int, float)) else None
        )
        st.markdown(
            f"**{item.get('horizon_label')} / {threshold}% Materialschwelle:** "
            f"Anstieg (Brutto) ≥ +{gross_boundary:.2f}% {probabilities.get('up')}%, "
            f"Mitte {probabilities.get('middle')}%, "
            f"Rückgang (Brutto) ≤ -{gross_boundary:.2f}% {probabilities.get('down')}%"
        )
        st.caption(
            f"Modellfamilie {item.get('model_family') or 'independent-threshold-v1'} · "
            "95% aggregate calibration-error interval approximation from fixed "
            "OOS predictions; not an individual stock outcome interval. "
            "Down/Mitte/Up: "
            f"{(intervals.get('down') or ['—', '—'])[0]}–"
            f"{(intervals.get('down') or ['—', '—'])[1]}% / "
            f"{(intervals.get('middle') or ['—', '—'])[0]}–"
            f"{(intervals.get('middle') or ['—', '—'])[1]}% / "
            f"{(intervals.get('up') or ['—', '—'])[0]}–"
            f"{(intervals.get('up') or ['—', '—'])[1]}% · "
            f"Baseline {item.get('baseline_rates_pct')} · "
            f"Brier skill {_number(item.get('brier_skill'), 3)} · "
            f"ECE {item.get('classwise_ece')} · "
            f"{item.get('full_test_fold_count') or item.get('fold_count')} volle "
            f"Test-Folds · {_number(item.get('history_years'), 1)} nutzbare Jahre · "
            f"min. Train {_number(item.get('min_usable_train_years'), 1)} Jahre · "
            f"n={item.get('sample_size')} · "
            f"OOS-Bootstrap {bootstrap.get('completed')}/"
            f"{bootstrap.get('requested')} abgeschlossen"
            f" ({bootstrap.get('skipped')} übersprungen) · "
            f"{(item.get('threshold_monotonicity') or {}).get('disclosure') or ''}"
        )
    if not forecast.get("forecasts") and forecast.get("baselines"):
        st.caption(
            "Nur historische Basisraten; keine aktienspezifische Modellfreigabe. "
            "Die Werte sind keine Prognose für dieses Unternehmen."
        )
        st.dataframe(
            pd.DataFrame(
                [
                    {
                        "Horizont": item.get("horizon_label"),
                        "Schwelle": item.get("threshold_pct"),
                        "Down %": (item.get("rates_pct") or {}).get("down"),
                        "Mitte %": (item.get("rates_pct") or {}).get("middle"),
                        "Up %": (item.get("rates_pct") or {}).get("up"),
                        "Freigabe": item.get("accepted_stock_specific_model"),
                        "Folds": item.get("fold_count"),
                    }
                    for item in forecast["baselines"]
                ]
            ),
            hide_index=True,
            width="stretch",
        )
    st.caption(
        f"Signal: {forecast.get('signal_timestamp') or '—'} · "
        f"Listing-Währung: {forecast.get('listing_currency') or '—'} · "
        "Einstieg: erster adjustierter/raw-äquivalenter Open strikt nach t · "
        "Kosten: 30 bp Round Trip. "
        f"{forecast.get('survivorship_warning') or ''}"
    )


def _research_card(row, rank=None):
    prefix = f"{rank}. " if rank is not None else ""
    with st.container(border=True):
        valuation = row.get("valuation_context") or {}
        valuation_thesis = row.get("valuation_thesis") or {}
        entry_thesis = row.get("entry_thesis") or {}
        jurisdiction = row.get("jurisdiction_risk") or {}
        st.caption("Vollständiger Name")
        st.subheader(
            f"{prefix}{row.get('display_name_full') or row.get('short_name') or row.get('symbol')}"
        )
        st.markdown(f"**{row.get('symbol')}**")
        st.caption(
            f"Branche: {row.get('industry_display') or 'nicht verfügbar'} · "
            f"Sektor: {row.get('sector_display') or 'nicht verfügbar'} · "
            f"Hauptsitz (Provider): {row.get('headquarters_country') or 'nicht verfügbar'} · "
            f"Wirtschaftliches Exposure: {row.get('economic_exposure_country') or 'nicht verfügbar'}"
            f"/{row.get('economic_exposure_region') or 'nicht verfügbar'} · "
            f"Börsenland/Markt: {row.get('listing_country') or 'nicht verfügbar'} / "
            f"{row.get('listing_market') or 'nicht verfügbar'}"
        )
        if row.get("legal_domicile_verified"):
            st.caption(
                f"Juristischer Sitz (verifiziert): {row.get('legal_domicile')} · "
                f"{row.get('legal_domicile_source') or ''}"
            )
        if row.get("economic_exposure_country") == "China":
            st.error("China-Risikokontext (heuristic_unvalidated; kein bewiesener Abschlag)")
            for reason in jurisdiction.get("reasons") or []:
                st.warning(reason)
        jobs_signal = row.get("jobs_signal") or {}
        if jobs_signal.get("status") == "without_jobs_signal":
            st.caption("Jobs-Signal: ohne Jobs-Signal (keine zuverlässig konfigurierte direkte Karriereseite)")
        elif jobs_signal.get("status") == "collecting_history":
            st.caption(
                f"Jobs-Signal: {jobs_signal.get('open_jobs')} offene Stellen · "
                "Baseline wird gesammelt (mindestens 7 Tage)"
            )
        elif jobs_signal.get("status") == "ok":
            st.caption(
                f"Jobs-Signal: {jobs_signal.get('open_jobs')} offen · "
                f"Trend {jobs_signal.get('change_pct')}% seit "
                f"{jobs_signal.get('baseline_date')}"
            )
        expert = row.get("expert_analysis") or {}
        if expert:
            st.markdown("#### Experten-Composite")
            expert_metrics = st.columns(4)
            expert_metrics[0].metric(
                "Long Term",
                _number((expert.get("long_term") or {}).get("score"), 1, "/100"),
                f"{_number((expert.get('long_term') or {}).get('coverage_pct'), 1, '%')} Abdeckung",
            )
            expert_metrics[1].metric(
                "Short Term",
                _number((expert.get("short_term") or {}).get("score"), 1, "/100"),
                f"{_number((expert.get('short_term') or {}).get('coverage_pct'), 1, '%')} Abdeckung",
            )
            expert_metrics[2].metric("Signal", expert.get("signal") or "insufficient_data")
            expert_metrics[3].metric(
                "Konfidenz", expert.get("evidence_quality") or "low"
            )
            expert_valuation = expert.get("valuation") or {}
            fair_range = expert_valuation.get("fair_value_range") or {}
            st.caption(
                f"Bewertung: {expert_valuation.get('verdict') or 'unavailable'} · "
                + (
                    f"fairer Heuristik-Bereich {_number(fair_range.get('lower'), 2)}–"
                    f"{_number(fair_range.get('upper'), 2)} {fair_range.get('currency') or ''}"
                    if fair_range
                    else "kein belastbarer fairer Bereich"
                )
            )
            if expert_valuation.get("missing_note"):
                st.info(expert_valuation["missing_note"])
            expert_risks = (expert.get("risks") or {}).get("top_risks") or []
            for risk in expert_risks:
                st.warning(risk)
            st.caption(
                "Szenario-Wahrscheinlichkeiten: "
                f"{(expert.get('outlook') or {}).get('probabilities_status') or 'withheld'}"
            )
        _render_probability_forecast(
            row, data.get("probability_baselines") or []
        )
        _render_sweet_spot(row)
        header = st.columns(6)
        header[0].metric("Completed-bar close (USD)", _number(row.get("price"), 2))
        header[1].metric("Completed bar", row.get("bar_date") or "—")
        header[2].metric("Timing", _number(row.get("entry_timing_score"), 0, "/100"))
        header[3].metric("Trend", _number(row.get("longterm_score"), 0, "/100"))
        header[4].metric("Tageskontext", row.get("daily_signal_direction") or "—")
        header[5].metric("Asset", ASSET_LABELS.get(row.get("asset_type"), row.get("asset_type")))
        st.markdown(f"**Research-Fazit:** {row.get('research_summary') or row.get('heuristic_summary') or '—'}")
        st.caption(
            f"Timing: {row.get('entry_timing_label') or '—'} · "
            f"{row.get('entry_timing_reason') or 'keine ausreichenden Eingaben'}"
        )
        st.markdown("#### Entry-These")
        if entry_thesis.get("available"):
            entry_metrics = st.columns(4)
            entry_metrics[0].metric("Timing", _number(entry_thesis.get("timing_score"), 0, "/100"))
            entry_metrics[1].metric("Trend", _number(entry_thesis.get("trend"), 0, "/100"))
            entry_metrics[2].metric("Regime", entry_thesis.get("regime") or "—")
            entry_metrics[3].metric(
                "Knife/Boden",
                entry_thesis.get("falling_knife_bottoming_status") or "—",
            )
            _render_points(
                "Warum das Timing konstruktiv wirken kann",
                entry_thesis.get("why_timing_may_be_good"),
                kind="success",
            )
            _render_points(
                "Benötigte Bestätigung",
                entry_thesis.get("what_confirms"),
            )
            _render_points(
                "Invalidation",
                entry_thesis.get("what_invalidates"),
                kind="error",
            )
            _render_points(
                "Stärkste Timing-Evidenz",
                entry_thesis.get("strongest_supporting_evidence"),
                kind="success",
            )
            _render_points(
                "Stärkste Gegenargumente",
                entry_thesis.get("strongest_counterarguments"),
                kind="warning",
            )
        else:
            st.warning("Keine vollständigen technischen Tagesdaten; keine Entry-These.")
        for action in row.get("research_actions") or []:
            tone = action.get("tone")
            text = action.get("text") or ""
            if tone == "neg":
                st.error(text)
            elif tone == "pos":
                st.success(text)
            else:
                st.info(text)

        analyst = row.get("analyst_context") or {}
        if analyst.get("available"):
            st.info(
                f"Analystenkonsens ({analyst.get('analyst_count')} Stimmen): "
                f"{analyst.get('consensus') or '—'} · Ziel {_number(analyst.get('target_price'), 2)} USD · "
                f"Abstand {_number(analyst.get('upside_pct'), 1, '%')}. "
                "Separater Analystenkontext, keine Modellprognose."
            )

        if row.get("falling_knife"):
            st.error(row["falling_knife"].get("warning"))
        if row.get("bottoming"):
            st.warning(
                "Spekulative Bodenbildungsbeobachtung: "
                + " · ".join(row["bottoming"].get("signals") or [])
            )
        if row.get("bull_thesis"):
            st.success(f"These/Chancen: {row['bull_thesis']}")
        if row.get("priced_in_note"):
            st.warning(row["priced_in_note"])
        for warning in row.get("risk_warnings") or []:
            st.error(warning)

        downside = row.get("downside_structure") or {}
        if downside:
            st.write(
                f"**Abwärtsstruktur:** Risiko {downside.get('risk') or '—'} · "
                f"{downside.get('verdict') or ''} · Unterstützung 1 "
                f"{_number(downside.get('support1'), 2)} USD "
                f"({_number(downside.get('support1_pct'), 1, '%')})"
            )
        scores = pd.DataFrame(
            [
                {
                    "completed-daily trend": row.get("longterm_score"),
                    "daily momentum context": row.get("daily_signal_direction"),
                    "company fundamental": valuation.get("fundamental_score"),
                    "value": valuation.get("value_score"),
                    "quality": valuation.get("quality_score"),
                    "growth": valuation.get("growth_score"),
                }
            ]
        )
        st.dataframe(scores, hide_index=True, width="stretch")
        if valuation.get("available"):
            st.caption("Fundamental: " + " · ".join(valuation.get("reasons") or []))
        else:
            st.caption(valuation.get("unavailable_reason") or "Fundamentaldaten nicht verfügbar.")
        st.markdown("#### Bewertungsthese")
        if valuation_thesis.get("available"):
            valuation_metrics = st.columns(4)
            valuation_metrics[0].metric(
                "Raw Value/Quality",
                _number(valuation_thesis.get("raw_score"), 1, "/100"),
            )
            valuation_metrics[1].metric(
                "Risikoabschlag",
                _number(valuation_thesis.get("risk_penalty"), 1),
            )
            valuation_metrics[2].metric(
                "Risikoadjustiert",
                _number(valuation_thesis.get("risk_adjusted_score"), 1, "/100"),
            )
            valuation_metrics[3].metric(
                "Jurisdiktion / Value Trap",
                f"{jurisdiction.get('level') or 'low'} / "
                f"{valuation_thesis.get('value_trap_risk') or '—'}",
            )
            st.caption(valuation_thesis.get("formula") or "")
            _render_points(
                "Warum es günstig aussieht",
                valuation_thesis.get("why_it_looks_cheap"),
                kind="success",
            )
            _render_points(
                "Warum der Abschlag gerechtfertigt sein kann",
                valuation_thesis.get("why_discount_may_be_justified"),
                kind="warning",
            )
            _render_points(
                "Stärkste positive Evidenz",
                valuation_thesis.get("strongest_positive_evidence"),
                kind="success",
            )
            _render_points(
                "Gegenargumente / Value-Trap-Risiken",
                valuation_thesis.get("strongest_counterarguments"),
                kind="error",
            )
        else:
            st.warning(
                valuation.get("unavailable_reason")
                or "Keine aktuellen vollständigen Fundamentaldaten; keine Bewertungsthese."
            )

        technical = pd.DataFrame(
            [{
                "RSI": row.get("rsi"),
                "MACD": row.get("macd"),
                "MACD-Signal": row.get("macd_signal"),
                "20T %": row.get("ret_20d"),
                "60T %": row.get("ret_60d"),
                "Abstand Hoch %": row.get("pct_from_high52"),
                "ATR %": row.get("atr_pct"),
                "Volatilität p.a. %": row.get("vol_annual_pct"),
                "Minervini": row.get("minervini_score"),
                "Weinstein": row.get("weinstein_label"),
            }]
        )
        st.dataframe(technical, hide_index=True, width="stretch")
        st.caption(
            f"Source timestamp: {row.get('bar_timestamp') or '—'} · "
            f"source interval: {row.get('source_interval') or '—'} · "
            f"currency converted from {row.get('currency')} at "
            f"{_number(row.get('fx_usd'), 8)} USD/unit"
        )
        with st.expander("Feature coverage and optional context"):
            st.json(row.get("feature_coverage") or {})
            if row.get("news"):
                st.write("Age-filtered issuer-feed headlines (context only):")
                for item in row["news"]:
                    st.write(
                        f"- {item.get('published_at', '—')} — {item.get('title', '')} "
                        f"({item.get('link', '')})"
                    )
            if row.get("macro_notes"):
                st.write("Macro heuristic context only:", " · ".join(row["macro_notes"]))
        scenarios = _scenario_table(row)
        if not scenarios.empty:
            st.write("Heuristische Szenariospannen – nicht statistisch kalibriert")
            st.dataframe(scenarios, hide_index=True, width="stretch")
        if row.get("next_earnings"):
            st.caption(
                f"Nächster Ergebnistermin: {row.get('next_earnings')} "
                f"({row.get('earnings_in_days')} Tage)"
            )


tabs = st.tabs(
    [
        "Sweet Spot",
        "Tages-Setups",
        "Unterbewertet",
        "Potenzial",
        "Einstiegs-Timing",
        "Fallende Messer",
        "Bodenbildung",
        "Risiken",
        "Alle suchen",
        "Datenqualität",
        "Paper",
        "Validierung",
        "Experten-Scores",
        "Trefferquote",
    ]
)

rankings = data.get("rankings_by_currency_asset") or {}
insight_categories = (data.get("insight_rankings") or {}).get("categories") or {}
rows_by_symbol = {
    row.get("symbol"): row for row in data.get("all", []) if row.get("symbol")
}


def _render_insight_category(category_key, *, key):
    category = insight_categories.get(category_key) or {}
    partitions = category.get("items_by_currency") or {}
    st.caption(
        f"{category.get('label') or category_key} · Formel: {category.get('formula') or '—'} · "
        "heuristic_unvalidated · keine Empfehlung"
    )
    currencies = [
        currency
        for currency in sorted(partitions)
        if partitions.get(currency)
    ]
    if not currencies:
        st.info("Keine Instrumente erfüllen aktuell die transparenten Mindestkriterien.")
        return
    default_index = currencies.index("USD") if "USD" in currencies else 0
    currency = st.selectbox(
        "Handelswährung",
        currencies,
        index=default_index,
        key=f"{key}_currency",
        help="Listen werden nicht währungsübergreifend gemischt.",
    )
    items = partitions[currency]
    overview = pd.DataFrame(
        [
            {
                "Rang": index,
                "Symbol": item.get("symbol"),
                "Vollständiger Name": (
                    rows_by_symbol.get(item.get("symbol")) or {}
                ).get("display_name_full"),
                "Hauptsitz (Provider) / Exposure": " / ".join(
                    filter(
                        None,
                        [
                            (rows_by_symbol.get(item.get("symbol")) or {}).get(
                                "headquarters_country"
                            ),
                            (rows_by_symbol.get(item.get("symbol")) or {}).get(
                                "economic_exposure_country"
                            ),
                        ],
                    )
                ),
                "Branche": (
                    rows_by_symbol.get(item.get("symbol")) or {}
                ).get("industry_display"),
                "Insight-Score": item.get("score"),
                "Raw": item.get("raw_score"),
                "Risikoabschlag": item.get("risk_penalty"),
                "Risiko": item.get("risk_level"),
                "Komponenten": " · ".join(
                    f"{name}: {_number(value, 1)}"
                    for name, value in (item.get("components") or {}).items()
                ),
                "Gründe": " · ".join(item.get("reasons") or []),
            }
            for index, item in enumerate(items, 1)
        ]
    )
    st.dataframe(overview, hide_index=True, width="stretch")
    symbols = [item.get("symbol") for item in items if item.get("symbol") in rows_by_symbol]
    if symbols:
        selected_symbol = st.selectbox("Detail", symbols, key=f"{key}_{currency}_symbol")
        _research_card(rows_by_symbol[selected_symbol], symbols.index(selected_symbol) + 1)


with tabs[0]:
    currencies = sorted(
        {
            row.get("currency")
            for row in rows_by_symbol.values()
            if (row.get("sweet_spot") or {}).get("combined_status")
            in {
                "in_zone_confirmed",
                "in_zone_risk_filtered",
                "approaching",
                "setup_waiting_confirmation",
            }
            and row.get("currency")
        }
    )
    st.caption(
        "Bestätigtes Grün zuerst, danach Amber-Beobachtungen · nach Handelswährung "
        "getrennt · Beobachtungszone, keine Ordermarke."
    )
    if not currencies:
        st.info("Keine bestätigten oder sich nähernden Sweet-Spot-Beobachtungen.")
    else:
        default_index = currencies.index("USD") if "USD" in currencies else 0
        currency = st.selectbox(
            "Handelswährung",
            currencies,
            index=default_index,
            key="sweet_currency",
        )
        currency_rows = [
            row for row in rows_by_symbol.values() if row.get("currency") == currency
        ]
        confirmed_rows = sorted(
            [
                row
                for row in currency_rows
                if (row.get("sweet_spot") or {}).get("combined_status")
                == "in_zone_confirmed"
            ],
            key=lambda row: (
                -(row.get("sweet_spot") or {}).get("reliability_score", 0),
                row.get("symbol") or "",
            ),
        )
        approaching_rows = sorted(
            [
                row
                for row in currency_rows
                if (row.get("sweet_spot") or {}).get("combined_status")
                in {
                    "in_zone_risk_filtered",
                    "approaching",
                    "setup_waiting_confirmation",
                }
            ],
            key=lambda row: (
                -(row.get("sweet_spot") or {}).get("reliability_score", 0),
                row.get("symbol") or "",
            ),
        )
        items = [
            {
                "symbol": row.get("symbol"),
                "score": (row.get("sweet_spot") or {}).get("reliability_score"),
            }
            for row in (*confirmed_rows, *approaching_rows)
        ]
        overview = pd.DataFrame(
            [
                {
                    "Rang": index,
                    "Symbol": item.get("symbol"),
                    "Vollständiger Name": (
                        rows_by_symbol.get(item.get("symbol")) or {}
                    ).get("display_name_full"),
                    "Status": (
                        (rows_by_symbol.get(item.get("symbol")) or {}).get("sweet_spot")
                        or {}
                    ).get("combined_status"),
                    "Evidenzqualität": item.get("score"),
                    "Branche": (
                        rows_by_symbol.get(item.get("symbol")) or {}
                    ).get("industry_display"),
                    "Land/Exposure": (
                        rows_by_symbol.get(item.get("symbol")) or {}
                    ).get("economic_exposure_country"),
                }
                for index, item in enumerate(items, 1)
            ]
        )
        st.dataframe(overview, hide_index=True, width="stretch")
        symbols = [
            item.get("symbol")
            for item in items
            if item.get("symbol") in rows_by_symbol
        ]
        if symbols:
            selected = st.selectbox("Detail", symbols, key="sweet_symbol")
            _research_card(rows_by_symbol[selected], symbols.index(selected) + 1)

with tabs[1]:
    _render_insight_category("daily_setups", key="daily")

with tabs[2]:
    _render_insight_category("undervalued_quality", key="value")

with tabs[3]:
    _render_insight_category("analyst_potential", key="potential")

with tabs[4]:
    _render_insight_category("entry_watchlist", key="entry")

with tabs[5]:
    _render_insight_category("falling_knives", key="knives")

with tabs[6]:
    _render_insight_category("bottoming_watch", key="bottom")

with tabs[7]:
    _render_insight_category("risk_watch", key="risk")

with tabs[8]:
    columns = [
        "symbol",
        "name",
        "display_name_full",
        "headquarters_country",
        "legal_domicile",
        "economic_exposure_country",
        "listing_country",
        "sector_display",
        "industry_display",
        "asset_type",
        "bar_date",
        "bar_age_days",
        "currency",
        "price",
        "radar_score",
        "longterm_score",
        "fundamental_score",
        "daily_signal_direction",
    ]
    frame = pd.DataFrame(
        [{column: row.get(column) for column in columns} for row in data.get("all", [])]
    )
    frame = frame.rename(
        columns={"radar_score": "local_partition_signal_score"}
    )
    query = st.text_input("Symbol, Name, Sektor oder Branche", "")
    if query and not frame.empty:
        match = frame.astype(str).apply(
            lambda column: column.str.contains(query, case=False, regex=False)
        ).any(axis=1)
        frame = frame[match]
    st.dataframe(frame, hide_index=True, width="stretch")
    st.caption(
        "The signal score is only comparable inside the same currency and asset-class "
        "partition; this table is not a global ranking."
    )
    if query and not frame.empty:
        symbols = frame["symbol"].dropna().tolist()
        if symbols:
            selected = st.selectbox("Detailansicht", symbols, key="search_detail")
            _research_card(rows_by_symbol[selected])

with tabs[10]:
    st.warning(
        "Simulation is UNVALIDATED and performance is non-actionable. Orders fill only "
        "on a completed bar dated at least two UTC dates after order observation, with "
        "a session open later than creation and configured costs. Corporate actions "
        "are best effort; legacy accounting remains explicitly marked."
    )
    try:
        portfolio = validate_portfolio_contract(
            load_json(PORTFOLIO, required=True, expected_type=dict)
        )
    except (PersistenceError, DataContractError) as exc:
        st.error(f"Portfolio cannot be read and was not reset: {exc}")
        portfolio = None
    if portfolio:
        paper_summary = data.get("paper") or {}
        simulation_metrics = st.columns(3)
        simulation_metrics[0].metric(
            "Simulation equity (non-actionable)",
            _number(paper_summary.get("equity"), 2, " USD"),
        )
        simulation_metrics[1].metric(
            "Cost-aware max drawdown",
            (
                "— (legacy frozen)"
                if paper_summary.get("legacy_migrated")
                else _number(paper_summary.get("max_drawdown_pct"), 2, "%")
            ),
        )
        simulation_metrics[2].metric(
            "Pending next-bar orders",
            paper_summary.get("pending_orders", 0),
        )
        st.json(
            {
                "schema": portfolio.get("schema", "legacy"),
                "schema_version": portfolio.get("schema_version"),
                "simulation_status": portfolio.get("simulation_status", "legacy_unvalidated"),
                "performance_actionable": portfolio.get("performance_actionable", False),
                "assumptions": portfolio.get("assumptions"),
                "legacy_migrated": portfolio.get("legacy_migrated", False),
            }
        )
        positions = portfolio.get("positions") or {}
        st.dataframe(
            pd.DataFrame(
                [
                    {
                        "symbol": symbol,
                        "quantity": position.get("quantity"),
                        "entry price": position.get("entry_price"),
                        "last price": position.get("last_price"),
                        "entry bar": position.get("entry_bar_date"),
                        "legacy": position.get("legacy"),
                    }
                    for symbol, position in positions.items()
                ]
            ),
            hide_index=True,
            width="stretch",
        )
        curve = pd.DataFrame(portfolio.get("equity_curve") or [])
        benchmark_columns = [
            column for column in ("bench_sp500", "bench_ndx", "bench_world") if column in curve
        ]
        if len(curve) >= 2 and "equity" in curve:
            if "as_of_bar_date" not in curve:
                curve["as_of_bar_date"] = None
            curve["as_of_bar_date"] = pd.to_datetime(curve["as_of_bar_date"])
            for column in benchmark_columns:
                date_column = f"{column}_bar_date"
                if date_column in curve:
                    aligned = pd.to_datetime(curve[date_column]) == curve["as_of_bar_date"]
                    curve.loc[~aligned, column] = None
                else:
                    curve[column] = None
            comparison = curve.dropna(subset=["equity", *benchmark_columns]).copy()
            if len(comparison) >= 2:
                common_start = comparison.iloc[0]
                rebased = pd.DataFrame(index=comparison["as_of_bar_date"])
                rebased["portfolio"] = comparison["equity"].to_numpy() / common_start["equity"] * 100
                for column in benchmark_columns:
                    rebased[column] = comparison[column].to_numpy() / common_start[column] * 100
                st.caption("All displayed series use the same first common observation.")
                st.line_chart(rebased)
        fills = [
            item for item in (portfolio.get("ledger") or []) if item.get("type") == "FILL"
        ]
        total_cost = sum(float(item.get("commission") or 0) for item in fills)
        st.metric("Recorded commissions", f"${total_cost:,.2f}")
        st.dataframe(pd.DataFrame(fills[-50:]), hide_index=True, width="stretch")

with tabs[11]:
    st.warning(
        "The deployed composite remains UNVALIDATED. The available backtest covers "
        "technical score only and must not be interpreted as alpha evidence."
    )
    try:
        backtest = load_json(BACKTEST, required=False, expected_type=dict, default={})
    except PersistenceError as exc:
        st.error(f"Backtest file is corrupt: {exc}")
        backtest = {}
    if backtest.get("schema") == "stock-radar-backtest":
        st.json(
            {
                "model_status": backtest.get("model_status"),
                "deployed_composite_status": backtest.get("deployed_composite_status"),
                "manifest": backtest.get("manifest"),
            }
        )
        st.dataframe(
            pd.DataFrame.from_dict(backtest.get("by_horizon") or {}, orient="index"),
            width="stretch",
        )
    elif backtest:
        st.info("Legacy backtest artifact detected; rerun it before relying on its methodology.")
    else:
        st.info("No v2 backtest has been run.")
    st.subheader("Probability release validation")
    probability_validation = data.get("probability_validation") or {}
    st.json(probability_validation)
    probability_models = probability_validation.get("models") or {}
    if probability_models:
        st.dataframe(
            pd.DataFrame.from_dict(probability_models, orient="index"),
            width="stretch",
        )
    st.subheader("Forward Validation")
    st.warning(
        "The ordered shadow model was rejected retrospectively and is being "
        "monitored prospectively. No shadow probabilities are shown; the current "
        "page remains baseline-only."
    )
    forward_metrics = st.columns(4)
    forward_metrics[0].metric("Status", forward_validation["status"])
    forward_metrics[1].metric("Weekly anchors", forward_validation["weeks_captured"])
    forward_metrics[2].metric(
        "Matured 1M",
        forward_validation["matured_outcomes"]["21"],
    )
    forward_metrics[3].metric(
        "Matured 12M",
        forward_validation["matured_outcomes"]["252"],
    )
    st.caption(
        "Earliest meaningful 1M assessment: "
        f"{forward_validation['schedule']['meaningful_1m_assessment_not_before']}; "
        "final 12M schedule: "
        f"{forward_validation['schedule']['final_12m_assessment_not_before']}. "
        "A pass can only create a candidate for independent review."
    )

with tabs[9]:
    st.subheader("Completed-bar age distribution")
    st.json(status.get("bar_age_distribution") or {})
    st.subheader("FX source status")
    st.dataframe(
        pd.DataFrame.from_dict(data.get("fx_status") or {}, orient="index"),
        width="stretch",
    )

with tabs[12]:
    expert_layer = data.get("expert_layer") or {}
    expert_rankings = expert_layer.get("rankings") or {}
    configured_weights = expert_rankings.get("weights") or {}
    st.warning(
        "Beide Composite-Scores sind heuristic_unvalidated, getrennt vom konservativen "
        "Radar-Ranking und keine Anlageberatung. Fehlende Faktoren werden nicht erfunden; "
        "die sichtbare Abdeckung sinkt entsprechend."
    )

    horizon_options = {
        "Long Term (6–24 Monate)": "long_term",
        "Short Term (Tage–3 Monate)": "short_term",
    }
    selected_label = st.radio(
        "Horizont",
        list(horizon_options),
        horizontal=True,
        key="expert_horizon",
    )
    selected_horizon = horizon_options[selected_label]
    default_weights = configured_weights.get(selected_horizon) or {}
    labels = {
        "value": "Bewertung",
        "quality": "Qualität",
        "growth": "Wachstum",
        "momentum": "Momentum",
        "sentiment": "Sentiment",
        "alternative_data": "Alt-Data",
        "technical_momentum": "Technik/Momentum",
        "catalysts": "Katalysatoren",
        "valuation": "Bewertung",
    }
    weight_columns = st.columns(min(3, max(1, len(default_weights))))
    tuned_weights = {}
    for index, (factor, default) in enumerate(default_weights.items()):
        tuned_weights[factor] = weight_columns[index % len(weight_columns)].slider(
            labels.get(factor, factor),
            min_value=0,
            max_value=100,
            value=int(round(default)),
            key=f"expert_weight_{selected_horizon}_{factor}",
        )

    def tuned_score(row):
        analysis = row.get("expert_analysis") or {}
        detail = analysis.get(selected_horizon) or {}
        components = detail.get("components") or {}
        available = [
            (components.get(factor, {}).get("value"), weight)
            for factor, weight in tuned_weights.items()
            if isinstance(components.get(factor, {}).get("value"), (int, float))
            and weight > 0
        ]
        denominator = sum(weight for _, weight in available)
        return (
            round(sum(value * weight for value, weight in available) / denominator, 1)
            if denominator
            else None
        )

    expert_rows = []
    for row in rows_by_symbol.values():
        analysis = row.get("expert_analysis") or {}
        detail = analysis.get(selected_horizon) or {}
        score = tuned_score(row)
        if score is None:
            continue
        expert_rows.append(
            {
                "Symbol": row.get("symbol"),
                "Name": row.get("display_name_full") or row.get("name"),
                "Score (getunt)": score,
                "Basis-Score": detail.get("score"),
                "Abdeckung %": detail.get("coverage_pct"),
                "Signal": analysis.get("signal"),
                "Konfidenz": analysis.get("evidence_quality"),
                "Fehlend": ", ".join(detail.get("missing_components") or []),
            }
        )
    expert_rows.sort(
        key=lambda item: (
            -(item["Score (getunt)"] or -1),
            item["Symbol"] or "",
        )
    )
    st.caption(
        "Slider ändern nur diese Ansicht. Dauerhafte Tagesgewichte stehen in "
        "`data/expert_score_weights.json` und werden beim nächsten Pipeline-Lauf angewendet."
    )
    st.dataframe(pd.DataFrame(expert_rows[:100]), hide_index=True, width="stretch")
    symbols = [item["Symbol"] for item in expert_rows[:100]]
    if symbols:
        selected = st.selectbox("Detail", symbols, key="expert_detail")
        _research_card(rows_by_symbol[selected], symbols.index(selected) + 1)

with tabs[13]:
    journal = ((data.get("expert_layer") or {}).get("recommendation_journal") or {})
    st.warning(
        "Trefferquote bedeutet hier ausschließlich positiver lokaler Kursreturn nach "
        "21/63/126/252 Handelssitzungen. Sie ist noch kein Alpha-, Benchmark- oder "
        "Kausalitätsnachweis."
    )
    summary_columns = st.columns(4)
    for column, horizon in zip(summary_columns, ("1m", "3m", "6m", "12m")):
        outcome = (journal.get("by_horizon") or {}).get(horizon) or {}
        column.metric(
            horizon.upper(),
            (
                f"{outcome.get('hit_rate_pct'):.1f}%"
                if isinstance(outcome.get("hit_rate_pct"), (int, float))
                else "noch offen"
            ),
            f"{outcome.get('evaluated', 0)} ausgewertet",
        )
    st.json(journal)
    st.subheader("Probability data health")
    st.json(data.get("probability_validation") or {"status": "unavailable"})
    st.subheader("Forward Validation")
    st.json(forward_validation)
    failed = status.get("failed_symbols") or {}
    st.subheader(f"Failed symbols ({len(failed)})")
    st.dataframe(
        pd.DataFrame(
            [{"symbol": symbol, "failure": reason} for symbol, reason in failed.items()]
        ),
        hide_index=True,
        width="stretch",
    )
