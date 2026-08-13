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

LATEST = ROOT / "data" / "output" / "latest.json"
PORTFOLIO = ROOT / "data" / "portfolio.json"
BACKTEST = ROOT / "data" / "backtest.json"

st.set_page_config(page_title="Stock Radar Research", page_icon="📊", layout="wide")
st.title("📊 Stock Radar — completed-daily-bar research")
st.caption(
    "Unvalidated heuristic research only. No intraday data, probability, confidence, "
    "profitability or buy/sell claim is made."
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
allowed, blockers = dashboard_gate(data)

if not allowed:
    st.error("Research cards are blocked because the freshness/completeness contract failed.")
    for reason in blockers:
        st.write(f"- {reason}")
    st.json({"data_status": status, "model_status": model})
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
    "heuristic volatility illustrations, not expected returns, medians, confidence "
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


def _research_card(row, rank=None):
    prefix = f"{rank}. " if rank is not None else ""
    with st.container(border=True):
        valuation = row.get("valuation_context") or {}
        st.subheader(f"{prefix}{row.get('symbol')} · {row.get('name') or ''}")
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
        zone = row.get("technical_observation_zone")
        if zone:
            st.caption(
                f"{zone.get('label')}: {_number(zone.get('lower'), 2)}–"
                f"{_number(zone.get('upper'), 2)} USD. {zone.get('note')}"
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
        "Tipps des Tages",
        "Unterbewertet",
        "Potenzial",
        "Guter Einstieg",
        "Fallende Messer",
        "Bodenbildung",
        "Risiken",
        "Alle suchen",
        "Datenqualität",
        "Paper",
        "Validierung",
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
                "Name": (rows_by_symbol.get(item.get("symbol")) or {}).get("name"),
                "Insight-Score": item.get("score"),
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
    _render_insight_category("daily_setups", key="daily")

with tabs[1]:
    _render_insight_category("undervalued_quality", key="value")

with tabs[2]:
    _render_insight_category("analyst_potential", key="potential")

with tabs[3]:
    _render_insight_category("entry_watchlist", key="entry")

with tabs[4]:
    _render_insight_category("falling_knives", key="knives")

with tabs[5]:
    _render_insight_category("bottoming_watch", key="bottom")

with tabs[6]:
    _render_insight_category("risk_watch", key="risk")

with tabs[7]:
    columns = [
        "symbol",
        "name",
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

with tabs[9]:
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

with tabs[10]:
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

with tabs[8]:
    st.subheader("Completed-bar age distribution")
    st.json(status.get("bar_age_distribution") or {})
    st.subheader("FX source status")
    st.dataframe(
        pd.DataFrame.from_dict(data.get("fx_status") or {}, orient="index"),
        width="stretch",
    )
    failed = status.get("failed_symbols") or {}
    st.subheader(f"Failed symbols ({len(failed)})")
    st.dataframe(
        pd.DataFrame(
            [{"symbol": symbol, "failure": reason} for symbol, reason in failed.items()]
        ),
        hide_index=True,
        width="stretch",
    )
