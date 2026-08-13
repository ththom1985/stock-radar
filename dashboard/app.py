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
        "Run `python -m src.analyze` to create a validated v2 snapshot."
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
                "horizon": item.get("label"),
                "reference change (heuristic)": _number(item.get("reference_change_pct"), 1, "%"),
                "range low USD": _number(item.get("range_low_price"), 2),
                "range high USD": _number(item.get("range_high_price"), 2),
                "status": item.get("model_status"),
            }
        )
    return pd.DataFrame(records)


def _research_card(row, rank=None):
    prefix = f"{rank}. " if rank is not None else ""
    with st.container(border=True):
        st.subheader(f"{prefix}{row.get('symbol')} · {row.get('name') or ''}")
        header = st.columns(5)
        header[0].metric("Completed-bar close (USD)", _number(row.get("price"), 2))
        header[1].metric("Completed bar", row.get("bar_date") or "—")
        header[2].metric("Bar age", _number(row.get("bar_age_days"), 0, " days"))
        header[3].metric("Core heuristic", _number(row.get("radar_score"), 0, "/100"))
        header[4].metric("Asset type", ASSET_LABELS.get(row.get("asset_type"), row.get("asset_type")))
        st.write(row.get("heuristic_summary") or "")
        scores = pd.DataFrame(
            [
                {
                    "completed-daily trend": row.get("longterm_score"),
                    "daily momentum context": row.get("daily_signal_direction"),
                    "company fundamental": row.get("fundamental_score"),
                    "value": row.get("value_score"),
                    "quality": row.get("quality_score"),
                    "growth": row.get("growth_score"),
                }
            ]
        )
        st.dataframe(scores, hide_index=True, width="stretch")
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
            st.write("Heuristic scenario range — not statistically calibrated")
            st.dataframe(scenarios, hide_index=True, width="stretch")


tabs = st.tabs(
    [
        "Company research",
        "ETFs / funds",
        "Crypto / other",
        "All instruments",
        "Paper simulation",
        "Validation",
        "Data health",
    ]
)

rankings = data.get("rankings_by_currency_asset") or {}


def _render_partitioned(asset_types, *, key):
    currencies = [
        currency
        for currency in sorted(rankings)
        if any((rankings.get(currency) or {}).get(asset_type) for asset_type in asset_types)
    ]
    if not currencies:
        st.info("No partition passed the configured completeness/feature gates.")
        return

    default_index = currencies.index("USD") if "USD" in currencies else 0
    currency = st.selectbox(
        "Trading currency",
        currencies,
        index=default_index,
        key=f"{key}_currency",
        help="Signals are comparable only inside the selected currency and asset class.",
    )
    for asset_type in asset_types:
        members = (rankings.get(currency) or {}).get(asset_type) or []
        if not members:
            continue
        st.subheader(
            f"{currency} · {ASSET_LABELS.get(asset_type, asset_type)} "
            "(local-currency technical partition)"
        )
        overview = pd.DataFrame(
            [
                {
                    "rank": index,
                    "symbol": row.get("symbol"),
                    "name": row.get("name"),
                    "completed close (USD)": row.get("price"),
                    "bar date": row.get("bar_date"),
                    "signal score": row.get("radar_score"),
                    "daily context": row.get("daily_signal_direction"),
                }
                for index, row in enumerate(members, 1)
            ]
        )
        st.dataframe(overview, hide_index=True, width="stretch")

        symbols = [row.get("symbol") for row in members if row.get("symbol")]
        if not symbols:
            continue
        selected_symbol = st.selectbox(
            "Instrument details",
            symbols,
            key=f"{key}_{currency}_{asset_type}_symbol",
        )
        selected = next(row for row in members if row.get("symbol") == selected_symbol)
        selected_rank = next(
            index
            for index, row in enumerate(members, 1)
            if row.get("symbol") == selected_symbol
        )
        _research_card(selected, selected_rank)


with tabs[0]:
    st.info(
        "Overall company ranking uses completed-daily technical context only. "
        "Generic fundamental bands are descriptive and excluded until robust "
        "sector-neutral point-in-time peer ranks exist."
    )
    _render_partitioned(["company_equity"], key="company")

with tabs[1]:
    st.info("Funds are ranked separately using completed-daily technical context only.")
    _render_partitioned(["etf_fund"], key="fund")

with tabs[2]:
    st.info("Crypto and other instruments are separate and have no company-fundamental score.")
    _render_partitioned(["crypto", "index_other", "unknown"], key="other")

with tabs[3]:
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
    query = st.text_input("Filter symbol/name/type", "")
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

with tabs[4]:
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

with tabs[5]:
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

with tabs[6]:
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
