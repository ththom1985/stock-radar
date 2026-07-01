"""Streamlit dashboard. Run:  streamlit run dashboard/app.py"""
import json
from pathlib import Path

import pandas as pd
import streamlit as st

ROOT = Path(__file__).resolve().parent.parent
LATEST = ROOT / "data" / "output" / "latest.json"

st.set_page_config(page_title="Stock Radar", page_icon="📈", layout="wide")
st.title("📈 Stock Radar – Tägliche Top-Chancen")

if not LATEST.exists():
    st.warning("Noch keine Analyse vorhanden. Bitte zuerst die Pipeline laufen lassen:\n\n"
               "`python -m src.analyze`")
    st.stop()

data = json.loads(LATEST.read_text(encoding="utf-8"))

c1, c2, c3 = st.columns(3)
c1.metric("Stand (UTC)", data["generated_at"].replace("T", " "))
c2.metric("Universum", data["universe_size"])
c3.metric("Analysiert", data["analyzed"])

st.caption("⚠️ Research-Werkzeug, keine Anlageberatung. Signale sind technisch & fundamental berechnet.")

_DIR = {"LONG": "🟢 LONG", "SHORT": "🔴 SHORT", "NEUTRAL": "⚪ NEUTRAL"}


def _sub_scores(r):
    """Small colored bars for Value/Quality/Growth if present."""
    cols = st.columns(4)
    for col, key, label in [
        (cols[0], "value_score", "Value"),
        (cols[1], "quality_score", "Quality"),
        (cols[2], "growth_score", "Growth"),
        (cols[3], "magic_score", "Magic"),
    ]:
        v = r.get(key)
        col.metric(label, f"{v:.0f}" if isinstance(v, (int, float)) else "–")


def render_cards(picks, score_key, score_label, reason_keys, dir_key=None, show_fund=False):
    for i, r in enumerate(picks, 1):
        with st.container(border=True):
            head = st.columns([4, 1, 1, 1, 1])
            title = f"#{i}  **{r['symbol']}** · {r.get('name') or '—'}"
            if dir_key:
                title += f"  {_DIR.get(r[dir_key], r[dir_key])}"
            head[0].markdown(f"### {title}")
            head[1].metric(score_label, r.get(score_key))
            head[2].metric("Kurs", r.get("price"))
            head[3].metric("KGV", r.get("pe") if r.get("pe") else "–")
            head[4].metric("ROE %", r.get("roe_pct") if r.get("roe_pct") is not None else "–")

            if show_fund:
                _sub_scores(r)

            reasons = []
            for k in reason_keys:
                reasons += r.get(k) or []
            if reasons:
                st.markdown("**Warum:**")
                for reason in reasons:
                    st.markdown(f"- {reason}")

            news = r.get("news") or []
            if news:
                with st.expander(f"📰 News ({len(news)})"):
                    for n in news:
                        pub = f" · _{n['published']}_" if n.get("published") else ""
                        st.markdown(f"- [{n['title']}]({n['link']}){pub}")


tab1, tab2, tab3, tab4 = st.tabs([
    "🚀 Daytrading Top 10", "🏦 Langzeit Top 10", "📊 Fundamental Top 10", "🔎 Alle Werte",
])

with tab1:
    st.caption("Kurzfristige Momentum-, Breakout- und Volumen-Setups (Long **und** Short).")
    render_cards(data["top_daytrade"], "daytrade_score", "Score",
                 ["daytrade_reasons"], dir_key="daytrade_direction")

with tab2:
    st.caption("Kombi-Score: langfristiger Trend (Technik) **+** Bewertung/Qualität (Fundamental).")
    render_cards(data["top_longterm"], "investment_score", "Invest",
                 ["longterm_reasons", "fundamental_reasons"], show_fund=True)

with tab3:
    st.caption("Reine Fundamentalbewertung: Value (KGV/KBV/EV-EBITDA), Quality (ROE/Margen/Schulden), "
               "Growth, plus Greenblatt „Magic Formula\".")
    render_cards(data.get("top_fundamental", []), "fundamental_score", "Fund",
                 ["fundamental_reasons"], show_fund=True)

with tab4:
    df = pd.DataFrame([
        {k: v for k, v in r.items()
         if k not in ("daytrade_reasons", "longterm_reasons", "fundamental_reasons", "news")}
        for r in data["all"]
    ])
    if not df.empty:
        st.dataframe(
            df.sort_values("investment_score", ascending=False),
            use_container_width=True, hide_index=True,
        )
