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

st.caption("⚠️ Research-Werkzeug, keine Anlageberatung. Signale sind rein technisch berechnet.")

tab1, tab2, tab3 = st.tabs(["🚀 Daytrading Top 10", "🏦 Langzeit Top 10", "🔎 Alle Werte"])

_DIR_COLOR = {"LONG": "🟢 LONG", "SHORT": "🔴 SHORT", "NEUTRAL": "⚪ NEUTRAL"}


def render_cards(picks, score_key, reasons_key, dir_key=None):
    for i, r in enumerate(picks, 1):
        with st.container(border=True):
            head = st.columns([4, 1, 1, 1, 1])
            title = f"#{i}  **{r['symbol']}** · {r['name'] or '—'}"
            if dir_key:
                title += f"  {_DIR_COLOR.get(r[dir_key], r[dir_key])}"
            head[0].markdown(f"### {title}")
            head[1].metric("Score", r[score_key])
            head[2].metric("Kurs", r["price"])
            head[3].metric("Tag %", r.get("daily_return_pct"))
            head[4].metric("RSI", r.get("rsi"))

            st.markdown("**Warum diese Chance:**")
            for reason in r[reasons_key]:
                st.markdown(f"- {reason}")

            news = r.get("news") or []
            if news:
                with st.expander(f"📰 News ({len(news)})"):
                    for n in news:
                        pub = f" · _{n['published']}_" if n.get("published") else ""
                        st.markdown(f"- [{n['title']}]({n['link']}){pub}")


with tab1:
    st.caption("Kurzfristige Momentum-, Breakout- und Volumen-Setups (Long **und** Short).")
    render_cards(data["top_daytrade"], "daytrade_score", "daytrade_reasons", "daytrade_direction")

with tab2:
    st.caption("Intakte Aufwärtstrends mit gesundem Rücksetzer – gute Einstiegspunkte.")
    render_cards(data["top_longterm"], "longterm_score", "longterm_reasons")

with tab3:
    df = pd.DataFrame([
        {k: v for k, v in r.items()
         if k not in ("daytrade_reasons", "longterm_reasons", "news")}
        for r in data["all"]
    ])
    if not df.empty:
        st.dataframe(
            df.sort_values("daytrade_score", ascending=False),
            use_container_width=True, hide_index=True,
        )
