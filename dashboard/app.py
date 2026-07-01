"""Stock Radar dashboard — compact cards with ELO rating, plain-German
reasoning, action ideas and a clearly separated Aschenbrenner section.
Run:  streamlit run dashboard/app.py
"""
import html
import json
from pathlib import Path

import pandas as pd
import streamlit as st

ROOT = Path(__file__).resolve().parent.parent
LATEST = ROOT / "data" / "output" / "latest.json"

st.set_page_config(page_title="Stock Radar", page_icon="📈", layout="wide")

CSS = """
<style>
.radar-grid{display:flex;flex-wrap:wrap;gap:10px;margin-top:6px;}
.card{flex:1 1 470px;max-width:100%;border:1px solid #e5e7eb;border-left:6px solid #6b7280;
  border-radius:12px;padding:10px 13px;background:#ffffff;font-size:13px;color:#111827;
  box-shadow:0 1px 2px rgba(0,0,0,.04);}
.card.asch{background:#faf5ff;border-color:#ddd6fe;}
.card .top{display:flex;align-items:center;gap:10px;flex-wrap:wrap;}
.card .elo{font-weight:800;font-size:22px;line-height:1;display:flex;flex-direction:column;}
.card .elo .rating{font-size:10px;font-weight:700;letter-spacing:.03em;text-transform:uppercase;}
.card .ttl{font-size:14px;flex:1;min-width:120px;}
.card .ttl b{font-size:15px;}
.card .meta{color:#6b7280;font-size:11px;margin:3px 0 5px;}
.card .bars{display:flex;gap:7px;margin:5px 0;}
.card .bar{flex:1;}
.card .bar .lbl{font-size:9px;color:#6b7280;margin-bottom:2px;text-align:center;}
.card .bar .track{background:#eef0f2;height:6px;border-radius:3px;overflow:hidden;}
.card .bar .fill{height:6px;border-radius:3px;}
.card .summary{color:#374151;margin:5px 0;line-height:1.4;}
.card .chips span{display:inline-block;padding:2px 9px;border-radius:11px;font-size:11px;
  margin:2px 5px 2px 0;font-weight:600;}
.chip-pos{background:#dcfce7;color:#166534;}
.chip-neg{background:#fee2e2;color:#991b1b;}
.chip-neutral{background:#f3f4f6;color:#374151;}
.asch-badge{background:#ede9fe;color:#6d28d9;border:1px solid #c4b5fd;padding:2px 9px;
  border-radius:11px;font-size:11px;font-weight:700;}
.card .news{font-size:11px;margin-top:6px;color:#6b7280;}
.card .news a{color:#2563eb;text-decoration:none;}
.grp-h{font-size:15px;font-weight:700;margin:14px 0 2px;}
</style>
"""
st.markdown(CSS, unsafe_allow_html=True)

st.title("📈 Stock Radar")

if not LATEST.exists():
    st.warning("Noch keine Analyse vorhanden. Bitte zuerst laufen lassen: `python -m src.analyze`")
    st.stop()

data = json.loads(LATEST.read_text(encoding="utf-8"))
meta = data.get("aschenbrenner_meta", {})

h1, h2, h3, h4 = st.columns(4)
h1.metric("Stand (UTC)", data["generated_at"].replace("T", " "))
h2.metric("Universum", data["universe_size"])
h3.metric("Analysiert", data["analyzed"])
h4.metric("Aschenbrenner-13F", meta.get("quarter") or "–")
st.caption("⚠️ Research-Werkzeug, keine Anlageberatung. ELO = Gesamt-Chancen-Rating aus Technik, "
           "Fundamentaldaten und Smart-Money-Signal.")


def _bar(label, val, color):
    v = val if isinstance(val, (int, float)) else 0
    return (f'<div class="bar"><div class="lbl">{label}</div>'
            f'<div class="track"><div class="fill" style="width:{max(0,min(100,v))}%;'
            f'background:{color}"></div></div></div>')


def _esc(x):
    return html.escape(str(x)) if x is not None else ""


def card_html(r, idx=None):
    color = r.get("radar_color", "#6b7280")
    asch = r.get("aschenbrenner")
    asch_cls = " asch" if asch else ""
    asch_badge = (f'<span class="asch-badge">{_esc(asch["label"])} · {asch.get("weight_pct")}%</span>'
                  if asch else "")
    bars = (_bar("Technik", r.get("longterm_score"), "#0ea5e9")
            + _bar("Value", r.get("value_score"), "#22c55e")
            + _bar("Quality", r.get("quality_score"), "#8b5cf6")
            + _bar("Growth", r.get("growth_score"), "#f59e0b"))
    chips = "".join(f'<span class="chip-{a["tone"]}">{_esc(a["text"])}</span>'
                    for a in (r.get("actions") or []))
    news = r.get("news") or []
    news_html = " · ".join(f'<a href="{_esc(n["link"])}" target="_blank">{_esc(n["title"][:58])}</a>'
                           for n in news[:2]) if news else ""
    news_div = f'<div class="news">📰 {news_html}</div>' if news_html else ""
    pe = r.get("pe")
    roe = r.get("roe_pct")
    meta_line = (f'{_esc(r.get("sector") or "")} · Kurs {r.get("price")} · '
                 f'KGV {pe if pe else "–"} · ROE {roe if roe is not None else "–"}%')
    rank = f"#{idx} " if idx else ""
    return (
        f'<div class="card{asch_cls}" style="border-left-color:{color}">'
        f'<div class="top">'
        f'<div class="elo" style="color:{color}">{r.get("radar_elo")}'
        f'<span class="rating">{_esc(r.get("radar_rating"))}</span></div>'
        f'<div class="ttl">{rank}<b>{_esc(r["symbol"])}</b> · {_esc(r.get("name") or "")}</div>'
        f'{asch_badge}</div>'
        f'<div class="meta">{meta_line}</div>'
        f'<div class="bars">{bars}</div>'
        f'<div class="summary">{_esc(r.get("plain_summary",""))}</div>'
        f'<div class="chips">{chips}</div>'
        f'{news_div}'
        f'</div>'
    )


def grid(picks, numbered=True):
    cards = "".join(card_html(r, i if numbered else None) for i, r in enumerate(picks, 1))
    st.markdown(f'<div class="radar-grid">{cards}</div>', unsafe_allow_html=True)


tabs = st.tabs([
    "🚀 Daytrading", "🏦 Langzeit", "📊 Fundamental", "🧠 Aschenbrenner", "🔎 Alle",
])

with tabs[0]:
    st.caption("Kurzfristige Momentum-, Breakout- und Volumen-Setups (Long **und** Short).")
    grid(data["top_daytrade"])

with tabs[1]:
    st.caption("Gesamt-Rating: langfristiger Trend (Technik) **+** Bewertung & Qualität (Fundamental).")
    grid(data["top_longterm"])

with tabs[2]:
    st.caption("Reine Fundamentalbewertung: Value, Quality, Growth + Greenblatt „Magic Formula\".")
    grid(data.get("top_fundamental", []))

with tabs[3]:
    holds = data.get("aschenbrenner_holdings", [])
    st.caption(f"🧠 Leopold Aschenbrenners Fonds *Situational Awareness LP* — Positionen aus dem "
               f"SEC-13F ({meta.get('quarter') or '?'}, eingereicht {meta.get('filed') or '?'}). "
               f"**Long** = setzt auf steigende Kurse · **Short-Wette** = Put-Optionen gegen die Aktie.")
    longs = [r for r in holds if r["aschenbrenner"]["stance"] == "LONG"]
    shorts = [r for r in holds if r["aschenbrenner"]["stance"] == "SHORT_BET"]
    mixed = [r for r in holds if r["aschenbrenner"]["stance"] == "MIXED"]
    if longs:
        st.markdown('<div class="grp-h">🟢 Long-Wetten (KI-Infrastruktur)</div>', unsafe_allow_html=True)
        grid(longs, numbered=False)
    if shorts:
        st.markdown('<div class="grp-h">🔴 Short-Wetten (wettet dagegen)</div>', unsafe_allow_html=True)
        grid(shorts, numbered=False)
    if mixed:
        st.markdown('<div class="grp-h">🟡 Gemischt (Put + Call / Absicherung)</div>', unsafe_allow_html=True)
        grid(mixed, numbered=False)

with tabs[4]:
    cols = ["symbol", "name", "sector", "radar_elo", "radar_rating", "price",
            "investment_score", "fundamental_score", "daytrade_score", "daytrade_direction",
            "value_score", "quality_score", "growth_score", "pe", "roe_pct"]
    df = pd.DataFrame([{k: r.get(k) for k in cols} for r in data["all"]])
    q = st.text_input("Filter (Symbol/Name/Sektor)", "")
    if q:
        m = df.apply(lambda row: q.lower() in " ".join(str(v).lower() for v in row.values), axis=1)
        df = df[m]
    if not df.empty:
        st.dataframe(df.sort_values("radar_elo", ascending=False),
                     use_container_width=True, hide_index=True)
