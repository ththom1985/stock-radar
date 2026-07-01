"""Stock Radar dashboard — readable compact cards with a 0-100 score, star
rating, plain-German reasoning, action ideas, a horizon projection and a
clearly separated Aschenbrenner section. Run: streamlit run dashboard/app.py
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
.radar-grid{display:flex;flex-wrap:wrap;gap:12px;margin-top:8px;}
.card{flex:1 1 500px;max-width:100%;border:1px solid #e5e7eb;border-left:7px solid #6b7280;
  border-radius:14px;padding:13px 16px;background:#ffffff;color:#111827;font-size:14px;
  box-shadow:0 1px 3px rgba(0,0,0,.05);}
.card.asch{background:#faf7ff;border-color:#e2d8fb;}
.card .hd{display:flex;align-items:center;gap:14px;flex-wrap:wrap;}
.card .score{display:flex;flex-direction:column;align-items:center;min-width:66px;}
.card .score .num{font-weight:800;font-size:30px;line-height:1;}
.card .score .num small{font-size:13px;font-weight:600;color:#9ca3af;}
.card .score .stars{font-size:13px;letter-spacing:1px;margin-top:2px;}
.card .score .elo{font-size:10px;color:#9ca3af;margin-top:2px;}
.card .hd .name{flex:1;min-width:130px;}
.card .hd .name .tk{font-size:17px;font-weight:700;}
.card .hd .name .rt{font-size:12px;font-weight:700;text-transform:uppercase;letter-spacing:.04em;}
.card .meta{color:#6b7280;font-size:12px;margin:7px 0 6px;}
.card .bars{display:flex;gap:9px;margin:7px 0;}
.card .bar{flex:1;}
.card .bar .lbl{font-size:10px;color:#6b7280;margin-bottom:3px;text-align:center;}
.card .bar .track{background:#eef0f2;height:7px;border-radius:4px;overflow:hidden;}
.card .bar .fill{height:7px;border-radius:4px;}
.card .summary{color:#1f2937;margin:8px 0;line-height:1.55;font-size:13.5px;}
.card .proj{background:#f8fafc;border:1px solid #eef2f7;border-radius:9px;padding:7px 10px;margin:7px 0;
  font-size:12.5px;color:#334155;}
.card .proj .row{margin:2px 0;}
.card .chips{margin-top:8px;}
.card .chips span{display:inline-block;padding:3px 11px;border-radius:12px;font-size:12px;
  margin:3px 6px 3px 0;font-weight:600;}
.chip-pos{background:#dcfce7;color:#166534;}
.chip-neg{background:#fee2e2;color:#991b1b;}
.chip-neutral{background:#f3f4f6;color:#374151;}
.asch-badge{background:#ede9fe;color:#6d28d9;border:1px solid #c4b5fd;padding:3px 10px;
  border-radius:12px;font-size:12px;font-weight:700;}
.card .news{font-size:12px;margin-top:8px;color:#6b7280;}
.card .news a{color:#2563eb;text-decoration:none;}
.grp-h{font-size:16px;font-weight:800;margin:16px 0 2px;}
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

with st.expander("ℹ️ Legende – was bedeuten die Zahlen?"):
    st.markdown("""
- **Score 0–100 & ⭐ Sterne** – die *eine* einfache Gesamtnote pro Aktie (Chance aus Technik +
  Fundamentaldaten + Smart-Money-Signal). Faustregel: **80+ Top · 65+ Stark · 50+ Solide ·
  35+ Neutral · darunter schwach**.
- **ELO** (kleine graue Zahl) – dieselbe Aussage auf einer Schach-Skala (~1000–2000), praktisch zum
  Feinvergleich zweier Aktien.
- **Balken:** **Trend** (technischer Aufwärtstrend) · **Value** (Bewertung, hoch = günstig) ·
  **Quality** (Ertragskraft/Bilanz) · **Growth** (Wachstum) – je 0–100.
- **🟣 Aschenbrenner-Badge** – Position im Fonds *Situational Awareness LP*. **LONG** = er setzt auf
  steigende Kurse · **Short-Wette** = Put-Option (gegen die Aktie) · **gemischt** = Absicherung.
- **Handlungs-Chips** – konkrete Ideen. 🟢 Grün = Chance · 🔴 Rot = Vorsicht · ⚪ Grau = neutral.
- **📅 Projektion** – erwartete Kurs-Spanne aus der tatsächlichen Schwankungsbreite
  (**≈ 2 von 3 Fällen**), plus Richtungs-Tendenz & Konfidenz aus den Signalen.
  Zeiträume: **Daytrading 1 Tag / 1 Woche**, **Langzeit 1 / 3 / 12 / 24 Monate**.
  ⚠️ **Statistische Schätzung, keine Kursprognose oder Garantie.**
""")

st.caption("⚠️ Research-Werkzeug, keine Anlageberatung.")

_ARROW = {"eher aufwärts": "↗", "eher abwärts": "↘", "eher seitwärts": "→"}


def _esc(x):
    return html.escape(str(x)) if x is not None else ""


def _bar(label, val, color):
    v = val if isinstance(val, (int, float)) else 0
    return (f'<div class="bar"><div class="lbl">{label}</div>'
            f'<div class="track"><div class="fill" style="width:{max(0,min(100,v))}%;'
            f'background:{color}"></div></div></div>')


def _stars(n):
    n = n or 0
    return "★" * n + "☆" * (5 - n)


def _proj_html(proj):
    if not proj:
        return ""
    rows = ""
    for p in proj:
        arr = _ARROW.get(p["direction"], "")
        rows += (f'<div class="row">📅 <b>{_esc(p["label"])}</b>: {arr} {_esc(p["direction"])} · '
                 f'Spanne {p["low_price"]}–{p["high_price"]} (±{p["high_pct"]}%, ≈2 von 3) · '
                 f'Konfidenz {_esc(p["confidence_label"])} ({p["confidence_pct"]}%)</div>')
    return f'<div class="proj">{rows}</div>'


def card_html(r, idx=None, proj_key="projection_long"):
    color = r.get("radar_color", "#6b7280")
    asch = r.get("aschenbrenner")
    asch_cls = " asch" if asch else ""
    asch_badge = (f'<span class="asch-badge">{_esc(asch["label"])} · {asch.get("weight_pct")}%</span>'
                  if asch else "")
    bars = (_bar("Trend", r.get("longterm_score"), "#0ea5e9")
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
        f'<div class="hd">'
        f'<div class="score">'
        f'<div class="num" style="color:{color}">{r.get("radar_score")}<small>/100</small></div>'
        f'<div class="stars" style="color:#f59e0b">{_stars(r.get("stars"))}</div>'
        f'<div class="elo">ELO {r.get("radar_elo")}</div>'
        f'</div>'
        f'<div class="name"><div class="tk">{rank}{_esc(r["symbol"])} · {_esc(r.get("name") or "")}</div>'
        f'<div class="rt" style="color:{color}">{_esc(r.get("radar_rating"))}</div></div>'
        f'{asch_badge}</div>'
        f'<div class="meta">{meta_line}</div>'
        f'<div class="bars">{bars}</div>'
        f'<div class="summary">{_esc(r.get("plain_summary",""))}</div>'
        f'{_proj_html(r.get(proj_key))}'
        f'<div class="chips">{chips}</div>'
        f'{news_div}'
        f'</div>'
    )


def grid(picks, numbered=True, proj_key="projection_long"):
    cards = "".join(card_html(r, i if numbered else None, proj_key) for i, r in enumerate(picks, 1))
    st.markdown(f'<div class="radar-grid">{cards}</div>', unsafe_allow_html=True)


tabs = st.tabs(["🚀 Daytrading", "🏦 Langzeit", "📊 Fundamental", "🧠 Aschenbrenner",
                "💼 Paper-Depot", "🔎 Alle"])

with tabs[0]:
    st.caption("Kurzfristige Momentum-, Breakout- und Volumen-Setups (Long **und** Short). "
               "Projektion auf **1 Tag & 1 Woche**.")
    grid(data["top_daytrade"], proj_key="projection_short")

with tabs[1]:
    st.caption("Gesamt-Rating: langfristiger Trend (Technik) **+** Bewertung & Qualität (Fundamental).")
    grid(data["top_longterm"])

with tabs[2]:
    st.caption("Reine Fundamentalbewertung: Value, Quality, Growth + Greenblatt „Magic Formula\".")
    grid(data.get("top_fundamental", []))

with tabs[3]:
    holds = data.get("aschenbrenner_holdings", [])
    st.caption(f"🧠 Leopold Aschenbrenners Fonds *Situational Awareness LP* — aus dem SEC-13F "
               f"({meta.get('quarter') or '?'}, eingereicht {meta.get('filed') or '?'}).")
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
    PORT = ROOT / "data" / "portfolio.json"
    if not PORT.exists():
        st.info("Noch kein Paper-Depot vorhanden – wird beim nächsten Analyse-Lauf erstellt.")
    else:
        pf = json.loads(PORT.read_text(encoding="utf-8"))
        pos = pf.get("positions", {})
        invested = sum(p.get("value_eur", 0) for p in pos.values())
        equity = pf.get("cash", 0) + invested
        ret = (equity / pf.get("start_capital", 10000) - 1) * 100
        st.caption(f"🤖 Vollautomatischer Selbst-Check: startet mit "
                   f"{pf.get('start_capital', 10000):,.0f} € virtuell, kauft täglich die Top-Tipps "
                   f"(Top {10}, gleichgewichtet), verkauft bei Stop −15 %, Ziel +30 %, Rating-Verfall "
                   f"oder Rauswurf aus den Top-Tipps. Wechselkurse ausgeklammert. Seit {pf.get('created')}.")

        m = st.columns(5)
        m[0].metric("Kontostand", f"{equity:,.0f} €", f"{ret:+.1f} %")
        m[1].metric("Realisiert G/V", f"{pf.get('realized_pnl', 0):,.0f} €")
        m[2].metric("Investiert", f"{invested:,.0f} €")
        m[3].metric("Cash", f"{pf.get('cash', 0):,.0f} €")
        m[4].metric("Positionen", len(pos))

        curve = pf.get("equity_curve", [])
        if len(curve) >= 2:
            cdf = pd.DataFrame(curve).set_index("date")[["equity"]]
            st.line_chart(cdf, height=220)
        elif curve:
            st.caption("📈 Kurve baut sich ab dem 2. Handelstag auf.")

        st.markdown("**Aktuelle Positionen**")
        if pos:
            hold = pd.DataFrame([{
                "Symbol": s, "Name": p.get("name"), "seit": p.get("entry_date"),
                "Kaufkurs": p.get("entry_price"), "akt. Kurs": p.get("last_price"),
                "Rendite %": p.get("pnl_pct"), "Einsatz €": p.get("stake_eur"),
                "Wert €": p.get("value_eur"), "G/V €": p.get("pnl_eur"),
            } for s, p in pos.items()])
            st.dataframe(hold.sort_values("G/V €", ascending=False),
                         use_container_width=True, hide_index=True)
        else:
            st.caption("Aktuell keine offenen Positionen.")

        st.markdown("**Handels-Protokoll** (neueste zuerst)")
        log = pf.get("trade_log", [])
        if log:
            ldf = pd.DataFrame(list(reversed(log)))[
                ["date", "action", "symbol", "name", "price", "stake_eur",
                 "value_eur", "pnl_eur", "pnl_pct", "reason"]]
            ldf.columns = ["Datum", "Aktion", "Symbol", "Name", "Kurs", "Einsatz €",
                           "Wert €", "G/V €", "G/V %", "Grund"]
            st.dataframe(ldf.head(40), use_container_width=True, hide_index=True)
        else:
            st.caption("Noch keine Trades.")

with tabs[5]:
    cols = ["symbol", "name", "sector", "radar_score", "radar_rating", "radar_elo", "price",
            "investment_score", "fundamental_score", "daytrade_score", "daytrade_direction",
            "value_score", "quality_score", "growth_score", "pe", "roe_pct"]
    df = pd.DataFrame([{k: r.get(k) for k in cols} for r in data["all"]])
    q = st.text_input("Filter (Symbol/Name/Sektor)", "")
    if q:
        m = df.apply(lambda row: q.lower() in " ".join(str(v).lower() for v in row.values), axis=1)
        df = df[m]
    if not df.empty:
        st.dataframe(df.sort_values("radar_score", ascending=False),
                     use_container_width=True, hide_index=True)
