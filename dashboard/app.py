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
.sector-badge{background:#e0f2fe;color:#075985;border:1px solid #bae6fd;padding:2px 9px;
  border-radius:11px;font-size:11px;font-weight:600;}
.card .qp{font-size:12px;color:#475569;margin:5px 0 2px;}
.card .qp b{color:#111827;}
.tier-h{font-size:18px;font-weight:800;margin:20px 0 2px;padding-bottom:3px;
  border-bottom:2px solid #e5e7eb;}
.tier-sub{font-size:12px;color:#6b7280;margin-bottom:4px;}
.card .stats{display:flex;gap:8px;align-items:center;flex-wrap:wrap;margin:7px 0 3px;font-size:12.5px;}
.card .stat{background:#f1f5f9;border-radius:9px;padding:3px 9px;color:#334155;}
.card .stat b{color:#0f172a;}
.urg{padding:3px 10px;border-radius:11px;font-size:11.5px;font-weight:800;}
.urg-urgent{background:#fee2e2;color:#991b1b;}
.urg-soon{background:#ffedd5;color:#9a3412;}
.urg-calm{background:#dcfce7;color:#166534;}
.day-h{font-size:19px;font-weight:800;margin:16px 0 4px;}
.rankbadge{font-size:24px;font-weight:900;min-width:46px;height:46px;border-radius:12px;
  display:flex;align-items:center;justify-content:center;color:#fff;flex:0 0 auto;}
.dir{font-size:15px;font-weight:800;padding:4px 12px;border-radius:12px;white-space:nowrap;}
.dir-up{background:#dcfce7;color:#15803d;}
.dir-down{background:#fee2e2;color:#b91c1c;}
.dir-side{background:#f1f5f9;color:#475569;}
.entry{padding:3px 10px;border-radius:11px;font-size:11.5px;font-weight:800;}
.entry-up{background:#dcfce7;color:#15803d;}
.entry-soon{background:#fef9c3;color:#854d0e;}
.entry-down{background:#fee2e2;color:#b91c1c;}
.entry-calm{background:#f1f5f9;color:#475569;}
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
- **🎬 Einstieg jetzt (0–100)** – wie gut der **aktuelle Kurs** zum Einsteigen ist: **hoch** = gesunder
  Rücksetzer / günstig bewertet / nahe am Tief in intaktem Trend; **niedrig** = heißgelaufen/überkauft/
  an den Höchstständen (schlechter Einstieg). Grün = gut, Gelb = okay, Rot = eher teuer.
- **🎯 Konfidenz · 🚀 Potenzial · ⏱️ Dringlichkeit** (oben auf jeder Karte): wie **sicher** das Signal
  ist (Übereinstimmung aller Faktoren), wie viel **Kurspotenzial** drin ist (Analysten-Ziel bzw. 12-Monats-
  Prognose), und wie **schnell** zu handeln ist (Sofort heute / Diese Woche / In Ruhe langfristig).
- **🏷️ Sektor-Badge & Branche** – Wirtschaftssektor und feinere Branche der Aktie.
- **🎯 Analysten** – Konsens-Empfehlung + Ø-Kursziel in % zum aktuellen Kurs (n = Anzahl Analysten).
- **Handlungs-Chips** – konkrete Ideen. 🟢 Grün = Chance · 🔴 Rot = Vorsicht · ⚪ Grau = neutral.
- **🟢/🔴 News (n)** – Stimmung der aktuellen Schlagzeilen (n = Anzahl), fließt in den Score ein.
  **📅 Zahlen** – nächster Termin der Geschäftszahlen; „in X T." = Tage bis dahin (davor oft mehr Schwankung).
- **📅 Projektion** – erwartete Kurs-Spanne aus der tatsächlichen Schwankungsbreite
  (**≈ 2 von 3 Fällen**), plus Richtungs-Tendenz & Konfidenz aus den Signalen.
  Zeiträume: **Daytrading 1 Tag / 1 Woche**, **Langzeit 1 / 3 / 12 / 24 Monate**.
  ⚠️ **Statistische Schätzung, keine Kursprognose oder Garantie.**
""")

mn = data.get("market_news", {})
if mn.get("headlines"):
    _mlabel = {"positiv": "🟢 positiv", "negativ": "🔴 negativ", "neutral": "⚪ neutral"}.get(
        mn.get("market_label"), "–")
    with st.expander(f"📰 Markt- & Wirtschafts-News — Gesamtstimmung: {_mlabel}"):
        for h in mn["headlines"]:
            src = f" · _{h.get('source')}_" if h.get("source") else ""
            st.markdown(f"- [{h.get('title')}]({h.get('link')}){src}")

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


def _qplabel(v):
    if not isinstance(v, (int, float)):
        return "–"
    return "hoch" if v >= 67 else "mittel" if v >= 45 else "niedrig"


def _entry(es):
    """(label, css) for the entry-timing traffic light."""
    if not isinstance(es, (int, float)):
        return "–", "calm"
    if es >= 70:
        return "sehr gut", "up"
    if es >= 55:
        return "gut", "up"
    if es >= 45:
        return "okay", "soon"
    if es >= 32:
        return "abwarten", "soon"
    return "teuer/heiß", "down"


def _proj_html(proj, context="invest"):
    """Clear, direction-first projection. No confusing symmetric ranges."""
    if not proj:
        return ""
    if context == "invest":
        parts = []
        for p in proj:
            lbl = p["label"].replace(" Monate", "M").replace(" Monat", "M")
            parts.append(f'{lbl} <b>{p["center_pct"]:+.0f}%</b>')
        return ('<div class="proj">📈 <b>Kursprognose</b> (Ø-Erwartung, keine Garantie): '
                + " · ".join(parts)
                + ' <span style="color:#94a3b8">– je weiter weg, desto unsicherer</span></div>')
    # trade: only the short-term swing magnitude (no false precision)
    wk = next((p for p in proj if "Woche" in p["label"]), proj[-1])
    swing = abs(wk["high_pct"] - wk["center_pct"])
    return (f'<div class="proj">⚡ Kurzfrist-Bewegungsspielraum (1 Woche): '
            f'typisch <b>±{swing:.0f}%</b></div>')


def _direction(r, context):
    """Return (label, css_class) for the up/down verdict."""
    if context == "trade":
        dd = r.get("daytrade_direction")
        if dd == "LONG":
            return "📈 STEIGEND", "up"
        if dd == "SHORT":
            return "📉 FALLEND", "down"
        return "➡️ SEITWÄRTS", "side"
    centre = None
    for p in (r.get("projection_long") or []):
        if str(p.get("label", "")).startswith("12"):
            centre = p.get("center_pct")
    base = centre if isinstance(centre, (int, float)) else ((r.get("investment_score") or 50) - 50)
    if base >= 3:
        return "📈 STEIGEND", "up"
    if base <= -3:
        return "📉 FALLEND", "down"
    return "➡️ SEITWÄRTS", "side"


def card_html(r, idx=None, context="invest"):
    color = r.get("radar_color", "#6b7280")
    proj_key = "projection_short" if context == "trade" else "projection_long"
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
    sector = r.get("sector") or ""
    industry = r.get("industry") or ""
    sector_badge = f'<span class="sector-badge">🏷️ {_esc(sector)}</span>' if sector else ""
    meta_line = ((f'{_esc(industry)} · ' if industry else "")
                 + f'Kurs {r.get("price")} · KGV {pe if pe else "–"} · '
                 f'ROE {roe if roe is not None else "–"}%')
    q, pot = r.get("quality"), r.get("potential")
    qp_line = (f'<div class="qp">🏅 Qualität: <b>{_qplabel(q)}</b> '
               f'({q if q is not None else "–"}/100) · 🚀 Potenzial: <b>{_qplabel(pot)}</b> '
               f'({pot if pot is not None else "–"}/100)</div>')

    # --- direction + headline stats ---
    dlabel, dcls = _direction(r, context)
    conv = r.get("conviction")
    if context == "invest":
        up = r.get("upside_pct")
        pot_stat = (f'<span class="stat">🚀 Kursziel <b>{up:+.0f}%</b></span>'
                    if isinstance(up, (int, float)) else "")
    else:
        atr = r.get("atr_pct")
        pot_stat = (f'<span class="stat">⚡ Tagesschwankung <b>±{atr:.0f}%</b></span>'
                    if isinstance(atr, (int, float)) else "")
    ut = r.get("urgency_tone") or "calm"
    es = r.get("entry_score")
    elab, ecls = _entry(es)
    entry_pill = (f'<span class="entry entry-{ecls}">🎬 Einstieg jetzt: {elab} '
                  f'({es if es is not None else "–"}/100)</span>')
    stat_row = (f'<div class="stats"><span class="dir dir-{dcls}">{dlabel}</span>'
                f'{entry_pill}'
                f'<span class="stat">🎯 Sicherheit <b>{conv if conv is not None else "–"}%</b></span>'
                f'{pot_stat}'
                f'<span class="urg urg-{ut}">{_esc(r.get("urgency") or "")}</span></div>')

    news_lbl = {"positiv": "🟢 News +", "negativ": "🔴 News −",
                "neutral": "⚪ News ="}.get(r.get("news_sentiment"))
    if news_lbl and r.get("news_mode") == "KI":
        news_lbl += " 🤖"
    llm_reason = r.get("news_llm_reason") or ""
    llm_div = f'<div class="meta">🤖 {_esc(llm_reason)}</div>' if llm_reason else ""
    earn = r.get("next_earnings")
    ed = r.get("earnings_in_days")
    earn_txt = ""
    if earn:
        earn_txt = f'📅 Zahlen {_esc(earn)}' + (f' (in {ed} T.)' if isinstance(ed, int) and ed >= 0 else "")
    hype_txt = ""
    if r.get("hype_rank"):
        fire = "🔥 " if r.get("hype_surging") else ""
        chg = r.get("hype_change_pct")
        hype_txt = (f'{fire}Reddit #{r.get("hype_rank")}'
                    + (f' ({chg:+.0f}%)' if isinstance(chg, (int, float)) else ""))
    ana_txt = ""
    au, an = r.get("analyst_upside_pct"), r.get("analyst_n")
    if context == "invest" and isinstance(au, (int, float)) and an:
        rk = {"strong_buy": "Kauf++", "buy": "Kauf", "hold": "Halten",
              "underperform": "Reduzieren", "sell": "Verkauf"}.get(
                  r.get("analyst_rating"), r.get("analyst_rating") or "")
        ana_txt = f'🎯 Analysten {_esc(rk)} {au:+.0f}% (n={an})'
    sig_bits = [b for b in [(f'{news_lbl} ({r.get("news_n")})' if news_lbl and r.get("news_n") else None),
                            ana_txt, hype_txt, earn_txt] if b]
    sig_line = f'<div class="meta">{" · ".join(sig_bits)}</div>' if sig_bits else ""

    rank_badge = (f'<div class="rankbadge" style="background:{color}">{idx}</div>' if idx else "")
    return (
        f'<div class="card{asch_cls}" style="border-left-color:{color}">'
        f'<div class="hd">'
        f'{rank_badge}'
        f'<div class="score">'
        f'<div class="num" style="color:{color}">{r.get("radar_score")}<small>/100</small></div>'
        f'<div class="stars" style="color:#f59e0b">{_stars(r.get("stars"))}</div>'
        f'<div class="elo">ELO {r.get("radar_elo")}</div>'
        f'</div>'
        f'<div class="name"><div class="tk">{_esc(r["symbol"])} · {_esc(r.get("name") or "")}</div>'
        f'<div class="rt" style="color:{color}">{_esc(r.get("radar_rating"))}</div></div>'
        f'{sector_badge}{asch_badge}</div>'
        f'{stat_row}'
        f'<div class="meta">{meta_line}</div>'
        f'{sig_line}'
        f'<div class="bars">{bars}</div>'
        f'{qp_line}'
        f'<div class="summary">{_esc(r.get("plain_summary",""))}</div>'
        f'{llm_div}'
        f'{_proj_html(r.get(proj_key), context)}'
        f'<div class="chips">{chips}</div>'
        f'{news_div}'
        f'</div>'
    )


def grid(picks, numbered=True, context="invest"):
    cards = "".join(card_html(r, i if numbered else None, context) for i, r in enumerate(picks, 1))
    st.markdown(f'<div class="radar-grid">{cards}</div>', unsafe_allow_html=True)


def grid_split(picks):
    """Trading view split into Long (rising) and Short (falling), each ranked."""
    longs = [r for r in picks if r.get("daytrade_direction") == "LONG"]
    shorts = [r for r in picks if r.get("daytrade_direction") == "SHORT"]
    if longs:
        st.markdown('<div class="day-h" style="color:#15803d">📈 Steigen erwartet – Long/Kaufen '
                    f'({len(longs)})</div>', unsafe_allow_html=True)
        grid(longs, numbered=True, context="trade")
    if shorts:
        st.markdown('<div class="day-h" style="color:#b91c1c">📉 Fallen erwartet – Short/Verkaufen '
                    f'({len(shorts)})</div>', unsafe_allow_html=True)
        grid(shorts, numbered=True, context="trade")
    if not longs and not shorts:
        grid(picks, numbered=True, context="trade")


tabs = st.tabs(["🎯 Heute", "🚀 Daytrading", "🏦 Langzeit", "📊 Fundamental", "🧠 Aschenbrenner",
                "🔥 Social", "💼 Paper-Depot", "🔎 Alle"])

with tabs[0]:
    st.caption("**Deine Top-Chancen heute**, nach Rang. Jede Karte zeigt oben: **Richtung** "
               "(📈 steigend / 📉 fallend), 🎯 **Sicherheit** (wie überzeugt), 🚀 **Potenzial/Ziel** und "
               "⏱️ **Dringlichkeit**. Die große Zahl links ist der Rang.")
    st.markdown('<div class="day-h">🚀 Zum Handeln – kurzfristig</div>', unsafe_allow_html=True)
    grid_split(data["top_daytrade"])
    st.markdown('<div class="day-h">🏦 Fürs Depot – langfristig (Rang 1 = beste Chance)</div>',
                unsafe_allow_html=True)
    grid(data["top_longterm"], numbered=True, context="invest")

with tabs[1]:
    st.caption("Kurzfristige Setups, getrennt in **Long (steigend)** und **Short (fallend)**, je nach "
               "Chance gereiht. Große Zahl = Rang.")
    grid_split(data["top_daytrade"])

with tabs[2]:
    st.caption("Langzeit-Rangliste (**#1 = beste Chance**): Technik + Fundamental + Analysten + News. "
               "Rating-Stufe steht je Karte (Top-Chance/Stark/…).")
    grid(data["top_longterm"], numbered=True)

with tabs[3]:
    st.caption("Reine Fundamentalbewertung (**#1 = beste**): Value, Quality, Growth + Greenblatt „Magic Formula\".")
    grid(data.get("top_fundamental", []), numbered=True)

with tabs[4]:
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

with tabs[5]:
    st.caption("🔥 Retail-Aufmerksamkeit auf Reddit (r/wallstreetbets & Co, via ApeWisdom) – "
               "frühe Momentum-/Meme-Bewegungen. 🔥 = Erwähnungen/Rang stark gestiegen. "
               "Vorsicht: Hype ≠ Qualität.")
    hype = data.get("top_hype", [])
    if hype:
        grid(hype)
    else:
        st.info("Aktuell keine deiner Titel in den Reddit-Trends (ApeWisdom ist US-Reddit-fokussiert).")

with tabs[6]:
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
                         width="stretch", hide_index=True)
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
            st.dataframe(ldf.head(40), width="stretch", hide_index=True)
        else:
            st.caption("Noch keine Trades.")

with tabs[7]:
    cols = ["symbol", "name", "sector", "industry", "radar_score", "radar_rating",
            "entry_score", "conviction", "upside_pct", "urgency", "quality", "potential",
            "analyst_rating", "analyst_upside_pct", "price", "investment_score",
            "fundamental_score", "daytrade_score", "daytrade_direction", "pe", "roe_pct"]
    df = pd.DataFrame([{k: r.get(k) for k in cols} for r in data["all"]])
    q = st.text_input("Filter (Symbol/Name/Sektor)", "")
    if q:
        m = df.apply(lambda row: q.lower() in " ".join(str(v).lower() for v in row.values), axis=1)
        df = df[m]
    if not df.empty:
        st.dataframe(df.sort_values("radar_score", ascending=False),
                     width="stretch", hide_index=True)
