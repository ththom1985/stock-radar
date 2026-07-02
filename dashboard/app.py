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
.card .expert-line{font-size:12px;margin:4px 0 2px;color:#334155;}
.card .expert-line b{color:#0f172a;}
.card details{margin-top:8px;font-size:12px;}
.card details summary{cursor:pointer;color:#2563eb;font-weight:700;padding:2px 0;}
.card .pro-grp{margin:7px 0;}
.card .pro-grp .h{font-weight:700;color:#0f172a;margin-bottom:2px;}
.card .pro-grid{display:grid;grid-template-columns:1fr 1fr;gap:1px 18px;color:#475569;}
.card .pro-grid b{color:#111827;}
.card .plan{margin:9px 0;padding:9px 11px;border-radius:9px;background:#f8fafc;border:1px solid #e2e8f0;}
.card .plan-pos{background:#f0fdf4;border-color:#bbf7d0;}
.card .plan-neg{background:#fef2f2;border-color:#fecaca;}
.card .plan-neutral{background:#fffbeb;border-color:#fde68a;}
.card .plan-act{font-weight:800;font-size:13.5px;color:#0f172a;margin-bottom:7px;}
.card .plan-grid{display:grid;grid-template-columns:1fr 1fr;gap:7px 14px;font-size:12px;color:#64748b;}
.card .plan-grid .pv{display:block;font-size:14px;font-weight:800;color:#0f172a;margin-top:1px;}
.card .plan-foot{display:flex;flex-wrap:wrap;gap:7px 10px;margin-top:9px;align-items:center;}
.card .plan-foot .gp{background:#16a34a;color:#fff;font-weight:800;padding:3px 10px;border-radius:999px;font-size:13px;}
.card .plan-foot .rk{color:#b91c1c;font-weight:700;font-size:12px;}
.card .plan-foot .crv{color:#334155;font-weight:700;font-size:12px;background:#e2e8f0;padding:2px 9px;border-radius:999px;}
.card .expert-badge{display:inline-block;background:#fef3c7;color:#92400e;border:1px solid #fcd34d;font-weight:700;font-size:11px;padding:2px 8px;border-radius:999px;margin:2px 0 0 4px;}
.card .vol-badge{display:inline-block;background:#fef2f2;color:#b91c1c;border:1px solid #fecaca;font-weight:700;font-size:11px;padding:2px 8px;border-radius:999px;margin:2px 0 0 4px;}
.card .riskbar{margin:8px 0 0;padding:7px 11px;border-radius:8px;background:#fef2f2;border:1px solid #fca5a5;border-left:4px solid #dc2626;color:#b91c1c;font-weight:700;font-size:12px;line-height:1.5;}
.card .entry-why{margin:6px 0 0;font-size:12px;font-weight:600;padding:4px 9px;border-radius:7px;}
.card .entry-why-up{background:#f0fdf4;color:#15803d;}
.card .entry-why-soon{background:#fffbeb;color:#a16207;}
.card .entry-why-down{background:#fef2f2;color:#b91c1c;}
.card .px{font-size:13px;color:#0f172a;margin-top:3px;}
.card .px b{font-size:16px;font-weight:800;}
.card .px .px-up{font-weight:800;color:#16a34a;}
.card .px .px-down{font-weight:800;color:#dc2626;}
.card .downside{margin:6px 0 0;font-size:12px;font-weight:600;padding:5px 9px;border-radius:7px;line-height:1.5;}
.card .downside b{font-weight:800;}
.card .downside-up{background:#f0fdf4;color:#15803d;}
.card .downside-soon{background:#fffbeb;color:#a16207;}
.card .downside-down{background:#fef2f2;color:#b91c1c;}
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
- **🎬 Einstieg jetzt (0–100) + „Timing"-Zeile** – wie gut der **aktuelle Moment** zum Kaufen ist, nach
  Profi-Logik: **Trend-Regime zuerst** (Dips kauft man nur im Aufwärtstrend – im Abwärtstrend ist jeder
  Rücksetzer Fallende-Messer-Risiko und wird stark abgewertet), **kontrollierter Rücksetzer** an einen
  steigenden Durchschnitt (nicht überdehnt/nachlaufend), **Momentum-Bestätigung** (dreht MACD/Stochastik
  wieder hoch, Kurs über EMA9?) und **Malus, wenn kurzfristig noch fallend**. Die **grün/gelb/rote
  Timing-Zeile** sagt in Klartext, *warum* – z.B. „gesunder Rücksetzer, Momentum dreht" vs. „noch fallend –
  auf Stabilisierung warten".
- **🎯 Sicherheit · 💰 Gewinnpotenzial · ⏱️ Dringlichkeit** (oben auf jeder Karte): wie **sicher** das Signal
  ist (Übereinstimmung aller Faktoren), das **Gewinnpotenzial in +%** aus dem Handlungsplan, und wie
  **schnell** zu handeln ist (Sofort heute / Diese Woche / In Ruhe langfristig).
- **💶 Kurs** (im Karten-Kopf) – der aktuelle Kurs (Tagesschluss, ~15 Min verzögert) mit Tagesveränderung
  (grün/rot). Währung je nach Börse (US = $, .DE = €, …), ohne Symbol dargestellt.
- **📉 Abwärtsrisiko** (grün/gelb/rote Zeile) – **wie weit könnte die Aktie noch fallen?** Zeigt die
  **nächste(n) Unterstützung(en)** unter dem Kurs (Kurslevel + % Abstand) und ein Urteil: *„Boden
  wahrscheinlicher"* (nahe Unterstützung + Momentum dreht) vs. *„viel Luft nach unten – Rückschlagrisiko"*
  vs. *„Abwärtstrend intakt – weiterer Rückgang wahrscheinlich"*. Genau die Frage „ist der Boden nah oder
  fällt sie noch weiter?".
- **⚠️ Rote Warn-Leiste** (über dem Handlungsplan) – macht Risiken sichtbar, die ein hoher Score sonst
  verdeckt: **„Spitzenzyklus möglich"** (zyklische Branche wie Chips/Rohstoffe/Auto mit explodierten
  Gewinnen → niedriges KGV kann trügen, Gewinne evtl. nicht dauerhaft) und **„Langfristig positiv, aber
  kurzfristig fallend"** (das Timing ist schlecht – du kaufst in fallende Kurse). Dazu das rote Badge
  **⚠️ Sehr schwankungsstark** bei hoher Tagesschwankung/Beta.
- **📋 Handlungsplan** (farbiger Kasten) – die **konkrete Empfehlung mit echten Kursen**: **🎬 Einstieg**
  (von–bis, gute Kaufzone aus Unterstützungen + ATR) · **🎯 Ausstieg/Ziel** (von–bis, nächste Widerstände
  bzw. Analysten-Ziel) · **🛑 Stop-Loss** (Absicherung, 1×ATR unter der Zone) · **⏳ Haltedauer** (geschätzte
  Zeit bis zum Ziel). Unten: **💰 Gewinnpotenzial in +%** (eine klare positive Zahl, kein ±), **Risiko** und
  **Chance-Risiko-Verhältnis** (z.B. 3,6:1 = 3,6-mal mehr Chance als Risiko; ab ~2:1 attraktiv). Grün =
  kaufbar · Gelb = gestaffelt/abwarten · Rot = meiden.
- **⭐ Experten-Badge** – dieser Titel wurde zuletzt von echten Experten/Medien empfohlen (Maydorn/
  Der Aktionär, Handelsblatt, DZ Bank, LYNX, echtgeld.tv, „Alles auf Aktien", Aktienwelt360, Abilitato).
  Eigener Tab **⭐ Experten** listet alle – nach Radar-Score gereiht, filterbar nach Quelle. Keine
  Kaufempfehlung: das Radar bewertet jeden Titel unabhängig.
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


def _fmt(x, dec=0, suf=""):
    if isinstance(x, bool):
        return "ja" if x else "nein"
    if isinstance(x, (int, float)):
        return f"{x:.{dec}f}{suf}"
    return _esc(x) if x else "–"


def _altman_zone(z):
    if not isinstance(z, (int, float)):
        return ""
    return " (sicher)" if z > 2.99 else " (Graubereich)" if z >= 1.81 else " (kritisch!)"


def _expert_line(r):
    bits = []
    met = len(r.get("minervini_met") or [])
    tot = met + len(r.get("minervini_failed") or [])
    if tot:
        bits.append(f'Minervini <b>{met}/{tot}</b>')
    wl = r.get("weinstein_label")
    if wl:
        bits.append(f'<b>{_esc(wl.split("–")[0].strip())}</b>')
    if isinstance(r.get("piotroski"), (int, float)):
        bits.append(f'Piotroski <b>{r["piotroski"]}/9</b>')
    if isinstance(r.get("altman_z"), (int, float)):
        bits.append(f'Altman-Z <b>{r["altman_z"]}</b>{_altman_zone(r["altman_z"])}')
    return f'<div class="expert-line">📐 {" · ".join(bits)}</div>' if bits else ""


def _pro_details(r):
    def row(label, val):
        return f'<div>{label}: <b>{val}</b></div>'

    trend = "".join([
        row("ADX (Trendstärke)", _fmt(r.get("adx"), 0)),
        row("+DI / −DI", f'{_fmt(r.get("plus_di"),0)} / {_fmt(r.get("minus_di"),0)}'),
        row("EMA9 &gt; EMA21", _fmt(r.get("ema9_above_21"))),
        row("Ichimoku", _fmt(r.get("ichimoku"))),
        row("Supertrend", "aufwärts" if r.get("supertrend_up") else "abwärts" if r.get("supertrend_up") is not None else "–"),
        row("Parabolic SAR", "long" if r.get("psar_bull") else "short" if r.get("psar_bull") is not None else "–"),
        row("Trend-Score", _fmt(r.get("tech_trend"), 0, "/100")),
    ])
    mom = "".join([
        row("RSI (14)", _fmt(r.get("rsi"), 0)),
        row("Stochastik %K/%D", f'{_fmt(r.get("stoch_k"),0)} / {_fmt(r.get("stoch_d"),0)}'),
        row("Williams %R", _fmt(r.get("williams_r"), 0)),
        row("CCI (20)", _fmt(r.get("cci"), 0)),
        row("ROC (10)", _fmt(r.get("roc10"), 1, "%")),
        row("Aroon ↑/↓", f'{_fmt(r.get("aroon_up"),0)} / {_fmt(r.get("aroon_down"),0)}'),
        row("Momentum-Score", _fmt(r.get("tech_momentum"), 0, "/100")),
    ])
    vol = "".join([
        row("OBV", "steigend" if r.get("obv_rising") else "fallend" if r.get("obv_rising") is not None else "–"),
        row("MFI (14)", _fmt(r.get("mfi"), 0)),
        row("Chaikin Money Flow", _fmt(r.get("cmf"), 2)),
        row("Rel. Volumen", _fmt(r.get("rvol"), 1, "×")),
        row("Volumen-Score", _fmt(r.get("tech_volume"), 0, "/100")),
    ])
    risk = "".join([
        row("ATR (Tagesspanne)", _fmt(r.get("atr_pct"), 1, "%")),
        row("Volatilität p.a.", _fmt(r.get("vol_annual_pct"), 0, "%")),
        row("Max. Drawdown 1J", _fmt(r.get("max_drawdown_pct"), 0, "%")),
        row("Beta", _fmt(r.get("beta"), 2)),
        row("Position 52W-Range", _fmt(r.get("range_pos_pct"), 0, "%")),
        row("Nächstes Fib-Level", _fmt(r.get("fib_nearest"))),
        row("Pivot / R1 / S1", f'{_fmt(r.get("pivot"),2)} / {_fmt(r.get("pivot_r1"),2)} / {_fmt(r.get("pivot_s1"),2)}'),
    ])
    fund = "".join([
        row("Piotroski F-Score", _fmt(r.get("piotroski"), 0, "/9")),
        row("Altman Z-Score", _fmt(r.get("altman_z"), 2) + _altman_zone(r.get("altman_z"))),
        row("Graham-Number", _fmt(r.get("graham_number"), 2)),
        row("Graham-Sicherheitsmarge", _fmt(r.get("graham_margin_pct"), 0, "%")),
        row("FCF-Rendite", _fmt(r.get("fcf_yield_pct"), 1, "%")),
        row("Rule of 40", _fmt(r.get("rule40"), 0)),
        row("Magic-Formula-Rang", _fmt(r.get("magic_score"), 0, "/100")),
    ])
    mv_met = r.get("minervini_met") or []
    minervini = (f'<div class="pro-grp"><div class="h">Minervini-Trendtemplate (erfüllt)</div>'
                 f'<div>{" · ".join(_esc(m) for m in mv_met) or "–"}</div></div>') if mv_met else ""

    return (
        '<details><summary>🔬 Profi-Analyse (alle Indikatoren)</summary>'
        f'<div class="pro-grp"><div class="h">📈 Trend</div><div class="pro-grid">{trend}</div></div>'
        f'<div class="pro-grp"><div class="h">⚡ Momentum</div><div class="pro-grid">{mom}</div></div>'
        f'<div class="pro-grp"><div class="h">📊 Volumen</div><div class="pro-grid">{vol}</div></div>'
        f'<div class="pro-grp"><div class="h">🎢 Volatilität &amp; Marken</div><div class="pro-grid">{risk}</div></div>'
        f'<div class="pro-grp"><div class="h">🏛️ Fundamental (Profi)</div><div class="pro-grid">{fund}</div></div>'
        f'{minervini}'
        '</details>'
    )


def _entry_why(r):
    """Plain-language timing verdict shown right under the header stats."""
    why = r.get("entry_reason")
    if not why:
        return ""
    es = r.get("entry_score")
    tone = "up" if isinstance(es, (int, float)) and es >= 55 else \
           "down" if isinstance(es, (int, float)) and es < 38 else "soon"
    return f'<div class="entry-why entry-why-{tone}">🎬 Timing: {_esc(why)}</div>'


def _downside_html(r):
    """How far it could fall to real support — answers 'fällt sie noch weiter?'"""
    dn = r.get("downside")
    if not dn:
        return ""
    risk = dn.get("risk")
    tone = {"hoch": "down", "mittel": "soon", "gering": "up"}.get(risk, "soon")
    s1, s1p = dn.get("support1"), dn.get("support1_pct")
    s2, s2p = dn.get("support2"), dn.get("support2_pct")
    sup = ""
    if s1 is not None:
        sup = f' Nächste Unterstützung <b>{s1}</b> ({s1p:+.0f} %)'
        if s2 is not None:
            sup += f', dann <b>{s2}</b> ({s2p:+.0f} %)'
    return (f'<div class="downside downside-{tone}">📉 Abwärtsrisiko <b>{_esc(risk)}</b> – '
            f'{_esc(dn.get("verdict",""))}{sup}</div>')


def _risk_bar(r):
    """Loud red warnings that a high score can hide (cyclical peak, bad timing)."""
    rw = r.get("risk_warnings") or []
    if not rw:
        return ""
    return f'<div class="riskbar">{"<br>".join(_esc(w) for w in rw)}</div>'


def _plan_html(r, context="invest"):
    """Prominent, number-based action plan: entry / target / stop / hold +
    a single positive profit potential (no ± ranges)."""
    tp = r.get("trade_plan_short") if context == "trade" else r.get("trade_plan_long")
    if not tp:
        return ""
    pot, risk, rrr = tp.get("potential_pct"), tp.get("risk_pct"), tp.get("rrr")
    is_short = tp.get("side") == "short"
    buy_lbl = "Verkaufen (Short)" if is_short else "Einstieg – kaufen"
    sell_lbl = "Eindecken (Ziel)" if is_short else "Ausstieg – Ziel-Zone"
    pot_txt = f'+{pot:.0f}%' if isinstance(pot, (int, float)) else '–'
    risk_txt = f'−{risk:.0f}%' if isinstance(risk, (int, float)) else '–'
    rrr_txt = (f'{rrr:.1f}:1'.replace('.', ',')) if isinstance(rrr, (int, float)) else '–'
    tone = tp.get("action_tone", "neutral")
    return (
        f'<div class="plan plan-{tone}">'
        f'<div class="plan-act">{_esc(tp.get("action",""))}</div>'
        f'<div class="plan-grid">'
        f'<div>🎬 {buy_lbl}<span class="pv">{tp.get("entry_low")} – {tp.get("entry_high")}</span></div>'
        f'<div>🎯 {sell_lbl}<span class="pv">{tp.get("target_low")} – {tp.get("target_high")}</span></div>'
        f'<div>🛑 Stop-Loss<span class="pv">{tp.get("stop")}</span></div>'
        f'<div>⏳ Haltedauer<span class="pv">{_esc(tp.get("hold",""))}</span></div>'
        f'</div>'
        f'<div class="plan-foot">'
        f'<span class="gp">💰 Gewinnpotenzial {pot_txt}</span>'
        f'<span class="rk">Risiko {risk_txt}</span>'
        f'<span class="crv">Chance-Risiko {rrr_txt}</span>'
        f'</div>'
        f'</div>'
    )


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
    # trade: the concrete plan box already carries entry/target/stop — no ± here
    return ""


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
    esrc = r.get("expert_sources") or []
    expert_badge = (f'<span class="expert-badge">⭐ {_esc(" · ".join(esrc))}</span>' if esrc else "")
    atrp, beta = r.get("atr_pct"), r.get("beta")
    _volatile = (isinstance(atrp, (int, float)) and atrp >= 5) or (isinstance(beta, (int, float)) and beta >= 1.8)
    vol_badge = ""
    if _volatile:
        swing = f' · ±{atrp:.0f}%/Tag' if isinstance(atrp, (int, float)) else ""
        vol_badge = f'<span class="vol-badge">⚠️ Sehr schwankungsstark{swing}</span>'
    meta_line = ((f'{_esc(industry)} · ' if industry else "")
                 + f'KGV {pe if pe else "–"} · '
                 f'ROE {roe if roe is not None else "–"}%')
    px = r.get("price")
    dchg = r.get("daily_return_pct")
    chg_txt = ""
    if isinstance(dchg, (int, float)):
        pcls = "up" if dchg >= 0 else "down"
        chg_txt = f' <span class="px-{pcls}">{dchg:+.1f}%</span>'
    price_line = f'<div class="px">💶 Kurs <b>{px}</b>{chg_txt}</div>' if px is not None else ""
    q, pot = r.get("quality"), r.get("potential")
    qp_line = (f'<div class="qp">🏅 Qualität: <b>{_qplabel(q)}</b> '
               f'({q if q is not None else "–"}/100) · 🚀 Potenzial: <b>{_qplabel(pot)}</b> '
               f'({pot if pot is not None else "–"}/100)</div>')

    # --- direction + headline stats ---
    dlabel, dcls = _direction(r, context)
    conv = r.get("conviction")
    if context == "invest":
        tp = r.get("trade_plan_long") or {}
        gp = tp.get("potential_pct")
        pot_stat = (f'<span class="stat">💰 Gewinnpotenzial <b>+{gp:.0f}%</b></span>'
                    if isinstance(gp, (int, float)) else "")
    else:
        tp = r.get("trade_plan_short") or {}
        gp = tp.get("potential_pct")
        pot_stat = (f'<span class="stat">💰 Gewinnpotenzial <b>+{gp:.0f}%</b></span>'
                    if isinstance(gp, (int, float)) else "")
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
        f'{price_line}'
        f'<div class="rt" style="color:{color}">{_esc(r.get("radar_rating"))}</div></div>'
        f'{sector_badge}{expert_badge}{vol_badge}{asch_badge}</div>'
        f'{stat_row}'
        f'{_entry_why(r)}'
        f'{_downside_html(r)}'
        f'{_risk_bar(r)}'
        f'{_plan_html(r, context)}'
        f'<div class="meta">{meta_line}</div>'
        f'{sig_line}'
        f'<div class="bars">{bars}</div>'
        f'{qp_line}'
        f'{_expert_line(r)}'
        f'<div class="summary">{_esc(r.get("plain_summary",""))}</div>'
        f'{llm_div}'
        f'{_proj_html(r.get(proj_key), context)}'
        f'<div class="chips">{chips}</div>'
        f'{news_div}'
        f'{_pro_details(r)}'
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
                "🔥 Social", "💼 Paper-Depot", "🔎 Alle", "⭐ Experten"])

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
    df = pd.DataFrame([{**{k: r.get(k) for k in cols},
                        "experten": " · ".join(r.get("expert_sources") or [])}
                       for r in data["all"]])
    q = st.text_input("Filter (Symbol/Name/Sektor)", "")
    if q:
        m = df.apply(lambda row: q.lower() in " ".join(str(v).lower() for v in row.values), axis=1)
        df = df[m]
    if not df.empty:
        st.dataframe(df.sort_values("radar_score", ascending=False),
                     width="stretch", hide_index=True)

with tabs[8]:
    st.caption("⭐ **Aktien, die echte Experten & Medien zuletzt empfohlen haben** — recherchiert aus "
               "öffentlich zugänglichen Quellen (Maydorn / Der Aktionär, Handelsblatt, DZ Bank, LYNX, "
               "echtgeld.tv, „Alles auf Aktien\", Aktienwelt360, Abilitato). Jede Karte trägt ihr "
               "⭐-Herkunfts-Badge. **Wichtig:** Das ist KEINE Kaufempfehlung — das Radar bewertet jeden "
               "Titel unabhängig; die Reihung folgt dem eigenen Score (#1 = bester Radar-Score).")
    picks = [r for r in data["all"] if r.get("expert_sources")]
    all_sources = sorted({s for r in picks for s in (r.get("expert_sources") or [])})
    c1, c2 = st.columns([2, 3])
    sel = c1.selectbox("Quelle filtern", ["Alle Quellen"] + all_sources)
    if sel != "Alle Quellen":
        picks = [r for r in picks if sel in (r.get("expert_sources") or [])]
    picks = sorted(picks, key=lambda r: r.get("radar_score") or 0, reverse=True)
    c2.markdown(f"<div style='padding-top:26px;color:#64748b'>{len(picks)} Titel · "
                f"{len(all_sources)} Quellen</div>", unsafe_allow_html=True)
    grid(picks, numbered=True, context="invest")
