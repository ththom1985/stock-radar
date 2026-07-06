"""Market-wide macro context: VIX (fear), rates (Fed/yields), USD, oil, gold,
plus the FOMC calendar and the latest FOMC statement tone.

This is CONTEXT, not prediction: it produces a risk regime that nudges scores
(risk-off punishes high-beta speculation; rising rates punish expensive growth;
near an FOMC decision, raise caution). Everything is free (yfinance + Fed RSS)
and cached, refreshed on every pipeline run (~3x/trading day via GitHub Actions).
"""
import json
import re
import urllib.request
from datetime import datetime, timezone, timedelta

import yfinance as yf

from .config import DATA

MACRO_CACHE = DATA / "macro.json"
MAX_AGE_HOURS = 2

# 2026 FOMC decision days (second day of each meeting)
FOMC_2026 = ["2026-01-28", "2026-03-18", "2026-04-29", "2026-06-17",
             "2026-07-29", "2026-09-16", "2026-10-28", "2026-12-09"]

_HAWK = ("raise", "hike", "tighten", "restrictive", "elevated inflation", "higher for longer",
         "firmer", "upside risks to inflation")
_DOVE = ("cut", "lower", "ease", "accommodative", "softening", "cooling", "rate reduction",
         "downside risks", "moderat")


def _load():
    if MACRO_CACHE.exists():
        try:
            return json.loads(MACRO_CACHE.read_text(encoding="utf-8"))
        except Exception:  # noqa: BLE001
            return {}
    return {}


def _fresh(cache):
    ts = cache.get("fetched_at")
    if not ts:
        return False
    try:
        return datetime.fromisoformat(ts) > datetime.now(timezone.utc) - timedelta(hours=MAX_AGE_HOURS)
    except Exception:  # noqa: BLE001
        return False


def _last_and_prev(symbol, days=20):
    """Latest close and the close ~days ago (for trend)."""
    try:
        h = yf.Ticker(symbol).history(period="2mo")
        if h is None or h.empty:
            return None, None
        close = h["Close"].dropna()
        last = float(close.iloc[-1])
        prev = float(close.iloc[-min(days + 1, len(close))])
        return round(last, 2), round(prev, 2)
    except Exception:  # noqa: BLE001
        return None, None


def _next_fomc(today):
    for d in FOMC_2026:
        if d >= today:
            days = (datetime.strptime(d, "%Y-%m-%d") - datetime.strptime(today, "%Y-%m-%d")).days
            return d, days
    return None, None


def _fomc_tone():
    """Heuristic hawkish/dovish read of the latest FOMC statement (Fed RSS)."""
    try:
        req = urllib.request.Request("https://www.federalreserve.gov/feeds/press_all.xml",
                                     headers={"User-Agent": "Mozilla/5.0"})
        xml = urllib.request.urlopen(req, timeout=15).read().decode("utf-8", "ignore")
        items = re.findall(r"<item>(.*?)</item>", xml, re.S)
        for it in items:
            title = (re.search(r"<title>(.*?)</title>", it, re.S) or [None, ""])[1]
            if "FOMC" in title or "Federal Open Market" in title or "monetary policy" in title.lower():
                blob = it.lower()
                h = sum(blob.count(w) for w in _HAWK)
                d = sum(blob.count(w) for w in _DOVE)
                tone = "hawkish" if h > d else "dovish" if d > h else "neutral"
                date = (re.search(r"<pubDate>(.*?)</pubDate>", it, re.S) or [None, ""])[1][:16]
                return {"tone": tone, "headline": re.sub(r"<[^>]+>", "", title).strip()[:120], "date": date}
    except Exception:  # noqa: BLE001
        pass
    return None


def fetch_macro(today=None, verbose=True):
    today = today or datetime.now(timezone.utc).strftime("%Y-%m-%d")
    cache = _load()
    if _fresh(cache):
        return cache

    if verbose:
        print("Makro-Umfeld laden (VIX, Zinsen, Fed) …")
    vix, vix_prev = _last_and_prev("^VIX")
    tnx, tnx_prev = _last_and_prev("^TNX")          # 10Y yield (sometimes ×10, sometimes direct)
    spx, spx_prev = _last_and_prev("^GSPC")
    ndx, _ = _last_and_prev("^NDX")            # Nasdaq 100
    world, _ = _last_and_prev("URTH")          # MSCI World (iShares ETF proxy)
    dxy, dxy_p = _last_and_prev("DX-Y.NYB")
    oil, oil_p = _last_and_prev("CL=F")
    gold, gold_p = _last_and_prev("GC=F")
    copper, copper_p = _last_and_prev("HG=F")
    silver, silver_p = _last_and_prev("SI=F")
    btc, btc_p = _last_and_prev("BTC-USD")

    def _yld(x):  # normalise to percent whether Yahoo returns 4.4 or 44.0
        return None if x is None else round(x / 10, 2) if x > 20 else round(x, 2)
    tnx_pct, tnx_prev_pct = _yld(tnx), _yld(tnx_prev)

    def _dir(last, prev, thr=3.0):  # rising/falling/flat over ~20 trading days
        if last is None or not prev:
            return "flat"
        chg = (last / prev - 1) * 100
        return "rising" if chg >= thr else "falling" if chg <= -thr else "flat"

    # VIX regime
    if vix is None:
        regime, regime_label = "neutral", "unbekannt"
    elif vix >= 30:
        regime, regime_label = "risk_off", "Risk-Off (Angst)"
    elif vix >= 22:
        regime, regime_label = "risk_off", "erhöhte Nervosität"
    elif vix <= 15:
        regime, regime_label = "risk_on", "Risk-On (ruhig)"
    else:
        regime, regime_label = "neutral", "neutral"

    # Rate direction from 10Y yield trend (percent)
    rate_dir = "flat"
    if tnx_pct is not None and tnx_prev_pct:
        chg = tnx_pct - tnx_prev_pct
        rate_dir = "rising" if chg >= 0.15 else "falling" if chg <= -0.15 else "flat"

    fomc_date, fomc_days = _next_fomc(today)
    tone = _fomc_tone()

    macro = {
        "fetched_at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "vix": vix, "vix_prev": vix_prev,
        "tnx_pct": tnx_pct, "tnx_prev_pct": tnx_prev_pct,
        "rate_dir": rate_dir,
        "spx": spx, "spx_up": (spx is not None and spx_prev is not None and spx > spx_prev),
        "ndx": ndx, "world": world,
        "dxy": dxy, "oil": oil, "gold": gold, "copper": copper, "silver": silver, "btc": btc,
        "dxy_dir": _dir(dxy, dxy_p, 1.5), "oil_dir": _dir(oil, oil_p),
        "gold_dir": _dir(gold, gold_p), "copper_dir": _dir(copper, copper_p),
        "silver_dir": _dir(silver, silver_p), "btc_dir": _dir(btc, btc_p, 6.0),
        "regime": regime, "regime_label": regime_label,
        "fomc_next": fomc_date, "fomc_in_days": fomc_days,
        "fomc_tone": tone,
    }
    try:
        MACRO_CACHE.write_text(json.dumps(macro, ensure_ascii=False, indent=1), encoding="utf-8")
    except Exception:  # noqa: BLE001
        pass
    return macro


def macro_adjust(row, macro):
    """Per-stock macro nudge (ELO points, small) + human notes. Context, not a forecast."""
    pts, notes = 0, []
    regime = macro.get("regime")
    vix = macro.get("vix")
    rate_dir = macro.get("rate_dir")
    beta = row.get("beta")
    sector = (row.get("sector") or "")
    themes = row.get("themes") or []

    if regime == "risk_off":
        if isinstance(beta, (int, float)) and beta >= 1.5:
            pts -= 8
            notes.append(f"Risk-Off (VIX {vix:.0f}): hohe Schwankung (Beta {beta:.1f}) belastet")
        elif isinstance(beta, (int, float)) and beta <= 0.9:
            pts += 4
            notes.append(f"Risk-Off (VIX {vix:.0f}): defensiv (niedrige Beta) im Vorteil")
    elif regime == "risk_on" and isinstance(beta, (int, float)) and beta >= 1.3:
        pts += 5
        notes.append(f"Risk-On (VIX {vix:.0f}): Rückenwind für Wachstum/Beta")

    fpe, vs = row.get("forward_pe"), row.get("value_score")
    if rate_dir == "rising":
        if (isinstance(fpe, (int, float)) and fpe > 40) or (isinstance(vs, (int, float)) and vs < 35):
            pts -= 6
            notes.append("Steigende Zinsen: teure Bewertung unter Druck")
        if "Financial" in sector or sector == "Financial Services":
            pts += 4
            notes.append("Steigende Zinsen: Banken/Versicherer profitieren")
    elif rate_dir == "falling":
        if "Real Estate" in sector or any(t in themes for t in ("REIT", "Versorger")):
            pts += 5
            notes.append("Fallende Zinsen: zinssensitiv (REIT/Versorger) im Vorteil")
        if isinstance(row.get("growth_score"), (int, float)) and row["growth_score"] >= 70:
            pts += 3
            notes.append("Fallende Zinsen: Rückenwind für Wachstum")

    # --- Commodity / FX tailwinds ---
    industry = (row.get("industry") or "").lower()
    sec = sector.lower()
    od = macro.get("oil_dir")
    if sec == "energy" or "oil" in industry or "gas" in industry:
        if od == "rising":
            pts += 5
            notes.append("Ölpreis steigt: Rückenwind für Energiewerte")
        elif od == "falling":
            pts -= 4
            notes.append("Ölpreis fällt: Gegenwind für Energiewerte")
    if "airline" in industry and od == "rising":
        pts -= 4
        notes.append("Ölpreis steigt: höhere Spritkosten (Airlines)")
    if "gold" in industry:
        gd = macro.get("gold_dir")
        if gd == "rising":
            pts += 5
            notes.append("Goldpreis steigt: Rückenwind für Goldminen")
        elif gd == "falling":
            pts -= 4
            notes.append("Goldpreis fällt: Gegenwind für Goldminen")
    if "silver" in industry and macro.get("silver_dir") == "rising":
        pts += 5
        notes.append("Silberpreis steigt: Rückenwind für Silberminen")
    if ("copper" in industry or "industrial metals" in industry):
        cd = macro.get("copper_dir")
        if cd == "rising":
            pts += 4
            notes.append("Kupferpreis steigt: Rückenwind")
        elif cd == "falling":
            pts -= 3
            notes.append("Kupferpreis fällt: Gegenwind")
    _DEV = {"us", "de", "fr", "gb", "ch", "jp", "ca", "nl", "it", "es", "se", "au",
            "nz", "dk", "no", "fi", "at", "ie", "be", "pt"}
    is_em = "Emerging Markets" in themes or (row.get("cc") and row.get("cc") not in _DEV)
    dd = macro.get("dxy_dir")
    if is_em:
        if dd == "rising":
            pts -= 5
            notes.append("Starker Dollar: Gegenwind für Emerging Markets")
        elif dd == "falling":
            pts += 4
            notes.append("Schwacher Dollar: Rückenwind für Emerging Markets")

    # Bitcoin -> crypto miners / proxies (not the coins themselves)
    if "Krypto" in themes and not (row.get("symbol") or "").upper().endswith("-USD"):
        bd = macro.get("btc_dir")
        if bd == "rising":
            pts += 6
            notes.append("Bitcoin steigt: Rückenwind für Krypto-Aktien/Miner")
        elif bd == "falling":
            pts -= 6
            notes.append("Bitcoin fällt: Gegenwind für Krypto-Aktien/Miner")

    fd = macro.get("fomc_in_days")
    if isinstance(fd, int) and 0 <= fd <= 3:
        notes.append(f"📅 Fed-Entscheidung in {fd} T – bis dahin oft erhöhte Schwankung")

    return max(-18, min(18, pts)), notes
