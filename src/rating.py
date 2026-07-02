"""Human-facing layer: a chess-style Radar-ELO rating, a plain-German
explanation anyone can read, and concrete action ideas per stock.

All inputs come from the already-computed row dict (technical + fundamental
scores + optional Aschenbrenner stance).
"""


def _has(x):
    return isinstance(x, (int, float))


def radar_elo(row):
    """Overall opportunity rating on a ~700-2200 scale (like a chess ELO).

    Anchored on the long-term investment score, nudged by short-term momentum
    and by whether smart money (Aschenbrenner) is long or betting against it.
    Returns (elo:int, rating_label:str, color:str).
    """
    base = row.get("investment_score")
    if not _has(base):
        base = row.get("longterm_score") or 50
    elo = 1000 + base * 10

    dt = row.get("daytrade_score") or 0
    if row.get("daytrade_direction") == "LONG" and dt > 40:
        elo += min(80, (dt - 40) * 2)
    elif row.get("daytrade_direction") == "SHORT" and dt > 40:
        elo -= min(60, (dt - 40) * 1.5)

    asch = row.get("aschenbrenner")
    if asch:
        if asch["stance"] == "LONG":
            elo += 60
        elif asch["stance"] == "SHORT_BET":
            elo -= 70

    ns = row.get("news_score")
    if _has(ns) and (row.get("news_n") or 0) >= 2:
        elo += max(-55, min(55, (ns - 50) * 1.4))

    if row.get("hype_surging") and _has(row.get("hype_score")):
        elo += min(40, row["hype_score"] * 0.4)  # retail momentum (short-term)

    an = row.get("analyst_n") or 0
    if an >= 3:
        au = row.get("analyst_upside_pct")
        if _has(au):
            elo += max(-45, min(55, au * 1.1))          # analyst target upside = potential
        am = row.get("analyst_mean")
        if _has(am):
            elo += max(-40, min(40, (3 - am) * 20))     # consensus: 1(strong buy)→+40, 5→-40

    # Expert frameworks
    mv = row.get("minervini_score")
    if _has(mv):
        elo += (mv - 50) * 0.4                          # ±20 trend-template alignment
    pio = row.get("piotroski")
    if _has(pio):
        elo += (pio - 4.5) * 7                          # Piotroski 9→+31, 0→-31
    alt = row.get("altman_z")
    if _has(alt):
        if alt < 1.81:
            elo -= 45                                   # bankruptcy distress zone
        elif alt > 2.99:
            elo += 12
    stg = row.get("weinstein_stage")
    if stg == 4:
        elo -= 30
    elif stg == 2:
        elo += 15

    elo = int(round(max(700, min(2200, elo)) / 5) * 5)

    for thr, label, color in [
        (1950, "Top-Chance", "#16a34a"),
        (1800, "Stark", "#65a30d"),
        (1650, "Solide", "#ca8a04"),
        (1500, "Neutral", "#6b7280"),
        (1350, "Schwach", "#ea580c"),
        (0, "Meiden", "#dc2626"),
    ]:
        if elo >= thr:
            return elo, label, color
    return elo, "Meiden", "#dc2626"


def radar_score(elo):
    """Simple, intuitive 0-100 score derived from the ELO (ELO 1000→0, 2000→100)."""
    return max(0, min(100, int(round((elo - 1000) / 10))))


def stars(score_0_100):
    """0-5 star rating from the 0-100 score."""
    return int(round(score_0_100 / 20))


def _label3(v):
    if v is None:
        return None
    return "hoch" if v >= 67 else "mittel" if v >= 45 else "niedrig"


def quality_score(row):
    """0-100: how solid/safe the company is (fundamentals, trend, analyst consensus)."""
    parts = []
    fs = row.get("fundamental_score")
    if _has(fs):
        parts.append((fs, 0.5))
    lt = row.get("longterm_score")
    if _has(lt):
        parts.append((lt, 0.3))
    am, an = row.get("analyst_mean"), row.get("analyst_n") or 0
    if _has(am) and an >= 3:
        parts.append((max(0, min(100, (5 - am) / 4 * 100)), 0.2))
    pio = row.get("piotroski")
    if _has(pio):
        parts.append((pio / 9 * 100, 0.2))          # Piotroski fundamental strength
    if not parts:
        return None
    return round(sum(v * w for v, w in parts) / sum(w for _, w in parts))


def potential_score(row):
    """0-100: upside potential (analyst target, momentum, room to run, growth, hype)."""
    parts = []
    au, an = row.get("analyst_upside_pct"), row.get("analyst_n") or 0
    if _has(au) and an >= 3:
        parts.append((max(0, min(100, 50 + au * 1.25)), 0.4))
    dt, ddir = row.get("daytrade_score"), row.get("daytrade_direction")
    if _has(dt):
        mom = 50 + (dt - 50) * (1 if ddir == "LONG" else -1 if ddir == "SHORT" else 0)
        parts.append((max(0, min(100, mom)), 0.2))
    pfh = row.get("pct_from_high52")
    if _has(pfh):
        parts.append((max(0, min(100, 50 - pfh)), 0.15))  # further below 52w-high = more room
    gs = row.get("growth_score")
    if _has(gs):
        parts.append((gs, 0.15))
    if row.get("hype_surging"):
        parts.append((80, 0.1))
    if not parts:
        return None
    return round(sum(v * w for v, w in parts) / sum(w for _, w in parts))


def _band(x, bands):
    for thr, sc in bands:
        if x <= thr:
            return sc
    return bands[-1][1]


def entry_score(row):
    """0-100: how good the CURRENT price is as an entry point right now.
    High = healthy pullback / near a low / cheap / room to analyst target,
    in an intact trend. Low = overbought / at the highs / chasing hype."""
    parts = []
    rsi = row.get("rsi")
    if _has(rsi):  # timing: pullback good, overbought bad
        parts.append((_band(rsi, [(30, 75), (45, 92), (55, 74), (65, 55), (72, 34), (200, 15)]), 0.30))
    pfh = row.get("pct_from_high52")
    if _has(pfh):  # distance below 52w-high: discount = closer to a low
        parts.append((_band(pfh, [(-45, 45), (-25, 72), (-10, 88), (-3, 55), (200, 32)]), 0.25))
    vs = row.get("value_score")
    if _has(vs):
        parts.append((vs, 0.20))
    au, an = row.get("analyst_upside_pct"), row.get("analyst_n") or 0
    if _has(au) and an >= 3:  # room to analyst target
        parts.append((_band(au, [(0, 25), (10, 50), (25, 72), (45, 90), (1e9, 100)]), 0.15))
    lt = row.get("longterm_score")
    if _has(lt):  # trend support: a dip in an uptrend is buyable
        parts.append((lt, 0.10))
    if not parts:
        return None
    score = sum(v * w for v, w in parts) / sum(w for _, w in parts)
    if row.get("hype_surging") and _has(rsi) and rsi > 68:
        score -= 12  # buying into froth is a poor entry
    # Volatility dampening: a "good entry level" on a very swingy stock is far
    # less precise — a normal single day can move several %. Wild names must not
    # read as an unqualified "sehr gut".
    atrp = row.get("atr_pct")
    if _has(atrp) and atrp > 4:
        score -= min(22, (atrp - 4) * 2.5)
    beta = row.get("beta")
    if _has(beta) and beta > 1.6:
        score -= min(10, (beta - 1.6) * 8)
    return int(round(max(0, min(100, score))))


def entry_label(score):
    if score is None:
        return "–", "calm"
    if score >= 70:
        return "sehr gut", "up"
    if score >= 55:
        return "gut", "up"
    if score >= 45:
        return "okay", "soon"
    if score >= 32:
        return "eher abwarten", "soon"
    return "schlecht (teuer/heiß)", "down"


def conviction(row):
    """0-100: how SURE the call is = agreement across signals (either direction).
    High when trend, fundamentals, analysts, news and momentum point the same way."""
    sig = []
    lt = row.get("longterm_score")
    if _has(lt):
        sig.append(1 if lt >= 55 else -1 if lt <= 45 else 0)
    fs = row.get("fundamental_score")
    if _has(fs):
        sig.append(1 if fs >= 60 else -1 if fs <= 40 else 0)
    au, an = row.get("analyst_upside_pct"), row.get("analyst_n") or 0
    if _has(au) and an >= 3:
        sig.append(1 if au >= 10 else -1 if au <= -5 else 0)
    ns = row.get("news_score")
    if _has(ns) and (row.get("news_n") or 0) >= 2:
        sig.append(1 if ns >= 60 else -1 if ns <= 40 else 0)
    dt, dd = row.get("daytrade_score") or 0, row.get("daytrade_direction")
    if dt >= 45:
        sig.append(1 if dd == "LONG" else -1 if dd == "SHORT" else 0)
    if not sig:
        return 50
    net = abs(sum(sig)) / len(sig)  # 0 (conflict) … 1 (full agreement)
    return int(round(50 + net * 45))


def urgency(row):
    """How fast to act. Returns (label, tone: urgent/soon/calm)."""
    dt, dd = row.get("daytrade_score") or 0, row.get("daytrade_direction")
    ed = row.get("earnings_in_days")
    if (dd in ("LONG", "SHORT") and dt >= 55) or row.get("hype_surging") or (_has(ed) and 0 <= ed <= 2):
        return "⏱️ Sofort (heute)", "urgent"
    if dt >= 45 or (_has(ed) and 0 <= ed <= 7):
        return "📅 Diese Woche", "soon"
    return "🧘 In Ruhe (Langzeit)", "calm"


def upside_pct(row):
    """Headline upside potential: analyst target if available, else 12M projection."""
    au, an = row.get("analyst_upside_pct"), row.get("analyst_n") or 0
    if _has(au) and an >= 3:
        return au
    for p in (row.get("projection_long") or []):
        if str(p.get("label", "")).startswith("12"):
            return p.get("center_pct")
    return None


def _pos(x):
    return isinstance(x, (int, float)) and x > 0


def _eta_label(days, context):
    """Friendly holding-period range from an estimated number of trading days."""
    if not isinstance(days, (int, float)) or days <= 0:
        return "wenige Tage" if context == "trade" else "einige Monate"
    for thr, lab in [(3, "1–3 Handelstage"), (10, "1–2 Wochen"), (25, "2–5 Wochen"),
                     (65, "1–3 Monate"), (130, "3–6 Monate"), (260, "6–12 Monate")]:
        if days <= thr:
            return lab
    return "12–24 Monate"


def trade_plan(row, context="invest"):
    """Concrete, number-based action plan for one stock.

    Uses real price structure (support/resistance from SMAs, pivots, breakout
    and 52-week levels + analyst target) and ATR for the money-management stop.
    Returns entry zone, target zone, stop, a SINGLE positive profit potential
    (%), the risk (%), the reward:risk ratio and an estimated holding period —
    plus a plain action recommendation. Returns None if price data is missing.
    """
    P = row.get("price")
    if not _pos(P):
        return None
    a = row.get("atr")
    if not _pos(a):
        ap = row.get("atr_pct")
        a = (ap / 100 * P) if _pos(ap) else 0.02 * P
    a = max(a, 0.005 * P)  # floor so the zones never collapse to a point

    side = "short" if (context == "trade" and row.get("daytrade_direction") == "SHORT") else "long"
    alt = row.get("altman_z")
    avoid = side == "long" and (
        row.get("weinstein_stage") == 4
        or (isinstance(alt, (int, float)) and alt < 1.81)
        or row.get("radar_rating") == "Meiden"
        or (row.get("longterm_score") or 50) < 30
    )

    supports = [row.get(k) for k in ("sma20", "sma50", "sma150", "sma200",
                                     "pivot", "pivot_s1", "low20", "ema21", "low52")]
    resist = [row.get(k) for k in ("sma20", "sma50", "sma150", "sma200",
                                   "pivot_r1", "high20", "high52", "ema9")]
    tgt = row.get("target_price")

    def below(cands):
        return sorted([c for c in cands if isinstance(c, (int, float)) and 0 < c < P], reverse=True)

    def above(cands):
        return sorted([c for c in cands if isinstance(c, (int, float)) and c > P])

    if side == "long":
        near_sup = next((s for s in below(supports) if s >= P - 2.2 * a), None)
        entry_low = min(near_sup if near_sup else P - a, P - 0.3 * a)
        entry_high = P + 0.2 * a
        res_above = above(resist)
        t1 = next((r for r in res_above if r >= P + 2 * a), None) or (P + 3 * a)
        hi = [t1 + 2 * a, P + 6 * a]
        if _pos(tgt) and tgt > P:
            hi.append(tgt)
        farther = next((r for r in res_above if r > t1 + 0.5 * a), None)
        if farther:
            hi.append(farther)
        target_low, target_high = t1, max(hi)
        stop = entry_low - 1.0 * a
        entry_mid, target_mid = (entry_low + entry_high) / 2, (target_low + target_high) / 2
        potential = (target_mid / entry_mid - 1) * 100
        risk = (entry_mid - stop) / entry_mid * 100
    else:  # short
        near_res = next((r for r in above(resist) if r <= P + 2.2 * a), None)
        entry_high = max(near_res if near_res else P + a, P + 0.3 * a)
        entry_low = P - 0.2 * a
        t1 = next((s for s in below(supports) if s <= P - 2 * a), None) or (P - 3 * a)
        lo = [t1 - 2 * a, P - 6 * a]
        farther = next((s for s in below(supports) if s < t1 - 0.5 * a), None)
        if farther:
            lo.append(farther)
        target_high, target_low = t1, min([x for x in lo if x > 0] or [P * 0.5])
        stop = entry_high + 1.0 * a
        entry_mid, target_mid = (entry_low + entry_high) / 2, (target_low + target_high) / 2
        potential = (1 - target_mid / entry_mid) * 100
        risk = (stop - entry_mid) / entry_mid * 100

    potential = max(potential, 0.0)
    rrr = round(potential / risk, 1) if risk and risk > 0 else None
    days = abs(target_mid - entry_mid) / (0.35 * a) if a else None
    hold = _eta_label(days, context)

    es = row.get("entry_score")
    if avoid:
        action, tone = "🚫 Meiden – aktuell kein sauberes Kaufsetup", "neg"
    elif side == "short":
        action, tone = "📉 Nur für Profis: Wette auf fallende Kurse (Short)", "neg"
    elif isinstance(es, (int, float)) and es >= 55:
        action, tone = "✅ Jetzt in der Einstiegszone kaufen", "pos"
    elif isinstance(es, (int, float)) and es >= 42:
        action, tone = "🟡 Gestaffelt kaufen (Teilpositionen)", "neutral"
    else:
        action, tone = "⏳ Auf Rücksetzer in die Zone warten", "neutral"

    def r2(x):
        return round(x, 2)

    return {
        "side": side,
        "avoid": bool(avoid),
        "entry_low": r2(min(entry_low, entry_high)),
        "entry_high": r2(max(entry_low, entry_high)),
        "target_low": r2(min(target_low, target_high)),
        "target_high": r2(max(target_low, target_high)),
        "stop": r2(stop),
        "potential_pct": round(potential, 1),
        "risk_pct": round(risk, 1),
        "rrr": rrr,
        "hold": hold,
        "action": action,
        "action_tone": tone,
    }


def plain_summary(row):
    """2-4 short sentences in plain German that a non-expert understands."""
    parts = []
    lt = row.get("longterm_score") or 0
    rsi = row.get("rsi")
    fs = row.get("fundamental_score")
    vs = row.get("value_score")
    qs = row.get("quality_score")

    # Trend + timing
    if lt >= 70:
        s = "Die Aktie steckt in einem stabilen Aufwärtstrend"
    elif lt >= 45:
        s = "Der Trend ist grundsätzlich intakt, aber nicht überragend"
    else:
        s = "Der langfristige Trend ist derzeit schwach"
    if _has(rsi):
        if 40 <= rsi <= 55 and lt >= 55:
            s += ", und der Kurs ist gerade etwas zurückgekommen – oft ein günstiger Einstiegsmoment."
        elif rsi > 70:
            s += " – der Kurs ist zuletzt aber heiß gelaufen (überkauft)."
        elif rsi < 30:
            s += " – aktuell ist er stark abverkauft, eine Gegenbewegung ist möglich."
        else:
            s += "."
    else:
        s += "."
    parts.append(s)

    # Fundamentals
    if _has(fs):
        if _has(qs) and qs >= 70:
            f = "Das Unternehmen verdient solide"
        elif _has(qs) and qs < 40:
            f = "Die Ertragskraft ist schwach"
        else:
            f = "Die Geschäftszahlen sind durchschnittlich"
        if _has(vs):
            if vs >= 65:
                f += " und ist dabei günstig bewertet."
            elif vs <= 35:
                f += ", ist aber hoch bewertet (teuer)."
            else:
                f += " zu einer fairen Bewertung."
        else:
            f += "."
        parts.append(f)

    # Analyst consensus
    au, an = row.get("analyst_upside_pct"), row.get("analyst_n") or 0
    if _has(au) and an >= 3:
        rating_de = {"strong_buy": "starker Kauf", "buy": "Kauf", "hold": "Halten",
                     "underperform": "Untergewichten", "sell": "Verkauf"}.get(
                         row.get("analyst_rating"), row.get("analyst_rating") or "—")
        arrow = f"+{au:.0f}%" if au >= 0 else f"{au:.0f}%"
        parts.append(f"{an} Analysten: Konsens „{rating_de}\", Ø-Kursziel {arrow} zum aktuellen Kurs.")

    # Expert frameworks
    if _has(row.get("minervini_score")) and row["minervini_score"] >= 80:
        parts.append("Erfüllt Minervinis Trend-Template (bestätigter Aufwärtstrend).")
    stg = row.get("weinstein_stage")
    if stg == 4:
        parts.append("Weinstein-Phase 4 (Abwärtstrend) – für Käufe meiden.")
    elif stg == 2 and not (_has(row.get("minervini_score")) and row["minervini_score"] >= 80):
        parts.append("Weinstein-Phase 2 (Aufwärtstrend).")
    pio = row.get("piotroski")
    if _has(pio):
        if pio >= 8:
            parts.append(f"Bilanz sehr solide (Piotroski {pio}/9).")
        elif pio <= 2:
            parts.append(f"Bilanz schwach (Piotroski {pio}/9).")
    alt = row.get("altman_z")
    if _has(alt) and alt < 1.81:
        parts.append(f"⚠️ Erhöhtes Pleiterisiko (Altman-Z {alt}).")

    # Short term
    dt = row.get("daytrade_score") or 0
    ddir = row.get("daytrade_direction")
    if ddir == "LONG" and dt >= 45:
        parts.append("Kurzfristig greifen gerade Käufer zu.")
    elif ddir == "SHORT" and dt >= 45:
        parts.append("Kurzfristig haben die Verkäufer die Oberhand.")

    # News sentiment
    ns = row.get("news_sentiment")
    if ns == "positiv":
        parts.append("Die aktuellen Schlagzeilen sind überwiegend positiv.")
    elif ns == "negativ":
        parts.append("Achtung: die aktuellen Schlagzeilen sind überwiegend negativ.")

    # Social / retail hype
    if row.get("hype_surging"):
        rank = row.get("hype_rank")
        chg = row.get("hype_change_pct")
        extra = f", Erwähnungen {chg:+.0f}%" if _has(chg) else ""
        parts.append(f"Auf Reddit stark diskutiert (Rang {rank}{extra}) – hohe "
                     f"Retail-Aufmerksamkeit, aber auch Hype-Risiko.")

    # Earnings
    ed = row.get("earnings_in_days")
    if _has(ed) and 0 <= ed <= 10:
        when = "heute" if ed == 0 else f"in {ed} Tagen"
        parts.append(f"📅 Zahlen {when} ({row.get('next_earnings')}) – bis dahin oft erhöhte Schwankung.")

    # Aschenbrenner
    asch = row.get("aschenbrenner")
    if asch:
        w = asch.get("weight_pct")
        if asch["stance"] == "LONG":
            parts.append(f"Der KI-Investor Leopold Aschenbrenner setzt hier auf steigende Kurse "
                         f"(rund {w}% seines Fonds).")
        elif asch["stance"] == "SHORT_BET":
            parts.append("Achtung: Aschenbrenner wettet über Put-Optionen gegen diese Aktie – "
                         "er rechnet eher mit fallenden Kursen.")
        elif asch["stance"] == "MIXED":
            parts.append("Aschenbrenner hält hier gleichzeitig auf steigende und fallende Kurse "
                         "gerichtete Positionen (Absicherung).")
    return " ".join(parts)


def suggest_actions(row):
    """Up to 3 concrete action ideas. Each: {text, tone} (pos/neg/neutral)."""
    out = []
    rsi = row.get("rsi")
    lt = row.get("longterm_score") or 0
    fs = row.get("fundamental_score")
    vs = row.get("value_score")
    dt = row.get("daytrade_score") or 0
    ddir = row.get("daytrade_direction")
    asch = row.get("aschenbrenner")

    if asch and asch["stance"] == "SHORT_BET":
        out.append({"text": "Vorsicht: Smart Money wettet dagegen", "tone": "neg"})
    if ddir == "LONG" and dt >= 50:
        out.append({"text": "Momentum-Long (kurzfristig)", "tone": "pos"})
    elif ddir == "SHORT" and dt >= 50:
        out.append({"text": "Short-Setup (kurzfristig)", "tone": "neg"})
    if lt >= 65 and _has(rsi) and 38 <= rsi <= 57:
        out.append({"text": "Rücksetzer-Einstieg beobachten", "tone": "pos"})
    if _has(fs) and fs >= 70 and _has(vs) and vs >= 60:
        out.append({"text": "Günstige Qualität – fürs Langfrist-Depot", "tone": "pos"})
    if _has(rsi) and rsi >= 72 and _has(vs) and vs <= 40:
        out.append({"text": "Abwarten – teuer & überkauft", "tone": "neg"})
    if _has(rsi) and rsi <= 28:
        out.append({"text": "Spekulativer Rebound möglich", "tone": "neutral"})
    ns = row.get("news_score")
    nn = row.get("news_n") or 0
    if _has(ns) and nn >= 2 and ns <= 38:
        out.append({"text": "Negative News – abwarten", "tone": "neg"})
    elif _has(ns) and nn >= 2 and ns >= 62:
        out.append({"text": "Positive News – Rückenwind", "tone": "pos"})
    ed = row.get("earnings_in_days")
    if _has(ed) and 0 <= ed <= 7:
        out.append({"text": f"Zahlen in {ed} T. – Vorsicht", "tone": "neutral"})
    if row.get("hype_surging"):
        out.append({"text": f"🔥 Reddit-Hype (Rang {row.get('hype_rank')})", "tone": "neutral"})
    if _has(row.get("minervini_score")) and row["minervini_score"] >= 80:
        out.append({"text": "Minervini Stage-2 ✓", "tone": "pos"})
    alt = row.get("altman_z")
    if _has(alt) and alt < 1.81:
        out.append({"text": "⚠️ Pleiterisiko (Altman)", "tone": "neg"})
    pio = row.get("piotroski")
    if _has(pio) and pio >= 8:
        out.append({"text": f"Bilanz stark (Piotroski {pio}/9)", "tone": "pos"})
    au, an = row.get("analyst_upside_pct"), row.get("analyst_n") or 0
    if _has(au) and an >= 5 and au >= 20:
        out.append({"text": f"Analysten-Kursziel +{au:.0f}%", "tone": "pos"})
    elif _has(au) and an >= 5 and au <= -5:
        out.append({"text": "Kurs über Analysten-Ziel", "tone": "neg"})
    if asch and asch["stance"] == "LONG":
        out.append({"text": "Aschenbrenner-Konviktion (long)", "tone": "pos"})

    # dedup by text, keep order, cap 3
    seen, dedup = set(), []
    for a in out:
        if a["text"] not in seen:
            seen.add(a["text"])
            dedup.append(a)
    if not dedup:
        dedup.append({"text": "Beobachten", "tone": "neutral"})
    return dedup[:3]
