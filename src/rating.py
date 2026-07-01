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

    # Short term
    dt = row.get("daytrade_score") or 0
    ddir = row.get("daytrade_direction")
    if ddir == "LONG" and dt >= 45:
        parts.append("Kurzfristig greifen gerade Käufer zu.")
    elif ddir == "SHORT" and dt >= 45:
        parts.append("Kurzfristig haben die Verkäufer die Oberhand.")

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
