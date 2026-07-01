"""Fundamental scoring: Value, Quality, Growth + Greenblatt Magic Formula.

Each sub-score is 0-100 and averages only over the metrics that are actually
available, so missing data does not unfairly penalise a stock. Every score
comes with plain-German reasons for full transparency.
"""


def _ok(x):
    return isinstance(x, (int, float)) and not (isinstance(x, float) and x != x)


def _band(x, bands):
    """bands = list of (threshold, score) sorted ascending by threshold.
    Returns score for the first threshold x <= threshold, else last score."""
    for thr, sc in bands:
        if x <= thr:
            return sc
    return bands[-1][1]


def score_value(f):
    """Cheapness. Low multiples = high score. Returns (score|None, reasons)."""
    parts, reasons = [], []

    pe = f.get("pe")
    if _ok(pe) and pe > 0:
        s = _band(pe, [(10, 100), (15, 85), (20, 70), (25, 55), (35, 35), (50, 20), (1e9, 5)])
        parts.append(s)
        if pe <= 15:
            reasons.append(f"Günstiges KGV ({pe:.0f})")
        elif pe >= 40:
            reasons.append(f"Hohes KGV ({pe:.0f}) – teuer bewertet")

    pb = f.get("pb")
    if _ok(pb) and pb > 0:
        parts.append(_band(pb, [(1, 100), (2, 80), (3, 65), (5, 45), (8, 25), (1e9, 10)]))
        if pb <= 1.5:
            reasons.append(f"Kurs-Buchwert niedrig ({pb:.1f})")

    ps = f.get("ps")
    if _ok(ps) and ps > 0:
        parts.append(_band(ps, [(1, 100), (2, 82), (4, 65), (7, 45), (12, 25), (1e9, 10)]))
        if ps <= 1.5:
            reasons.append(f"Kurs-Umsatz niedrig ({ps:.1f})")

    ev = f.get("ev_ebitda")
    if _ok(ev) and ev > 0:
        parts.append(_band(ev, [(8, 100), (12, 80), (16, 62), (22, 42), (35, 22), (1e9, 8)]))
        if ev <= 10:
            reasons.append(f"EV/EBITDA attraktiv ({ev:.0f})")

    peg = f.get("peg")
    if _ok(peg) and peg > 0:
        parts.append(_band(peg, [(1, 100), (1.5, 80), (2, 60), (3, 38), (1e9, 15)]))
        if peg <= 1:
            reasons.append(f"PEG < 1 ({peg:.2f}) – Wachstum günstig bezahlt")

    if not parts:
        return None, reasons
    return round(sum(parts) / len(parts), 1), reasons


def score_quality(f):
    """Profitability & balance-sheet strength. Returns (score|None, reasons)."""
    parts, reasons = [], []

    roe = f.get("roe")
    if _ok(roe):
        r = roe * 100
        parts.append(_band(r, [(0, 5), (8, 40), (15, 70), (20, 85), (1e9, 100)]))
        if r >= 20:
            reasons.append(f"Sehr hohe Eigenkapitalrendite (ROE {r:.0f}%)")
        elif r < 0:
            reasons.append("Negative Eigenkapitalrendite – Verlust")

    roa = f.get("roa")
    if _ok(roa):
        r = roa * 100
        parts.append(_band(r, [(0, 5), (2, 40), (5, 65), (10, 85), (1e9, 100)]))

    pm = f.get("profit_margin")
    if _ok(pm):
        m = pm * 100
        parts.append(_band(m, [(0, 5), (5, 45), (10, 65), (20, 85), (1e9, 100)]))
        if m >= 20:
            reasons.append(f"Hohe Nettomarge ({m:.0f}%)")

    de = f.get("debt_to_equity")
    if _ok(de) and de >= 0:
        parts.append(_band(de, [(30, 100), (80, 80), (150, 55), (250, 30), (1e9, 12)]))
        if de <= 50:
            reasons.append("Solide Bilanz (wenig Schulden)")
        elif de >= 250:
            reasons.append(f"Hohe Verschuldung (D/E {de:.0f}%)")

    cr = f.get("current_ratio")
    if _ok(cr):
        parts.append(_band(cr, [(1, 30), (1.5, 70), (2, 90), (1e9, 100)]))

    if not parts:
        return None, reasons
    return round(sum(parts) / len(parts), 1), reasons


def score_growth(f):
    """Revenue & earnings growth. Returns (score|None, reasons)."""
    parts, reasons = [], []

    rg = f.get("revenue_growth")
    if _ok(rg):
        g = rg * 100
        parts.append(_band(g, [(0, 20), (10, 60), (20, 85), (1e9, 100)]))
        if g >= 20:
            reasons.append(f"Starkes Umsatzwachstum (+{g:.0f}%)")
        elif g < 0:
            reasons.append(f"Umsatz schrumpft ({g:.0f}%)")

    eg = f.get("earnings_growth")
    if _ok(eg):
        g = eg * 100
        parts.append(_band(g, [(0, 20), (10, 60), (25, 85), (1e9, 100)]))
        if g >= 25:
            reasons.append(f"Starkes Gewinnwachstum (+{g:.0f}%)")

    if not parts:
        return None, reasons
    return round(sum(parts) / len(parts), 1), reasons


def combine_fundamental(value, quality, growth):
    """Blend available sub-scores: Quality 40%, Value 35%, Growth 25%."""
    weights = [(quality, 0.40), (value, 0.35), (growth, 0.25)]
    num = sum(s * w for s, w in weights if s is not None)
    den = sum(w for s, w in weights if s is not None)
    if den == 0:
        return None
    return round(num / den, 1)


def magic_formula_ranks(fund_by_symbol):
    """Greenblatt-style rank (approximation): combines earnings yield
    (EBITDA/EV) and return on capital (ROA). Lower combined rank = better.
    Returns dict symbol -> score 0-100 (higher = better)."""
    ey, roc = {}, {}
    for sym, f in fund_by_symbol.items():
        ev = f.get("enterprise_value")
        ebitda = f.get("ebitda")
        if _ok(ev) and ev > 0 and _ok(ebitda):
            ey[sym] = ebitda / ev
        roa = f.get("roa")
        if _ok(roa):
            roc[sym] = roa

    def _rank(d):  # higher value -> better (rank 1)
        order = sorted(d, key=lambda s: d[s], reverse=True)
        return {s: i + 1 for i, s in enumerate(order)}

    ey_rank, roc_rank = _rank(ey), _rank(roc)
    common = set(ey_rank) & set(roc_rank)
    if not common:
        return {}
    combined = {s: ey_rank[s] + roc_rank[s] for s in common}
    n = len(combined)
    order = sorted(combined, key=lambda s: combined[s])  # best first
    return {s: round(100 * (n - i) / n, 1) for i, s in enumerate(order)}
