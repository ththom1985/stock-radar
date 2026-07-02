"""Social / retail-hype layer via ApeWisdom (free, no key).

Aggregates Reddit mention data (r/wallstreetbets, r/stocks, …): rank,
mention count, and 24h change. Turns it into a hype signal that flags stocks
gaining unusual retail attention — an early-momentum / meme signal.

Note: ApeWisdom is US-Reddit focused, so it mostly matches US tickers. Foreign
listings (.DE/.PA/…) simply get no hype data (treated as neutral). Robinhood
per-stock popularity is NOT available (RobinTrack shut down 2020).
"""
import urllib.request
import json

BASE = "https://apewisdom.io/api/v1.0/filter/all-stocks/page/{page}"
PAGES = 4  # top ~400 most-mentioned tickers
UA = "StockRadar/1.0"


def fetch_social(pages=PAGES):
    """Return {ticker: {rank, mentions, upvotes, mentions_24h_ago, rank_24h_ago}}."""
    out = {}
    for p in range(1, pages + 1):
        try:
            req = urllib.request.Request(BASE.format(page=p), headers={"User-Agent": UA})
            with urllib.request.urlopen(req, timeout=15) as resp:
                data = json.loads(resp.read().decode("utf-8"))
        except Exception:  # noqa: BLE001
            break
        rows = data.get("results", [])
        if not rows:
            break
        for r in rows:
            t = (r.get("ticker") or "").upper()
            if t:
                out[t] = {
                    "rank": r.get("rank"),
                    "mentions": r.get("mentions"),
                    "upvotes": r.get("upvotes"),
                    "mentions_24h_ago": r.get("mentions_24h_ago"),
                    "rank_24h_ago": r.get("rank_24h_ago"),
                }
    return out


def social_signal(symbol, data):
    """Return hype signal dict for a symbol, or None if not on the board."""
    e = data.get((symbol or "").upper())
    if not e or not e.get("rank"):
        return None
    rank = e["rank"]
    hype_score = round(max(5, min(100, 100 - rank * 0.9)), 0)

    mentions = e.get("mentions") or 0
    prev = e.get("mentions_24h_ago") or 0
    change_pct = round((mentions / prev - 1) * 100, 0) if prev else None

    rank_prev = e.get("rank_24h_ago")
    rank_jump = (rank_prev - rank) if rank_prev else 0  # positive = climbed

    surging = bool(
        (prev and mentions >= 1.5 * prev and mentions >= 30) or rank_jump >= 15
    )
    return {
        "hype_rank": rank,
        "hype_mentions": mentions,
        "hype_change_pct": change_pct,
        "hype_score": hype_score,
        "hype_surging": surging,
    }
