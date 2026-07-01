"""News layer: per-ticker headlines + heuristic sentiment, plus broad market /
economic feeds. Sentiment flows into the rating; a market gauge gives context.

Heuristic (free) now; a Claude-API mode can be added later behind a flag.
"""
import concurrent.futures as cf

import feedparser

YF_RSS = "https://feeds.finance.yahoo.com/rss/2.0/headline?s={sym}&region=US&lang=en-US"

# Broad economic / market feeds for context + market sentiment
MARKET_FEEDS = [
    "http://feeds.marketwatch.com/marketwatch/topstories/",
    "https://www.investing.com/rss/news.rss",
    "https://www.cnbc.com/id/100003114/device/rss/rss.html",   # CNBC markets
    "https://www.tagesschau.de/wirtschaft/index~rss2.xml",     # DE Wirtschaft
]

# Finance sentiment lexicon (EN + DE). Lowercase, matched as substrings on words.
POS = {
    "beat", "beats", "surge", "surges", "jump", "jumps", "soar", "soars", "rally",
    "rallies", "upgrade", "upgraded", "raise", "raised", "record", "strong", "growth",
    "profit", "profits", "wins", "win", "deal", "approval", "approved", "breakthrough",
    "outperform", "bullish", "gain", "gains", "rebound", "expand", "expansion",
    "buyback", "beat estimates", "tops", "top", "boost", "boosts", "optimistic",
    "steigt", "steigen", "gewinnt", "rekord", "stark", "starkes", "wächst", "wachstum",
    "gewinn", "übernahme", "hochgestuft", "erholung", "durchbruch", "dividende", "kaufen",
}
NEG = {
    "miss", "misses", "plunge", "plunges", "drop", "drops", "fall", "falls", "slump",
    "cut", "cuts", "downgrade", "downgraded", "warning", "warns", "lawsuit", "probe",
    "investigation", "recall", "layoff", "layoffs", "loss", "losses", "weak", "bankruptcy",
    "fraud", "decline", "declines", "slash", "halt", "delay", "delays", "bearish", "sinks",
    "tumble", "tumbles", "plummet", "slips", "concern", "concerns", "fear", "fears",
    "fällt", "fallen", "sinkt", "verlust", "warnung", "klage", "ermittlung", "rückruf",
    "entlassungen", "schwach", "insolvenz", "betrug", "gewinnwarnung", "abgestuft", "einbruch",
    "stürzt", "verkaufen",
}


def _score_titles(titles):
    """Return (score_0_100, label, n_headlines). 50 = neutral."""
    pos = neg = 0
    for t in titles:
        words = t.lower().replace(",", " ").replace(".", " ").split()
        wset = set(words)
        pos += len(wset & POS)
        neg += len(wset & NEG)
    net = pos - neg
    score = max(5, min(95, 50 + net * 6))
    label = "positiv" if score >= 60 else "negativ" if score <= 40 else "neutral"
    return score, label, len(titles)


def _fetch_one(symbol, limit):
    try:
        feed = feedparser.parse(YF_RSS.format(sym=symbol))
        items = [{"title": e.get("title", ""), "link": e.get("link", ""),
                  "published": e.get("published", "")} for e in feed.entries[:limit]]
        return symbol, items
    except Exception:  # noqa: BLE001
        return symbol, []


def fetch_all_ticker_news(symbols, limit=6, workers=12):
    """Threaded fetch of per-ticker headlines. Returns {symbol: [items]}."""
    out = {}
    with cf.ThreadPoolExecutor(max_workers=workers) as ex:
        for sym, items in ex.map(lambda s: _fetch_one(s, limit), symbols):
            out[sym] = items
    return out


def news_signal(items):
    """Turn a ticker's headlines into a dict {score, sentiment, n, headlines}."""
    titles = [i["title"] for i in items if i.get("title")]
    score, label, n = _score_titles(titles)
    return {"news_score": score, "news_sentiment": label, "news_n": n,
            "news": items[:3]}


def fetch_market_news(limit=10):
    """Aggregate broad market/economic headlines + an overall market sentiment."""
    headlines = []
    for url in MARKET_FEEDS:
        try:
            feed = feedparser.parse(url)
            src = feed.feed.get("title", "")
            for e in feed.entries[:limit]:
                headlines.append({"title": e.get("title", ""), "link": e.get("link", ""),
                                  "source": src, "published": e.get("published", "")})
        except Exception:  # noqa: BLE001
            continue
    score, label, n = _score_titles([h["title"] for h in headlines])
    return {"market_sentiment": score, "market_label": label,
            "headlines": headlines[:24]}
