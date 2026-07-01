"""News headlines per ticker via free RSS feeds.

Uses Yahoo Finance's public per-symbol RSS. No scraping of paywalled content:
only publicly syndicated headlines + links are stored. Open articles yourself.
"""
import feedparser

YF_RSS = "https://feeds.finance.yahoo.com/rss/2.0/headline?s={sym}&region=US&lang=en-US"


def fetch_news_for(symbol, limit=3):
    """Return up to `limit` recent headlines: list of {title, link, published}."""
    try:
        feed = feedparser.parse(YF_RSS.format(sym=symbol))
        items = []
        for e in feed.entries[:limit]:
            items.append({
                "title": e.get("title", ""),
                "link": e.get("link", ""),
                "published": e.get("published", ""),
            })
        return items
    except Exception:  # noqa: BLE001
        return []
