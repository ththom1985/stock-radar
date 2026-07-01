"""Load Leopold Aschenbrenner's (Situational Awareness LP) 13F stance per ticker.

Data comes from data/aschenbrenner.json (curated from the SEC 13F-HR). Refresh
each quarter from https://www.sec.gov/Archives/edgar/data/2045724/ ...
"""
import json
from .config import DATA

ASCH_FILE = DATA / "aschenbrenner.json"

# Human-readable labels for each stance
STANCE_LABEL = {
    "LONG": "🟣 Aschenbrenner: LONG",
    "SHORT_BET": "🟣 Aschenbrenner: wettet DAGEGEN (Put)",
    "MIXED": "🟣 Aschenbrenner: gemischt (Put+Call)",
}


def load_aschenbrenner():
    """Return the full dataset dict (or a minimal empty structure)."""
    if not ASCH_FILE.exists():
        return {"holdings": {}, "report_quarter": None}
    try:
        return json.loads(ASCH_FILE.read_text(encoding="utf-8"))
    except Exception:  # noqa: BLE001
        return {"holdings": {}, "report_quarter": None}


def stance_for(symbol, data=None):
    """Return {stance, weight_pct, name, label} for a symbol, or None."""
    data = data or load_aschenbrenner()
    h = data.get("holdings", {}).get(symbol)
    if not h:
        return None
    return {
        "stance": h["stance"],
        "weight_pct": h.get("weight_pct"),
        "name": h.get("name"),
        "is_etf": h.get("is_etf", False),
        "label": STANCE_LABEL.get(h["stance"], "🟣 Aschenbrenner"),
    }
