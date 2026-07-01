"""KI-gestützte News-Deutung mit Claude (Haiku 4.5) — optionaler Upgrade-Modus.

Aktiv nur wenn ANTHROPIC_API_KEY gesetzt ist. Liest die Schlagzeilen der
wichtigsten Titel und bewertet Materialität + Richtung strukturiert (JSON).
Fällt bei jedem Fehler still auf die kostenlose Heuristik zurück.
"""
import json
import os

# Günstigstes schnelles Modell für Klassifikation (siehe Anthropic-Preise: $1/$5 pro 1M).
MODEL = os.environ.get("STOCK_RADAR_LLM_MODEL", "claude-haiku-4-5")
BATCH = 12  # Titel pro API-Aufruf

_SCHEMA = {
    "type": "object",
    "properties": {
        "results": {
            "type": "array",
            "items": {
                "type": "object",
                "properties": {
                    "symbol": {"type": "string"},
                    "score": {"type": "integer"},
                    "direction": {"type": "string", "enum": ["bullish", "bearish", "neutral"]},
                    "reason": {"type": "string"},
                },
                "required": ["symbol", "score", "direction", "reason"],
                "additionalProperties": False,
            },
        }
    },
    "required": ["results"],
    "additionalProperties": False,
}

_DIR2SENT = {"bullish": "positiv", "bearish": "negativ", "neutral": "neutral"}

_SYSTEM = (
    "Du bist ein erfahrener Finanz-Nachrichtenanalyst. Für jede Aktie bekommst du "
    "aktuelle Schlagzeilen. Bewerte die voraussichtliche kurzfristige Auswirkung auf "
    "den Aktienkurs (Tage bis Wochen) auf einer Skala von 0 (sehr negativ) bis 100 "
    "(sehr positiv); 50 = neutral / keine kursrelevante Nachricht. Gib zusätzlich die "
    "Richtung (bullish/bearish/neutral) und eine sehr kurze Begründung auf DEUTSCH."
)


def _client():
    if not os.environ.get("ANTHROPIC_API_KEY"):
        return None
    try:
        import anthropic
        return anthropic.Anthropic()
    except Exception:  # noqa: BLE001
        return None


def llm_available():
    return _client() is not None


def _score_batch(client, batch):
    """batch = list of (symbol, name, [titles]). Returns {symbol: {...}}."""
    lines = []
    for sym, name, titles in batch:
        heads = " | ".join(t[:120] for t in titles[:4]) or "(keine aktuellen Schlagzeilen)"
        lines.append(f"{sym} ({name}): {heads}")
    user = "Bewerte diese Aktien anhand ihrer Schlagzeilen:\n\n" + "\n".join(lines)

    resp = client.messages.create(
        model=MODEL,
        max_tokens=1500,
        system=_SYSTEM,
        messages=[{"role": "user", "content": user}],
        output_config={"format": {"type": "json_schema", "schema": _SCHEMA}},
    )
    text = next((b.text for b in resp.content if b.type == "text"), "{}")
    data = json.loads(text)
    out = {}
    for r in data.get("results", []):
        sym = r.get("symbol")
        if not sym:
            continue
        score = max(0, min(100, int(r.get("score", 50))))
        out[sym] = {
            "news_score": score,
            "news_sentiment": _DIR2SENT.get(r.get("direction"), "neutral"),
            "news_llm_reason": r.get("reason", ""),
            "news_mode": "KI",
        }
    return out


def enhance(rows_subset, news_by):
    """Enhance the given rows' news signal with Claude. Returns {symbol: {...}}
    or {} if unavailable. Never raises."""
    client = _client()
    if client is None:
        return {}
    triples = []
    for r in rows_subset:
        sym = r["symbol"]
        titles = [i.get("title", "") for i in (news_by.get(sym) or [])]
        triples.append((sym, r.get("name") or sym, titles))

    result = {}
    for i in range(0, len(triples), BATCH):
        chunk = triples[i:i + BATCH]
        try:
            result.update(_score_batch(client, chunk))
        except Exception as exc:  # noqa: BLE001
            print(f"  KI-News Batch {i // BATCH + 1} fehlgeschlagen: {str(exc)[:100]}")
            continue
    return result
