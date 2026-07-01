"""Main pipeline: load universe -> fetch prices -> compute indicators ->
score -> write data/output/latest.json (+ dated history snapshot)."""
import json
from datetime import datetime, timezone

from .config import OUTPUT, HISTORY, TOP_N
from .universe import load_universe
from .fetch import fetch_prices
from .indicators import compute_features
from .score import score_daytrade, score_longterm
from .news import fetch_news_for


def _num(x):
    """Round floats for JSON, turn NaN into None."""
    if isinstance(x, float):
        if x != x:  # NaN
            return None
        return round(x, 4)
    return x


def run(with_news=True):
    universe = load_universe()
    symbols = [u["symbol"] for u in universe]
    name_map = {u["symbol"]: u["name"] for u in universe}
    print(f"Universum: {len(symbols)} Ticker. Lade Kurse …")

    prices = fetch_prices(symbols)
    print(f"Kursdaten erhalten für {len(prices)}/{len(symbols)} Ticker.")

    rows = []
    for sym, df in prices.items():
        f = compute_features(df)
        if not f:
            continue
        dscore, ddir, dreasons = score_daytrade(f)
        lscore, lreasons = score_longterm(f)
        rows.append({
            "symbol": sym,
            "name": name_map.get(sym, ""),
            "price": _num(f["price"]),
            "daily_return_pct": _num(f["daily_return"] * 100) if f.get("daily_return") == f.get("daily_return") else None,
            "rsi": _num(f.get("rsi")),
            "rvol": _num(f.get("rvol")),
            "atr_pct": _num(f.get("atr_pct")),
            "pct_from_high52": _num(f.get("pct_from_high52")),
            "daytrade_score": dscore,
            "daytrade_direction": ddir,
            "daytrade_reasons": dreasons,
            "longterm_score": lscore,
            "longterm_reasons": lreasons,
        })

    top_daytrade = sorted(rows, key=lambda r: r["daytrade_score"], reverse=True)[:TOP_N]
    top_longterm = sorted(rows, key=lambda r: r["longterm_score"], reverse=True)[:TOP_N]

    if with_news:
        print("Lade News für die Top-Picks …")
        for r in {id(x): x for x in top_daytrade + top_longterm}.values():
            r["news"] = fetch_news_for(r["symbol"], limit=3)

    result = {
        "generated_at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "universe_size": len(symbols),
        "analyzed": len(rows),
        "top_daytrade": top_daytrade,
        "top_longterm": top_longterm,
        "all": rows,
    }

    (OUTPUT / "latest.json").write_text(
        json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    stamp = datetime.now().strftime("%Y%m%d_%H%M")
    (HISTORY / f"{stamp}.json").write_text(
        json.dumps(result, ensure_ascii=False), encoding="utf-8"
    )

    print("\n=== TOP DAYTRADE ===")
    for i, r in enumerate(top_daytrade, 1):
        print(f"{i:>2}. {r['symbol']:<6} Score {r['daytrade_score']:>5} [{r['daytrade_direction']}]  {r['name']}")
    print("\n=== TOP LANGZEIT ===")
    for i, r in enumerate(top_longterm, 1):
        print(f"{i:>2}. {r['symbol']:<6} Score {r['longterm_score']:>5}  {r['name']}")
    print(f"\nGespeichert: {OUTPUT / 'latest.json'}")
    return result


if __name__ == "__main__":
    run()
