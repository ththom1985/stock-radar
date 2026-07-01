"""Main pipeline: load universe -> fetch prices -> technical indicators ->
fundamentals -> scoring -> write data/output/latest.json (+ history snapshot).

Ranking layers:
  - daytrade_score   : short-term technical (momentum/breakout/volume)
  - longterm_score   : long-term technical (trend + healthy entry)
  - fundamental_score: Value/Quality/Growth from company financials
  - investment_score : blend of long-term technical + fundamental
"""
import json
from datetime import datetime, timezone

from .config import OUTPUT, HISTORY, TOP_N
from .universe import load_universe
from .fetch import fetch_prices
from .indicators import compute_features
from .score import score_daytrade, score_longterm
from .fundamentals import fetch_fundamentals
from .fundamental_score import (
    score_value, score_quality, score_growth, combine_fundamental, magic_formula_ranks,
)
from .aschenbrenner import load_aschenbrenner, stance_for
from .rating import radar_elo, radar_score, stars, plain_summary, suggest_actions
from .projection import project
from .paper_trader import update_portfolio
from .news import fetch_news_for


def _num(x):
    """Round floats for JSON, turn NaN into None."""
    if isinstance(x, float):
        if x != x:  # NaN
            return None
        return round(x, 4)
    return x


def _pct(x):
    return round(x * 100, 1) if isinstance(x, (int, float)) and x == x else None


def _invest_score(longterm, fundamental):
    """Blend long-term technical with fundamental (50/50 when both present)."""
    if fundamental is None:
        return longterm
    return round(0.5 * longterm + 0.5 * fundamental, 1)


def run(with_news=True, with_fundamentals=True):
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
            "vol_daily": _num(f.get("vol_daily")),
            "vol_daily_pct": _pct(f.get("vol_daily")),
            "pct_from_high52": _num(f.get("pct_from_high52")),
            "daytrade_score": dscore,
            "daytrade_direction": ddir,
            "daytrade_reasons": dreasons,
            "longterm_score": lscore,
            "longterm_reasons": lreasons,
        })

    # --- Fundamental layer ---
    if with_fundamentals:
        fund = fetch_fundamentals([r["symbol"] for r in rows])
        magic = magic_formula_ranks(fund)
        for r in rows:
            f = fund.get(r["symbol"], {})
            vscore, vreasons = score_value(f)
            qscore, qreasons = score_quality(f)
            gscore, greasons = score_growth(f)
            fscore = combine_fundamental(vscore, qscore, gscore)
            r["sector"] = f.get("sector")
            r["pe"] = _num(f.get("pe"))
            r["roe_pct"] = _pct(f.get("roe"))
            r["revenue_growth_pct"] = _pct(f.get("revenue_growth"))
            r["value_score"] = vscore
            r["quality_score"] = qscore
            r["growth_score"] = gscore
            r["magic_score"] = magic.get(r["symbol"])
            r["fundamental_score"] = fscore
            r["fundamental_reasons"] = (qreasons + vreasons + greasons)[:5]
            r["investment_score"] = _invest_score(r["longterm_score"], fscore)
    else:
        for r in rows:
            r["fundamental_score"] = None
            r["investment_score"] = r["longterm_score"]

    # --- Aschenbrenner stance + human-facing rating layer ---
    asch_data = load_aschenbrenner()
    for r in rows:
        r["aschenbrenner"] = stance_for(r["symbol"], asch_data)
        elo, rating_label, color = radar_elo(r)
        r["radar_elo"] = elo
        r["radar_rating"] = rating_label
        r["radar_color"] = color
        r["radar_score"] = radar_score(elo)
        r["stars"] = stars(r["radar_score"])
        r["plain_summary"] = plain_summary(r)
        r["actions"] = suggest_actions(r)
        r["projection_short"] = project(r, "short")
        r["projection_long"] = project(r, "long")

    top_daytrade = sorted(rows, key=lambda r: r["daytrade_score"], reverse=True)[:TOP_N]
    top_longterm = sorted(rows, key=lambda r: r["investment_score"], reverse=True)[:TOP_N]
    top_fundamental = sorted(
        [r for r in rows if r.get("fundamental_score") is not None],
        key=lambda r: r["fundamental_score"], reverse=True,
    )[:TOP_N]

    aschenbrenner_holdings = sorted(
        [r for r in rows if r.get("aschenbrenner")],
        key=lambda r: r["aschenbrenner"].get("weight_pct") or 0, reverse=True,
    )

    # --- Virtual paper-trading self-check (persists in data/portfolio.json) ---
    paper = update_portfolio(rows)

    if with_news:
        print("Lade News für die Top-Picks …")
        picks = top_daytrade + top_longterm + top_fundamental + aschenbrenner_holdings
        for r in {id(x): x for x in picks}.values():
            r["news"] = fetch_news_for(r["symbol"], limit=3)

    result = {
        "generated_at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "universe_size": len(symbols),
        "analyzed": len(rows),
        "aschenbrenner_meta": {
            "quarter": asch_data.get("report_quarter"),
            "filed": asch_data.get("filed"),
        },
        "top_daytrade": top_daytrade,
        "top_longterm": top_longterm,
        "top_fundamental": top_fundamental,
        "aschenbrenner_holdings": aschenbrenner_holdings,
        "paper": paper,
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
        print(f"{i:>2}. {r['symbol']:<8} Score {r['daytrade_score']:>5} [{r['daytrade_direction']}]  {r['name']}")
    print("\n=== TOP LANGZEIT (Technik + Fundamental) ===")
    for i, r in enumerate(top_longterm, 1):
        fs = r.get("fundamental_score")
        fs = f"F{fs:>5}" if fs is not None else "F   - "
        print(f"{i:>2}. {r['symbol']:<8} Invest {r['investment_score']:>5}  ({fs})  {r['name']}")
    print("\n=== TOP FUNDAMENTAL (Value/Quality/Growth) ===")
    for i, r in enumerate(top_fundamental, 1):
        print(f"{i:>2}. {r['symbol']:<8} Score {r['fundamental_score']:>5}  {r['name']}")
    print(f"\n=== PAPER-DEPOT ===")
    print(f"Kontostand {paper['equity']:.2f} € ({paper['total_return_pct']:+.2f}%) · "
          f"Cash {paper['cash']:.2f} € · Positionen {paper['n_positions']} · "
          f"Trades gesamt {paper['n_trades']}")
    print(f"\nGespeichert: {OUTPUT / 'latest.json'}")
    return result


if __name__ == "__main__":
    run()
