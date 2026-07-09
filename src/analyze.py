"""Main pipeline: load universe -> fetch prices -> technical indicators ->
fundamentals -> scoring -> write data/output/latest.json (+ history snapshot).


Ranking layers:
  - daytrade_score   : short-term technical (momentum/breakout/volume)
  - longterm_score   : long-term technical (trend + healthy entry)
  - fundamental_score: Value/Quality/Growth from company financials
  - investment_score : blend of long-term technical + fundamental
"""
import json
import os
from datetime import datetime, timezone

from .config import OUTPUT, HISTORY, TOP_N, INVEST_W_TECH, INVEST_W_FUND, DATA
from .universe import load_universe
from .fetch import fetch_prices
from .indicators import compute_features
from .score import score_daytrade, score_longterm
from .fundamentals import fetch_fundamentals
from .fundamental_score import (
    score_value, score_quality, score_growth, combine_fundamental, magic_formula_ranks,
)
from .llm_news import enhance as llm_enhance_news, llm_available
from .aschenbrenner import load_aschenbrenner, stance_for
from .rating import (radar_elo, radar_score, stars, plain_summary, suggest_actions,
                     quality_score as calc_quality, potential_score as calc_potential,
                     conviction, urgency, upside_pct, entry_score, entry_reason,
                     downside_analysis, trade_plan, risk_warnings, bull_thesis, priced_in_note,
                     trend_phase, volume_signal, falling_knife)
from .geo import country_flag
from .fx import currency_for, get_fx_rates
from .macro import fetch_macro, macro_adjust
from .intraday import fetch_intraday_patterns, pattern_note
from .projection import project
from .social import fetch_social, social_signal
from .deep_fundamentals import fetch_deep
from .expert_signals import (minervini, weinstein_stage,
                             tech_trend_score, tech_momentum_score, tech_volume_score)
from .paper_trader import update_portfolio
from .news_engine import fetch_all_ticker_news, news_signal, fetch_market_news
from .earnings import fetch_earnings, days_until


def _load_json_map(filename):
    """Load a {symbol: [...]} JSON map from data/, empty dict on any failure."""
    p = DATA / filename
    if p.exists():
        try:
            return json.loads(p.read_text(encoding="utf-8"))
        except Exception:  # noqa: BLE001
            return {}
    return {}


def _load_expert_sources():
    """symbol -> list of expert/media sources that recommended it (badge)."""
    return _load_json_map("expert_sources.json")


def _load_prev_output():
    """Last written latest.json (to carry news forward on light intraday runs)."""
    try:
        return json.loads((OUTPUT / "latest.json").read_text(encoding="utf-8"))
    except Exception:  # noqa: BLE001
        return {}


def _num(x):
    """Round floats for JSON, turn NaN/Inf into None."""
    if isinstance(x, float):
        if x != x or x in (float("inf"), float("-inf")):  # NaN or Inf
            return None
        return round(x, 4)
    return x


def _pct(x):
    return round(x * 100, 1) if isinstance(x, (int, float)) and x == x else None


def _clean(v):
    """Make a feature value JSON-safe."""
    if isinstance(v, bool):
        return v
    if isinstance(v, float):
        return None if (v != v or v in (float("inf"), float("-inf"))) else round(v, 4)
    if isinstance(v, (int, str)):
        return v
    return None


def _invest_score(longterm, fundamental):
    """Blend long-term technical with fundamental using configurable weights."""
    if fundamental is None:
        return longterm
    return round(INVEST_W_TECH * longterm + INVEST_W_FUND * fundamental, 1)


def run(with_news=True, with_fundamentals=True):
    # Light intraday mode (frequent price refresh): skip the slow RSS/LLM news
    # fetch and carry the last full run's news forward instead.
    intraday = os.environ.get("STOCK_RADAR_INTRADAY") == "1"
    if intraday:
        with_news = False
    universe = load_universe()
    symbols = [u["symbol"] for u in universe]
    name_map = {u["symbol"]: u["name"] for u in universe}
    print(f"Universum: {len(symbols)} Ticker{' · INTRADAY (Kurse frisch, News weitergetragen)' if intraday else ''}. Lade Kurse …")

    prices = fetch_prices(symbols)
    print(f"Kursdaten erhalten für {len(prices)}/{len(symbols)} Ticker.")

    rows = []
    for sym, df in prices.items():
        f = compute_features(df)
        if not f:
            continue
        dscore, ddir, dreasons = score_daytrade(f)
        lscore, lreasons = score_longterm(f)
        row = {k: _clean(v) for k, v in f.items()}  # keep all indicators for the pro view
        row.update({
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
        row["ret_60d"] = _num(f.get("ret_60d"))
        rows.append(row)

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
            r["industry"] = f.get("industry")
            # Analyst consensus
            r["analyst_rating"] = f.get("rec_key")
            r["analyst_mean"] = _num(f.get("rec_mean"))
            r["analyst_n"] = f.get("analyst_n")
            tgt = f.get("target_price")
            r["target_price"] = _num(tgt)
            r["analyst_upside_pct"] = (
                round((tgt / r["price"] - 1) * 100, 1)
                if isinstance(tgt, (int, float)) and tgt > 0 and r.get("price") else None
            )
            r["pe"] = _num(f.get("pe"))
            r["forward_pe"] = _num(f.get("forward_pe"))
            r["earnings_growth"] = _num(f.get("earnings_growth"))
            r["roe_pct"] = _pct(f.get("roe"))
            r["revenue_growth_pct"] = _pct(f.get("revenue_growth"))
            r["value_score"] = vscore
            r["quality_score"] = qscore
            r["growth_score"] = gscore
            r["magic_score"] = magic.get(r["symbol"])
            r["fundamental_score"] = fscore
            r["fundamental_reasons"] = (qreasons + vreasons + greasons)[:5]
            r["investment_score"] = _invest_score(r["longterm_score"], fscore)
            # --- literature value metrics (from .info) ---
            r["beta"] = _num(f.get("beta"))
            eps, bvps, price = f.get("eps"), f.get("bvps"), r.get("price")
            if isinstance(eps, (int, float)) and isinstance(bvps, (int, float)) and eps > 0 and bvps > 0:
                graham = (22.5 * eps * bvps) ** 0.5
                r["graham_number"] = round(graham, 2)
                r["graham_margin_pct"] = round((graham / price - 1) * 100, 1) if price else None
            fcf, mcap, rev = f.get("free_cashflow"), f.get("market_cap"), f.get("revenue")
            if isinstance(fcf, (int, float)) and isinstance(mcap, (int, float)) and mcap:
                r["fcf_yield_pct"] = round(fcf / mcap * 100, 1)
            if isinstance(f.get("revenue_growth"), (int, float)) and isinstance(fcf, (int, float)) and rev:
                r["rule40"] = round(f["revenue_growth"] * 100 + fcf / rev * 100)
    else:
        for r in rows:
            r["fundamental_score"] = None
            r["investment_score"] = r["longterm_score"]

    # --- Convert every price level to USD (display consistency) ---
    # Ratios (%, upside, margins, atr_pct) were computed from local values and
    # stay correct; only absolute price fields are scaled.
    _PRICE_KEYS = ("price", "prev_close", "sma20", "sma50", "sma150", "sma200",
                   "sma150_1m_ago", "sma200_1m_ago", "ema9", "ema21", "high52", "low52",
                   "high20", "low20", "atr", "pivot", "pivot_r1", "pivot_s1",
                   "target_price", "graham_number")
    fx = get_fx_rates({currency_for(r["symbol"]) for r in rows})
    for r in rows:
        cur = currency_for(r["symbol"])
        rate = fx.get(cur) or 1.0
        r["currency"] = cur
        r["fx_usd"] = rate
        if rate != 1.0:
            for k in _PRICE_KEYS:
                v = r.get(k)
                if isinstance(v, (int, float)):
                    r[k] = round(v * rate, 2)

    # --- News layer (per-ticker sentiment) + earnings dates ---
    market = {"market_sentiment": None, "market_label": None, "headlines": []}
    if with_news:
        syms = [r["symbol"] for r in rows]
        print(f"Lade News für {len(syms)} Titel …")
        news_by = fetch_all_ticker_news(syms)
        earn = fetch_earnings(syms)
        for r in rows:
            sig = news_signal(news_by.get(r["symbol"], []))
            r.update(sig)
            nd = earn.get(r["symbol"], {}).get("next_earnings")
            r["next_earnings"] = nd
            r["earnings_in_days"] = days_until(nd)
        print("Lade Markt-News …")
        market = fetch_market_news()

        # --- KI-Upgrade: Top-N von Claude bewerten lassen (nur wenn API-Key gesetzt) ---
        if llm_available():
            top_n = int(os.environ.get("STOCK_RADAR_LLM_TOPN", "60"))
            subset = sorted(rows, key=lambda r: r.get("investment_score") or 0,
                            reverse=True)[:top_n]
            print(f"KI-News (Claude) für Top {len(subset)} Titel …")
            enh = llm_enhance_news(subset, news_by)
            for r in rows:
                if r["symbol"] in enh:
                    r.update(enh[r["symbol"]])
            print(f"  KI-News: {len(enh)} Titel bewertet.")
    else:
        # Intraday: carry news/earnings forward from the last full run (no re-fetch)
        prev = _load_prev_output()
        prev_by = {r.get("symbol"): r for r in prev.get("all", [])}
        _carry = ("news_score", "news_sentiment", "news_n", "news", "news_mode",
                  "news_llm_reason", "next_earnings", "earnings_in_days", "intraday_note")
        for r in rows:
            pr = prev_by.get(r["symbol"])
            if pr:
                for k in _carry:
                    if k in pr:
                        r[k] = pr[k]
        market = prev.get("market_news") or market
        print(f"  News aus letztem Volllauf übernommen ({len(prev_by)} Titel).")

    # --- Social / Reddit hype layer (ApeWisdom, free) ---
    print("Lade Reddit-/Social-Hype …")
    social = fetch_social()
    n_hype = 0
    for r in rows:
        sig = social_signal(r["symbol"], social)
        if sig:
            r.update(sig)
            n_hype += 1
    print(f"  Social: {n_hype} Titel mit Reddit-Erwähnungen.")

    # --- Relative strength, deep fundamentals & expert frameworks ---
    ranked = sorted([r for r in rows if isinstance(r.get("ret_60d"), (int, float))],
                    key=lambda r: r["ret_60d"])
    nrs = len(ranked)
    for i, r in enumerate(ranked):
        r["rs_rating"] = round((i + 1) / nrs * 100) if nrs else None
    if with_fundamentals:
        mcaps = {r["symbol"]: (fund.get(r["symbol"], {}) or {}).get("market_cap") for r in rows}
        # Prioritise the top picks so Piotroski/Altman fill for your best stocks first.
        ordered = [r["symbol"] for r in sorted(
            rows, key=lambda r: r.get("investment_score") or 0, reverse=True)]
        deep = fetch_deep(ordered, mcaps)
        for r in rows:
            d = deep.get(r["symbol"], {})
            r["piotroski"] = d.get("piotroski")
            r["altman_z"] = d.get("altman_z")
    print("Berechne Experten-Signale (Minervini, Weinstein, Trend/Momentum/Volumen) …")
    for r in rows:
        mv, met, failed = minervini(r, r.get("rs_rating"))
        r["minervini_score"], r["minervini_met"], r["minervini_failed"] = mv, met, failed
        stg, lbl = weinstein_stage(r)
        r["weinstein_stage"], r["weinstein_label"] = stg, lbl
        r["tech_trend"] = tech_trend_score(r)
        r["tech_momentum"] = tech_momentum_score(r)
        r["tech_volume"] = tech_volume_score(r)

    # --- Intraday time-of-day patterns (full run only; bounded + cached) ---
    # Prioritise the top picks so your most important stocks get the 🕐 pattern first.
    if not intraday:
        ordered = [r["symbol"] for r in sorted(
            rows, key=lambda r: r.get("investment_score") or 0, reverse=True)]
        pats = fetch_intraday_patterns(ordered)
        for r in rows:
            r["intraday_note"] = pattern_note(pats.get(r["symbol"]))

    # --- Macro context (VIX / rates / Fed) ---
    macro = fetch_macro()
    print(f"Marktumfeld: {macro.get('regime_label')} · VIX {macro.get('vix')} · "
          f"10J-Zins {macro.get('tnx_pct')}% ({macro.get('rate_dir')}) · "
          f"Fed in {macro.get('fomc_in_days')} T")

    # --- Aschenbrenner stance + human-facing rating layer ---
    asch_data = load_aschenbrenner()
    expert_src = _load_expert_sources()
    theme_map = _load_json_map("themes.json")
    for r in rows:
        r["expert_sources"] = expert_src.get(r["symbol"])
        r["themes"] = theme_map.get(r["symbol"])
        r["country"], r["cc"] = country_flag(r["symbol"])
        r["aschenbrenner"] = stance_for(r["symbol"], asch_data)
        r["macro_pts"], r["macro_notes"] = macro_adjust(r, macro)
        r["volume_pts"], r["volume_note"] = volume_signal(r)
        elo, rating_label, color = radar_elo(r)
        r["radar_elo"] = elo
        r["radar_rating"] = rating_label
        r["radar_color"] = color
        r["radar_score"] = radar_score(elo)
        r["stars"] = stars(r["radar_score"])
        r["quality"] = calc_quality(r)
        r["potential"] = calc_potential(r)
        r["plain_summary"] = plain_summary(r)
        r["actions"] = suggest_actions(r)
        r["projection_short"] = project(r, "short")
        r["projection_long"] = project(r, "long")
        r["trend_phase"] = trend_phase(r)
        r["exp_return_12m"] = next((p.get("center_pct") for p in (r["projection_long"] or [])
                                    if str(p.get("label", "")).startswith("12")), None)
        r["conviction"] = conviction(r)
        u_label, u_tone = urgency(r)
        r["urgency"], r["urgency_tone"] = u_label, u_tone
        r["upside_pct"] = upside_pct(r)
        r["knife_warn"] = falling_knife(r)
        r["entry_score"] = entry_score(r)
        r["entry_reason"] = entry_reason(r)
        r["risk_warnings"] = risk_warnings(r)
        r["bull_thesis"] = bull_thesis(r)
        r["priced_in"] = priced_in_note(r)
        r["downside"] = downside_analysis(r)
        r["trade_plan_long"] = trade_plan(r, "invest")
        r["trade_plan_short"] = trade_plan(r, "trade")

    top_daytrade = sorted(rows, key=lambda r: r["daytrade_score"], reverse=True)[:TOP_N]

    def _long_opportunity(r):
        # Quality (investment_score) AND expected 12M return, so a high-quality
        # but low-upside stock cannot outrank a strong-upside one.
        base = r.get("investment_score") or 0
        er = r.get("exp_return_12m")
        return base + (max(-30, min(50, er)) * 0.6 if isinstance(er, (int, float)) else 0)

    top_longterm = sorted(rows, key=_long_opportunity, reverse=True)[:TOP_N]
    top_fundamental = sorted(
        [r for r in rows if r.get("fundamental_score") is not None],
        key=lambda r: r["fundamental_score"], reverse=True,
    )[:TOP_N]

    aschenbrenner_holdings = sorted(
        [r for r in rows if r.get("aschenbrenner")],
        key=lambda r: r["aschenbrenner"].get("weight_pct") or 0, reverse=True,
    )

    top_hype = sorted(
        [r for r in rows if r.get("hype_rank")],
        key=lambda r: r.get("hype_score") or 0, reverse=True,
    )[:TOP_N]

    # --- Virtual paper-trading self-check (persists in data/portfolio.json) ---
    paper = update_portfolio(rows, benchmarks={
        "sp500": macro.get("spx"), "ndx": macro.get("ndx"), "world": macro.get("world")})

    result = {
        "generated_at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "universe_size": len(symbols),
        "analyzed": len(rows),
        "market_news": market,
        "macro": macro,
        "aschenbrenner_meta": {
            "quarter": asch_data.get("report_quarter"),
            "filed": asch_data.get("filed"),
        },
        "top_daytrade": top_daytrade,
        "top_longterm": top_longterm,
        "top_fundamental": top_fundamental,
        "aschenbrenner_holdings": aschenbrenner_holdings,
        "top_hype": top_hype,
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
