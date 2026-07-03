"""Autonomous virtual paper-trading portfolio — an honest self-check of the tips.

Runs continuously over months (never reset). Every run it does a full REVISION
of each holding against the *fresh* analysis and decides intelligently — it does
NOT use rigid fixed take-profit / stop-loss levels:

  - Winners are let run (no fixed +30% exit); protected by a TRAILING stop once
    they are meaningfully up, so a big move isn't cut short but gains are locked.
  - Losers / broken theses are cut early — sold as soon as the reason to own the
    stock is gone (trend turned down, rating dropped, fell out of the top tips),
    at whatever P&L that happens to be, instead of waiting for a −15% line.
  - A catastrophic hard stop remains only as a safety net.
  - Freed cash is reinvested into the best not-yet-held pick (skipping downtrends).
  - Trades at most once per calendar day; marks-to-market on every run.

Simplifications (documented for honesty): virtual/fractional sizing, no fees or
slippage. All prices are already converted to USD upstream.
"""
import json
from datetime import datetime, timezone

from .config import DATA

PORTFOLIO_FILE = DATA / "portfolio.json"

START_CAPITAL = 10000.0
K = 10                 # target number of holdings
HYSTERESIS = 6         # tolerance before a top-dropout is forced out
HARD_STOP = -22.0      # % — catastrophic safety net only
TRAIL_ARM = 12.0       # % — once up this much from entry, trailing stop is armed
TRAIL_GIVEBACK = 10.0  # % — sell if it gives back this much from its own peak
SELL_SCORE = 50        # rating below this -> thesis gone, sell
HOLD_SCORE = 60        # a top-dropout may be kept only while still this strong
BUY_SCORE = 60         # only buy fresh picks at least this strong
MIN_STAKE = 50.0


def _today():
    return datetime.now(timezone.utc).strftime("%Y-%m-%d")


def _init(today):
    return {
        "start_capital": START_CAPITAL, "created": today, "cash": START_CAPITAL,
        "last_trade_date": None, "positions": {}, "trade_log": [],
        "equity_curve": [], "realized_pnl": 0.0,
    }


def _load(today):
    if PORTFOLIO_FILE.exists():
        try:
            return json.loads(PORTFOLIO_FILE.read_text(encoding="utf-8"))
        except Exception:  # noqa: BLE001
            pass
    return _init(today)


def _save(p):
    PORTFOLIO_FILE.write_text(json.dumps(p, ensure_ascii=False, indent=1), encoding="utf-8")


def _mark(pos, price):
    """Update a position's live value + peak (for the trailing stop)."""
    if price and pos["entry_price"]:
        pos["last_price"] = round(price, 4)
        pos["peak_price"] = round(max(pos.get("peak_price") or pos["entry_price"], price), 4)
        pos["value_eur"] = round(pos["stake_eur"] * price / pos["entry_price"], 2)
        pos["pnl_eur"] = round(pos["value_eur"] - pos["stake_eur"], 2)
        pos["pnl_pct"] = round((price / pos["entry_price"] - 1) * 100, 2)


def _sell_reason(pos, cur, score, phase, in_top_hyst):
    """Intelligent, signal-based revision. Returns a reason string or None."""
    entry = pos["entry_price"]
    peak = pos.get("peak_price") or entry
    ret = (cur / entry - 1) * 100
    give = (cur / peak - 1) * 100          # <= 0, drawdown from the peak
    peak_gain = (peak / entry - 1) * 100
    down = phase.startswith("Abwärtstrend")

    if ret <= HARD_STOP:
        return f"Not-Stop ({ret:+.0f}%)"
    if peak_gain >= TRAIL_ARM and give <= -TRAIL_GIVEBACK:
        return f"Trailing-Stop – Gewinn gesichert ({ret:+.0f}%, vom Hoch {give:.0f}%)"
    if down:
        return f"Trend gedreht (Abwärtstrend) → raus ({ret:+.0f}%)"
    if score < SELL_SCORE:
        return f"Rating auf {score} gefallen → These weg ({ret:+.0f}%)"
    if not in_top_hyst and score < HOLD_SCORE:
        return f"aus den Top-Tipps gefallen ({ret:+.0f}%)"
    return None


def update_portfolio(rows, today=None):
    """Run the intelligent paper strategy against today's analysed rows."""
    today = today or _today()
    p = _load(today)

    price_by = {r["symbol"]: r.get("price") for r in rows}
    name_by = {r["symbol"]: r.get("name") for r in rows}
    score_by = {r["symbol"]: r.get("radar_score") for r in rows}
    asch_by = {r["symbol"]: r.get("aschenbrenner") for r in rows}
    phase_by = {r["symbol"]: (r.get("trend_phase") or {}).get("phase", "") for r in rows}

    ranked = [r["symbol"] for r in sorted(
        [r for r in rows if r.get("radar_score") is not None],
        key=lambda r: r["radar_score"], reverse=True)]
    top_k = set(ranked[:K])
    top_hyst = set(ranked[:K + HYSTERESIS])

    for sym, pos in p["positions"].items():
        _mark(pos, price_by.get(sym))

    trade_today = today != p.get("last_trade_date")
    if trade_today:
        # --- REVISION: re-check every holding against fresh signals ---
        for sym in list(p["positions"]):
            pos = p["positions"][sym]
            cur = price_by.get(sym)
            if not cur:
                continue
            reason = _sell_reason(pos, cur, score_by.get(sym) or 0,
                                  phase_by.get(sym, ""), sym in top_hyst)
            if reason:
                value = pos["stake_eur"] * cur / pos["entry_price"]
                pnl = value - pos["stake_eur"]
                ret = (cur / pos["entry_price"] - 1) * 100
                p["cash"] += value
                p["realized_pnl"] += pnl
                p["trade_log"].append({
                    "date": today, "action": "SELL", "symbol": sym,
                    "name": pos.get("name"), "price": round(cur, 4),
                    "stake_eur": round(pos["stake_eur"], 2), "value_eur": round(value, 2),
                    "pnl_eur": round(pnl, 2), "pnl_pct": round(ret, 2), "reason": reason,
                })
                del p["positions"][sym]

        # --- BUY: refill empty slots with the best fresh picks (skip downtrends) ---
        invested = sum(pos["value_eur"] for pos in p["positions"].values())
        equity = p["cash"] + invested
        target_stake = equity / K
        slots = K - len(p["positions"])
        candidates = [s for s in ranked
                      if s in top_k and s not in p["positions"] and price_by.get(s)
                      and (score_by.get(s) or 0) >= BUY_SCORE
                      and not phase_by.get(s, "").startswith("Abwärtstrend")
                      and not (asch_by.get(s) and asch_by[s]["stance"] == "SHORT_BET")]
        for sym in candidates[:max(0, slots)]:
            stake = min(target_stake, p["cash"])
            if stake < MIN_STAKE:
                break
            cur = price_by[sym]
            p["cash"] -= stake
            p["positions"][sym] = {
                "name": name_by.get(sym), "entry_price": round(cur, 4),
                "peak_price": round(cur, 4), "entry_date": today,
                "stake_eur": round(stake, 2), "score_at_entry": score_by.get(sym),
            }
            _mark(p["positions"][sym], cur)
            p["trade_log"].append({
                "date": today, "action": "BUY", "symbol": sym, "name": name_by.get(sym),
                "price": round(cur, 4), "stake_eur": round(stake, 2), "value_eur": round(stake, 2),
                "pnl_eur": 0.0, "pnl_pct": 0.0, "reason": "Top-Tipp",
            })

        p["last_trade_date"] = today

    invested = sum(pos["value_eur"] for pos in p["positions"].values())
    equity = round(p["cash"] + invested, 2)
    snap = {"date": today, "equity": equity, "cash": round(p["cash"], 2),
            "invested": round(invested, 2), "n_positions": len(p["positions"])}
    if p["equity_curve"] and p["equity_curve"][-1]["date"] == today:
        p["equity_curve"][-1] = snap
    else:
        p["equity_curve"].append(snap)

    _save(p)

    total_return = (equity / p["start_capital"] - 1) * 100
    return {
        "equity": equity, "cash": round(p["cash"], 2), "invested": round(invested, 2),
        "total_return_pct": round(total_return, 2), "realized_pnl": round(p["realized_pnl"], 2),
        "n_positions": len(p["positions"]), "n_trades": len(p["trade_log"]),
        "start_capital": p["start_capital"], "created": p["created"],
    }
