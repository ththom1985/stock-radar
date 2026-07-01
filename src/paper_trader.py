"""Autonomous virtual paper-trading portfolio — an honest self-check of the tips.

Strategy (long-only, mechanical):
  - Universe of candidates = ranked by radar_score.
  - HOLD the top K equal-weighted (~start_capital/K each).
  - SELL a position when it drops out of the top (K+hysteresis), its rating
    falls below SCORE_FLOOR, hits STOP_LOSS, or reaches TAKE_PROFIT.
  - BUY the best not-yet-held candidates to refill empty slots.
  - Trades once per calendar day; marks-to-market on every run.

Simplifications (documented for honesty):
  - FX ignored: each position is a EUR bucket that grows by the stock's own
    native-currency return -> measures stock-PICKING, not currency effects.
  - Fractional/virtual sizing, no fees/slippage.
"""
import json
from datetime import datetime, timezone

from .config import DATA

PORTFOLIO_FILE = DATA / "portfolio.json"

START_CAPITAL = 10000.0
K = 10                 # target number of holdings
HYSTERESIS = 6         # sell only if it falls out of top (K+HYSTERESIS)
SCORE_FLOOR = 55       # minimum radar_score to hold/buy
STOP_LOSS = -15.0      # %
TAKE_PROFIT = 30.0     # %
MIN_STAKE = 50.0       # don't buy with less than this cash


def _today():
    return datetime.now(timezone.utc).strftime("%Y-%m-%d")


def _init(today):
    return {
        "start_capital": START_CAPITAL,
        "created": today,
        "cash": START_CAPITAL,
        "last_trade_date": None,
        "positions": {},          # symbol -> {...}
        "trade_log": [],
        "equity_curve": [],
        "realized_pnl": 0.0,
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
    """Update a position's live value from current price."""
    if price and pos["entry_price"]:
        pos["last_price"] = round(price, 4)
        pos["value_eur"] = round(pos["stake_eur"] * price / pos["entry_price"], 2)
        pos["pnl_eur"] = round(pos["value_eur"] - pos["stake_eur"], 2)
        pos["pnl_pct"] = round((price / pos["entry_price"] - 1) * 100, 2)


def update_portfolio(rows, today=None):
    """Run the paper strategy against today's analysed rows. Returns summary dict."""
    today = today or _today()
    p = _load(today)

    price_by = {r["symbol"]: r.get("price") for r in rows}
    name_by = {r["symbol"]: r.get("name") for r in rows}
    score_by = {r["symbol"]: r.get("radar_score") for r in rows}
    asch_by = {r["symbol"]: r.get("aschenbrenner") for r in rows}

    ranked = [r["symbol"] for r in sorted(
        [r for r in rows if r.get("radar_score") is not None],
        key=lambda r: r["radar_score"], reverse=True)]
    top_k = set(ranked[:K])
    top_hyst = set(ranked[:K + HYSTERESIS])

    # Always mark existing positions to market
    for sym, pos in p["positions"].items():
        _mark(pos, price_by.get(sym))

    trade_today = today != p.get("last_trade_date")
    if trade_today:
        # --- SELLS ---
        for sym in list(p["positions"]):
            pos = p["positions"][sym]
            cur = price_by.get(sym)
            if not cur:
                continue
            ret = (cur / pos["entry_price"] - 1) * 100
            score = score_by.get(sym)
            reason = None
            if ret <= STOP_LOSS:
                reason = f"Stop-Loss {STOP_LOSS:.0f}%"
            elif ret >= TAKE_PROFIT:
                reason = f"Ziel +{TAKE_PROFIT:.0f}% erreicht"
            elif sym not in top_hyst:
                reason = "aus Top-Tipps gefallen"
            elif score is not None and score < SCORE_FLOOR:
                reason = f"Rating unter {SCORE_FLOOR}"
            if reason:
                value = pos["stake_eur"] * cur / pos["entry_price"]
                pnl = value - pos["stake_eur"]
                p["cash"] += value
                p["realized_pnl"] += pnl
                p["trade_log"].append({
                    "date": today, "action": "SELL", "symbol": sym,
                    "name": pos.get("name"), "price": round(cur, 4),
                    "stake_eur": round(pos["stake_eur"], 2), "value_eur": round(value, 2),
                    "pnl_eur": round(pnl, 2), "pnl_pct": round(ret, 2), "reason": reason,
                })
                del p["positions"][sym]

        # --- BUYS: refill empty slots ---
        invested = sum(pos["value_eur"] for pos in p["positions"].values())
        equity = p["cash"] + invested
        target_stake = equity / K
        slots = K - len(p["positions"])
        candidates = [s for s in ranked
                      if s in top_k and s not in p["positions"]
                      and price_by.get(s) and (score_by.get(s) or 0) >= SCORE_FLOOR
                      and not (asch_by.get(s) and asch_by[s]["stance"] == "SHORT_BET")]
        for sym in candidates[:max(0, slots)]:
            stake = min(target_stake, p["cash"])
            if stake < MIN_STAKE:
                break
            cur = price_by[sym]
            p["cash"] -= stake
            p["positions"][sym] = {
                "name": name_by.get(sym), "entry_price": round(cur, 4),
                "entry_date": today, "stake_eur": round(stake, 2),
                "score_at_entry": score_by.get(sym),
            }
            _mark(p["positions"][sym], cur)
            p["trade_log"].append({
                "date": today, "action": "BUY", "symbol": sym, "name": name_by.get(sym),
                "price": round(cur, 4), "stake_eur": round(stake, 2), "value_eur": round(stake, 2),
                "pnl_eur": 0.0, "pnl_pct": 0.0, "reason": "Top-Tipp",
            })

        p["last_trade_date"] = today

    # --- equity snapshot (one per day, overwrite same day) ---
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
