"""Load the ticker universe from data/tickers.csv."""
import csv
from .config import SP1500_CSV, SP1500_STAGE, TICKERS_CSV


def load_universe():
    """Return a list of dicts: {symbol, name, exchange}.

    CSV columns expected: symbol,name,exchange
    Lines whose symbol is empty or starts with '#' are ignored.
    """
    tickers = []
    seen = set()
    for path in (TICKERS_CSV, SP1500_CSV):
        if not path.exists():
            continue
        with open(path, newline="", encoding="utf-8-sig") as f:
            reader = csv.DictReader(f)
            for row in reader:
                if path == SP1500_CSV:
                    try:
                        stage = int(row.get("stage") or 1)
                    except ValueError:
                        continue
                    if stage > SP1500_STAGE:
                        continue
                sym = (row.get("symbol") or "").strip().upper()
                if not sym or sym.startswith("#") or sym in seen:
                    continue
                seen.add(sym)
                tickers.append({
                    "symbol": sym,
                    "name": (row.get("name") or "").strip(),
                    "exchange": (row.get("exchange") or "").strip(),
                })
    return tickers
