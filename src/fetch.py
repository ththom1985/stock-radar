"""Download daily OHLCV price history via yfinance (free, ~15 min delayed)."""
import time
import yfinance as yf
from .config import HISTORY_PERIOD, FETCH_CHUNK, FETCH_PAUSE


def _chunks(seq, n):
    for i in range(0, len(seq), n):
        yield seq[i:i + n]


def fetch_prices(symbols, period=HISTORY_PERIOD):
    """Download daily bars for many symbols.

    Returns dict: symbol -> DataFrame with columns Open/High/Low/Close/Volume.
    Symbols that fail or return no data are silently skipped.
    """
    result = {}
    total = len(symbols)
    for idx, chunk in enumerate(_chunks(symbols, FETCH_CHUNK), 1):
        try:
            data = yf.download(
                tickers=chunk,
                period=period,
                interval="1d",
                group_by="ticker",
                auto_adjust=True,
                threads=True,
                progress=False,
            )
        except Exception as exc:  # noqa: BLE001
            print(f"  Batch {idx} fehlgeschlagen: {exc}")
            continue

        for sym in chunk:
            try:
                if len(chunk) == 1:
                    df = data
                else:
                    df = data[sym]
                df = df.dropna(how="all")
                if df is not None and not df.empty and "Close" in df.columns:
                    result[sym] = df
            except Exception:  # noqa: BLE001
                continue

        print(f"  Batch {idx}: {min(idx * FETCH_CHUNK, total)}/{total} Ticker verarbeitet")
        time.sleep(FETCH_PAUSE)
    return result
