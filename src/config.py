"""Central configuration: paths and analysis parameters."""
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
DATA = ROOT / "data"
PRICES = DATA / "prices"
OUTPUT = DATA / "output"
HISTORY = OUTPUT / "history"
TICKERS_CSV = DATA / "tickers.csv"

for _d in (DATA, PRICES, OUTPUT, HISTORY):
    _d.mkdir(parents=True, exist_ok=True)

# --- Data fetching ---
HISTORY_PERIOD = "1y"   # how much daily history to pull (needs >= 200 days for SMA200)
FETCH_CHUNK = 100       # tickers per yfinance batch request
FETCH_PAUSE = 1.0       # seconds between batches (be gentle to Yahoo)

# --- Output ---
TOP_N = 10

# --- Indicator windows ---
RSI_PERIOD = 14
ATR_PERIOD = 14
SMA_WINDOWS = (20, 50, 200)
BB_WINDOW = 20
BB_STD = 2
VOL_AVG_WINDOW = 20
BREAKOUT_WINDOW = 20
