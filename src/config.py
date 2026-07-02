"""Central configuration: paths and analysis parameters."""
import os
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent


def _load_dotenv():
    """Minimal .env loader (no dependency): KEY=value lines into os.environ."""
    env = ROOT / ".env"
    if not env.exists():
        return
    for line in env.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        k, v = line.split("=", 1)
        os.environ.setdefault(k.strip(), v.strip().strip('"').strip("'"))


_load_dotenv()
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

# --- Feintuning: Gewichte für den Investment-Score (Langzeit) ---
# Anteil Technik (Trend) vs. Fundamental. Müssen zusammen 1.0 ergeben.
INVEST_W_TECH = 0.5
INVEST_W_FUND = 0.5

# --- Indicator windows ---
RSI_PERIOD = 14
ATR_PERIOD = 14
SMA_WINDOWS = (20, 50, 200)
BB_WINDOW = 20
BB_STD = 2
VOL_AVG_WINDOW = 20
BREAKOUT_WINDOW = 20
