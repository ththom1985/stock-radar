# Stock Radar — completed-daily-bar research

Stock Radar is a **research tool**, not a trading system or investment adviser. It
builds a conservative snapshot from completed Yahoo Finance daily bars and
separately presents company equities, funds/ETFs, crypto, and other instruments.

## Validation status

The deployed composite is **UNVALIDATED** and its output is always marked
`actionable: false`.

- No profitability, alpha, probability, confidence, expected-return, or
  intraday claim is made.
- Scenario ranges are uncalibrated log-return/volatility illustrations. They are
  excluded from ranking and cannot produce negative prices.
- The available backtest can validate only the technical score because reliable
  point-in-time histories for fundamentals, analyst targets, and news are not
  available. It cannot validate the deployed composite.
- Optional news, analyst, macro, social/expert, and deep-fundamental context does
  not change the comparable core ranking.

## Daily data contract

The pipeline downloads `1d` Yahoo bars with raw prices and available corporate
actions. It preserves a provider timezone when present; naive indexes use
conservative symbol-based US/Europe/Asia/24x7 session profiles. A same-date bar
is accepted only after the mapped close plus a 90-minute buffer. Unknown and
24x7 markets lag until the UTC day and buffer have ended.

Every output row contains:

- completed `bar_date` and source timestamp;
- completed-bar age;
- source interval (`1d`);
- local and USD price provenance;
- asset type and feature-coverage flags.

`data/output/latest.json` uses schema `stock-radar-output`, version 2, and contains:

- `data_status`: price/fresh-bar coverage, SLA, age distribution, failures, and
  blocking reasons;
- `model_status`: explicit validation and non-actionability metadata;
- `rankings_by_asset`: separate, coverage-consistent research rankings;
- `all`: the complete analyzed-row contract.

The dashboard refuses to render research cards when the snapshot is corrupt,
stale, incomplete, or uses an unsupported schema.

## Reliability model

- JSON state is replaced atomically using a flushed sibling temporary file.
- Existing corrupt JSON raises an explicit error; it is never silently replaced
  with an empty cache or fresh portfolio.
- Failed refreshes retain stale-good fundamentals, deep fundamentals, earnings,
  FX, patterns, and news while recording failure metadata.
- Missing non-USD FX excludes affected symbols. It never assumes a 1:1 USD rate.
- Price ingestion retries, splits failed batches, retries missing names
  individually, and writes `data/failed_symbols.json`.
- Rankings are suppressed when coverage/freshness is below the configured SLA.

## Asset comparability

Asset type uses provider `quoteType` when available and conservative symbol/name
fallbacks. Company fundamental scoring is disabled for ETFs, funds, crypto,
indices, futures, and other non-company instruments.

All overall rankings now use completed-daily technical context only. Generic
absolute fundamental bands remain descriptive because robust sector-neutral,
point-in-time peer ranks are not implemented; banks, insurers, REITs, and
industrial companies are therefore not forced into one generic valuation rank.
The output exposes complete/current peer counts and feature coverage.

Rank-eligible, technical, and company-fundamental descriptive coverage are
blocking data gates with configurable minima by asset class.

There is no global cross-currency ordering. Research lists are partitioned by
trading currency and asset class because current FX cannot make historical
local-currency indicators point-in-time comparable.

## Paper simulation

The paper module is an **UNVALIDATED, non-actionable simulation**:

- a completed-bar signal creates a pending long-only order at observation time;
- no bar before UTC creation-date + two calendar dates may fill, and a verified
  session-open timestamp must be later than order creation;
- fills store signal/fill timestamps, quantity, raw/execution prices, commission,
  slippage, not-before date, and fill-observation time;
- only USD company equities are eligible until point-in-time historical FX exists;
- issuer uniqueness, sector/country caps, minimum dollar volume, and maximum
  ATR/annualized volatility apply (no correlation-optimization claim);
- bounded sparse action history is replayed using stable symbol/type/ex-date/value
  keys; late-reported actions remain eligible and corrected values apply explicit
  delta/correction ledger entries exactly once;
- legacy portfolio data is preserved and marked during migration, never reset.
  Legacy positions are frozen because their historical fills are incompatible
  with v2 accounting; starting a separate clean v2 simulation requires an
  explicit user decision (`STOCK_RADAR_START_NEW_PAPER=1`). Legacy data remains
  under `legacy_archive` / `legacy_frozen_positions`.

Corporate-action coverage cannot be guaranteed across missed runs. Consequently,
paper performance remains explicitly non-actionable.

Portfolio benchmarks use the same completed-session ingestion contract. Values
are stored with their own bar dates and rebased only on common portfolio/benchmark
as-of dates.

## Automation

`.github/workflows/daily.yml` runs once on weekdays at **23:15 UTC**, after the
major US markets are closed in both daylight-saving seasons. The legacy
`intraday.yml` filename is manual-only and has no schedule or intraday mode.
Both workflows skip their job unless `github.ref` is exactly
`refs/heads/main`, explicitly check out `main`, and can therefore never publish
a feature-branch dispatch into `main`.

Workflows use verified action commit SHAs, pinned direct Python dependencies,
non-persisted checkout credentials, complete cache publication, and failing
pull/push retries. The deterministic unit suite is a blocking step before
analysis. Publication failures are not converted into successful jobs.

## Setup and execution

```powershell
cd "C:\path\to\Stock-Radar"
python -m venv .venv
.\.venv\Scripts\python.exe -m pip install -r requirements-ci.txt
.\.venv\Scripts\python.exe -m src.analyze
```

The first v2 run migrates legacy caches as they are refreshed and migrates the
paper portfolio with its original accounting archived in-place.

Run the dashboard:

```powershell
.\.venv\Scripts\python.exe -m pip install -r requirements.txt
.\.venv\Scripts\streamlit.exe run dashboard/app.py
```

Run deterministic tests:

```powershell
.\.venv\Scripts\python.exe -m unittest discover -s tests -v
```

### Safe bounded smoke run

This exercises the real pipeline on a bounded universe while redirecting every
JSON/cache/portfolio write away from tracked production data:

```powershell
$env:STOCK_RADAR_DRY_RUN = "1"
$env:STOCK_RADAR_DRY_RUN_DIR = "$PWD\data\dry-run"
$env:STOCK_RADAR_MAX_SYMBOLS = "25"
.\.venv\Scripts\python.exe -m src.analyze
```

Unset those variables before a production run. `data/dry-run/` is ignored by Git.

### Full-universe provider-backed market-data contract run

This scans every configured market-data symbol and validates classification,
session mappings, completed-bar and feature coverage. Slow enrichment and paper
simulation are explicitly skipped and declared in output; it does not claim full
model readiness.

```powershell
Remove-Item Env:STOCK_RADAR_MAX_SYMBOLS -ErrorAction SilentlyContinue
$env:STOCK_RADAR_DRY_RUN = "1"
$env:STOCK_RADAR_DRY_RUN_DIR = "$PWD\data\dry-run-market-full"
$env:STOCK_RADAR_MARKET_DATA_ONLY = "1"
.\.venv\Scripts\python.exe -m src.analyze
```

### Full-universe full-pipeline dry run

```powershell
Remove-Item Env:STOCK_RADAR_MAX_SYMBOLS -ErrorAction SilentlyContinue
Remove-Item Env:STOCK_RADAR_MARKET_DATA_ONLY -ErrorAction SilentlyContinue
$env:STOCK_RADAR_DRY_RUN = "1"
$env:STOCK_RADAR_DRY_RUN_DIR = "$PWD\data\dry-run-pipeline-full"
.\.venv\Scripts\python.exe -m src.analyze
```

Both commands redirect caches, output, and portfolio state. Neither writes
tracked production data and neither requires Anthropic.

Run the technical-only validation backtest explicitly; it is intentionally not a
production prerequisite:

```python
from src.backtest import run_backtest
run_backtest(["AAPL", "MSFT", "..."], round_trip_cost_bps=20)
```

## Configuration

Environment variables:

- `STOCK_RADAR_MIN_COVERAGE_PCT` (default `97.0`)
- `STOCK_RADAR_MAX_BAR_AGE_DAYS` (default `4`)
- `STOCK_RADAR_MAX_OUTPUT_AGE_HOURS` (default `36`)
- `STOCK_RADAR_MIN_RANK_COVERAGE_COMPANY_PCT` (default `70`)
- `STOCK_RADAR_MIN_RANK_COVERAGE_FUND_PCT` (default `70`)
- `STOCK_RADAR_MIN_RANK_COVERAGE_CRYPTO_PCT` (default `70`)
- `STOCK_RADAR_MIN_RANK_COVERAGE_OTHER_PCT` (default `70`)
- `STOCK_RADAR_MIN_COMPANY_FUNDAMENTAL_COVERAGE_PCT` (default `60`)
- `STOCK_RADAR_DEEP_MAX` (default `60`)
- `STOCK_RADAR_MIN_DOLLAR_VOLUME` (default `20000000`)
- `STOCK_RADAR_MAX_PAPER_ATR_PCT` (default `5`)
- `STOCK_RADAR_MAX_PAPER_ANNUAL_VOL_PCT` (default `60`)
- `STOCK_RADAR_MAX_PAPER_PER_SECTOR` (default `3`)
- `STOCK_RADAR_MAX_PAPER_PER_COUNTRY` (default `4`)
- `STOCK_RADAR_PAPER_ORDER_MAX_AGE_DAYS` (default `7`)
- `STOCK_RADAR_PAPER_SLIPPAGE_BPS` (default `10`)
- `STOCK_RADAR_PAPER_COMMISSION_BPS` (default `5`)
- `STOCK_RADAR_START_NEW_PAPER` (default unset; explicit legacy-migration decision)
- `STOCK_RADAR_DRY_RUN`, `STOCK_RADAR_DRY_RUN_DIR`, `STOCK_RADAR_MAX_SYMBOLS`
  (safe bounded smoke controls)
- `STOCK_RADAR_MARKET_DATA_ONLY` (skip slow enrichment/paper and declare skipped layers)

The ticker universe remains `data/tickers.csv`.

## Session-map limitation

Every suffix currently present in `data/tickers.csv` has an explicit
conservative IANA-timezone/open/close profile. Additional reviewed profiles
cover Saudi Arabia, Austria, Brazil, Mexico, South Africa, and Canadian NEO/CSE.
Unknown suffixes and unverified derivative sessions are explicitly blocked,
not silently treated as UTC. Exchange holiday calendars are not bundled; the
90-minute close buffer and freshness gates remain conservative but cannot
replace a full holiday calendar.
