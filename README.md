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
- a separated identity contract: configured `short_name`, provider-preferred
  `display_name_full`, normalized provider headquarters (`provider_country` /
  `headquarters_country`), nullable verified `legal_domicile`, listing
  market/country, sector/industry, and conservatively mapped economic exposure.
  Provider country is not legal domicile. `issuer_country` is retained only as an
  explicitly deprecated alias of verified `legal_domicile`; listing country is
  never silently used as domicile or exposure. Provider headquarters is also
  never used as an economic-exposure proxy: without a documented ticker override
  or the explicit Hong Kong listing rule, exposure remains `Nicht verfügbar`.

`data/output/latest.json` uses schema `stock-radar-output`, version 3, and contains:

- `data_status`: price/fresh-bar coverage, SLA, age distribution, failures, and
  blocking reasons;
- `model_status`: explicit validation and non-actionability metadata;
- `rankings_by_currency_asset`: conservative technical partitions;
- `insight_rankings`: transparent, research-only category formulas and items;
- `insight_metadata`: explicit `heuristic_unvalidated` provenance;
- `all`: the complete analyzed-row contract.

The dashboard refuses to render research cards when the snapshot is corrupt,
stale, incomplete, or uses an unsupported schema.

## Login-free online dashboard

The public GitHub Pages dashboard is available without a Streamlit account:

**https://ththom1985.github.io/stock-radar/**

`python -m src.export_static` creates the compact schema-v3 `docs/data.json`
payload from the validated output-v3 snapshot. Both analysis workflows regenerate and publish this
payload after a successful run, so the Pages dashboard stays synchronized with
`data/output/latest.json`.
The exporter keeps all rendered identity, metric, scenario, news (up to three),
jurisdiction, valuation-thesis and entry-thesis content. Repeated per-row
provenance/actionability fields and non-rendered compatibility/context duplicates
are represented once by the top-level `instrument_contract` and
`insight_metadata.provenance_catalog`. It serializes one deterministic compact
UTF-8 byte sequence and measures that exact sequence. The hard write guard remains
10 MiB; the real-payload regression target is at most 8.5 MiB to retain at least
15% operational headroom.

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

## Insight cockpit

The useful insight layer is separate from the conservative core technical
partition. Every group is marked `heuristic_unvalidated`, records its inputs and
missing inputs, and remains `actionable: false`.

- **Tages-Setups / `daily_setups`**: 45% completed-daily trend, 35% entry
  timing, 20% completed-daily momentum context; falling knives and high critical
  downside structures are excluded.
- **Unterbewertet / `undervalued_quality`**: raw score = 55% Value + 45% Quality,
  only for company equities with complete/current fundamentals. The visible
  risk-adjusted score is
  `clamp(raw - min(45, jurisdiction[0..20] + size/liquidity[0..12] +
  cyclical-peak[0..8] + shrinking-fundamentals[0..10] +
  weak-trend/downside[0..10]), 0, 100)`. Every component remains dimensionless
  and is exported with its reason/evidence IDs; raw Value and Quality scores are
  unchanged. Cyclical-peak evidence uses positive peak-cycle conditions and cannot
  reuse the negative-growth evidence owned by the shrinking component.
  Banks, insurers, REITs and other generic non-comparable cases remain excluded.
- **Potenzial / `analyst_potential`**: analyst target gap with at least five
  analysts, plus visible trend/timing components and explicit overbought or
  weak-trend penalties. Analyst consensus is not a model forecast.
- **Einstiegs-Timing / `entry_watchlist`**: timing and trend observation with
  nearby support, non-negative completed-day context and no falling knife.
- **Fallende Messer**: warning severity from 5/20-day deterioration without
  stabilization; never an opportunity recommendation.
- **Bodenbildung**: multi-signal watchlist, always labelled speculative.
- **Risiko-Watch**: transparent downside, volatility, earnings and knife flags.

Per instrument the output includes German research summary, timing score/reason,
falling-knife and bottoming state, support/downside structure, analyst and
valuation context, risks, thesis, priced-in warning, technical observation zone,
news and 1M/6M/12M/24M heuristic scenario ranges. Scenario ranges remain
excluded from every core comparable rank and are never described as probable,
median or expected outcomes.

`jurisdiction_risk` is a bounded `heuristic_unvalidated` context, not a precise
DCF or mathematically proven discount. China economic exposure is explicitly
separated from provider headquarters (for example PDD/Ireland and
TCOM/Singapore) and
records policy/data regulation, capital-control/state-influence, geopolitical,
audit/delisting context. Cayman/VIE wording is emitted only for the verified
BABA, PDD and TCOM legal-domicile/structure overrides, sourced to their cited
2026 SEC Form 20-F accessions; a configured ADR without that evidence is labelled
only as ADR context.
Hong Kong/China listings are distinguished from US listings. Explicit Argentina,
Brazil/state-linked and selected emerging-market exposures receive their own
currency/policy/governance context rather than a blanket non-US penalty.

Current complete comparable fundamentals also produce `valuation_thesis`
(`why_it_looks_cheap`, justified-discount evidence, strongest evidence and
counterarguments, raw/penalty/adjusted scores and value-trap risk). Completed
technical inputs produce `entry_thesis` with concrete RSI, SMA50/200, MACD,
20/60-day, support/ATR and earnings context, plus confirmation and invalidation.
Analyst context stays separate. Both groups are always `actionable: false`;
stale/incomplete fundamentals produce no valuation score or ranking.
User-facing setup and timing language is observational (`Tages-Setups`,
`Einstiegs-Timing`); recommendation-oriented legacy wording is rejected by the
deep output contract. Provider analyst keys are translated only as explicitly
attributed analyst consensus labels such as `Buy-Konsens`.

If current/complete company fundamentals are unavailable, retained cache scores
are never used in narratives or the static UI: valuation, profitability and
quality are shown as unavailable. Speculative bottoming observations remain
separate from ordinary setup/timing lists and cannot override final downtrend,
late-stage, weak-trend or falling-knife caps.

The static cockpit applies the same fail-closed contract before rendering any
tips: status must be `ok`, `data_actionable` true, blocking reasons empty, model
and insight actionability false, and `generated_at` no older than 36 hours.

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

The reliability cache/portfolio schemas remain independently versioned. Output
schema v3 carries the provider-free insight, identity, jurisdiction and thesis
contract.

Run the dashboard:

```powershell
.\.venv\Scripts\python.exe -m pip install -r requirements.txt
.\.venv\Scripts\streamlit.exe run dashboard/app.py
```

Build the login-free static dashboard payload:

```powershell
.\.venv\Scripts\python.exe -m src.export_static
```

Provider-free enrichment of an existing v2/v3 snapshot and static export:

```powershell
.\.venv\Scripts\python.exe -m src.enrich_snapshot --in-place --export-static
```

Without `--in-place`, the utility writes
`data/output/latest.enriched.json` for review. All writes are atomic and no
provider request is made. The command may merge identity, listing and USD-normalized
market-cap context from the existing local `data/fundamentals.json` cache and
`data/tickers.csv`; it does not refresh either source. It rehydrates every core
ranking member from the enriched symbol row while preserving exact
currency/asset/symbol order.

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
