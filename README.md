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
jurisdiction, valuation-thesis, entry-thesis, and Sweet-Spot reason content. Repeated per-row
provenance/actionability fields and non-rendered compatibility/context duplicates
are represented once by the top-level `instrument_contract` and
`insight_metadata.provenance_catalog`; the exact zone formula, thresholds, and
provenance are additionally represented once by `sweet_spot_contract`. It
serializes one deterministic compact UTF-8 byte sequence and measures that exact
sequence. The hard ceiling remains 10 MiB, while publication is refused above
the stricter 8.5 MiB operational target.

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

- **Sweet Spot / `in_sweet_spot`**: only combined `in_zone_confirmed` rows;
  the current USD close must be inside the mathematical zone and every technical
  and applicable investor safety filter must pass. `approaching_sweet_spot`
  contains only amber observations. Both lists are partitioned by trading
  currency; their score is heuristic evidence quality, not a probability.
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
- **Einstiegs-Timing / `entry_watchlist`**: timing and trend observation that is
  mathematically inside the Sweet-Spot zone or at most 1 ATR above it, with
  non-negative completed-day context and no falling knife.
- **Fallende Messer**: warning severity from 5/20-day deterioration without
  stabilization; never an opportunity recommendation.
- **Bodenbildung**: multi-signal watchlist, always labelled speculative.
- **Risiko-Watch**: transparent downside, volatility, earnings and knife flags.

Per instrument the output includes German research summary, timing score/reason,
falling-knife and bottoming state, support/downside structure, analyst and
valuation context, risks, thesis, priced-in warning, Sweet-Spot observation zone,
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

### Sweet-Spot-Beobachtungszone

`src/sweet_spot.py` is a pure deterministic completed-daily model. Every row
receives a `sweet_spot` contract, including unavailable rows. All absolute input
levels and output marks are USD-normalized upstream. The display wording is
**Sweet-Spot-Beobachtungszone / technische Einstiegsbeobachtung** and always
states **Beobachtungszone, keine Ordermarke**.

Candidate references are positive finite SMA20, SMA50, SMA150, SMA200, EMA21,
prior pivot, Pivot S1, and 20-day low. The current price is never a reference
source; 52-week low is intentionally excluded. A level is tactical only from
`-4.0 ATR` below through `+2.0 ATR` above the current close. Exact/near duplicates
within `max(0.02 ATR, 0.02% of price)` count once. Relevance weights are SMA20
1.00, SMA50 1.20, SMA150 0.85, SMA200 1.15, EMA21 1.10, prior pivot 1.00,
Pivot S1 1.15, and 20-day low 1.05. A level at/below price keeps full weight; an
overhead level receives role factor 0.85 in Stage 2 and 0.70 otherwise.
Independence is counted by conservative source family, not by raw level:
`pivot` = prior pivot/Pivot S1, `moving_average_fast` = SMA20/EMA21,
`moving_average_medium` = SMA50, `moving_average_long` = SMA150/SMA200, and
`price_structure` = 20-day low. Two derivatives from one family can influence
the zone envelope but never satisfy the two-family green gate.

All contiguous candidate windows whose envelope is at most `0.90 ATR` are
evaluated. The deterministic cluster score is:

```text
34 * independent family count
+ 6 * raw level count
+ 12 * sum(max weight per family)
+ 18 * max(0, 1 - abs(weighted center - price) / (4 ATR))
+ 8 * support-family weight share
```

`IDEAL = sum(level * role-adjusted weight) / sum(weight)`. Raw zone bounds are
`min(IDEAL - 0.35 ATR, cluster low - 0.10 ATR)` and
`max(IDEAL + 0.35 ATR, cluster high + 0.10 ATR)`. Width is therefore at least
`0.70 ATR` and is proportionally capped at `1.20 ATR`. If no two-family cluster
exists, a deterministic non-current single anchor is used. Tactical structural
MA/EMA/20-day-low anchors rank first, then structural anchors down to `-10 ATR`,
then pivot-only anchors; within a tier relevance weight and proximity break ties.
A complete Stage-2 case is labelled `single_anchor`; all other numeric fallbacks
are `reference_only`, with `strategic_reference/far_below` provenance when the
extended range was required. A pivot equal to the current close is rejected as
degenerate. Single-anchor/reference-only bands retain nonzero `0.70 ATR` width,
positive prices, at most 49 evidence points, and can never be green. Only missing
finite positive price/ATR or the absence of every non-current valid level remains
unavailable. The technical invalidation reference is the lower zone boundary
minus `0.35 ATR`; it is an observation reference only.

Evidence quality is the bounded sum
`min(32, 12*family count) + 22*(1-exp(-family-weight/2.5)) +
18*cluster tightness + 16*price proximity + 12*data completeness`. Family
weights use only the maximum weight in each family, so correlated derivatives
have diminishing/no duplicate effect. Two families can exceed 65 when strong,
tight, near price, and complete, but do not pass automatically.

Green is possible only with at least 2 independent families, evidence quality at
least 65, completed bars no older than 4 days, no falling knife or speculative
bottoming, no Stage 3/4/down/top regime, price not below an available SMA200,
long-term score at least 60, entry-timing score at least 55, non-negative daily
context, RSI 32–70, no severe ATR-scaled MACD deterioration, no high downside
structure, ATR below 5%, annualized volatility below 60%, and no earnings in the
next 7 calendar days. A mathematically in-zone row can therefore be amber or red.
Statuses are deterministic: green `in_zone_confirmed`; amber
`in_zone_risk_filtered`, `approaching`, or `setup_waiting_confirmation`; red
`broken_below` or `safety_blocked`; neutral `far_above`,
`reference_only_far`, or `unavailable`. `reference_only` rows never enter the
confirmed or approaching Sweet-Spot categories, even when price lies inside the
mathematical fallback band.

For company equities the technical result is separate from valuation. A company
does not need to look cheap for technical green, but combined green is downgraded
to amber for high jurisdiction risk, high value-trap risk, incomplete/stale
fundamentals, Altman Z below 1.81, or a defined major structural counterargument.
The contract explicitly reports whether technical location and a descriptive
Value score of at least 65 align. ETFs and crypto use technical-only status and
are labeled as having no company-fundamental overlay.

One centralized invariant checker recomputes all green gates for model
classification, category membership, full-output validation, and compact-static
validation. `why_green_or_not` contains the complete untruncated failed-gate list.
Static gate evidence is compactly exported and revalidated rather than trusting
stored color/status fields.

Zone numbers retain full floating-point precision in the contracts. Both UIs
choose decimals by price magnitude and increase precision until internally
distinct lower/IDEAL/upper values remain visibly distinct.

This geometry has not been backtested as an optimal entry edge. Its evidence
quality score measures only input confluence and proximity; it is not confidence,
probability, expected return, a recommendation, or a guarantee.

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
