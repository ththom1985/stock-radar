# Stock Radar — completed-daily-bar research

Stock Radar is a **research tool**, not a trading system or investment adviser. It
builds a conservative snapshot from completed Yahoo Finance daily bars and
separately presents company equities, funds/ETFs, crypto, and other instruments.

## Validation status

The deployed composite is **UNVALIDATED** and its output is always marked
`actionable: false`.

- No profitability, alpha, expected-return, or intraday claim is made. The
  separate probability engine publishes a stock-specific number only for an
  individually accepted horizon/threshold model; otherwise it says
  `No validated stock-specific probability edge` and displays baseline rates
  only when a fitted baseline exists.
- Scenario ranges are uncalibrated log-return/volatility illustrations. They are
  excluded from ranking and cannot produce negative prices.
- The available backtest can validate only the technical score because reliable
  point-in-time histories for fundamentals, analyst targets, and news are not
  available. It cannot validate the deployed composite.
- Optional news, analyst, macro, social/expert, and deep-fundamental context does
  not change the comparable core ranking.

## Calibrated material-move probability engine

`src/probability_*.py` is a separate, fail-closed model family. It never reads or
changes radar scores, insight ranks, Sweet-Spot evidence/colors, fundamentals,
analysts, news, earnings, identity, jurisdiction, or ranking inputs. Its MVP
partition is one primary **USD company-equity listing** per available issuer key.
ETFs, funds, crypto, non-USD listings, duplicate issuer listings, and histories
without at least 252 completed prior bars are withheld.

The current checked-in `data/probability_models.json` and
`data/probability_validation.json` are intentionally withheld. Both
independent-threshold-v1 and the completed ordered-vector-v1 retrospective
validation accepted zero models. Their artifacts remain preserved under
`data/probability_experiments/`; no rejected probability is promoted.

### Target and timing convention

For a completed adjusted daily close at session `t`:

1. features use data available at or before `t`;
2. hypothetical entry is the first adjusted/raw-equivalent open strictly after
   `t`;
3. exit is adjusted close at `t + H` sessions, for `H = 21, 63, 126, 252`;
4. gross total return is `adjusted_close[t+H] / adjusted_open[t+1] - 1`;
5. a direction-symmetric 30 bp friction dead-band defines material moves:
   `DOWN` when gross `<= -(X + 0.003)`, `UP` when gross `>= X + 0.003`, and
   `MIDDLE` otherwise.

Threshold grids are 21 sessions `[3,5,10]%`, 63 `[5,10,20]%`, 126
`[10,15,25]%`, and 252 `[10,20,30]%`. Classes are mutually exclusive and sum
to one. The dataset also stores long-only `gross - 0.003`, but **P(positive net
return) is not published**: that target needs its own model and calibration and
must not be inferred from a material-threshold class.

### Dataset and point-in-time boundary

The restartable builder uses the **current universe membership observed at
retrieval time** and downloads Yahoo history from `2008-01-01` by default,
with actions and `auto_adjust=False`, then reconstructs adjusted OHLC from
`Adj Close / Close`. It stores checksummed, atomic, per-symbol files plus a
manifest under ignored `data/probability_cache/panel_v1/`. The manifest records
retrieval time, source range, symbols, failures, checksums, feature version,
schema hash, and code hash. Failed batches retry, split recursively, and retry
individual symbols. The 2008 boundary leaves sufficient calendar depth, where a
symbol actually traded, for 252-bar warm-up, exact 252-session labels/purging,
five usable training years, calibration, embargo, and at least five full annual
tests through the current date.

The assembled dataframe checkpoint is
`weekly_dataset_v3.pkl.gz`: an atomic gzip-compressed pandas pickle plus file
SHA-256 and semantic dataset hash. Dates, dtypes, floating-point values, and
nullable labels round-trip exactly across processes. **Trust boundary:** pickle
may execute code; only repository-generated files under ignored
`data/probability_cache/` may be loaded. Never place downloaded/untrusted pickle
there. Legacy `weekly_dataset_v2.pkl.gz` is ignored and must be rebuilt,
not migrated.

One anchor is retained per ISO week: the final completed session in that week.
Features include adjusted-return horizons; SMA/EMA distances and slopes; RSI;
MACD-histogram/ATR; ATR%; 20/60/252 volatility; upside/downside semivolatility;
drawdown and 52-week high/low distance; Bollinger %B/bandwidth; Wilder ADX/DMI;
relative volume; raw contemporaneous dollar liquidity; prior pivot/S1/20-day-low
ATR distances; and point-in-time-aligned SPY return, volatility, drawdown, and
trend fields.
The ordered challenger adds only five frozen PIT interactions: SPY-above-SMA200
times stock 60-day return, SPY-above-SMA200 times stock SMA200 distance, SPY
vol60 times stock vol60, SPY vol60 times trailing drawdown, and
SPY-above-SMA200 times downside semivolatility. SPY inputs use an explicit
backward as-of join (`SPY timestamp <= stock timestamp`), including non-US and
weekend stock dates.

Yahoo exposes a current adjusted history, not a historical snapshot of its
action tape. A later split/dividend can uniformly rescale pre-event adjusted
OHLC. All price features are dimensionless and invariant to that uniform
scaling; liquidity uses contemporaneous `RawClose * Volume`. Tests enforce
future-row/action and scale invariance. Full explicit point-in-time action
reconstruction is not claimed.

### Model, calibration, and validation

Independent-threshold v1 uses one three-class model per threshold. The
preregistered `ordered-vector-v1` challenger instead fits one L2 multinomial
seven-bin distribution per horizon and derives all three threshold outputs by
disjoint tail sums. Exact negative/positive boundary equalities follow the
documented seven-bin contract, so simplex and threshold monotonicity hold by
construction without caps or projection. In every outer fold, 0.5/99.5%
winsorization, medians,
needed missing indicators, scaling, coefficients, and SPY trend/volatility
regime buckets are fit on training rows only. The challenger selects regularized
vector scaling from the frozen penalty grid `[0.01,0.1,1,10,100]`: candidates
fit on calibration months 1–9, months 10–12 select by seven-class log loss with
Brier tie-break, then the winner refits on the full calibration year. Biases are
zero-mean and L2 shrinkage targets unit scales/zero biases. Production JSON
stores every numeric transform, coefficient, intercept, calibrator, OOD
reference, and hash, so artifact inference is deterministic NumPy only.

Outer validation is purged expanding walk-forward: at least five actual usable
feature-date years after the 252-bar warm-up and exact-label purging, 12 months
calibration, the latest exact calibration-label exit plus a one-week embargo,
then 12 full untouched
test months, rolling annually. Exact per-row `max_exit_date` must precede the
next segment by the one-week embargo; no feature-date approximation substitutes
for a known label interval. Training label intervals may not enter calibration;
calibration label intervals may not enter test. Climatology is the required comparator. A
training-only SPY trend × training-volatility-tercile baseline, shrunk by 100
observations toward climatology, is reported separately.

Reported OOS metrics include multiclass Brier and skill, log loss and
improvement, adaptive equal-count classwise reliability/ECE and maximum gap,
one-vs-rest calibration slope/intercept, prevalence, class/issuer/date counts,
coverage, every fold, and regime ECE. Challenger reliability starts with ten
equal-count bins and merges adjacent bins until supported (500 rows, 50
positives, 50 negatives), requiring five supported bins; unsupported extreme
tails retain Wilson intervals and do not enter ECE/gap. Regimes are available
only with 26 dates, 8 quarter blocks, 100 outcomes/class, and 100 issuers.
Release uses 1,000 fixed-seed two-way bootstrap repetitions over issuer clusters
and calendar-quarter blocks. Reports record requested, attempted, completed, and
skipped draws. Invalid draws retry deterministically up to three times the
requested count; release requires all requested draws (at least 1,000) to
complete, otherwise the horizon is withheld. Models are not refit inside that metric bootstrap; metric
intervals and per-class aggregate calibration-residual offsets are therefore
explicit approximations from fixed OOS predictions, not individual-return
intervals. A separate configurable full model+calibrator bootstrap must complete
at least 200 refits for release; smoke may record an explicit smaller
development-only override, which can never create a production artifact.

Every horizon/threshold must independently satisfy all gates:

- at least 8 usable years, 5 test folds, 200 issuers, 100 forecast dates;
- every counted fold has at least 5.0 actual usable training years and a full
  untouched 12-month test window;
- minimum per fold/class: 1,000 train, 300 calibration, 200 untouched test;
- at least 80% inference coverage;
- at least 80% successful requested eligible-issuer provider coverage and at
  least 200 successful issuers, with unavailable symbols/reasons reported;
- aggregate Brier skill at least 2% versus climatology and its 95% block
  bootstrap lower bound strictly above zero;
- log-loss improvement at least 1%;
- every class ECE at most 3%, maximum reliability gap at most 8%;
- every meaningful class slope 0.8–1.2 and intercept -0.1–0.1;
- no two consecutive folds below -2% Brier skill;
- supported regime ECE at most 5%; unsupported regimes are unavailable rather
  than passing or failing.

No gate is relaxed automatically. The MVP production transform is serialized
as `raw-temperature-scaled-identity-v1`: the exact temperature-scaled softmax
probabilities used by OOS metrics, bootstrap, calibration diagnostics, and
acceptance are published without caps or projection. A complete horizon grid
records temporal OOS inversion rate and violating-magnitude mean/p50/p95/max
separately for UP and DOWN. Each direction must have at most 1% violating
adjacent-threshold comparisons, p95 magnitude at most 1 percentage point, and
maximum magnitude at most 3 points (magnitude quantiles use violating
comparisons only). At current-row inference, raw values are
never changed: an inversion above 0.5 point, or any smaller inversion that would
make whole-percent display non-monotonic, withholds that horizon with
`current_threshold_non_monotonic`. A smaller inversion may be displayed
unchanged only when whole-percent values remain monotonic/equal, with an
independent-threshold tolerance disclosure. Display rounding preserves a
three-class sum of 100. The labeled interval is a **95% aggregate calibration-error interval
approximation from fixed OOS predictions; not an individual stock outcome
interval**. A row is withheld for acceptance failure, missing/invalid hashes,
model age over 45 days, stale/incomplete bars, missing features, unsupported
partition, fewer than 252 prior bars, more than two features outside training
0.5/99.5% bounds, robust distance beyond training p99.5, or interval width over
20 percentage points.

For `ordered-vector-v1`, `ordered-vector-tail-sum-identity-v1` publishes the
validated vector-scaled seven-class tail sums unchanged. Its monotonicity
assertion uses float64 machine epsilon; any horizon-level inference or interval
failure withholds all three stock-specific thresholds for that horizon while
retaining baseline rows.

The ordered family and tail-sum identity are also mandatory at the artifact
root; model/root family or transform disagreement invalidates the complete
artifact. Completed challenger runs first archive versioned validation/model
copies under `data/probability_experiments/`, then atomically replace the
canonical model followed by canonical `data/probability_validation.json`.
Failed completed releases publish canonical `no_model_passed` / `withheld`,
never a stale `not_run`.

### Prospective forward validation (shadow only)

`src.probability_forward` freezes and monitors the rejected ordered-vector-v1
without retuning it. `freeze` writes an immutable preregistration **before**
fitting all four final horizons from the exact cached dataset bounded at
`2026-07-10`. It records feature/specification, gate, source, dependency,
dataset, validation, and production-artifact hashes. The resulting artifact has
the distinct `stock-radar-probability-shadow-artifact` schema, is marked
`REJECTED_SHADOW_NOT_FORECAST`, `shadow_only: true`, and `actionable: false`,
contains no accepted-model index, and is rejected by the normal production
loader. It never overwrites `data/probability_models.json`.

Detailed coefficients, weekly predictions, OOD diagnostics, and matured outcomes
live only under ignored `data/probability_forward/cohorts/<cohort>/`. The
authoritative SQLite ledger uses WAL, full synchronous transactions, immutable
prediction/outcome triggers, HMAC-SHA256 records, a canonical append-only event
chain covering metadata, anchors, predictions, exclusions, outcomes, resolution
attempts, and candidate reports, plus a monotonic separately stored signed
snapshot root. Event transactions atomically stage a pending seal; manifest
publication then finalizes that exact head in a short transaction. Startup and
`recover` deterministically finish either crash window (DB commit before
manifest, or manifest before finalization) only after full reconciliation.
Deterministic immutable gzip snapshots and verified SQLite
backups include the corresponding sealed manifest. Back up both the cohort directory and its
`manifest-signing.key`; losing the local directory loses the prospective
history. The HMAC protects integrity while the key remains separate from an
attacker, but it is not encryption.
Do not open the same SQLite cohort concurrently on multiple synced machines;
WAL protects local processes, not cross-device OneDrive synchronization.
There is no external cryptographic timestamp or transparency log: rollback
detection is only as strong as an independently retained newer sealed manifest
(including the manifest sidecar written with every backup). An attacker able to
replace the database, seal, and local key together can rewrite local history.

Only `data/probability_forward_status.json` may be committed. It contains counts,
dates, schedules, and state—never coefficients, feature vectors, prices,
returns, labels, or shadow probability values. GitHub Actions can inject this
already-committed aggregate into `latest.json` and `docs/data.json`; a public
runner has no private durable ledger and therefore **cannot** capture, mature, or
evaluate unseen history. GitHub cache/artifact retention is deliberately not
misrepresented as durable storage. There is no external cloud dependency.

Run the local commands from PowerShell:

```powershell
Set-Location "C:\Users\ththomas\OneDrive\09_Software_Tools_Projekte\01_Aktive_Software_Projekte\Stock-Radar"

# One time, after reviewing the preregistration inputs. Restartable checkpoints
# are reused only when every frozen binding is identical.
.\.venv\Scripts\python.exe -m src.probability_forward freeze

# Once per ISO week, preferably Saturday 01:30 Europe/Berlin. Capture enforces
# Friday 23:00 UTC or later, the current ISO week, and feature_date > freeze date.
.\.venv\Scripts\python.exe -m src.probability_forward capture

# Run after capture and at least weekly; labels are inserted only at exact t+H.
.\.venv\Scripts\python.exe -m src.probability_forward evaluate

# Local integrity/backup and, only after all minimums, a 1,000-draw candidate report.
.\.venv\Scripts\python.exe -m src.probability_forward verify
.\.venv\Scripts\python.exe -m src.probability_forward recover
.\.venv\Scripts\python.exe -m src.probability_forward backup
.\.venv\Scripts\python.exe -m src.probability_forward report
```

Production CLI commands accept no clock, historical-date, test-mode, or reduced
bootstrap override. They use timezone-aware system UTC; injected clocks exist
only in internal development tests and produce artifacts permanently marked
`development_test_mode`. Capture requires the selected stock and SPY session to
equal the latest completed US session inferred from the current US-equity/SPY
histories (so stale Thursday data is rejected when a Friday session exists, while
a genuine Friday holiday can retain Thursday). Each anchor records requested,
successful, and failed provider issuers. Candidate metrics use a stable SQLite
read snapshot; the report event is rejected/retried if the event head changes
before sealing, and aggregate publication rechecks the same bound head under a
write lock. Candidate review additionally requires
at least 80% provider coverage and at least 200 successful issuers. `report`
always uses exactly 1,000 fixed-prediction bootstrap repetitions, recomputes all
gates from immutable ledger rows, and republishes aggregate status; a stale
candidate JSON is never trusted.

The freeze performs four multinomial fits and four vector-calibrator fits but no
retrospective search/bootstrap; plan roughly **10–45 minutes and 2–6 GiB** on a
typical workstation. Actual time depends on BLAS, CPU, and the 203 MiB cached
dataframe.

No Windows task is created by this repository. If the operator later chooses to
schedule it, use two Task Scheduler actions whose working directory is the
repository: weekly Saturday around 01:30 local for `capture`, followed around
02:30 by `evaluate`. The capture command itself rejects early, stale-week, and
backfill attempts. Keep `verify` and an off-machine backup of the ignored cohort
and signing key in the operator's backup policy.

Assuming implementation/freeze on 2026-08-18, the first permitted completed
weekly anchor is Friday **2026-08-21**. Its 21-session outcome is expected around
**2026-09-22**, but that is only the first outcome. Release review still requires
at least 104 weekly anchors, 200 issuers, 200 outcomes per class, 26 dates, eight
quarters, all unchanged metric gates, and a complete 1,000-draw bootstrap. The
earliest meaningful 1M review is therefore approximately **2028-09-12**; the
252-session cohort cannot complete its 104th-anchor outcomes before the frozen
NYSE weekday/holiday schedule estimate **2029-08-14**. Exchange closures, halts,
and missing bars can only delay these
dates. A complete pass creates a non-actionable candidate report for independent
review; it never promotes automatically.

### Commands, checkpointing, and cost

Run from PowerShell:

```powershell
Set-Location "C:\Users\ththomas\OneDrive\09_Software_Tools_Projekte\01_Aktive_Software_Projekte\Stock-Radar"
python -m pip install -r requirements-ci.txt

# Bounded in-memory learnable/random check; writes no production artifact
python -m src.probability_train smoke
python -m src.probability_train smoke --model-family ordered-vector-v1 --dev-refit-override

# Restartable ignored Yahoo panel + weekly dataset
python -m src.probability_train build --start 2008-01-01

# Development reports only (200 bootstrap repetitions)
python -m src.probability_train validation-only --model-family ordered-vector-v1 --bootstrap 200

# Ordered release from an existing cache (1,000 fixed OOS resamples, 200 refits)
python -m src.probability_train train --model-family ordered-vector-v1 --bootstrap 1000 --refit-bootstrap 200

# Build, validate, and release in one restartable run
python -m src.probability_train full --model-family ordered-vector-v1 --start 2008-01-01 --bootstrap 1000 --refit-bootstrap 200
```

`--no-resume` discards compatible checkpoints; `--horizon` and `--threshold`
bound a diagnostic run. Resume keys bind the full dataframe content hash, panel
source/manifest hash, requested/success/failure coverage, symbol/date/count
summary, model family, feature schema/version, portable code/dependency hash, C,
calibrator/grid, fixed and refit bootstrap counts, seed, fold settings,
acceptance gates, and publish-transform version; any
change forces recomputation. On a typical 8-core workstation, smoke is roughly
2–8 minutes and under 1 GiB; Yahoo panel build is roughly 30–120 minutes and
1–4 GiB; a four-horizon ordered release with 200 full refits is roughly
8–24 hours with 4–12 GiB peak working space. These are planning
estimates—provider throttling,
eligible-universe size, CPU, and memory dominate.

Limitations remain material: the training membership is today's observable
universe (survivorship bias); probabilities are for listing-currency USD total
returns, not an investor's home-currency return; Yahoo adjustment history is not
a true vintage action tape; bootstrap probability bands are aggregate
calibration-error approximations; and no causal/structural return claim is made.
True point-in-time fundamentals, estimates, news, and earnings are a future
model phase and may not be backfilled into this price/SPY MVP.

Daily SPY inference uses the same backward as-of rule as dataset features: the
selected completed SPY session must be at or before the stock signal session.
Weekend/non-US-holiday gaps are allowed through 4 calendar and 2 business days;
future-only or older SPY history is withheld.

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
- `forward_validation_status`: leak-safe prospective cohort counts and schedule;
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
interns probability reasons/model/baseline metadata and packs row-varying whole
percentages, directly quantized validated interval bounds, and conservative OOD
summaries into a bounded base64 byte stream.
The browser hydrates and revalidates sums, directly stored interval bounds, OOD
status, and threshold monotonicity before rendering. It serializes one deterministic compact UTF-8
byte sequence and measures that exact
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
quality score measures only input confluence and proximity; it does not measure
likelihood, probability, expected return, or recommendation strength.

The static cockpit applies the same fail-closed contract before rendering any
tips: status must be `ok`, `data_actionable` true, blocking reasons empty, model
and insight actionability false, and `generated_at` no older than 36 hours.

## Free expert-analysis layer

The additive expert layer never changes `radar_score`, core technical partitions,
Sweet Spot, or released/withheld probability models. It remains
`heuristic_unvalidated`, `actionable: false`, and is analysis support rather than
investment advice.

- `data/expert_score_weights.json` configures separate long-term and short-term
  composites. Missing factors are omitted with visible weight coverage; they are
  never guessed or filled with a neutral score.
- The Streamlit dashboard exposes temporary weight sliders for scenario testing.
  Persistent daily weights remain reviewable source-controlled JSON.
- `data/recommendation_log.jsonl` records the daily top observations with the
  exact completed-bar price, score components, evidence quality, and alternative
  signals. `data/recommendation_outcomes.jsonl` appends matured 21/63/126/252
  session outcomes. A positive-return hit rate is not an alpha or benchmark claim.
- `data/valuation_history.json` stores one point-in-time multiple snapshot per
  month. An own five-year average is withheld until at least 48 monthly
  observations exist. Current sector medians require at least five available peers.
- Fair-value ranges are unvalidated interquartile implied-price ranges from
  available sector medians and complete point-in-time own-history averages. If
  fewer than two references exist, no range or verdict is produced.
- Existing 1M/6M/12M/24M scenario ranges remain uncalibrated. Scenario
  probabilities are explicitly withheld while the strict probability engine has
  no accepted model.

### Free source status

| Source | Role | Refresh / delay | Current integration |
| --- | --- | --- | --- |
| Yahoo Finance via yfinance | Completed daily OHLCV, global fundamentals, analyst context | Daily bars; fundamentals weekly | Active primary source |
| SEC EDGAR Companyfacts | Official US annual accounting facts and five-period history | 30-day cache; filing-time data | Active when `SEC_USER_AGENT` is set |
| SEC EDGAR Form 4 | Open-market insider purchases/sales and 21-day cluster purchases | 2-day cache; normally filed within two business days | Active when `SEC_USER_AGENT` is set |
| Stooq | Secondary EOD price source | Daily | Active only after Yahoo failure for conservatively mapped ordinary US tickers |
| FRED | 10Y–2Y curve, high-yield spread, NFCI risk regime | Daily cache; source-series delay | Active when free `FRED_API_KEY` is set |
| ECB / Bundesbank | EU macro regime | Series dependent | Not yet wired; explicit gap |
| FINRA | Short interest / days to cover | Twice monthly | Not yet wired; free credentials required |
| House / Senate disclosures | Congressional transactions | Up to 45-day reporting lag | Not yet wired; explicit gap |
| CFTC / CBOE | Positioning and put/call regime | Publication dependent | Not yet wired; explicit gap |
| Yahoo options | Self-computed Black-Scholes GEX from IV/open interest | 18-hour cache; five symbols per run by default | Active, bounded, US USD company equities only |
| Wikimedia / Reddit / careers pages | Attention and hiring trends | Daily/periodic | Not yet wired; explicit gap |

SEC requests require a descriptive Fair Access user agent containing a contact
email:

```text
SEC_USER_AGENT=Stock-Radar research your.name@example.com
STOCK_RADAR_SEC_MAX=25
STOCK_RADAR_INSIDER_MAX=15
FRED_API_KEY=your_free_fred_key
STOCK_RADAR_OPTIONS_MAX=5
```

The GitHub workflows read `SEC_USER_AGENT` from a repository secret. Without it,
both SEC modules report `disabled` and retain stale-good cached data; they do not
send anonymous requests.

Equibles is not a mandatory runtime dependency. Its free self-hosted core is
technically useful for SEC/FINRA/Congress enrichment, but its persistent
ParadeDB plus .NET/Docker services are incompatible with ephemeral GitHub Actions.
It may be added later as an optional local MCP adapter only.

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

Corporate-action coverage may remain incomplete across missed runs. Consequently,
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
