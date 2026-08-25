from __future__ import annotations

import copy
import io
import json
import shutil
import sqlite3
import threading
import uuid
from datetime import datetime
from contextlib import redirect_stderr
from pathlib import Path
from unittest import TestCase
from unittest.mock import patch

import numpy as np
import pandas as pd

from src.probability_contract import HORIZONS, THRESHOLD_GRIDS, ordered_label_column
from src.probability_dataset import (
    EligibilityResult,
    dataset_content_hash,
    select_eligible_universe,
)
from src.probability_features import FEATURE_NAMES, latest_probability_features
from src.probability_forward import (
    _expected_completed_us_session,
    _parser,
    _point_metrics,
    _prospective_schedule,
    _prospective_support,
    _trusted_now,
    build_candidate_report,
    build_public_status,
    capture_from_histories,
    cohort_paths,
    evaluate_from_histories,
    freeze_shadow_cohort,
    load_shadow_artifact,
    main as forward_main,
    publish_local_status,
    recover_cohort,
    validate_preregistration,
    verify_cohort,
)
from src.probability_forward_public import (
    finalize_forward_validation_status,
    initial_forward_validation_status,
    public_status_json_contains_private_values,
    validate_forward_validation_status,
)
from src.probability_forward_store import (
    ForwardLedger,
    ForwardIntegrityError,
    canonical_json_bytes,
    read_gzip_json,
    sha256_bytes,
    signed_digest,
    write_immutable_bytes,
)
from src.probability_inference import (
    ProbabilityArtifactError,
    load_probability_artifact,
    validate_probability_artifact,
)
from src.probability_model import canonical_hash
from tests.helpers import ProjectTempMixin, ROOT

COHORT = "ordered-vector-v1-forward-synthetic-test"
FREEZE_TIME = "2026-08-18T07:15:43+00:00"
CAPTURE_TIME = "2026-08-21T23:30:00+00:00"


def clock_at(value: str):
    parsed = datetime.fromisoformat(value)
    return lambda: parsed


def price_history(
    *,
    seed: int,
    end: str = "2026-08-21",
    periods: int = 720,
) -> pd.DataFrame:
    index = pd.bdate_range(end=end, periods=periods)
    rng = np.random.default_rng(seed)
    close = 100.0 * np.exp(np.cumsum(rng.normal(0.0002, 0.011, len(index))))
    open_price = close * np.exp(rng.normal(0.0, 0.002, len(index)))
    frame = pd.DataFrame(
        {
            "Open": open_price,
            "High": np.maximum(open_price, close) * 1.006,
            "Low": np.minimum(open_price, close) * 0.994,
            "Close": close,
            "RawOpen": open_price,
            "RawClose": close,
            "Volume": rng.integers(800_000, 3_000_000, len(index)).astype(float),
            "Dividends": np.zeros(len(index)),
            "Stock Splits": np.zeros(len(index)),
        },
        index=index,
    )
    return frame


def extend_history(frame: pd.DataFrame, sessions: int, *, seed: int) -> pd.DataFrame:
    if sessions <= 0:
        return frame.copy()
    rng = np.random.default_rng(seed)
    index = pd.bdate_range(
        start=pd.Timestamp(frame.index[-1]) + pd.offsets.BDay(1),
        periods=sessions,
    )
    close = float(frame["Close"].iloc[-1]) * np.exp(
        np.cumsum(rng.normal(0.0003, 0.012, len(index)))
    )
    open_price = close * np.exp(rng.normal(0.0, 0.002, len(index)))
    extension = pd.DataFrame(
        {
            "Open": open_price,
            "High": np.maximum(open_price, close) * 1.006,
            "Low": np.minimum(open_price, close) * 0.994,
            "Close": close,
            "RawOpen": open_price,
            "RawClose": close,
            "Volume": rng.integers(800_000, 3_000_000, len(index)).astype(float),
            "Dividends": np.zeros(len(index)),
            "Stock Splits": np.zeros(len(index)),
        },
        index=index,
    )
    return pd.concat([frame, extension])


def synthetic_freeze_dataset(
    stock: pd.DataFrame,
    spy: pd.DataFrame,
) -> pd.DataFrame:
    _timestamp, center_dict = latest_probability_features(stock, spy)
    center = np.asarray([center_dict[name] for name in FEATURE_NAMES], dtype=float)
    rng = np.random.default_rng(20260818)
    weeks = pd.date_range(end="2026-07-10", periods=320, freq="W-FRI")
    issuers = [f"issuer-{index:02d}" for index in range(14)]
    records = []
    scale = np.maximum(np.abs(center) * 0.06, 0.012)
    for week_index, feature_date in enumerate(weeks):
        for issuer_index, issuer in enumerate(issuers):
            vector = center + rng.normal(size=len(center)) * scale
            ordered_label = (week_index + issuer_index) % 7
            record = {
                "symbol": f"S{issuer_index:02d}",
                "issuer_key": issuer,
                "feature_date": feature_date,
                "feature_timestamp": feature_date,
                "history_start": pd.Timestamp("2012-01-03"),
                "history_bars_before": 600,
                "max_exit_date": feature_date + pd.Timedelta(days=358),
            }
            record.update(
                {name: float(vector[index]) for index, name in enumerate(FEATURE_NAMES)}
            )
            for horizon in HORIZONS:
                record[ordered_label_column(horizon)] = ordered_label
            records.append(record)
    return pd.DataFrame.from_records(records).sort_values(
        ["feature_date", "issuer_key"]
    ).reset_index(drop=True)


def synthetic_context(dataset: pd.DataFrame) -> dict:
    digest = dataset_content_hash(dataset)
    return {
        "training_cutoff": "2026-07-10",
        "dataset_binding": {"dataset_content_hash": digest},
        "canonical_artifact_hash": "a" * 64,
        "canonical_artifact_document_hash": "b" * 64,
        "canonical_validation_document_hash": "c" * 64,
        "ordered_experiment_artifact_hash": "d" * 64,
        "ordered_validation_document_hash": "e" * 64,
        "ordered_preregistration_document_hash": "f" * 64,
        "rejection_reason_hash": "1" * 64,
    }


class ForwardValidationTests(ProjectTempMixin, TestCase):
    @classmethod
    def setUpClass(cls):
        super().setUpClass()
        cls.stock = price_history(seed=81)
        cls.spy = price_history(seed=82)
        cls.dataset = synthetic_freeze_dataset(cls.stock, cls.spy)
        cls.context = synthetic_context(cls.dataset)
        cls.manifest = {
            "schema": "synthetic-forward-dataset",
            "schema_version": 1,
            "dataset_content_hash": cls.context["dataset_binding"][
                "dataset_content_hash"
            ],
        }
        cls.fixture_root = ROOT / "tests" / ".r" / uuid.uuid4().hex[:8]
        cls.fixture_root.mkdir(parents=True)
        cls.artifact = freeze_shadow_cohort(
            cls.dataset,
            cls.manifest,
            cls.context,
            cohort_id=COHORT,
            root=cls.fixture_root,
            signing_key=b"k" * 32,
            clock=clock_at(FREEZE_TIME),
            test_mode=True,
        )

    @classmethod
    def tearDownClass(cls):
        shutil.rmtree(cls.fixture_root, ignore_errors=True)
        runtime = cls.fixture_root.parent
        try:
            runtime.rmdir()
        except OSError:
            pass
        super().tearDownClass()

    def setUp(self):
        super().setUp()
        self.short_work = ROOT / "tests" / ".r" / uuid.uuid4().hex[:8]
        self.short_work.mkdir(parents=True)
        self.forward_root = self.short_work / "f"
        shutil.copytree(self.fixture_root, self.forward_root)

    def tearDown(self):
        shutil.rmtree(self.short_work, ignore_errors=True)
        super().tearDown()

    def _eligibility(self, *, include_bbb: bool = False) -> EligibilityResult:
        symbols = ("AAA", "BBB") if include_bbb else ("AAA",)
        return EligibilityResult(
            symbols=symbols,
            issuer_keys={symbol: f"issuer:{symbol.lower()}" for symbol in symbols},
            excluded={},
        )

    def _capture(
        self,
        *,
        stock: pd.DataFrame | None = None,
        spy: pd.DataFrame | None = None,
        eligibility: EligibilityResult | None = None,
        now: str = CAPTURE_TIME,
    ):
        stock = stock if stock is not None else self.stock
        spy = spy if spy is not None else self.spy
        histories = {"AAA": stock, "SPY": spy}
        if eligibility and "BBB" in eligibility.symbols:
            histories["BBB"] = stock.copy()
        return capture_from_histories(
            histories,
            eligibility or self._eligibility(),
            cohort_id=COHORT,
            root=self.forward_root,
            clock=clock_at(now),
            test_mode=True,
        )

    @staticmethod
    def _dummy_report(label: str) -> dict:
        report = {
            "schema": "synthetic-candidate-report",
            "schema_version": 1,
            "generated_at": CAPTURE_TIME,
            "label": label,
            "all_gates_passed": False,
            "development_test_mode": True,
        }
        report["report_hash"] = canonical_hash(report)
        return report

    def test_freeze_is_distinct_shadow_only_and_restartable(self):
        paths = cohort_paths(COHORT, self.forward_root)
        artifact = load_shadow_artifact(paths.artifact)
        preregistration = json.loads(
            paths.preregistration.read_text(encoding="utf-8")
        )
        self.assertTrue(preregistration["development_test_mode"])
        self.assertTrue(artifact["development_test_mode"])
        self.assertTrue(artifact["shadow_only"])
        self.assertFalse(artifact["actionable"])
        self.assertFalse(artifact["production_loader_compatible"])
        self.assertNotIn("accepted_model_keys", artifact)
        self.assertEqual(len(artifact["models"]), 4)
        self.assertTrue(all(not model["accepted"] for model in artifact["models"].values()))
        with self.assertRaises(ProbabilityArtifactError):
            validate_probability_artifact(artifact)
        with self.assertRaises(ProbabilityArtifactError):
            load_probability_artifact(paths.artifact)

        repeated = freeze_shadow_cohort(
            self.dataset,
            self.manifest,
            self.context,
            cohort_id=COHORT,
            root=self.forward_root,
            signing_key=b"k" * 32,
            clock=clock_at(FREEZE_TIME),
            test_mode=True,
        )
        self.assertEqual(repeated["artifact_hash"], artifact["artifact_hash"])

    def test_production_cli_rejects_time_and_bootstrap_spoofing(self):
        for command in ("freeze", "capture", "evaluate", "report", "recover"):
            with (
                self.subTest(command=command),
                redirect_stderr(io.StringIO()),
                self.assertRaises(SystemExit),
            ):
                _parser().parse_args([command, "--now", FREEZE_TIME])
        with redirect_stderr(io.StringIO()), self.assertRaises(SystemExit):
            _parser().parse_args(["report", "--bootstrap-repetitions", "1"])
        with self.assertRaisesRegex(RuntimeError, "test-mode"):
            _trusted_now(
                clock=clock_at(FREEZE_TIME),
                test_mode=False,
            )
        with self.assertRaisesRegex(RuntimeError, "development test-mode"):
            verify_cohort(cohort_id=COHORT, root=self.forward_root)
        for repetitions in (999, 1001):
            with (
                self.subTest(repetitions=repetitions),
                self.assertRaisesRegex(RuntimeError, "exactly 1000"),
            ):
                build_candidate_report(
                    cohort_id=COHORT,
                    root=self.forward_root,
                    bootstrap_repetitions=repetitions,
                )
        with (
            patch(
                "src.probability_forward.publish_local_status",
                return_value=initial_forward_validation_status(),
            ) as publish,
            patch("builtins.print"),
        ):
            forward_main(["report"])
        publish.assert_called_once()

    def test_freeze_cutoff_must_precede_trusted_clock(self):
        context = copy.deepcopy(self.context)
        context["training_cutoff"] = "2026-08-18"
        shifted = self.dataset.copy()
        shifted["feature_date"] = pd.to_datetime(shifted["feature_date"]) + pd.Timedelta(
            days=39
        )
        self.assertEqual(
            pd.to_datetime(shifted["feature_date"]).max().date().isoformat(),
            "2026-08-18",
        )
        with self.assertRaisesRegex(RuntimeError, "strictly earlier"):
            freeze_shadow_cohort(
                shifted,
                self.manifest,
                context,
                cohort_id="ordered-vector-v1-forward-bad-time",
                root=self.short_work / "bad-time",
                signing_key=b"t" * 32,
                clock=clock_at("2026-08-18T23:00:00+00:00"),
                test_mode=True,
            )

    def test_preregistration_is_written_before_fit_and_cutoff_is_fail_closed(self):
        root = self.short_work / "failed"
        with patch(
            "src.probability_forward._fit_shadow_model",
            side_effect=RuntimeError("synthetic fit interruption"),
        ):
            with self.assertRaisesRegex(RuntimeError, "synthetic fit interruption"):
                freeze_shadow_cohort(
                    self.dataset,
                    self.manifest,
                    self.context,
                    cohort_id="ordered-vector-v1-forward-interrupted",
                    root=root,
                    signing_key=b"i" * 32,
                    clock=clock_at(FREEZE_TIME),
                    test_mode=True,
                )
        paths = cohort_paths("ordered-vector-v1-forward-interrupted", root)
        self.assertTrue(paths.preregistration.exists())
        self.assertFalse(paths.artifact.exists())

        future = pd.concat(
            [
                self.dataset,
                self.dataset.iloc[[0]].assign(
                    feature_date=pd.Timestamp("2026-07-17")
                ),
            ],
            ignore_index=True,
        )
        with self.assertRaisesRegex(RuntimeError, "exact frozen cutoff"):
            freeze_shadow_cohort(
                future,
                self.manifest,
                self.context,
                cohort_id="ordered-vector-v1-forward-future-row",
                root=self.short_work / "future",
                signing_key=b"f" * 32,
                clock=clock_at(FREEZE_TIME),
                test_mode=True,
            )

    def test_code_change_requires_new_cohort(self):
        prereg = json.loads(
            cohort_paths(COHORT, self.forward_root)
            .preregistration.read_text(encoding="utf-8")
        )
        changed = copy.deepcopy(
            prereg["frozen_hashes"]["forward_source_files"]
        )
        changed[next(iter(changed))] = "0" * 64
        with patch(
            "src.probability_forward.forward_code_binding",
            return_value={
                "files": changed,
                "forward_code_hash": "0" * 64,
            },
        ):
            with self.assertRaisesRegex(RuntimeError, "new cohort"):
                validate_preregistration(prereg)

    def test_capture_is_weekly_idempotent_and_mismatch_fails(self):
        first = self._capture()
        self.assertTrue(first["created"])
        self.assertEqual(first["prediction_count"], 4)
        self.assertEqual(first["eligible_count"], 4)
        second = self._capture()
        self.assertTrue(second["idempotent"])
        self.assertEqual(
            verify_cohort(
                cohort_id=COHORT,
                root=self.forward_root,
                test_mode=True,
            )["anchors"],
            1,
        )
        paths = cohort_paths(COHORT, self.forward_root)
        artifact = load_shadow_artifact(paths.artifact)
        prereg = json.loads(paths.preregistration.read_text(encoding="utf-8"))
        weekly_path = paths.predictions / "2026-08-21.json.gz"
        envelope = read_gzip_json(weekly_path)
        digest = sha256_bytes(weekly_path.read_bytes())
        handles = [
            ForwardLedger(
                paths.ledger,
                cohort_id=COHORT,
                artifact_hash=artifact["artifact_hash"],
                preregistration_hash=prereg["preregistration_hash"],
                signing_key=b"k" * 32,
            )
            for _ in range(2)
        ]
        try:
            self.assertFalse(handles[0].insert_capture(envelope, file_digest=digest))
            self.assertFalse(handles[1].insert_capture(envelope, file_digest=digest))
            with self.assertRaisesRegex(RuntimeError, "different content"):
                handles[1].insert_capture(envelope, file_digest="0" * 64)
            self.assertEqual(
                handles[0].connection.execute("SELECT COUNT(*) FROM anchors").fetchone()[0],
                1,
            )
        finally:
            for handle in handles:
                handle.close()

        revised = self.stock.copy()
        revised.loc[revised.index[-1], "Close"] *= 1.15
        revised.loc[revised.index[-1], "RawClose"] *= 1.15
        revised.loc[revised.index[-1], "High"] = max(
            revised.loc[revised.index[-1], "High"],
            revised.loc[revised.index[-1], "Close"],
        )
        with self.assertRaisesRegex(RuntimeError, "refusing rewrite"):
            self._capture(stock=revised)

    def test_hash_chain_appends_and_backfill_is_rejected(self):
        self._capture()
        manifest_path = cohort_paths(COHORT, self.forward_root).manifest
        first_seal = manifest_path.read_bytes()
        later_stock = extend_history(self.stock, 5, seed=83)
        later_spy = extend_history(self.spy, 5, seed=84)
        second = self._capture(
            stock=later_stock,
            spy=later_spy,
            now="2026-08-28T23:30:00+00:00",
        )
        self.assertTrue(second["created"])
        self.assertEqual(
            verify_cohort(
                cohort_id=COHORT,
                root=self.forward_root,
                test_mode=True,
            )["anchors"],
            2,
        )
        with self.assertRaisesRegex(RuntimeError, "backfill"):
            self._capture(now="2026-08-29T12:00:00+00:00")
        manifest_path.write_bytes(first_seal)
        with self.assertRaisesRegex(
            RuntimeError,
            "does not match|rollback|diverges",
        ):
            verify_cohort(
                cohort_id=COHORT,
                root=self.forward_root,
                test_mode=True,
            )

    def test_feature_asof_ignores_stock_rows_after_anchor(self):
        baseline_root = self.short_work / "base"
        future_root = self.short_work / "asof"
        shutil.copytree(self.fixture_root, baseline_root)
        shutil.copytree(self.fixture_root, future_root)
        baseline = capture_from_histories(
            {"AAA": self.stock, "SPY": self.spy},
            self._eligibility(),
            cohort_id=COHORT,
            root=baseline_root,
            clock=clock_at(CAPTURE_TIME),
            test_mode=True,
        )
        stock_with_future = extend_history(self.stock, 1, seed=91)
        future = capture_from_histories(
            {"AAA": stock_with_future, "SPY": self.spy},
            self._eligibility(),
            cohort_id=COHORT,
            root=future_root,
            clock=clock_at(CAPTURE_TIME),
            test_mode=True,
        )
        self.assertEqual(baseline["prediction_count"], future["prediction_count"])
        baseline_file = cohort_paths(COHORT, baseline_root).predictions / "2026-08-21.json.gz"
        future_file = cohort_paths(COHORT, future_root).predictions / "2026-08-21.json.gz"
        self.assertEqual(baseline_file.read_bytes(), future_file.read_bytes())

    def test_spy_and_stock_must_equal_expected_completed_us_session(self):
        stale_spy = self.spy.iloc[:-1].copy()
        with self.assertRaisesRegex(RuntimeError, "SPY is stale"):
            self._capture(spy=stale_spy)
        with self.assertRaisesRegex(RuntimeError, "histories are stale"):
            self._capture(stock=self.stock.iloc[:-1], spy=stale_spy)

        stale_stock = self.stock.iloc[:-1].copy()
        result = self._capture(stock=stale_stock)
        self.assertEqual(result["prediction_count"], 0)
        self.assertGreaterEqual(result["exclusion_count"], 1)

        holiday_now = datetime.fromisoformat("2026-04-03T23:30:00+00:00")
        holiday_stock = self.stock.loc[self.stock.index <= "2026-04-02"]
        holiday_spy = self.spy.loc[self.spy.index <= "2026-04-02"]
        self.assertEqual(
            _expected_completed_us_session(
                {"AAA": holiday_stock, "SPY": holiday_spy},
                us_session_symbols={"AAA"},
                now=holiday_now,
            ).isoformat(),
            "2026-04-02",
        )

    def test_maturity_schedule_uses_frozen_us_sessions(self):
        self._capture()
        paths = cohort_paths(COHORT, self.forward_root)
        artifact = load_shadow_artifact(paths.artifact)
        prereg = json.loads(paths.preregistration.read_text(encoding="utf-8"))
        with ForwardLedger(
            paths.ledger,
            cohort_id=COHORT,
            artifact_hash=artifact["artifact_hash"],
            preregistration_hash=prereg["preregistration_hash"],
            signing_key=b"k" * 32,
        ) as ledger:
            row = ledger.connection.execute(
                """
                SELECT maturity_session_date, maturity_schedule_version
                FROM predictions WHERE horizon_sessions = 21
                """
            ).fetchone()
        self.assertEqual(row["maturity_session_date"], "2026-09-22")
        self.assertEqual(
            row["maturity_schedule_version"],
            "nyse-weekday-holidays-v1",
        )
        schedule = _prospective_schedule(datetime.fromisoformat(FREEZE_TIME).date())
        self.assertEqual(schedule["first_1m_maturity_estimate"], "2026-09-22")

    def test_duplicate_issuer_and_ood_are_withheld(self):
        selection = select_eligible_universe(
            [
                {"symbol": "AAA", "exchange": "NYSE"},
                {"symbol": "AAB", "exchange": "NASDAQ"},
            ],
            {
                "AAA": {
                    "quote_type": "EQUITY",
                    "reported_currency": "USD",
                    "issuer_uuid": "same",
                },
                "AAB": {
                    "quote_type": "EQUITY",
                    "reported_currency": "USD",
                    "issuer_uuid": "same",
                },
            },
        )
        self.assertEqual(selection.symbols, ("AAA",))
        self.assertIn("duplicate issuer", selection.excluded["AAB"])

        extreme = self.stock.copy()
        extreme.loc[extreme.index[-60]:, ["Open", "High", "Low", "Close", "RawOpen", "RawClose"]] *= 20
        result = self._capture(stock=extreme)
        self.assertEqual(result["eligible_count"], 0)
        self.assertEqual(result["withheld_count"], 4)
        ledger = sqlite3.connect(cohort_paths(COHORT, self.forward_root).ledger)
        try:
            total, unique_total = ledger.execute(
                """
                SELECT COUNT(*), COUNT(DISTINCT anchor_date || '|' || symbol || '|' || reason)
                FROM exclusions
                """
            ).fetchone()
            self.assertEqual(total, unique_total)
        finally:
            ledger.close()

    def test_maturity_is_exact_never_early_and_missing_is_unresolved(self):
        self._capture(eligibility=self._eligibility(include_bbb=True))
        early_stock = extend_history(self.stock, 20, seed=101)
        early_spy = extend_history(self.spy, 20, seed=102)
        early = evaluate_from_histories(
            {"AAA": early_stock, "BBB": early_stock, "SPY": early_spy},
            cohort_id=COHORT,
            root=self.forward_root,
            clock=clock_at("2026-09-18T23:30:00+00:00"),
            test_mode=True,
        )
        self.assertEqual(early["newly_labeled"], 0)

        mature_stock = extend_history(self.stock, 260, seed=103)
        mature_spy = extend_history(self.spy, 260, seed=104)
        mature = evaluate_from_histories(
            {"AAA": mature_stock, "SPY": mature_spy},
            cohort_id=COHORT,
            root=self.forward_root,
            clock=clock_at("2027-09-01T23:30:00+00:00"),
            test_mode=True,
        )
        self.assertEqual(mature["newly_labeled"], 4)
        self.assertEqual(mature["newly_unresolved"], 4)
        self.assertEqual(mature["integrity"]["labels"], 4)

        paths = cohort_paths(COHORT, self.forward_root)
        prereg = json.loads(paths.preregistration.read_text(encoding="utf-8"))
        artifact = load_shadow_artifact(paths.artifact)
        with ForwardLedger(
            paths.ledger,
            cohort_id=COHORT,
            artifact_hash=artifact["artifact_hash"],
            preregistration_hash=prereg["preregistration_hash"],
            signing_key=b"k" * 32,
        ) as ledger:
            h21 = ledger.connection.execute(
                """
                SELECT p.feature_date, l.entry_timestamp, l.exit_timestamp
                FROM labels AS l JOIN predictions AS p
                ON p.prediction_id = l.prediction_id
                WHERE p.symbol = 'AAA' AND p.horizon_sessions = 21
                """
            ).fetchone()
            feature_position = np.flatnonzero(
                mature_stock.index == pd.Timestamp(h21["feature_date"])
            )[0]
            self.assertEqual(
                pd.Timestamp(h21["entry_timestamp"]),
                mature_stock.index[feature_position + 1],
            )
            self.assertEqual(
                pd.Timestamp(h21["exit_timestamp"]),
                mature_stock.index[feature_position + 21],
            )
            with self.assertRaises(sqlite3.IntegrityError):
                ledger.connection.execute(
                    "UPDATE labels SET gross_return = gross_return + 1"
                )

    def test_tampering_is_detected_and_backup_is_valid(self):
        self._capture()
        paths = cohort_paths(COHORT, self.forward_root)
        artifact = load_shadow_artifact(paths.artifact)
        prereg = json.loads(paths.preregistration.read_text(encoding="utf-8"))
        with ForwardLedger(
            paths.ledger,
            cohort_id=COHORT,
            artifact_hash=artifact["artifact_hash"],
            preregistration_hash=prereg["preregistration_hash"],
            signing_key=b"k" * 32,
        ) as ledger:
            backup = ledger.backup(
                paths.backups,
                manifest_path=paths.manifest,
            )
        check = sqlite3.connect(backup)
        try:
            self.assertEqual(check.execute("PRAGMA integrity_check").fetchone()[0], "ok")
        finally:
            check.close()
        sidecar = backup.with_name(backup.name.replace(".db", ".manifest.json"))
        self.assertTrue(sidecar.exists())
        backed_manifest = json.loads(sidecar.read_text(encoding="utf-8"))
        live_manifest = json.loads(paths.manifest.read_text(encoding="utf-8"))
        self.assertEqual(
            backed_manifest["event_head_hash"],
            live_manifest["event_head_hash"],
        )

        raw = sqlite3.connect(paths.ledger)
        try:
            raw.execute("DROP TRIGGER predictions_no_update")
            raw.execute(
                "UPDATE predictions SET raw_ordered_json = '[1,0,0,0,0,0,0]' "
                "WHERE rowid = (SELECT MIN(rowid) FROM predictions)"
            )
            raw.commit()
        finally:
            raw.close()
        with self.assertRaisesRegex(RuntimeError, "reconcile|signature mismatch"):
            verify_cohort(
                cohort_id=COHORT,
                root=self.forward_root,
                test_mode=True,
            )

    def test_sealed_events_detect_metadata_anchor_chain_and_manifest_tampering(self):
        def captured_root(label: str) -> Path:
            root = self.short_work / label
            shutil.copytree(self.fixture_root, root)
            capture_from_histories(
                {"AAA": self.stock, "SPY": self.spy},
                self._eligibility(),
                cohort_id=COHORT,
                root=root,
                clock=clock_at(CAPTURE_TIME),
                test_mode=True,
            )
            return root

        probes = {
            "metadata": [
                "DROP TRIGGER metadata_no_update",
                "UPDATE metadata SET value_json='\"edited\"' WHERE key='artifact_hash'",
            ],
            "anchor": [
                "DROP TRIGGER anchors_no_update",
                "UPDATE anchors SET spy_asof='2026-08-20'",
            ],
            "valid_prefix_rollback": [
                "DROP TRIGGER events_no_delete",
                "DROP TRIGGER predictions_no_delete",
                "DROP TRIGGER exclusions_no_delete",
                "DROP TRIGGER anchors_no_delete",
                "DELETE FROM predictions",
                "DELETE FROM exclusions",
                "DELETE FROM anchors",
                "DELETE FROM events WHERE sequence > 1",
            ],
            "empty_chain": [
                "DROP TRIGGER events_no_delete",
                "DELETE FROM events",
            ],
        }
        for label, statements in probes.items():
            with self.subTest(label=label):
                root = captured_root(label)
                ledger_path = cohort_paths(COHORT, root).ledger
                connection = sqlite3.connect(ledger_path)
                try:
                    for statement in statements:
                        connection.execute(statement)
                    connection.commit()
                finally:
                    connection.close()
                with self.assertRaises(RuntimeError):
                    verify_cohort(
                        cohort_id=COHORT,
                        root=root,
                        test_mode=True,
                    )

    def test_two_phase_seal_recovers_both_crash_windows(self):
        paths = cohort_paths(COHORT, self.forward_root)
        artifact = load_shadow_artifact(paths.artifact)
        prereg = json.loads(paths.preregistration.read_text(encoding="utf-8"))

        def append_report(label: str):
                    with ForwardLedger(
                        paths.ledger,
                        cohort_id=COHORT,
                        artifact_hash=artifact["artifact_hash"],
                        preregistration_hash=prereg["preregistration_hash"],
                        signing_key=b"k" * 32,
                    ) as ledger:
                        ledger.verify_sealed_manifest(
                            paths.manifest,
                            b"k" * 32,
                            candidate_report_path=paths.candidate_report,
                        )
                        snapshot = ledger.snapshot_identity()
                        ledger.record_candidate_report(
                            self._dummy_report(label),
                            expected_snapshot=snapshot,
                        )

        append_report("before-manifest")
        with ForwardLedger(
                    paths.ledger,
                    cohort_id=COHORT,
                    artifact_hash=artifact["artifact_hash"],
                    preregistration_hash=prereg["preregistration_hash"],
                    signing_key=b"k" * 32,
        ) as ledger:
                    with self.assertRaisesRegex(RuntimeError, "simulated-before-manifest"):
                        ledger.recover_seal(
                            paths.manifest,
                            b"k" * 32,
                            candidate_report_path=paths.candidate_report,
                            crash_injector=lambda phase: (
                                (_ for _ in ()).throw(
                                    RuntimeError("simulated-before-manifest")
                                )
                                if phase == "after_db_commit_before_manifest"
                                else None
                            ),
                        )
                    seal = ledger.connection.execute(
                        "SELECT * FROM seal_state WHERE singleton = 1"
                    ).fetchone()
                    self.assertIsNotNone(seal["pending_event_count"])
        recovered = recover_cohort(
                    cohort_id=COHORT,
                    root=self.forward_root,
                    test_mode=True,
        )
        self.assertEqual(
                    recovered["latest_candidate_report_hash"],
                    self._dummy_report("before-manifest")["report_hash"],
        )
        self.assertEqual(
                    recover_cohort(
                        cohort_id=COHORT,
                        root=self.forward_root,
                        test_mode=True,
                    )["event_head_hash"],
                    recovered["event_head_hash"],
        )

        append_report("before-finalize")
        with ForwardLedger(
                    paths.ledger,
                    cohort_id=COHORT,
                    artifact_hash=artifact["artifact_hash"],
                    preregistration_hash=prereg["preregistration_hash"],
                    signing_key=b"k" * 32,
        ) as ledger:
                    with self.assertRaisesRegex(RuntimeError, "simulated-before-finalize"):
                        ledger.recover_seal(
                            paths.manifest,
                            b"k" * 32,
                            candidate_report_path=paths.candidate_report,
                            crash_injector=lambda phase: (
                                (_ for _ in ()).throw(
                                    RuntimeError("simulated-before-finalize")
                                )
                                if phase == "after_manifest_before_finalize"
                                else None
                            ),
                        )
                    manifest = json.loads(paths.manifest.read_text(encoding="utf-8"))
                    seal = ledger.connection.execute(
                        "SELECT * FROM seal_state WHERE singleton = 1"
                    ).fetchone()
                    self.assertEqual(
                        manifest["event_count"],
                        seal["pending_event_count"],
                    )
        recovered = recover_cohort(
                    cohort_id=COHORT,
                    root=self.forward_root,
                    test_mode=True,
        )
        self.assertEqual(
                    recovered["latest_candidate_report_hash"],
                    self._dummy_report("before-finalize")["report_hash"],
        )
        verify_cohort(
                    cohort_id=COHORT,
                    root=self.forward_root,
                    test_mode=True,
        )

    def test_two_phase_recovery_rejects_manifest_ahead_or_tampered(self):
        paths = cohort_paths(COHORT, self.forward_root)
        manifest = json.loads(paths.manifest.read_text(encoding="utf-8"))
        core = {
                    key: value
                    for key, value in manifest.items()
                    if key not in {"manifest_hash", "manifest_signature"}
        }
        core["event_count"] += 1
        manifest_hash = sha256_bytes(canonical_json_bytes(core))
        ahead = {
                    **core,
                    "manifest_hash": manifest_hash,
                    "manifest_signature": signed_digest(manifest_hash, b"k" * 32),
        }
        paths.manifest.write_text(json.dumps(ahead), encoding="utf-8")
        with self.assertRaisesRegex(RuntimeError, "ahead"):
                    recover_cohort(
                        cohort_id=COHORT,
                        root=self.forward_root,
                        test_mode=True,
                    )

        root = self.short_work / "manifest-unsigned"
        shutil.copytree(self.fixture_root, root)
        capture_from_histories(
            {"AAA": self.stock, "SPY": self.spy},
            self._eligibility(),
            cohort_id=COHORT,
            root=root,
            clock=clock_at(CAPTURE_TIME),
            test_mode=True,
        )
        manifest_path = cohort_paths(COHORT, root).manifest
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        manifest["event_count"] = 0
        manifest_path.write_text(json.dumps(manifest), encoding="utf-8")
        with self.assertRaisesRegex(RuntimeError, "manifest signature"):
            verify_cohort(
                cohort_id=COHORT,
                root=root,
                test_mode=True,
            )

    def test_public_aggregate_has_no_private_values(self):
        self._capture()
        status = build_public_status(
            cohort_id=COHORT,
            root=self.forward_root,
            clock=clock_at(CAPTURE_TIME),
            test_mode=True,
        )
        validate_forward_validation_status(status)
        self.assertFalse(public_status_json_contains_private_values(status))
        encoded = json.dumps(status).casefold()
        for forbidden in (
            "raw_ordered_probabilities",
            "derived_probabilities",
            "feature_vector",
            "coefficient",
            "entry_open",
            "gross_return",
        ):
            self.assertNotIn(forbidden, encoded)

        leaked = initial_forward_validation_status()
        leaked["probabilities"] = [0.1, 0.2, 0.7]
        with self.assertRaises(ValueError):
            finalize_forward_validation_status(leaked)

    def test_public_status_recomputes_gates_and_ignores_stale_candidate(self):
        self._capture()
        status = build_public_status(
            cohort_id=COHORT,
            root=self.forward_root,
            clock=clock_at(CAPTURE_TIME),
            test_mode=True,
        )
        self.assertEqual(status["status"], "collecting")
        self.assertEqual(sum(status["matured_outcomes"].values()), 0)

        report_path = cohort_paths(COHORT, self.forward_root).candidate_report
        stale = json.loads(report_path.read_text(encoding="utf-8"))
        stale["all_gates_passed"] = True
        stale["status"] = "eligible_for_review"
        stale.pop("report_hash", None)
        stale["report_hash"] = canonical_hash(stale)
        report_path.write_text(json.dumps(stale), encoding="utf-8")
        recomputed = build_public_status(
            cohort_id=COHORT,
            root=self.forward_root,
            clock=clock_at("2026-08-22T12:00:00+00:00"),
            test_mode=True,
        )
        self.assertEqual(recomputed["status"], "collecting")

        mature_stock = extend_history(self.stock, 260, seed=131)
        mature_spy = extend_history(self.spy, 260, seed=132)
        evaluate_from_histories(
            {"AAA": mature_stock, "SPY": mature_spy},
            cohort_id=COHORT,
            root=self.forward_root,
            clock=clock_at("2027-09-01T23:30:00+00:00"),
            test_mode=True,
        )
        report = json.loads(report_path.read_text(encoding="utf-8"))
        self.assertFalse(report["all_gates_passed"])
        self.assertEqual(
            report["provider_support"]["minimum_anchor_successful_issuers"],
            1,
        )
        reasons = [
            reason
            for horizon in report["horizons"].values()
            for threshold in horizon["thresholds"].values()
            for reason in threshold["reasons"]
        ]
        self.assertTrue(
            any("provider successful-issuer count" in reason for reason in reasons)
        )

    def test_development_bootstrap_override_can_never_be_eligible(self):
        self._capture()
        report = build_candidate_report(
            cohort_id=COHORT,
            root=self.forward_root,
            bootstrap_repetitions=1,
            clock=clock_at(CAPTURE_TIME),
            test_mode=True,
        )
        self.assertTrue(report["development_test_mode"])
        self.assertFalse(report["all_gates_passed"])
        self.assertNotEqual(report["status"], "eligible_for_review")

    def test_candidate_report_is_covered_by_sealed_event_head(self):
        self._capture()
        mature_stock = extend_history(self.stock, 260, seed=141)
        mature_spy = extend_history(self.spy, 260, seed=142)
        evaluate_from_histories(
            {"AAA": mature_stock, "SPY": mature_spy},
            cohort_id=COHORT,
            root=self.forward_root,
            clock=clock_at("2027-09-01T23:30:00+00:00"),
            test_mode=True,
        )
        report_path = cohort_paths(COHORT, self.forward_root).candidate_report
        original = json.loads(report_path.read_text(encoding="utf-8"))
        tampered = copy.deepcopy(original)
        tampered["all_gates_passed"] = True
        tampered.pop("report_hash", None)
        tampered["report_hash"] = canonical_hash(tampered)
        report_path.write_text(json.dumps(tampered), encoding="utf-8")
        verify_cohort(
            cohort_id=COHORT,
            root=self.forward_root,
            test_mode=True,
        )
        self.assertEqual(
            json.loads(report_path.read_text(encoding="utf-8")),
            original,
        )

    def test_report_retries_if_event_head_changes_during_metrics_snapshot(self):
        self._capture()
        mature_stock = extend_history(self.stock, 260, seed=151)
        mature_spy = extend_history(self.spy, 260, seed=152)
        evaluate_from_histories(
            {"AAA": mature_stock, "SPY": mature_spy},
            cohort_id=COHORT,
            root=self.forward_root,
            clock=clock_at("2027-09-01T23:30:00+00:00"),
            test_mode=True,
            recompute_report=False,
        )
        paths = cohort_paths(COHORT, self.forward_root)
        artifact = load_shadow_artifact(paths.artifact)
        prereg = json.loads(paths.preregistration.read_text(encoding="utf-8"))
        injected = {"done": False, "event_count": None}
        original_metrics = _point_metrics

        def concurrent_metrics(labels, probabilities, baselines):
            if not injected["done"]:
                injected["done"] = True
                with ForwardLedger(
                    paths.ledger,
                    cohort_id=COHORT,
                    artifact_hash=artifact["artifact_hash"],
                    preregistration_hash=prereg["preregistration_hash"],
                    signing_key=b"k" * 32,
                ) as other:
                    other.verify_sealed_manifest(
                        paths.manifest,
                        b"k" * 32,
                        candidate_report_path=paths.candidate_report,
                    )
                    snapshot = other.snapshot_identity()
                    other.record_candidate_report(
                        self._dummy_report("concurrent"),
                        expected_snapshot=snapshot,
                    )
                    injected["event_count"] = snapshot["event_count"] + 1
            return original_metrics(labels, probabilities, baselines)

        with patch(
            "src.probability_forward._point_metrics",
            side_effect=concurrent_metrics,
        ):
            report = build_candidate_report(
                cohort_id=COHORT,
                root=self.forward_root,
                bootstrap_repetitions=0,
                clock=clock_at("2027-09-02T00:00:00+00:00"),
                test_mode=True,
            )
        self.assertTrue(injected["done"])
        self.assertEqual(
            report["source_snapshot"]["event_count"],
            injected["event_count"],
        )
        with ForwardLedger(
            paths.ledger,
            cohort_id=COHORT,
            artifact_hash=artifact["artifact_hash"],
            preregistration_hash=prereg["preregistration_hash"],
            signing_key=b"k" * 32,
        ) as ledger:
            latest = ledger.connection.execute(
                "SELECT event_type, entity_key FROM events "
                "ORDER BY sequence DESC LIMIT 1"
            ).fetchone()
        self.assertEqual(latest["event_type"], "candidate_report")
        self.assertEqual(latest["entity_key"], report["report_hash"])

        with patch(
            "src.probability_forward.build_candidate_report",
            return_value={
                "all_gates_passed": True,
                "report_hash": "stale-report-hash",
            },
        ):
            status = build_public_status(
                cohort_id=COHORT,
                root=self.forward_root,
                clock=clock_at("2027-09-02T01:00:00+00:00"),
                test_mode=True,
            )
        self.assertNotEqual(status["status"], "eligible_for_review")

    def test_public_status_snapshot_cas_retries_multiple_concurrent_captures(self):
        writes = {"count": 0}
        later_stock = extend_history(self.stock, 5, seed=161)
        later_spy = extend_history(self.spy, 5, seed=162)

        def concurrent_capture():
            if writes["count"] == 0:
                capture_from_histories(
                    {"AAA": self.stock, "SPY": self.spy},
                    self._eligibility(),
                    cohort_id=COHORT,
                    root=self.forward_root,
                    clock=clock_at(CAPTURE_TIME),
                    test_mode=True,
                )
            elif writes["count"] == 1:
                capture_from_histories(
                    {"AAA": later_stock, "SPY": later_spy},
                    self._eligibility(),
                    cohort_id=COHORT,
                    root=self.forward_root,
                    clock=clock_at("2026-08-28T23:30:00+00:00"),
                    test_mode=True,
                )
            writes["count"] += 1

        status_path = self.short_work / "status.json"
        status = publish_local_status(
            cohort_id=COHORT,
            root=self.forward_root,
            status_path=status_path,
            snapshot_path=self.short_work / "missing-latest.json",
            clock=clock_at("2026-08-29T12:00:00+00:00"),
            test_mode=True,
            _snapshot_hook=concurrent_capture,
        )
        self.assertEqual(writes["count"], 3)
        self.assertEqual(status["weeks_captured"], 2)
        self.assertEqual(
            json.loads(status_path.read_text(encoding="utf-8"))[
                "weeks_captured"
            ],
            2,
        )
        paths = cohort_paths(COHORT, self.forward_root)
        manifest = json.loads(paths.manifest.read_text(encoding="utf-8"))
        database = sqlite3.connect(paths.ledger)
        try:
            event_count = database.execute("SELECT COUNT(*) FROM events").fetchone()[0]
            anchor_count = database.execute("SELECT COUNT(*) FROM anchors").fetchone()[0]
            seal = database.execute(
                "SELECT * FROM seal_state WHERE singleton = 1"
            ).fetchone()
        finally:
            database.close()
        self.assertEqual(anchor_count, 2)
        self.assertEqual(manifest["event_count"], event_count)
        self.assertEqual(seal[1], event_count)
        self.assertNotEqual((status["weeks_captured"], event_count), (0, 2))

    def test_known_metrics_and_forward_count_gates(self):
        labels = np.asarray([0, 1, 2, 0, 1, 2])
        probabilities = np.asarray(
            [
                [0.8, 0.1, 0.1],
                [0.1, 0.8, 0.1],
                [0.1, 0.1, 0.8],
                [0.7, 0.2, 0.1],
                [0.2, 0.7, 0.1],
                [0.1, 0.2, 0.7],
            ]
        )
        baseline = np.full((len(labels), 3), 1 / 3)
        metrics = _point_metrics(labels, probabilities, baseline)
        expected_brier = float(
            np.mean(
                np.sum(
                    np.square(probabilities - np.eye(3)[labels]),
                    axis=1,
                )
            )
        )
        self.assertAlmostEqual(metrics["brier"], expected_brier)
        self.assertGreater(metrics["brier_skill"], 0)
        self.assertGreater(metrics["log_loss_improvement"], 0)

        rows = []
        gate_labels = []
        dates = pd.date_range("2026-01-02", periods=104, freq="W-FRI")
        for index in range(600):
            rows.append(
                {
                    "feature_date": dates[index % len(dates)].date().isoformat(),
                    "issuer_key": f"issuer-{index % 200}",
                }
            )
            gate_labels.append(index % 3)
        support = _prospective_support(rows, np.asarray(gate_labels))
        self.assertTrue(support["passed"])
        self.assertEqual(min(support["class_counts"].values()), 200)

    def test_schema_migration_and_binding_are_fail_closed(self):
        path = self.short_work / "new.db"
        with ForwardLedger(
            path,
            cohort_id="cohort-binding-test",
            artifact_hash="a" * 64,
            preregistration_hash="b" * 64,
            signing_key=b"s" * 32,
        ) as first:
            self.assertEqual(first.connection.execute("PRAGMA user_version").fetchone()[0], 3)
            self.assertEqual(
                first.connection.execute("PRAGMA journal_mode").fetchone()[0].casefold(),
                "wal",
            )
        with ForwardLedger(
            path,
            cohort_id="cohort-binding-test",
            artifact_hash="a" * 64,
            preregistration_hash="b" * 64,
            signing_key=b"s" * 32,
        ):
            pass
        first = ForwardLedger(
            path,
            cohort_id="cohort-binding-test",
            artifact_hash="a" * 64,
            preregistration_hash="b" * 64,
            signing_key=b"s" * 32,
        )
        second = ForwardLedger(
            path,
            cohort_id="cohort-binding-test",
            artifact_hash="a" * 64,
            preregistration_hash="b" * 64,
            signing_key=b"s" * 32,
        )
        try:
            first.connection.execute("BEGIN IMMEDIATE")
            second.connection.execute("PRAGMA busy_timeout = 50")
            with self.assertRaisesRegex(sqlite3.OperationalError, "locked"):
                second.connection.execute("BEGIN IMMEDIATE")
            first.connection.execute("ROLLBACK")
            with second.transaction():
                pass
        finally:
            first.close()
            second.close()
        with self.assertRaisesRegex(RuntimeError, "binding"):
            ForwardLedger(
                path,
                cohort_id="different-cohort",
                artifact_hash="a" * 64,
                preregistration_hash="b" * 64,
                signing_key=b"s" * 32,
            )

        newer = self.short_work / "newer.db"
        connection = sqlite3.connect(newer)
        connection.execute("PRAGMA user_version = 99")
        connection.close()
        with self.assertRaisesRegex(RuntimeError, "newer"):
            ForwardLedger(
                newer,
                cohort_id="newer-cohort",
                artifact_hash="a" * 64,
                preregistration_hash="b" * 64,
                signing_key=b"s" * 32,
            )

    def test_concurrent_immutable_file_creation_is_domain_safe(self):
        def run_pair(path: Path, values: tuple[bytes, bytes]):
            barrier = threading.Barrier(2)
            results = []
            errors = []

            def writer(value: bytes):
                try:
                    barrier.wait()
                    results.append(write_immutable_bytes(path, value))
                except Exception as exc:
                    errors.append(exc)

            threads = [
                threading.Thread(target=writer, args=(value,))
                for value in values
            ]
            for thread in threads:
                thread.start()
            for thread in threads:
                thread.join()
            return results, errors

        same_path = self.short_work / "same.bin"
        results, errors = run_pair(same_path, (b"same", b"same"))
        self.assertFalse(errors)
        self.assertEqual(sorted(results), [False, True])
        self.assertEqual(same_path.read_bytes(), b"same")

        different_path = self.short_work / "different.bin"
        results, errors = run_pair(different_path, (b"left", b"right"))
        self.assertEqual(results, [True])
        self.assertEqual(len(errors), 1)
        self.assertIsInstance(errors[0], ForwardIntegrityError)
        self.assertNotIsInstance(errors[0], FileExistsError)


if __name__ == "__main__":
    import unittest

    unittest.main()
