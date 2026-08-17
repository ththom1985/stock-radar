from __future__ import annotations

import copy
import subprocess
import sys
import unittest
from datetime import datetime, timezone

import numpy as np
import pandas as pd

from src.persistence import atomic_write_json
from src.export_static import (
    _compact_probability_data,
    _hydrate_compact_probability,
)
from src.probability_features import (
    FEATURE_NAMES,
    build_probability_features,
    latest_probability_features,
)
from src.probability_inference import (
    MAX_SPY_BUSINESS_LAG_DAYS,
    MAX_SPY_CALENDAR_LAG_DAYS,
    ProbabilityArtifactError,
    _select_spy_asof_history,
    attach_probability_forecasts,
    empty_probability_artifact,
    evaluate_current_threshold_grid,
    finalize_artifact,
    score_probability_row,
    validate_probability_artifact,
    validate_probability_forecast,
)
from src.probability_model import (
    GRID_MONOTONICITY_GATES,
    PUBLISH_TRANSFORM,
    canonical_hash,
    fit_multinomial_model,
    predict_probabilities,
)
from tests.helpers import ROOT, ProjectTempMixin


NOW = datetime(2026, 8, 13, 18, 0, tzinfo=timezone.utc)


def current_history(seed=31):
    index = pd.bdate_range(end="2026-08-12", periods=720)
    rng = np.random.default_rng(seed)
    close = 100 * np.exp(np.cumsum(rng.normal(0.0003, 0.01, len(index))))
    open_price = close * np.exp(rng.normal(0, 0.002, len(index)))
    return pd.DataFrame(
        {
            "Open": open_price,
            "High": np.maximum(open_price, close) * 1.008,
            "Low": np.minimum(open_price, close) * 0.992,
            "Close": close,
            "RawClose": close,
            "Volume": rng.integers(800_000, 3_000_000, len(index)).astype(float),
        },
        index=index,
    )


def accepted_artifact(history, spy):
    binding = {"dataset_content_hash": "a" * 64}
    binding["binding_hash"] = canonical_hash(binding)
    feature_history = build_probability_features(history, spy).dropna()
    as_of = history.index[-20]
    center = feature_history.loc[as_of].to_numpy(dtype=float)
    rng = np.random.default_rng(44)
    scale = np.maximum(np.abs(center) * 0.08, 0.01)
    matrix = center + rng.normal(size=(1800, len(center))) * scale
    features = pd.DataFrame(matrix, columns=FEATURE_NAMES)
    latent = (
        (matrix[:, 0] - center[0]) / scale[0]
        - 0.6 * (matrix[:, 1] - center[1]) / scale[1]
    )
    labels = np.where(latent < -0.4, 0, np.where(latent > 0.4, 2, 1))
    model = fit_multinomial_model(features, labels)
    model.update(
        {
            "temperature": {
                "version": "temporal-temperature-v1",
                "value": 1.0,
                "fit_source": "outer_fold_calibration_only",
                "fit_rows": 900,
            },
            "model_key": "h21_x3",
            "horizon_sessions": 21,
            "threshold_pct": 3,
            "round_trip_cost_bps": 30,
            "publish_transform": dict(PUBLISH_TRANSFORM),
            "grid_monotonicity": {
                "passed": True,
                "thresholds": [3, 5, 10],
                "action": "withhold; no projection",
                "gates": GRID_MONOTONICITY_GATES,
            },
            "dataset_binding_hash": binding["binding_hash"],
            "checkpoint_key": "checkpoint-test",
            "history_years": 12.0,
            "full_test_fold_count": 6,
            "min_usable_train_years": 5.2,
            "trained_at": "2026-08-12T18:00:00+00:00",
            "training_cutoff": "2026-07-15",
            "baseline_rates": {"down": 0.30, "middle": 0.40, "up": 0.30},
            "oos_sample_size": 12500,
            "oos_metrics": {
                "brier": 0.5,
                "climatology_brier": 0.55,
                "brier_skill": 0.09,
                "log_loss": 0.9,
                "climatology_log_loss": 1.0,
                "log_loss_improvement": 0.10,
                "classwise_ece": {
                    "down": 0.02,
                    "middle": 0.02,
                    "up": 0.02,
                },
                "maximum_gap": 0.05,
                "calibration": {
                    name: {"slope": 1.0, "intercept": 0.0}
                    for name in ("down", "middle", "up")
                },
            },
            "fold_count": 6,
            "bootstrap": {
                "probability_error_offsets_ci95": {
                    "down": [-0.02, 0.02],
                    "middle": [-0.02, 0.02],
                    "up": [-0.02, 0.02],
                }
            },
            "event_counts_calibration_test": {
                "down": 1500,
                "middle": 3000,
                "up": 1500,
            },
            "accepted": True,
            "acceptance_reasons": [],
        }
    )
    model.pop("model_hash", None)
    model["model_hash"] = canonical_hash(model)
    artifact = empty_probability_artifact(
        created_at="2026-08-12T18:00:00+00:00"
    )
    artifact.update(
        {
            "training_cutoff": "2026-07-15",
            "production_status": "accepted_partial_grid",
            "production_reasons": [],
            "dataset_binding": binding,
            "models": {"h21_x3": model},
            "baselines": {
                "h21_x3": {
                    "model_key": "h21_x3",
                    "horizon_sessions": 21,
                    "threshold_pct": 3,
                    "rates": {"down": 0.30, "middle": 0.40, "up": 0.30},
                    "sample_size": 12500,
                    "event_counts_calibration_test": {
                        "down": 1500,
                        "middle": 3000,
                        "up": 1500,
                    },
                    "validation": {
                        "accepted": True,
                        "reasons": [],
                        "brier_skill": 0.09,
                        "classwise_ece": {
                            "down": 0.02,
                            "middle": 0.02,
                            "up": 0.02,
                        },
                        "fold_count": 6,
                    },
                }
            },
            "accepted_model_keys": ["h21_x3"],
        }
    )
    return finalize_artifact(artifact), as_of


def row_for(as_of):
    return {
        "symbol": "AAA",
        "asset_type": "company_equity",
        "currency": "USD",
        "bar_date": as_of.date().isoformat(),
        "bar_timestamp": f"{as_of.date().isoformat()}T20:00:00+00:00",
        "bar_age_days": 1,
        "completed_bars_only": True,
        "source_interval": "1d",
        "radar_score": 73,
        "sweet_spot": {"combined_status": "approaching"},
    }


class ProbabilityInferenceTests(ProjectTempMixin, unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.history = current_history()
        cls.spy = current_history(seed=32)
        cls.artifact, cls.as_of = accepted_artifact(cls.history, cls.spy)

    def test_accepted_forecast_sums_and_deep_contract(self):
        forecast = score_probability_row(
            row_for(self.as_of),
            self.history,
            self.spy,
            self.artifact,
            now=NOW,
        )
        self.assertEqual(forecast["status"], "partial")
        self.assertEqual(len(forecast["forecasts"]), 1)
        item = forecast["forecasts"][0]
        self.assertEqual(sum(item["probabilities_pct"].values()), 100)
        self.assertEqual(item["sum_pct"], 100)
        self.assertIn("gross UP >= +3.30%", item["definition"])
        self.assertEqual(
            item["model_interval_method"],
            "95% aggregate calibration-error interval approximation from fixed "
            "OOS predictions; not an individual stock outcome interval",
        )
        self.assertEqual(item["publish_transform"], PUBLISH_TRANSFORM)
        _timestamp, feature_vector = latest_probability_features(
            self.history,
            self.spy,
            as_of=self.as_of,
        )
        expected = predict_probabilities(
            self.artifact["models"]["h21_x3"],
            pd.DataFrame([feature_vector], columns=FEATURE_NAMES),
            require_complete=True,
        )[0]
        np.testing.assert_allclose(
            [item["probabilities"][name] for name in ("down", "middle", "up")],
            expected,
            rtol=0,
            atol=0,
        )
        self.assertIsNone(forecast["positive_net_return_probability"])
        self.assertTrue(forecast["separate_from_radar_score"])
        self.assertTrue(forecast["separate_from_sweet_spot"])
        validate_probability_forecast(forecast)

        compact_row = {
            "currency": "USD",
            "bar_date": self.as_of.date().isoformat(),
            "probability_forecast": copy.deepcopy(forecast),
        }
        reasons, models, baselines = _compact_probability_data([compact_row])
        self.assertIsInstance(compact_row["pf"][2], str)
        hydrated = _hydrate_compact_probability(
            compact_row["pf"],
            reason_catalog=reasons,
            model_catalog=models,
            baseline_catalog=baselines,
            contract={
                "publish_transform": PUBLISH_TRANSFORM,
                "model_interval_scope": (
                    "95% aggregate calibration-error interval approximation "
                    "from fixed OOS predictions; not an individual stock "
                    "outcome interval"
                ),
                "row_defaults": {
                    key: forecast[key]
                    for key in (
                        "schema_version",
                        "actionable",
                        "separate_from_radar_score",
                        "separate_from_insight_ranking",
                        "separate_from_sweet_spot",
                        "supported_partition",
                        "entry_assumption",
                        "cost_assumption_bps_round_trip",
                        "outcome_definition",
                        "positive_net_return_probability",
                        "positive_net_return_note",
                        "artifact_created_at",
                        "training_cutoff",
                        "survivorship_warning",
                    )
                }
            },
            listing_currency="USD",
            signal_timestamp=self.as_of.date().isoformat(),
        )
        self.assertEqual(
            hydrated["forecasts"][0]["probabilities_pct"],
            forecast["forecasts"][0]["probabilities_pct"],
        )
        self.assertEqual(
            hydrated["forecasts"][0]["model_interval_95_pct"],
            forecast["forecasts"][0]["model_interval_95_pct"],
        )
        validate_probability_forecast(hydrated)

    def test_stale_model_withholds_instrument_probability_but_keeps_baseline(self):
        artifact = copy.deepcopy(self.artifact)
        model = artifact["models"]["h21_x3"]
        model["trained_at"] = "2026-06-01T00:00:00+00:00"
        model.pop("model_hash")
        model["model_hash"] = canonical_hash(model)
        artifact = finalize_artifact(artifact)
        forecast = score_probability_row(
            row_for(self.as_of), self.history, self.spy, artifact, now=NOW
        )
        self.assertEqual(forecast["status"], "withheld")
        self.assertFalse(forecast["forecasts"])
        self.assertTrue(forecast["baselines"])
        self.assertTrue(
            all(
                "acceptance_reasons" not in item
                for item in forecast["baselines"]
            )
        )
        self.assertTrue(any("model age" in reason for reason in forecast["reasons"]))

    def test_unaccepted_grid_reason_does_not_block_an_accepted_model(self):
        artifact = copy.deepcopy(self.artifact)
        artifact["production_reasons"] = ["h252_x30 did not pass acceptance"]
        artifact = finalize_artifact(artifact)
        forecast = score_probability_row(
            row_for(self.as_of), self.history, self.spy, artifact, now=NOW
        )
        self.assertEqual(forecast["status"], "partial")
        self.assertTrue(forecast["forecasts"])
        self.assertIn(
            "h252_x30 did not pass acceptance",
            forecast["reasons"],
        )

    def test_too_wide_interval_withholds(self):
        artifact = copy.deepcopy(self.artifact)
        model = artifact["models"]["h21_x3"]
        model["bootstrap"]["probability_error_offsets_ci95"] = {
            name: [-0.11, 0.11] for name in ("down", "middle", "up")
        }
        model.pop("model_hash")
        model["model_hash"] = canonical_hash(model)
        artifact = finalize_artifact(artifact)
        forecast = score_probability_row(
            row_for(self.as_of), self.history, self.spy, artifact, now=NOW
        )
        self.assertEqual(forecast["status"], "withheld")
        self.assertTrue(
            any("interval width" in reason for reason in forecast["reasons"])
        )

    def test_unsupported_partition_and_short_history_withhold(self):
        row = row_for(self.as_of)
        row["currency"] = "EUR"
        forecast = score_probability_row(
            row,
            self.history.iloc[-100:],
            self.spy,
            self.artifact,
            now=NOW,
        )
        self.assertEqual(forecast["status"], "withheld")
        self.assertTrue(any("not USD" in reason for reason in forecast["reasons"]))
        self.assertTrue(
            any("insufficient history" in reason for reason in forecast["reasons"])
        )

    def test_stale_spy_bar_withholds(self):
        forecast = score_probability_row(
            row_for(self.as_of),
            self.history,
            self.spy.loc[: self.as_of - pd.Timedelta(days=7)],
            self.artifact,
            now=NOW,
        )
        self.assertEqual(forecast["status"], "withheld")
        self.assertTrue(
            any("stale SPY bar" in reason for reason in forecast["reasons"])
        )

    def test_stock_holiday_uses_previous_spy_session(self):
        spy = self.spy.drop(index=self.as_of)
        forecast = score_probability_row(
            row_for(self.as_of),
            self.history,
            spy,
            self.artifact,
            now=NOW,
        )
        self.assertEqual(forecast["status"], "partial")
        self.assertTrue(forecast["forecasts"])
        self.assertLess(
            forecast["spy_asof"]["selected_session"],
            self.as_of.date().isoformat(),
        )

    def test_saturday_and_future_only_spy_asof_rules(self):
        friday = max(
            timestamp
            for timestamp in self.spy.index
            if timestamp <= self.as_of and timestamp.dayofweek == 4
        )
        saturday = friday + pd.Timedelta(days=1)
        selected, detail = _select_spy_asof_history(
            self.spy,
            saturday,
        )
        self.assertEqual(selected.index[-1], friday)
        self.assertEqual(detail["selected_session"], friday.date().isoformat())
        self.assertLessEqual(
            detail["calendar_lag_days"],
            MAX_SPY_CALENDAR_LAG_DAYS,
        )
        self.assertLessEqual(
            detail["business_lag_days"],
            MAX_SPY_BUSINESS_LAG_DAYS,
        )
        stock = self.history.loc[:friday].copy()
        weekend_row = stock.loc[[friday]].copy()
        weekend_row.index = pd.DatetimeIndex([saturday])
        stock = pd.concat([stock, weekend_row])
        forecast = score_probability_row(
            row_for(saturday),
            stock,
            self.spy,
            self.artifact,
            now=NOW,
        )
        self.assertEqual(
            forecast["spy_asof"]["selected_session"],
            friday.date().isoformat(),
        )
        self.assertFalse(
            any(
                "SPY" in reason or "future" in reason
                for reason in forecast["reasons"]
            )
        )

        future_only = self.spy.copy()
        future_only.index = future_only.index + pd.Timedelta(days=5000)
        with self.assertRaisesRegex(ValueError, "future SPY history"):
            _select_spy_asof_history(future_only, saturday)

    def test_current_threshold_tolerance_never_changes_probabilities(self):
        small = {
            10: np.array([0.20, 0.49, 0.31]),
            20: np.array([0.19, 0.496, 0.314]),
        }
        before = {key: value.copy() for key, value in small.items()}
        diagnostic = evaluate_current_threshold_grid(small)
        self.assertTrue(diagnostic["permitted"])
        self.assertTrue(
            diagnostic["tolerated_independent_threshold_inversion"]
        )
        for key in small:
            np.testing.assert_array_equal(small[key], before[key])

        large = evaluate_current_threshold_grid(
            {
                10: np.array([0.20, 0.49, 0.31]),
                20: np.array([0.19, 0.494, 0.316]),
            }
        )
        self.assertFalse(large["permitted"])
        self.assertEqual(
            large["reason_code"], "current_threshold_non_monotonic"
        )

        display_break = evaluate_current_threshold_grid(
            {
                10: np.array([0.20, 0.496, 0.304]),
                20: np.array([0.19, 0.502, 0.308]),
            }
        )
        self.assertFalse(display_break["permitted"])
        self.assertLessEqual(display_break["max_up_inversion"], 0.005)
        self.assertFalse(display_break["whole_percent_display_monotonic"])

    def test_artifact_and_model_hash_mismatches_are_rejected(self):
        first = finalize_artifact(copy.deepcopy(self.artifact))
        second = finalize_artifact(copy.deepcopy(self.artifact))
        self.assertEqual(first["artifact_hash"], second["artifact_hash"])

        artifact = copy.deepcopy(self.artifact)
        artifact["code_hash"] = "0" * 64
        artifact = finalize_artifact(artifact)
        with self.assertRaisesRegex(
            ProbabilityArtifactError, "code hash mismatch"
        ):
            validate_probability_artifact(artifact)

        artifact = copy.deepcopy(self.artifact)
        artifact["models"]["h21_x3"]["intercept"][0] += 1
        artifact = finalize_artifact(artifact)
        with self.assertRaisesRegex(ProbabilityArtifactError, "model hash mismatch"):
            validate_probability_artifact(artifact)

    def test_no_accepted_artifact_is_baseline_only_and_prominent(self):
        artifact = empty_probability_artifact(
            created_at="2026-08-12T18:00:00+00:00",
            reason="strict acceptance failed",
        )
        artifact["baselines"] = copy.deepcopy(self.artifact["baselines"])
        artifact = finalize_artifact(artifact)
        forecast = score_probability_row(
            row_for(self.as_of), self.history, self.spy, artifact, now=NOW
        )
        self.assertEqual(forecast["status"], "withheld")
        self.assertEqual(
            forecast["message"], "No validated stock-specific probability edge"
        )
        self.assertFalse(forecast["forecasts"])
        self.assertTrue(forecast["baselines"])
        broken = copy.deepcopy(forecast)
        broken["message"] = "This stock will rise"
        with self.assertRaisesRegex(ValueError, "forbidden"):
            validate_probability_forecast(broken)

    def test_withheld_artifact_needs_no_history_for_supported_baselines(self):
        artifact = empty_probability_artifact(
            created_at="2026-08-12T18:00:00+00:00",
            reason="forward validation required",
        )
        artifact["baselines"] = copy.deepcopy(self.artifact["baselines"])
        artifact = finalize_artifact(artifact)
        row = row_for(self.as_of)

        forecast = score_probability_row(
            row, None, None, artifact, now=NOW
        )
        self.assertEqual(forecast["status"], "withheld")
        self.assertTrue(forecast["baselines"])
        self.assertFalse(
            any(
                "insufficient history" in reason
                for reason in forecast["reasons"]
            )
        )

        unsupported = copy.deepcopy(row)
        unsupported["currency"] = "EUR"
        withheld = score_probability_row(
            unsupported, None, None, artifact, now=NOW
        )
        self.assertFalse(withheld["baselines"])

    def test_attach_does_not_change_score_or_sweet_spot(self):
        path = self.work / "probability_models.json"
        atomic_write_json(path, self.artifact)
        row = row_for(self.as_of)
        before = copy.deepcopy(row)
        rows = [row]
        attach_probability_forecasts(
            rows,
            {"AAA": self.history, "SPY": self.spy},
            artifact_path=path,
            now=NOW,
        )
        self.assertEqual(row["radar_score"], before["radar_score"])
        self.assertEqual(row["sweet_spot"], before["sweet_spot"])
        self.assertIn("probability_forecast", row)

    def test_malformed_artifact_boundary_withholds_instead_of_crashing(self):
        artifact = copy.deepcopy(self.artifact)
        model = artifact["models"]["h21_x3"]
        model["coefficient"] = [[1.0]]
        model.pop("model_hash")
        model["model_hash"] = canonical_hash(model)
        artifact = finalize_artifact(artifact)
        path = self.work / "malformed-model.json"
        atomic_write_json(path, artifact)
        row = row_for(self.as_of)
        attach_probability_forecasts(
            [row],
            {"AAA": self.history, "SPY": self.spy},
            artifact_path=path,
            now=NOW,
        )
        self.assertEqual(row["probability_forecast"]["status"], "withheld")
        self.assertTrue(
            any(
                "invalid_artifact" in reason
                for reason in row["probability_forecast"]["reasons"]
            )
        )

        path.write_text('{"schema":NaN}', encoding="utf-8")
        row = row_for(self.as_of)
        attach_probability_forecasts(
            [row],
            {"AAA": self.history, "SPY": self.spy},
            artifact_path=path,
            now=NOW,
        )
        self.assertEqual(row["probability_forecast"]["status"], "withheld")
        self.assertTrue(
            any(
                "invalid_artifact" in reason
                for reason in row["probability_forecast"]["reasons"]
            )
        )

    def test_inference_import_does_not_load_sklearn(self):
        command = (
            "import sys; import src.probability_inference; "
            "assert not any(n == 'sklearn' or n.startswith('sklearn.') "
            "for n in sys.modules); assert 'yfinance' not in sys.modules"
        )
        subprocess.run(
            [sys.executable, "-c", command],
            check=True,
            cwd=str(ROOT),
            capture_output=True,
            text=True,
        )


if __name__ == "__main__":
    unittest.main()
