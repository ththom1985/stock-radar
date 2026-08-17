from __future__ import annotations

import copy
import unittest

import numpy as np
import pandas as pd

from src.persistence import atomic_write_json
from src.probability_features import FEATURE_NAMES
from src.probability_model import (
    PUBLISH_TRANSFORM,
    apply_grid_release_gate,
    apply_publish_transform,
    assess_ood,
    canonical_hash,
    classwise_reliability,
    evaluate_acceptance,
    evaluate_predictions,
    evaluate_raw_grid_monotonicity,
    fit_multinomial_model,
    fit_preprocessor,
    fit_temperature,
    predict_probabilities,
    two_way_cluster_bootstrap,
    transform_preprocessor,
    whole_identity_percentages,
)
from src.probability_train import (
    _load_or_new_validation,
    build_dataset_binding,
    checkpoint_key,
    make_synthetic_dataset,
    run_smoke,
    train_dataset,
)
from tests.helpers import ProjectTempMixin


def passing_report():
    calibration = {
        name: {"slope": 1.0, "intercept": 0.0}
        for name in ("down", "middle", "up")
    }
    fold = {
        "metrics": {"brier_skill": 0.04},
        "train": {
            "class_counts": {"down": 1500, "middle": 1500, "up": 1500}
        },
        "calibration": {
            "class_counts": {"down": 400, "middle": 400, "up": 400}
        },
        "test": {
            "class_counts": {"down": 300, "middle": 300, "up": 300}
        },
    }
    return {
        "history_years": 9.0,
        "fold_count": 5,
        "full_test_fold_count": 5,
        "min_usable_train_years": 5.1,
        "issuer_count": 250,
        "forecast_date_count": 150,
        "min_train_class_count": 1500,
        "min_calibration_class_count": 400,
        "min_test_class_count": 300,
        "inference_coverage": 0.9,
        "publish_transform_identity_verified": True,
        "provider_success_coverage": 0.9,
        "provider_requested_issuer_count": 260,
        "provider_successful_issuer_count": 250,
        "aggregate": {
            "brier_skill": 0.04,
            "log_loss_improvement": 0.02,
            "classwise_ece": {"down": 0.02, "middle": 0.02, "up": 0.02},
            "maximum_gap": 0.06,
            "calibration": calibration,
        },
        "bootstrap": {"brier_skill_ci95": [0.005, 0.08]},
        "folds": [{**copy.deepcopy(fold), "fold": index} for index in range(5)],
        "regime": {
            "groups": [
                {
                    "regime": "trend_up|vol_low",
                    "sample_gate_passed": True,
                    "max_class_ece": 0.04,
                }
            ]
        },
    }


class ProbabilityModelTests(ProjectTempMixin, unittest.TestCase):
    def test_brier_and_reliability_known_perfect_example(self):
        labels = np.array([0, 1, 2, 0])
        probabilities = np.eye(3)[labels]
        baseline = np.full((len(labels), 3), 1 / 3)
        metrics = evaluate_predictions(labels, probabilities, baseline)
        self.assertEqual(metrics["brier"], 0.0)
        self.assertAlmostEqual(metrics["climatology_brier"], 2 / 3)
        self.assertEqual(metrics["brier_skill"], 1.0)
        reliability = classwise_reliability(labels, probabilities)
        self.assertEqual(reliability["maximum_gap"], 0.0)
        self.assertTrue(
            all(
                reliability["classes"][name]["ece"] == 0
                for name in ("down", "middle", "up")
            )
        )

    def test_fold_local_preprocessor_is_unchanged_by_test_outlier(self):
        rng = np.random.default_rng(2)
        train = pd.DataFrame(
            rng.normal(size=(200, len(FEATURE_NAMES))), columns=FEATURE_NAMES
        )
        train.iloc[0, 3] = np.nan
        first = fit_preprocessor(train)
        outlier = train.copy()
        outlier.loc[len(outlier)] = 1e9
        second = fit_preprocessor(outlier.iloc[:-1])
        self.assertEqual(canonical_hash(first), canonical_hash(second))
        transformed = transform_preprocessor(train, first)
        self.assertEqual(
            transformed.shape[1],
            len(FEATURE_NAMES) + 1,
        )

    def test_temperature_uses_calibration_logits_only_and_is_deterministic(self):
        logits = np.array(
            [
                [4.0, 0.0, -1.0],
                [-1.0, 4.0, 0.0],
                [0.0, -1.0, 4.0],
            ]
            * 40
        )
        labels = np.array([0, 1, 2] * 40)
        first = fit_temperature(logits, labels)
        second = fit_temperature(logits, labels)
        self.assertEqual(first, second)
        self.assertEqual(first["fit_source"], "outer_fold_calibration_only")
        self.assertLessEqual(
            first["calibrated_log_loss"],
            first["uncalibrated_log_loss"] + 1e-12,
        )

    def test_model_and_numpy_artifact_predictions_are_deterministic(self):
        rng = np.random.default_rng(9)
        features = pd.DataFrame(
            rng.normal(size=(900, len(FEATURE_NAMES))), columns=FEATURE_NAMES
        )
        score = features.iloc[:, 0] - 0.6 * features.iloc[:, 1]
        labels = np.where(score < -0.4, 0, np.where(score > 0.4, 2, 1))
        first = fit_multinomial_model(features, labels)
        second = fit_multinomial_model(features, labels)
        self.assertEqual(first["model_hash"], second["model_hash"])
        np.testing.assert_allclose(
            predict_probabilities(first, features.iloc[:20]),
            predict_probabilities(second, features.iloc[:20]),
            rtol=0,
            atol=0,
        )

    def test_ood_quantiles_distance_and_missing_are_fail_closed(self):
        rng = np.random.default_rng(10)
        train = pd.DataFrame(
            rng.normal(size=(1000, len(FEATURE_NAMES))), columns=FEATURE_NAMES
        )
        preprocessor = fit_preprocessor(train)
        ordinary = assess_ood(train.iloc[[500]], preprocessor)[0]
        self.assertFalse(ordinary["withhold"])
        extreme = train.iloc[[500]].copy()
        extreme.iloc[0, :4] = 1e6
        result = assess_ood(extreme, preprocessor)[0]
        self.assertTrue(result["withhold"])
        self.assertGreater(result["outside_count"], 2)
        missing = train.iloc[[500]].copy()
        missing.iloc[0, 0] = np.nan
        self.assertTrue(assess_ood(missing, preprocessor)[0]["withhold"])

    def test_publish_transform_is_identity_and_metrics_use_identity_values(self):
        raw = np.array([[0.02, 0.48, 0.50], [0.40, 0.40, 0.20]])
        transformed = apply_publish_transform(raw, PUBLISH_TRANSFORM)
        np.testing.assert_array_equal(transformed, raw)
        labels = np.array([2, 0])
        baseline = np.full((2, 3), 1 / 3)
        raw_metrics = evaluate_predictions(labels, transformed, baseline)
        capped = np.array([[0.10, 0.40, 0.50], [0.40, 0.40, 0.20]])
        capped_metrics = evaluate_predictions(labels, capped, baseline)
        self.assertNotEqual(raw_metrics["brier"], capped_metrics["brier"])
        self.assertEqual(whole_identity_percentages(raw[0]), [2, 48, 50])

    def test_raw_grid_violation_withholds_without_projection(self):
        keys = ["A|2025-01-03", "B|2025-01-03"]
        reports = {
            10: {
                "_oos": {
                    "keys": keys,
                    "probabilities": np.array(
                        [[0.20, 0.30, 0.50], [0.30, 0.40, 0.30]]
                    ),
                },
                "acceptance": {"accepted": True, "reasons": [], "gates": {}},
            },
            20: {
                "_oos": {
                    "keys": keys,
                    "probabilities": np.array(
                        [[0.15, 0.25, 0.60], [0.20, 0.55, 0.25]]
                    ),
                },
                "acceptance": {"accepted": True, "reasons": [], "gates": {}},
            },
        }
        result = evaluate_raw_grid_monotonicity(reports)
        self.assertFalse(result["passed"])
        self.assertEqual(result["up_violation_count"], 1)
        apply_grid_release_gate(reports)
        self.assertFalse(reports[10]["acceptance"]["accepted"])
        np.testing.assert_array_equal(
            reports[20]["_oos"]["probabilities"][0],
            [0.15, 0.25, 0.60],
        )

    def test_raw_monotonic_grid_passes_unchanged(self):
        keys = ["A|2025-01-03"]
        reports = {}
        for threshold, row in (
            (10, [0.30, 0.30, 0.40]),
            (20, [0.20, 0.50, 0.30]),
            (30, [0.10, 0.70, 0.20]),
        ):
            reports[threshold] = {
                "_oos": {
                    "keys": keys,
                    "probabilities": np.asarray([row]),
                },
                "acceptance": {"accepted": True, "reasons": [], "gates": {}},
            }
        before = {
            key: value["_oos"]["probabilities"].copy()
            for key, value in reports.items()
        }
        result = apply_grid_release_gate(reports)
        self.assertTrue(result["passed"])
        for key in reports:
            np.testing.assert_array_equal(
                reports[key]["_oos"]["probabilities"], before[key]
            )
            self.assertTrue(reports[key]["acceptance"]["accepted"])

    def test_nested_small_oos_inversions_pass_but_large_frequent_fail(self):
        row_count = 6159
        keys = [f"I{index}|2025-01-03" for index in range(row_count)]
        easy = np.tile([0.30, 0.30, 0.40], (row_count, 1))
        hard = np.tile([0.20, 0.50, 0.30], (row_count, 1))
        hard[:31, 2] = easy[:31, 2] + 0.004
        hard[:31, 1] -= 0.104
        reports = {
            10: {"_oos": {"keys": keys, "probabilities": easy}},
            20: {"_oos": {"keys": keys, "probabilities": hard}},
        }
        small = evaluate_raw_grid_monotonicity(reports)
        self.assertTrue(small["passed"])
        self.assertEqual(small["up_violation_count"], 31)
        self.assertAlmostEqual(small["up_violation_rate"], 31 / 6159)
        self.assertLessEqual(small["p95_up_excess"], 0.01)

        bad_hard = np.tile([0.20, 0.50, 0.30], (row_count, 1))
        bad_count = int(np.ceil(row_count * 0.05))
        bad_hard[:bad_count, 2] = easy[:bad_count, 2] + 0.05
        bad_hard[:bad_count, 1] -= 0.15
        bad = evaluate_raw_grid_monotonicity(
            {
                10: {"_oos": {"keys": keys, "probabilities": easy}},
                20: {"_oos": {"keys": keys, "probabilities": bad_hard}},
            }
        )
        self.assertFalse(bad["passed"])
        self.assertGreater(bad["up_violation_rate"], 0.01)
        self.assertGreater(bad["max_up_excess"], 0.03)

    def test_every_acceptance_gate_mutation_withholds(self):
        self.assertTrue(evaluate_acceptance(passing_report())["accepted"])
        mutations = {
            "history": lambda report: report.update(history_years=7.99),
            "folds": lambda report: report.update(fold_count=4),
            "full_test_folds": lambda report: report.update(
                full_test_fold_count=4
            ),
            "usable_train": lambda report: report.update(
                min_usable_train_years=4.99
            ),
            "issuers": lambda report: report.update(issuer_count=199),
            "dates": lambda report: report.update(forecast_date_count=99),
            "train": lambda report: report.update(min_train_class_count=999),
            "calibration": lambda report: report.update(
                min_calibration_class_count=299
            ),
            "test": lambda report: report.update(min_test_class_count=199),
            "coverage": lambda report: report.update(inference_coverage=0.799),
            "transform_mismatch": lambda report: report.update(
                publish_transform_identity_verified=False
            ),
            "provider_coverage": lambda report: report.update(
                provider_success_coverage=0.799
            ),
            "provider_issuers": lambda report: report.update(
                provider_successful_issuer_count=199
            ),
            "brier": lambda report: report["aggregate"].update(brier_skill=0.019),
            "bootstrap": lambda report: report["bootstrap"].update(
                brier_skill_ci95=[0.0, 0.1]
            ),
            "logloss": lambda report: report["aggregate"].update(
                log_loss_improvement=0.009
            ),
            "ece": lambda report: report["aggregate"]["classwise_ece"].update(
                down=0.031
            ),
            "gap": lambda report: report["aggregate"].update(maximum_gap=0.081),
            "slope": lambda report: report["aggregate"]["calibration"]["up"].update(
                slope=1.21
            ),
            "intercept": lambda report: report["aggregate"]["calibration"][
                "middle"
            ].update(intercept=0.11),
            "negative_folds": lambda report: [
                report["folds"][index]["metrics"].update(brier_skill=-0.021)
                for index in (1, 2)
            ],
            "regime": lambda report: report["regime"]["groups"][0].update(
                max_class_ece=0.051
            ),
        }
        for label, mutate in mutations.items():
            with self.subTest(label=label):
                report = passing_report()
                mutate(report)
                result = evaluate_acceptance(report)
                self.assertFalse(result["accepted"])
                self.assertTrue(result["reasons"])

    def test_learnable_synthetic_passes_and_random_withholds(self):
        result = run_smoke(bootstrap_repetitions=50)
        self.assertTrue(result["learnable"]["accepted"])
        self.assertFalse(result["random"]["accepted"])
        self.assertTrue(result["random"]["reasons"])

    def test_release_cannot_reduce_bootstrap_repetitions(self):
        with self.assertRaisesRegex(ValueError, "at least 1000"):
            train_dataset(
                pd.DataFrame(),
                bootstrap_repetitions=200,
                release=True,
            )

    def test_fixed_oos_bootstrap_retries_and_release_requires_completion(self):
        labels = np.tile(np.arange(3), 2)
        probabilities = np.full((len(labels), 3), 0.20)
        probabilities[np.arange(len(labels)), labels] = 0.60
        baseline = np.full((len(labels), 3), 1 / 3)
        dates = [
            "2024-01-15",
            "2024-01-15",
            "2024-01-15",
            "2024-04-15",
            "2024-04-15",
            "2024-04-15",
        ]
        issuers = ["A", "A", "A", "B", "B", "B"]
        first = two_way_cluster_bootstrap(
            labels,
            probabilities,
            baseline,
            dates,
            issuers,
            repetitions=25,
            seed=11,
            max_attempts=100,
        )
        second = two_way_cluster_bootstrap(
            labels,
            probabilities,
            baseline,
            dates,
            issuers,
            repetitions=25,
            seed=11,
            max_attempts=100,
        )
        self.assertEqual(first, second)
        self.assertEqual(first["requested_repetitions"], 25)
        self.assertEqual(first["completed_repetitions"], 25)
        self.assertGreater(first["attempted_repetitions"], 25)
        self.assertGreater(first["skipped_repetitions"], 0)
        self.assertEqual(
            first["skipped_repetitions"],
            first["attempted_repetitions"]
            - first["completed_repetitions"],
        )
        self.assertTrue(first["complete"])

        incomplete = two_way_cluster_bootstrap(
            labels,
            probabilities,
            baseline,
            dates,
            issuers,
            repetitions=10,
            seed=11,
            max_attempts=1,
        )
        self.assertFalse(incomplete["complete"])
        self.assertLess(incomplete["completed_repetitions"], 10)
        report = passing_report()
        report["bootstrap"] = incomplete
        report["fixed_oos_bootstrap_required_repetitions"] = 10
        acceptance = evaluate_acceptance(report)
        self.assertFalse(acceptance["accepted"])
        self.assertTrue(
            any(
                "fixed OOS bootstrap incomplete" in reason
                for reason in acceptance["reasons"]
            )
        )

    def test_checkpoint_key_binds_dataframe_manifest_and_hyperparameters(self):
        dataset = make_synthetic_dataset(
            learnable=True, issuer_count=3, week_count=30
        )
        first = build_dataset_binding(
            dataset,
            {"panel_manifest_sha256": "panel-a"},
        )
        changed = dataset.copy()
        changed.loc[0, FEATURE_NAMES[-1]] += 0.01
        second = build_dataset_binding(
            changed,
            {"panel_manifest_sha256": "panel-a"},
        )
        third = build_dataset_binding(
            dataset,
            {"panel_manifest_sha256": "panel-b"},
        )
        self.assertNotEqual(first["binding_hash"], second["binding_hash"])
        self.assertNotEqual(first["binding_hash"], third["binding_hash"])
        provider_limited = build_dataset_binding(
            dataset,
            {
                "panel_manifest_sha256": "panel-a",
                "provider_requested_issuer_count": 250,
                "provider_successful_issuer_count": 190,
                "provider_unavailable_symbols": {
                    "BAD": "provider timeout"
                },
            },
        )
        self.assertEqual(
            provider_limited["provider"]["success_coverage"], 0.76
        )
        self.assertEqual(
            provider_limited["provider"]["unavailable_symbols"]["BAD"],
            "provider timeout",
        )
        base_key = checkpoint_key(
            first,
            horizon=21,
            thresholds=[3, 5, 10],
            bootstrap_repetitions=200,
            seed=1729,
            c_value=0.1,
        )
        self.assertNotEqual(
            base_key,
            checkpoint_key(
                first,
                horizon=21,
                thresholds=[3, 5, 10],
                bootstrap_repetitions=201,
                seed=1729,
                c_value=0.1,
            ),
        )
        path = self.work / "validation.json"
        cached = _load_or_new_validation(
            path,
            now="2026-08-14T08:00:00+00:00",
            resume=False,
            binding=first,
        )
        cached["models"]["sentinel"] = {"cached": True}
        atomic_write_json(path, cached)
        reused = _load_or_new_validation(
            path,
            now="2026-08-14T08:00:00+00:00",
            resume=True,
            binding=first,
        )
        self.assertIn("sentinel", reused["models"])
        unrelated = _load_or_new_validation(
            path,
            now="2026-08-14T08:00:00+00:00",
            resume=True,
            binding=second,
        )
        self.assertNotIn("sentinel", unrelated["models"])


if __name__ == "__main__":
    unittest.main()
