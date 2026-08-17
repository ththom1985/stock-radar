from __future__ import annotations

import copy
import json
import subprocess
import sys
import unittest
from unittest.mock import patch

import numpy as np
import pandas as pd

from src.probability_contract import (
    CLASS_NAMES,
    ORDERED_MODEL_FAMILY,
    THRESHOLD_GRIDS,
    model_key,
    ordered_model_key,
)
from src.probability_dataset import classify_ordered_move
from src.probability_features import FEATURE_NAMES, build_probability_features
from src.probability_inference import (
    ProbabilityArtifactError,
    empty_probability_artifact,
    finalize_artifact,
    score_probability_row,
    validate_probability_artifact,
)
from src.probability_model import PUBLISH_TRANSFORM, canonical_hash, stable_softmax
from src.probability_ordered import (
    ADAPTIVE_MIN_BIN_ROWS,
    ADAPTIVE_MIN_NEGATIVES,
    ADAPTIVE_MIN_POSITIVES,
    ORDERED_MONOTONICITY_EPSILON,
    ORDERED_PUBLISH_TRANSFORM,
    VECTOR_CALIBRATION_VERSION,
    VECTOR_PENALTY_GRID,
    adaptive_classwise_reliability,
    apply_vector_scaling,
    assert_exact_ordered_monotonicity,
    derive_threshold_probabilities,
    fit_ordered_multinomial_model,
    fit_ordered_vector_calibration,
    fit_vector_scaling,
    full_refit_ordered_bootstrap,
    predict_ordered_probabilities,
    regime_support,
)
from src.probability_train import (
    _publish_ordered_completion,
    checkpoint_key,
    make_synthetic_dataset,
    run_ordered_smoke,
    train_dataset,
)
from tests.helpers import ROOT, ProjectTempMixin
from tests.test_probability_inference import NOW, current_history, row_for


def ordered_artifact():
    history = current_history()
    spy = current_history(seed=32)
    feature_history = build_probability_features(history, spy).dropna()
    as_of = history.index[-20]
    center = feature_history.loc[as_of].to_numpy(dtype=float)
    rng = np.random.default_rng(71)
    scale = np.maximum(np.abs(center) * 0.08, 0.01)
    matrix = center + rng.normal(size=(2100, len(center))) * scale
    latent = (
        (matrix[:, 0] - center[0]) / scale[0]
        - 0.6 * (matrix[:, 1] - center[1]) / scale[1]
    )
    edges = np.quantile(latent, np.linspace(0, 1, 8)[1:-1])
    labels = np.searchsorted(edges, latent, side="right")
    model = fit_ordered_multinomial_model(
        pd.DataFrame(matrix, columns=FEATURE_NAMES),
        labels,
    )
    model["vector_scaling"] = {
        "version": VECTOR_CALIBRATION_VERSION,
        "formula": "softmax(exp(s_j) * z_j + b_j)",
        "penalty_grid": list(VECTOR_PENALTY_GRID),
        "selected_penalty": 0.1,
        "log_scales": [0.0] * 7,
        "scales": [1.0] * 7,
        "biases": [0.0] * 7,
        "converged": True,
    }
    thresholds = THRESHOLD_GRIDS[21]
    exact = assert_exact_ordered_monotonicity(
        derive_threshold_probabilities(np.full(7, 1 / 7), thresholds)
    )
    offsets = {
        name: [-0.02, 0.02]
        for name in CLASS_NAMES
    }
    rates = {
        str(threshold): {
            "down": 0.30,
            "middle": 0.40,
            "up": 0.30,
        }
        for threshold in thresholds
    }
    metrics = {
        str(threshold): {
            "brier_skill": 0.09,
            "log_loss_improvement": 0.08,
            "classwise_ece": {
                name: 0.02 for name in CLASS_NAMES
            },
            "maximum_gap": 0.05,
        }
        for threshold in thresholds
    }
    model.update(
        {
            "model_key": ordered_model_key(21),
            "horizon_sessions": 21,
            "thresholds_pct": list(thresholds),
            "round_trip_cost_bps": 30,
            "publish_transform": dict(ORDERED_PUBLISH_TRANSFORM),
            "exact_monotonicity": exact,
            "trained_at": "2026-08-12T18:00:00+00:00",
            "training_cutoff": "2026-07-15",
            "history_years": 12.0,
            "full_test_fold_count": 6,
            "min_usable_train_years": 5.2,
            "fold_count": 6,
            "oos_sample_size": 12500,
            "baseline_rates_by_threshold": rates,
            "oos_metrics_by_threshold": metrics,
            "bootstrap_by_threshold": {
                str(threshold): {
                    "probability_error_offsets_ci95": offsets,
                    "requested_repetitions": 1000,
                    "attempted_repetitions": 1000,
                    "completed_repetitions": 1000,
                    "skipped_repetitions": 0,
                    "complete": True,
                }
                for threshold in thresholds
            },
            "refit_acceptance": {
                "dev_override": False,
                "completed_repetitions": 200,
                "production_release_eligible": True,
            },
            "full_refit_bootstrap": {
                "requested_repetitions": 200,
                "completed_repetitions": 200,
                "method": "full model+calibrator refit",
            },
            "accepted": True,
            "acceptance_reasons": [],
            "checkpoint_key": "ordered-checkpoint-test",
        }
    )
    binding = {"dataset_content_hash": "b" * 64}
    binding["binding_hash"] = canonical_hash(binding)
    model["dataset_binding_hash"] = binding["binding_hash"]
    model.pop("model_hash", None)
    model["model_hash"] = canonical_hash(model)
    artifact = empty_probability_artifact(
        created_at="2026-08-12T18:00:00+00:00",
        model_family=ORDERED_MODEL_FAMILY,
    )
    artifact.update(
        {
            "engine_version": "probability-ordered-vector-v1",
            "model_family": ORDERED_MODEL_FAMILY,
            "training_cutoff": "2026-07-15",
            "production_status": "accepted_partial_grid",
            "production_reasons": [],
            "dataset_binding": binding,
            "models": {ordered_model_key(21): model},
            "accepted_model_keys": [ordered_model_key(21)],
            "baselines": {
                model_key(21, threshold): {
                    "model_family": ORDERED_MODEL_FAMILY,
                    "source_model_key": ordered_model_key(21),
                    "model_key": model_key(21, threshold),
                    "horizon_sessions": 21,
                    "threshold_pct": threshold,
                    "rates": rates[str(threshold)],
                    "sample_size": 12500,
                    "validation": {
                        "accepted": True,
                        "reasons": [],
                        **metrics[str(threshold)],
                        "fold_count": 6,
                    },
                }
                for threshold in thresholds
            },
        }
    )
    return finalize_artifact(artifact), history, spy, as_of


class OrderedProbabilityTests(ProjectTempMixin, unittest.TestCase):
    def test_all_seven_boundary_equalities_and_adjacent_values(self):
        thresholds = (0.03, 0.05, 0.10)
        cost = 0.003
        equalities = np.asarray(
            [
                -(thresholds[2] + cost),
                -(thresholds[1] + cost),
                -(thresholds[0] + cost),
                0.0,
                thresholds[0] + cost,
                thresholds[1] + cost,
                thresholds[2] + cost,
            ]
        )
        np.testing.assert_array_equal(
            classify_ordered_move(equalities, thresholds),
            np.arange(7),
        )
        for boundary in equalities[[0, 1, 2, 4, 5, 6]]:
            below = np.nextafter(boundary, -np.inf)
            above = np.nextafter(boundary, np.inf)
            labels = classify_ordered_move(
                np.asarray([below, boundary, above]),
                thresholds,
            )
            self.assertLessEqual(labels[0], labels[1])
            self.assertLessEqual(labels[1], labels[2])

    def test_tail_sums_are_exact_simplexes_and_monotonic(self):
        ordered = np.asarray([0.05, 0.10, 0.15, 0.20, 0.15, 0.10, 0.25])
        derived = derive_threshold_probabilities(ordered, (3, 5, 10))
        np.testing.assert_array_equal(
            derived[3],
            [
                ordered[:3].sum(),
                ordered[3:4].sum(),
                ordered[4:].sum(),
            ],
        )
        np.testing.assert_array_equal(
            derived[5],
            [
                ordered[:2].sum(),
                ordered[2:5].sum(),
                ordered[5:].sum(),
            ],
        )
        np.testing.assert_array_equal(
            derived[10],
            [
                ordered[:1].sum(),
                ordered[1:6].sum(),
                ordered[6:].sum(),
            ],
        )
        diagnostic = assert_exact_ordered_monotonicity(derived)
        self.assertTrue(diagnostic["passed"])
        self.assertEqual(diagnostic["tolerance"], ORDERED_MONOTONICITY_EPSILON)
        self.assertEqual(diagnostic["up_violation_count"], 0)
        self.assertEqual(diagnostic["down_violation_count"], 0)

    def test_vector_scaling_identity_and_class_bias_correction(self):
        logits = np.asarray([[2.0, 1.0, 0.0, -1.0, -2.0, -3.0, -4.0]])
        identity = {
            "version": VECTOR_CALIBRATION_VERSION,
            "scales": [1.0] * 7,
            "biases": [0.0] * 7,
        }
        np.testing.assert_array_equal(
            apply_vector_scaling(logits, identity),
            stable_softmax(logits),
        )

        labels = np.tile(np.asarray([0, 0, 0, 1, 2, 3, 4, 5, 6]), 120)
        zero_logits = np.zeros((len(labels), 7))
        dates = pd.date_range("2025-01-01", "2025-12-31", periods=len(labels))
        calibrator = fit_vector_scaling(
            zero_logits,
            labels,
            dates,
            penalty_grid=(0.01,),
        )
        corrected = apply_vector_scaling(
            np.zeros((1, 7)),
            calibrator,
        )[0]
        self.assertGreater(corrected[0], corrected[1])
        self.assertAlmostEqual(sum(calibrator["biases"]), 0.0, places=12)

    def test_vector_penalty_selection_is_nine_three_and_test_free(self):
        rng = np.random.default_rng(19)
        labels = np.tile(np.arange(7), 120)
        logits = rng.normal(size=(len(labels), 7))
        dates = pd.date_range("2025-01-01", "2025-12-31", periods=len(labels))
        first = fit_vector_scaling(logits, labels, dates)
        second = fit_vector_scaling(logits, labels, dates)
        self.assertEqual(canonical_hash(first), canonical_hash(second))
        self.assertEqual(first["penalty_grid"], list(VECTOR_PENALTY_GRID))
        self.assertLess(
            first["penalty_fit_end"],
            first["penalty_selection_start"],
        )
        self.assertEqual(
            first["penalty_fit_rows"] + first["penalty_selection_rows"],
            len(labels),
        )
        untouched_test_start = pd.Timestamp("2026-01-01")
        self.assertLess(
            pd.Timestamp(first["calibration_interval_end"]),
            untouched_test_start,
        )

    def test_shared_calibration_path_is_deterministic_and_used_by_refits(self):
        dataset = make_synthetic_dataset(
            learnable=True,
            issuer_count=20,
            week_count=120,
            model_family=ORDERED_MODEL_FAMILY,
        )
        target = "ordered_label_h21"
        train = dataset.iloc[: 20 * 50]
        calibration = dataset.iloc[20 * 50 : 20 * 102].copy()
        calibration.iloc[0, calibration.columns.get_indexer(FEATURE_NAMES[:4])] = (
            1e6
        )
        base = fit_ordered_multinomial_model(
            train.loc[:, FEATURE_NAMES],
            train[target].to_numpy(dtype=int),
        )
        interval_start = calibration["feature_date"].min()
        first = fit_ordered_vector_calibration(
            copy.deepcopy(base),
            calibration,
            target=target,
            calibration_interval_start=interval_start,
        )
        second = fit_ordered_vector_calibration(
            copy.deepcopy(base),
            calibration,
            target=target,
            calibration_interval_start=interval_start,
        )
        np.testing.assert_array_equal(
            first["coverage_mask"],
            second["coverage_mask"],
        )
        self.assertFalse(first["coverage_mask"][0])
        self.assertEqual(
            canonical_hash(first["vector_scaling"]),
            canonical_hash(second["vector_scaling"]),
        )
        self.assertEqual(
            first["model"]["model_hash"],
            second["model"]["model_hash"],
        )

        fold = {
            "fold": 0,
            "train_indices": np.arange(0, 20 * 50),
            "calibration_indices": np.arange(20 * 50, 20 * 102),
            "test_indices": np.arange(20 * 102, len(dataset)),
        }
        with patch(
            "src.probability_ordered.fit_ordered_vector_calibration",
            side_effect=RuntimeError("shared-calibration-sentinel"),
        ):
            refit = full_refit_ordered_bootstrap(
                dataset,
                [fold],
                horizon=21,
                repetitions=1,
            )
        self.assertEqual(refit["completed_repetitions"], 0)
        self.assertIn(
            "shared-calibration-sentinel",
            refit["failures"][0]["reason"],
        )

    def test_adaptive_bins_enforce_support_and_supported_gap_only(self):
        count = 12000
        x = np.linspace(0.0, 1.0, count)
        probabilities = np.column_stack(
            [0.20 + 0.20 * x, 0.50 - 0.20 * x, np.full(count, 0.30)]
        )
        labels = np.arange(count) % 3
        reliability = adaptive_classwise_reliability(labels, probabilities)
        self.assertTrue(reliability["all_classes_supported"])
        supported_gaps = []
        for name in CLASS_NAMES:
            detail = reliability["classes"][name]
            self.assertGreaterEqual(detail["supported_bin_count"], 5)
            for row in detail["bins"]:
                if not row["supported"]:
                    continue
                self.assertGreaterEqual(row["count"], ADAPTIVE_MIN_BIN_ROWS)
                self.assertGreaterEqual(
                    row["positive_count"],
                    ADAPTIVE_MIN_POSITIVES,
                )
                self.assertGreaterEqual(
                    row["negative_count"],
                    ADAPTIVE_MIN_NEGATIVES,
                )
                supported_gaps.append(row["gap"])
        self.assertEqual(reliability["maximum_gap"], max(supported_gaps))

        unsupported = adaptive_classwise_reliability(
            labels[:2000],
            probabilities[:2000],
        )
        self.assertFalse(unsupported["all_classes_supported"])
        tails = unsupported["classes"]["down"][
            "unsupported_extreme_tails"
        ]
        self.assertTrue(tails)
        self.assertTrue(
            all(row["observed_rate_wilson95"] for row in tails)
        )

    def test_regime_support_rules_are_exact_and_unavailable_is_not_pass(self):
        dates = np.tile(
            pd.date_range("2022-01-31", periods=32, freq="ME"),
            12,
        )
        labels = np.arange(len(dates)) % 3
        issuers = np.asarray(
            [f"I{index % 110:03d}" for index in range(len(dates))]
        )
        supported = regime_support(labels, dates, issuers)
        self.assertTrue(supported["available"])
        self.assertGreaterEqual(supported["distinct_dates"], 26)
        self.assertGreaterEqual(supported["quarter_blocks"], 8)
        self.assertGreaterEqual(min(supported["class_counts"].values()), 100)
        self.assertGreaterEqual(supported["issuer_count"], 100)

        unavailable = regime_support(
            labels[:180],
            dates[:180],
            issuers[:180],
        )
        self.assertFalse(unavailable["available"])
        self.assertEqual(unavailable["status"], "unavailable")
        self.assertTrue(unavailable["unavailable_reasons"])

    def test_ordered_numpy_artifact_is_deterministic_without_training_imports(self):
        rng = np.random.default_rng(20)
        features = pd.DataFrame(
            rng.normal(size=(1400, len(FEATURE_NAMES))),
            columns=FEATURE_NAMES,
        )
        labels = np.tile(np.arange(7), 200)
        model = fit_ordered_multinomial_model(features, labels)
        model["vector_scaling"] = {
            "version": VECTOR_CALIBRATION_VERSION,
            "scales": [1.0] * 7,
            "biases": [0.0] * 7,
        }
        model.pop("model_hash", None)
        model["model_hash"] = canonical_hash(model)
        restored = json.loads(json.dumps(model))
        first = predict_ordered_probabilities(model, features.iloc[:20])
        second = predict_ordered_probabilities(restored, features.iloc[:20])
        np.testing.assert_array_equal(first, second)
        subprocess.run(
            [
                sys.executable,
                "-c",
                (
                    "import sys; import src.probability_ordered; "
                    "assert 'sklearn' not in sys.modules; "
                    "assert 'scipy' not in sys.modules"
                ),
            ],
            cwd=str(ROOT),
            check=True,
            capture_output=True,
            text=True,
        )

    def test_ordered_inference_derives_grid_and_horizon_fails_atomically(self):
        artifact, history, spy, as_of = ordered_artifact()
        input_row = row_for(as_of)
        before = copy.deepcopy(input_row)
        forecast = score_probability_row(
            input_row,
            history,
            spy,
            artifact,
            now=NOW,
        )
        self.assertEqual(input_row, before)
        self.assertEqual(input_row["radar_score"], before["radar_score"])
        self.assertEqual(input_row["sweet_spot"], before["sweet_spot"])
        self.assertEqual(len(forecast["forecasts"]), 3)
        self.assertTrue(
            all(
                item["model_family"] == ORDERED_MODEL_FAMILY
                for item in forecast["forecasts"]
            )
        )
        self.assertTrue(
            all(
                item["fixed_oos_bootstrap"]["completed"] == 1000
                and item["fixed_oos_bootstrap"]["requested"] == 1000
                and item["fixed_oos_bootstrap"]["skipped"] == 0
                for item in forecast["forecasts"]
            )
        )
        frame = pd.DataFrame(
            [
                build_probability_features(history, spy)
                .loc[as_of, FEATURE_NAMES]
                .to_dict()
            ],
            columns=FEATURE_NAMES,
        )
        ordered = predict_ordered_probabilities(
            artifact["models"][ordered_model_key(21)],
            frame,
            require_complete=True,
        )[0]
        expected = derive_threshold_probabilities(
            ordered,
            THRESHOLD_GRIDS[21],
        )
        for item in forecast["forecasts"]:
            np.testing.assert_array_equal(
                [
                    item["probabilities"][name]
                    for name in CLASS_NAMES
                ],
                expected[item["threshold_pct"]],
            )

        broken = copy.deepcopy(artifact)
        model = broken["models"][ordered_model_key(21)]
        model["bootstrap_by_threshold"]["5"][
            "probability_error_offsets_ci95"
        ]["up"] = [-0.2, 0.2]
        model.pop("model_hash")
        model["model_hash"] = canonical_hash(model)
        broken = finalize_artifact(broken)
        withheld = score_probability_row(
            row_for(as_of),
            history,
            spy,
            broken,
            now=NOW,
        )
        self.assertEqual(withheld["status"], "withheld")
        self.assertFalse(withheld["forecasts"])
        self.assertTrue(withheld["baselines"])

    def test_ordered_artifact_top_level_family_transform_and_hash_are_strict(self):
        artifact, _history, _spy, _as_of = ordered_artifact()
        self.assertEqual(artifact["model_family"], ORDERED_MODEL_FAMILY)
        self.assertEqual(
            artifact["publish_transform"],
            ORDERED_PUBLISH_TRANSFORM,
        )
        validate_probability_artifact(artifact)

        stale_hash = copy.deepcopy(artifact)
        stale_hash["publish_transform"] = dict(PUBLISH_TRANSFORM)
        with self.assertRaisesRegex(
            ProbabilityArtifactError,
            "artifact hash mismatch",
        ):
            validate_probability_artifact(stale_hash)

        inconsistent = copy.deepcopy(stale_hash)
        inconsistent = finalize_artifact(inconsistent)
        with self.assertRaisesRegex(
            ProbabilityArtifactError,
            "publish transform mismatch",
        ):
            validate_probability_artifact(inconsistent)

        incomplete = copy.deepcopy(artifact)
        model = incomplete["models"][ordered_model_key(21)]
        model["bootstrap_by_threshold"]["3"][
            "completed_repetitions"
        ] = 999
        model["bootstrap_by_threshold"]["3"]["complete"] = False
        model.pop("model_hash")
        model["model_hash"] = canonical_hash(model)
        incomplete = finalize_artifact(incomplete)
        with self.assertRaisesRegex(
            ProbabilityArtifactError,
            "ordered model",
        ):
            validate_probability_artifact(incomplete)

    def test_release_requires_preregistered_full_refits(self):
        with self.assertRaisesRegex(ValueError, "200 full model"):
            train_dataset(
                pd.DataFrame(),
                release=True,
                model_family=ORDERED_MODEL_FAMILY,
                bootstrap_repetitions=1000,
                refit_bootstrap_repetitions=199,
            )

    def test_completed_ordered_run_updates_canonical_and_experiment_copies(self):
        canonical_validation = self.work / "probability_validation.json"
        canonical_models = self.work / "probability_models.json"
        experiment_validation = (
            self.work
            / "probability_experiments"
            / "ordered-vector-v1_validation.json"
        )
        experiment_models = (
            self.work
            / "probability_experiments"
            / "ordered-vector-v1_models.json"
        )
        validation = {
            "model_family": ORDERED_MODEL_FAMILY,
            "publish_transform": dict(ORDERED_PUBLISH_TRANSFORM),
            "status": "no_model_passed",
            "production_status": "withheld",
        }
        artifact = {
            "model_family": ORDERED_MODEL_FAMILY,
            "publish_transform": dict(ORDERED_PUBLISH_TRANSFORM),
            "production_status": "withheld",
            "accepted_model_keys": [],
        }
        _publish_ordered_completion(
            validation,
            artifact,
            validation_path=canonical_validation,
            models_path=canonical_models,
            experiment_validation_path=experiment_validation,
            experiment_models_path=experiment_models,
        )
        for path, expected in (
            (canonical_validation, validation),
            (experiment_validation, validation),
            (canonical_models, artifact),
            (experiment_models, artifact),
        ):
            self.assertEqual(
                json.loads(path.read_text(encoding="utf-8")),
                expected,
            )

        accepted_validation = {
            **validation,
            "status": "accepted_partial_grid",
            "production_status": "accepted_partial_grid",
        }
        accepted_artifact = {
            **artifact,
            "production_status": "accepted_partial_grid",
            "accepted_model_keys": ["h21_ordered"],
        }
        _publish_ordered_completion(
            accepted_validation,
            accepted_artifact,
            validation_path=canonical_validation,
            models_path=canonical_models,
            experiment_validation_path=experiment_validation,
            experiment_models_path=experiment_models,
        )
        self.assertEqual(
            json.loads(
                canonical_validation.read_text(encoding="utf-8")
            )["status"],
            "accepted_partial_grid",
        )
        self.assertEqual(
            json.loads(canonical_models.read_text(encoding="utf-8"))[
                "accepted_model_keys"
            ],
            ["h21_ordered"],
        )

    def test_failed_release_runtime_publishes_canonical_withheld_state(self):
        canonical_validation = self.work / "runtime_validation.json"
        canonical_models = self.work / "runtime_models.json"
        experiment_validation = self.work / "runtime_experiment_validation.json"
        experiment_models = self.work / "runtime_experiment_models.json"
        validation, artifact = train_dataset(
            pd.DataFrame(),
            validation_path=canonical_validation,
            models_path=canonical_models,
            experiment_validation_path=experiment_validation,
            experiment_models_path=experiment_models,
            release=True,
            resume=False,
            model_family=ORDERED_MODEL_FAMILY,
            bootstrap_repetitions=1000,
            refit_bootstrap_repetitions=200,
        )
        self.assertEqual(validation["status"], "no_model_passed")
        self.assertEqual(validation["model_family"], ORDERED_MODEL_FAMILY)
        self.assertEqual(artifact["production_status"], "withheld")
        self.assertEqual(artifact["model_family"], ORDERED_MODEL_FAMILY)
        self.assertEqual(
            artifact["publish_transform"],
            ORDERED_PUBLISH_TRANSFORM,
        )
        for path in (
            canonical_validation,
            canonical_models,
            experiment_validation,
            experiment_models,
        ):
            self.assertTrue(path.exists(), path)

    def test_learnable_ordered_smoke_passes_dev_override_random_withholds(self):
        result = run_ordered_smoke(
            bootstrap_repetitions=20,
            refit_bootstrap_repetitions=1,
            dev_refit_override=True,
        )
        self.assertTrue(result["learnable"]["accepted"])
        self.assertFalse(
            result["learnable"]["production_release_eligible"]
        )
        self.assertTrue(result["learnable"]["exact_monotonicity"])
        self.assertFalse(result["random"]["accepted"])

    def test_checkpoint_binds_family_grid_and_refit_count(self):
        dataset = make_synthetic_dataset(
            learnable=True,
            issuer_count=3,
            week_count=30,
        )
        from src.probability_train import build_dataset_binding

        binding = build_dataset_binding(dataset)
        independent = checkpoint_key(
            binding,
            horizon=21,
            thresholds=[3, 5, 10],
            bootstrap_repetitions=20,
            seed=1729,
            c_value=0.1,
        )
        ordered = checkpoint_key(
            binding,
            horizon=21,
            thresholds=[3, 5, 10],
            bootstrap_repetitions=20,
            seed=1729,
            c_value=0.1,
            model_family=ORDERED_MODEL_FAMILY,
            refit_bootstrap_repetitions=1,
        )
        ordered_more_refits = checkpoint_key(
            binding,
            horizon=21,
            thresholds=[3, 5, 10],
            bootstrap_repetitions=20,
            seed=1729,
            c_value=0.1,
            model_family=ORDERED_MODEL_FAMILY,
            refit_bootstrap_repetitions=2,
        )
        self.assertNotEqual(independent, ordered)
        self.assertNotEqual(ordered, ordered_more_refits)

    def test_ui_displays_model_family_without_ranking_coupling(self):
        dashboard = (ROOT / "dashboard" / "app.py").read_text(
            encoding="utf-8"
        )
        static = (ROOT / "docs" / "index.html").read_text(
            encoding="utf-8"
        )
        self.assertIn("Modellfamilie", dashboard)
        self.assertIn("Modellfamilie", static)
        self.assertIn("OOS-Bootstrap", dashboard)
        self.assertIn("OOS-Bootstrap", static)
        ordered_source = (
            ROOT / "src" / "probability_ordered.py"
        ).read_text(encoding="utf-8")
        self.assertNotIn("radar_score", ordered_source)
        self.assertNotIn("sweet_spot", ordered_source)


if __name__ == "__main__":
    unittest.main()
