from __future__ import annotations

import unittest
from datetime import datetime, timezone
import hashlib
import subprocess
import sys

import numpy as np
import pandas as pd

from src.probability_dataset import (
    DATASET_FILENAME,
    DATASET_SCHEMA_VERSION,
    HORIZONS,
    DEFAULT_START,
    ROUND_TRIP_COST,
    build_symbol_dataset,
    classify_material_move,
    dataset_content_hash,
    download_probability_panel,
    load_probability_panel,
    load_weekly_dataset,
    read_exact_dataset_cache,
    make_purged_expanding_folds,
    select_eligible_universe,
    write_exact_dataset_cache,
)
from src.persistence import atomic_write_json
from src.probability_features import (
    FEATURE_NAMES,
    FEATURE_VERSION,
    INTERACTION_FEATURES,
    PROBABILITY_CODE_FILES,
    SPY_FEATURES,
    build_probability_features,
    feature_schema_hash,
    probability_code_hash,
)
from src.probability_train import make_synthetic_dataset
from src.probability_contract import ordered_label_column
from tests.helpers import ROOT, ProjectTempMixin


def history_frame(periods=720, seed=7):
    index = pd.bdate_range("2022-01-03", periods=periods)
    rng = np.random.default_rng(seed)
    close = 80.0 * np.exp(np.cumsum(rng.normal(0.0004, 0.012, periods)))
    open_price = close * np.exp(rng.normal(0.0, 0.002, periods))
    return pd.DataFrame(
        {
            "Open": open_price,
            "High": np.maximum(open_price, close) * 1.01,
            "Low": np.minimum(open_price, close) * 0.99,
            "Close": close,
            "RawClose": close,
            "Volume": rng.integers(500_000, 4_000_000, periods).astype(float),
            "Dividends": 0.0,
            "Stock Splits": 0.0,
        },
        index=index,
    )


class ProbabilityFeatureDatasetTests(ProjectTempMixin, unittest.TestCase):
    def test_stable_complete_feature_schema(self):
        history = history_frame()
        features = build_probability_features(history, history)
        self.assertEqual(tuple(features.columns), FEATURE_NAMES)
        self.assertEqual(features.iloc[-1].isna().sum(), 0)
        self.assertEqual(len(feature_schema_hash()), 64)

    def test_flat_history_has_neutral_rsi(self):
        history = history_frame()
        for column in ("Open", "High", "Low", "Close", "RawClose"):
            history[column] = 100.0
        features = build_probability_features(history, history)
        self.assertEqual(features.iloc[-1]["rsi14"], 50.0)

    def test_appending_future_data_and_actions_cannot_change_t(self):
        history = history_frame()
        cutoff = history.index[500]
        before = build_probability_features(
            history.loc[:cutoff], history.loc[:cutoff]
        ).loc[cutoff]

        appended = history.copy()
        appended.loc[appended.index[-1], "Stock Splits"] = 2.0
        after_append = build_probability_features(appended, appended).loc[cutoff]
        np.testing.assert_allclose(
            before.to_numpy(),
            after_append.to_numpy(),
            rtol=1e-12,
            atol=1e-12,
        )

        backscaled = appended.copy()
        price_columns = ["Open", "High", "Low", "Close"]
        backscaled.loc[:cutoff, price_columns] *= 0.25
        after_future_scale = build_probability_features(
            backscaled, backscaled
        ).loc[cutoff]
        np.testing.assert_allclose(
            before.to_numpy(),
            after_future_scale.to_numpy(),
            rtol=1e-10,
            atol=1e-10,
        )

    def test_market_alignment_never_uses_future_spy_session(self):
        stock = history_frame()
        spy = history_frame(seed=22).drop(stock.index[450])
        features = build_probability_features(stock, spy)
        prior = build_probability_features(
            stock.loc[: stock.index[449]], spy.loc[: stock.index[449]]
        ).iloc[-1]
        for name in [column for column in FEATURE_NAMES if column.startswith("spy_")]:
            self.assertAlmostEqual(features.loc[stock.index[450], name], prior[name])

        row = features.loc[stock.index[450]]
        above = float(row["spy_price_sma200"] >= 0)
        expected = {
            "interaction_spy_above_sma200_x_ret_60d": (
                above * row["ret_log_60"]
            ),
            "interaction_spy_above_sma200_x_price_to_sma200_minus1": (
                above * row["price_sma200"]
            ),
            "interaction_spy_vol60_x_vol60": (
                row["spy_vol_60"] * row["vol_60"]
            ),
            "interaction_spy_vol60_x_trailing_drawdown": (
                row["spy_vol_60"] * row["drawdown_252"]
            ),
            "interaction_spy_above_sma200_x_downside_semivol": (
                above * row["downside_semivol_60"]
            ),
        }
        self.assertEqual(set(expected), set(INTERACTION_FEATURES))
        for name, value in expected.items():
            self.assertAlmostEqual(row[name], value)

    def test_weekend_stock_session_uses_prior_spy_and_ignores_future_spy(self):
        stock = history_frame()
        spy = history_frame(seed=22)
        friday = next(
            timestamp
            for timestamp in stock.index[400:]
            if timestamp.dayofweek == 4
        )
        saturday = friday + pd.Timedelta(days=1)
        weekend_row = stock.loc[[friday]].copy()
        weekend_row.index = pd.DatetimeIndex([saturday])
        stock = pd.concat([stock, weekend_row]).sort_index()

        before = build_probability_features(
            stock.loc[:saturday],
            spy.loc[:friday],
        ).loc[saturday]
        future_spy = spy.copy()
        future_rows = future_spy.index > saturday
        future_spy.loc[future_rows, ["Open", "High", "Low", "Close", "RawClose"]] *= 50
        after = build_probability_features(stock, future_spy).loc[saturday]

        np.testing.assert_allclose(
            before.loc[list(SPY_FEATURES) + list(INTERACTION_FEATURES)],
            after.loc[list(SPY_FEATURES) + list(INTERACTION_FEATURES)],
            rtol=0,
            atol=0,
        )

    def test_labels_use_next_open_and_exact_horizon_close(self):
        history = history_frame()
        dataset = build_symbol_dataset("AAA", history, history)
        self.assertFalse(dataset.empty)
        sample = dataset.iloc[0]
        position = history.index.get_loc(sample["feature_date"])
        self.assertEqual(sample["entry_timestamp"], history.index[position + 1])
        for horizon in HORIZONS:
            expected = (
                history["Close"].iloc[position + horizon]
                / history["Open"].iloc[position + 1]
                - 1.0
            )
            self.assertAlmostEqual(sample[f"gross_return_h{horizon}"], expected)
            self.assertEqual(
                sample[f"exit_timestamp_h{horizon}"],
                history.index[position + horizon],
            )
            self.assertIn(
                int(sample[f"ordered_label_h{horizon}"]),
                range(7),
            )
        latest = dataset.iloc[-1]
        self.assertFalse(pd.isna(latest["label_h21_x3"]))
        self.assertFalse(pd.isna(latest["ordered_label_h21"]))
        self.assertTrue(pd.isna(latest["label_h252_x10"]))
        self.assertTrue(pd.isna(latest["ordered_label_h252"]))

    def test_nonfinite_exit_close_remains_unlabeled_not_sentinel(self):
        history = history_frame()
        baseline = build_symbol_dataset("AAA", history, history)
        feature_date = baseline.iloc[0]["feature_date"]
        position = history.index.get_loc(feature_date)
        exit_date = history.index[position + 21]
        broken = history.copy()
        broken.loc[exit_date, "Close"] = np.nan

        dataset = build_symbol_dataset("AAA", broken, history)
        row = dataset.loc[dataset["feature_date"] == feature_date].iloc[0]
        self.assertTrue(pd.isna(row[ordered_label_column(21)]))
        self.assertTrue(pd.isna(row["label_h21_x3"]))
        self.assertTrue(pd.isna(row["exit_timestamp_h21"]))

    def test_material_classes_are_coherent_at_cost_boundaries(self):
        threshold = 0.05
        values = np.array(
            [
                -threshold - ROUND_TRIP_COST,
                -threshold - ROUND_TRIP_COST + 1e-9,
                0.0,
                threshold + ROUND_TRIP_COST - 1e-9,
                threshold + ROUND_TRIP_COST,
            ]
        )
        self.assertEqual(
            classify_material_move(values, threshold).tolist(),
            [0, 1, 1, 1, 2],
        )

    def test_anchors_are_final_completed_session_per_iso_week(self):
        history = history_frame()
        dataset = build_symbol_dataset("AAA", history, history)
        for timestamp in pd.to_datetime(dataset["feature_date"]):
            iso = timestamp.isocalendar()
            same_week = [
                value
                for value in history.index
                if value.isocalendar().year == iso.year
                and value.isocalendar().week == iso.week
            ]
            self.assertEqual(timestamp, max(same_week))

    def test_purge_and_embargo_prevent_label_overlap(self):
        dataset = make_synthetic_dataset(
            learnable=True, issuer_count=12, week_count=760
        )
        folds = make_purged_expanding_folds(dataset, minimum_folds=5)
        for fold in folds:
            train = dataset.iloc[fold["train_indices"]]
            calibration = dataset.iloc[fold["calibration_indices"]]
            test = dataset.iloc[fold["test_indices"]]
            self.assertTrue(
                (
                    pd.to_datetime(train["max_exit_date"])
                    < fold["calibration_start"] - pd.Timedelta(days=7)
                ).all()
            )
            self.assertTrue(
                (
                    pd.to_datetime(calibration["max_exit_date"])
                    < fold["test_start"] - pd.Timedelta(days=7)
                ).all()
            )
            self.assertLess(
                pd.to_datetime(calibration["feature_date"]).max(),
                pd.to_datetime(test["feature_date"]).min(),
            )
            self.assertGreaterEqual(fold["usable_train_years"], 5.0)
            self.assertTrue(fold["full_test_window"])
            self.assertLessEqual(
                fold["test_end"],
                pd.to_datetime(dataset["feature_date"]).max(),
            )
        first = folds[0]
        candidate = int(first["train_indices"][-1])
        mutated = dataset.copy()
        mutated.loc[candidate, "max_exit_date"] = first["calibration_start"]
        rebuilt = make_purged_expanding_folds(mutated, minimum_folds=5)
        same_fold = next(
            fold
            for fold in rebuilt
            if fold["calibration_start"] == first["calibration_start"]
        )
        self.assertNotIn(candidate, set(same_fold["train_indices"]))

    def test_fold_builder_rejects_pre_warmup_rows_and_partial_tests(self):
        dataset = make_synthetic_dataset(
            learnable=True, issuer_count=12, week_count=760
        )
        broken = dataset.copy()
        broken.loc[0, "history_bars_before"] = 251
        with self.assertRaisesRegex(ValueError, "252-bar warm-up"):
            make_purged_expanding_folds(broken)
        cutoff = pd.to_datetime(dataset["feature_date"]).max() - pd.Timedelta(
            days=180
        )
        shortened = dataset.loc[
            pd.to_datetime(dataset["feature_date"]) <= cutoff
        ].copy()
        folds = make_purged_expanding_folds(shortened)
        self.assertTrue(
            all(
                fold["test_end"]
                <= pd.to_datetime(shortened["feature_date"]).max()
                for fold in folds
            )
        )

    def test_default_2008_range_supports_five_full_folds_every_horizon(self):
        self.assertEqual(DEFAULT_START, "2008-01-01")
        sessions = pd.bdate_range(DEFAULT_START, "2026-08-13")
        weekly_positions = {}
        for position, timestamp in enumerate(sessions):
            iso = timestamp.isocalendar()
            weekly_positions[(iso.year, iso.week)] = position
        counts = {}
        for horizon in HORIZONS:
            records = []
            for position in weekly_positions.values():
                if position < 252 or position + horizon >= len(sessions):
                    continue
                records.append(
                    {
                        "symbol": "SYN",
                        "issuer_key": "issuer-syn",
                        "feature_date": sessions[position],
                        "history_start": sessions[0],
                        "history_bars_before": position,
                        "max_exit_date": sessions[position + horizon],
                    }
                )
            dataset = pd.DataFrame(records)
            folds = make_purged_expanding_folds(dataset, minimum_folds=5)
            counts[horizon] = len(folds)
            self.assertTrue(all(fold["full_test_window"] for fold in folds))
            self.assertTrue(
                all(
                    fold["test_end"] <= dataset["feature_date"].max()
                    for fold in folds
                )
            )
        self.assertTrue(all(count >= 5 for count in counts.values()), counts)

    def test_usd_company_and_duplicate_issuer_filter(self):
        universe = [
            {"symbol": "AAA", "exchange": "NYSE"},
            {"symbol": "AAA.B", "exchange": "OTHER"},
            {"symbol": "ETF", "exchange": "NYSE"},
            {"symbol": "EUR", "exchange": "NYSE"},
            {"symbol": "MISS", "exchange": "NYSE"},
        ]
        metadata = {
            "AAA": {
                "quote_type": "EQUITY",
                "reported_currency": "USD",
                "issuer_uuid": "issuer-a",
            },
            "AAA.B": {
                "quote_type": "EQUITY",
                "reported_currency": "USD",
                "issuer_uuid": "issuer-a",
            },
            "ETF": {"quote_type": "ETF", "reported_currency": "USD"},
            "EUR": {"quote_type": "EQUITY", "reported_currency": "EUR"},
        }
        result = select_eligible_universe(universe, metadata)
        self.assertEqual(result.symbols, ("AAA",))
        self.assertIn("AAA.B", result.excluded)
        self.assertIn("ETF", result.excluded)
        self.assertIn("EUR", result.excluded)
        self.assertIn("MISS", result.excluded)

    def test_panel_batch_split_manifest_checksum_and_resume(self):
        frame = history_frame(periods=320)
        frame = frame.rename_axis("Date")
        frame["Adj Close"] = frame["Close"]
        calls = []

        def downloader(symbols, _start, _end):
            calls.append(tuple(symbols))
            if len(symbols) > 1:
                raise RuntimeError("synthetic batch failure")
            return frame

        now = datetime(2026, 8, 13, 22, 0, tzinfo=timezone.utc)
        manifest = download_probability_panel(
            ["AAA", "BBB"],
            start="2022-01-01",
            end="2026-08-14",
            cache_dir=self.work,
            batch_size=2,
            retries=0,
            now=now,
            downloader=downloader,
        )
        self.assertEqual(manifest["completed_symbols"], ["AAA", "BBB"])
        self.assertEqual(manifest["requested_symbol_count"], 2)
        self.assertEqual(manifest["successful_symbol_count"], 2)
        self.assertEqual(manifest["provider_success_coverage"], 1.0)
        self.assertIn(("AAA", "BBB"), calls)
        self.assertIn(("AAA",), calls)
        self.assertIn(("BBB",), calls)
        histories, loaded = load_probability_panel(self.work)
        self.assertEqual(set(histories), {"AAA", "BBB"})
        self.assertEqual(loaded["feature_schema_hash"], feature_schema_hash())

        calls.clear()
        download_probability_panel(
            ["AAA", "BBB"],
            start="2022-01-01",
            end="2026-08-14",
            cache_dir=self.work,
            batch_size=2,
            retries=0,
            now=now,
            downloader=downloader,
        )
        self.assertEqual(calls, [])

    def test_dataset_content_hash_binds_unrelated_values(self):
        dataset = make_synthetic_dataset(
            learnable=True, issuer_count=3, week_count=30
        )
        reordered = dataset.sample(frac=1.0, random_state=7)[
            list(reversed(dataset.columns))
        ]
        self.assertEqual(
            dataset_content_hash(dataset),
            dataset_content_hash(reordered),
        )
        changed = dataset.copy()
        changed.loc[0, FEATURE_NAMES[0]] += 1e-9
        self.assertNotEqual(
            dataset_content_hash(dataset),
            dataset_content_hash(changed),
        )

    def test_exact_dataset_cache_roundtrips_across_processes(self):
        dataset = make_synthetic_dataset(
            learnable=True, issuer_count=3, week_count=30
        )
        dataset.loc[0, FEATURE_NAMES[0]] = np.nextafter(
            dataset.loc[0, FEATURE_NAMES[0]], np.inf
        )
        path = self.work / "dataset.pkl.gz"
        write_exact_dataset_cache(path, dataset)
        loaded = read_exact_dataset_cache(path)
        pd.testing.assert_frame_equal(
            loaded,
            dataset,
            check_exact=True,
            check_dtype=True,
            check_freq=True,
        )
        expected_hash = dataset_content_hash(dataset)
        command = (
            "from pathlib import Path; "
            "from src.probability_dataset import "
            "read_exact_dataset_cache,dataset_content_hash; "
            f"d=read_exact_dataset_cache(Path({str(path)!r})); "
            "print(dataset_content_hash(d))"
        )
        completed = subprocess.run(
            [sys.executable, "-c", command],
            cwd=str(ROOT),
            check=True,
            capture_output=True,
            text=True,
        )
        self.assertEqual(completed.stdout.strip(), expected_hash)

    def test_dataset_loader_verifies_file_and_semantic_hashes(self):
        dataset = make_synthetic_dataset(
            learnable=True, issuer_count=2, week_count=12
        )
        path = self.work / DATASET_FILENAME
        write_exact_dataset_cache(path, dataset)
        file_hash = hashlib.sha256(path.read_bytes()).hexdigest()
        manifest = {
            "schema_version": DATASET_SCHEMA_VERSION,
            "feature_version": FEATURE_VERSION,
            "feature_schema_hash": feature_schema_hash(),
            "code_hash": probability_code_hash(),
            "file": DATASET_FILENAME,
            "sha256": file_hash,
            "dataset_content_hash": dataset_content_hash(dataset),
            "dataset_cache_key": "test-cache-key",
            "storage_format": "trusted-local-pandas-pickle-protocol5-gzip",
            "trust_boundary": "test repository-generated cache",
            "provider_requested_issuer_count": 2,
            "provider_successful_issuer_count": 2,
            "provider_success_coverage": 1.0,
            "provider_unavailable_symbols": {},
            "panel_manifest_sha256": "panel-test",
        }
        manifest_path = self.work / "dataset_manifest.json"
        atomic_write_json(manifest_path, manifest)
        loaded, _ = load_weekly_dataset(self.work)
        pd.testing.assert_frame_equal(loaded, dataset, check_exact=True)

        broken_semantic = dict(manifest)
        broken_semantic["dataset_content_hash"] = "0" * 64
        atomic_write_json(manifest_path, broken_semantic)
        with self.assertRaisesRegex(ValueError, "content hash mismatch"):
            load_weekly_dataset(self.work)

        atomic_write_json(manifest_path, manifest)
        path.write_bytes(path.read_bytes() + b"x")
        with self.assertRaisesRegex(ValueError, "checksum mismatch"):
            load_weekly_dataset(self.work)

    def test_panel_manifest_binds_provider_failures_and_coverage(self):
        frame = history_frame(periods=320)
        frame["Adj Close"] = frame["Close"]

        def downloader(symbols, _start, _end):
            if "BAD" in symbols:
                raise RuntimeError("provider denied BAD")
            return frame

        manifest = download_probability_panel(
            ["AAA", "BAD"],
            start="2022-01-01",
            end="2026-08-14",
            cache_dir=self.work,
            batch_size=2,
            retries=0,
            now=datetime(2026, 8, 13, 22, 0, tzinfo=timezone.utc),
            downloader=downloader,
        )
        self.assertEqual(manifest["requested_symbol_count"], 2)
        self.assertEqual(manifest["successful_symbol_count"], 1)
        self.assertEqual(manifest["provider_success_coverage"], 0.5)
        self.assertIn("BAD", manifest["failures"])
        self.assertIn("provider denied BAD", manifest["failures"]["BAD"])

    def test_code_hash_is_crlf_lf_and_platform_path_invariant(self):
        lf_root = self.work / "lf"
        crlf_root = self.work / "crlf"
        lf_root.mkdir()
        crlf_root.mkdir()
        (lf_root / "module.py").write_bytes(b"x = 1\ny = 2\n")
        (crlf_root / "module.py").write_bytes(b"x = 1\r\ny = 2\r\n")
        self.assertEqual(
            probability_code_hash(lf_root, names=("module.py",)),
            probability_code_hash(crlf_root, names=("module.py",)),
        )
        shared_root = self.work / "shared"
        (shared_root / "src").mkdir(parents=True)
        (shared_root / "src" / "fetch.py").write_text(
            "VALUE = 1\n", encoding="utf-8"
        )
        (shared_root / "requirements-ci.txt").write_text(
            "numpy==2.5.0\n", encoding="utf-8"
        )
        windows_style = probability_code_hash(
            shared_root,
            names=("src\\fetch.py", "requirements-ci.txt"),
        )
        unix_style = probability_code_hash(
            shared_root,
            names=("src/fetch.py", "requirements-ci.txt"),
        )
        self.assertEqual(windows_style, unix_style)
        (shared_root / "src" / "fetch.py").write_text(
            "VALUE = 2\n", encoding="utf-8"
        )
        self.assertNotEqual(
            windows_style,
            probability_code_hash(
                shared_root,
                names=("src/fetch.py", "requirements-ci.txt"),
            ),
        )

    def test_default_code_hash_binds_ordered_core_and_dependencies(self):
        required = {
            "src/probability_contract.py",
            "src/probability_dataset.py",
            "src/probability_features.py",
            "src/probability_model.py",
            "src/probability_ordered.py",
            "src/probability_inference.py",
            "src/probability_train.py",
            "src/config.py",
            "src/persistence.py",
            "requirements-ci.txt",
        }
        self.assertTrue(required.issubset(set(PROBABILITY_CODE_FILES)))
        root = self.work / "ordered-hash"
        ordered = root / "src" / "probability_ordered.py"
        ordered.parent.mkdir(parents=True)
        ordered.write_text("ORDERED_VERSION = 1\n", encoding="utf-8")
        first = probability_code_hash(root)
        ordered.write_text("ORDERED_VERSION = 2\n", encoding="utf-8")
        second = probability_code_hash(root)
        self.assertNotEqual(first, second)


if __name__ == "__main__":
    unittest.main()
