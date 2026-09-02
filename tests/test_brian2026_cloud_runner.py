from __future__ import annotations

from dataclasses import replace
from datetime import datetime, timezone
from pathlib import Path
import tempfile
import unittest

from brian2026.cloud_runner import (
    DEVELOPMENT_END,
    EXECUTION_DECLARATION,
    FINAL_HOLDOUT_DECLARATION,
    HOLDOUT_STATUS,
    SMOKE_END,
    build_cloud_summary,
    mode_range,
    run_cloud,
)
from brian2026.engine import BrianEngine
from brian2026.phase24 import MultiMonthDatasetManifest
from brian2026.portfolio import DEVELOPMENT_CUTOFF


def dataset(*, requested_end: str = "2024-02-01T00:00:00+00:00",
            actual_end: float = 1_706_745_599.999) -> MultiMonthDatasetManifest:
    return MultiMonthDatasetManifest(
        "BTCUSDT", "binance", "spot", "2024-01-01T00:00:00+00:00",
        requested_end, 1_704_067_200.0, actual_end, (),
        (("1m", 44_640), ("5m", 8_928), ("15m", 2_976), ("1h", 744)),
        "READY", "official-binance-test-double", 1_700_000_000.0,
    )


class CloudRunnerTests(unittest.TestCase):
    def test_ranges_are_fixed_and_never_cross_cutoff(self):
        self.assertEqual(SMOKE_END, datetime(2024, 2, 1, tzinfo=timezone.utc))
        self.assertEqual(mode_range("full-development")[1], DEVELOPMENT_END)
        self.assertEqual(mode_range("phase27-development")[1], DEVELOPMENT_END)
        self.assertEqual(mode_range("phase28-development")[1], DEVELOPMENT_END)
        self.assertEqual(mode_range("phase29-development")[1], DEVELOPMENT_END)
        self.assertEqual(DEVELOPMENT_END.timestamp(), DEVELOPMENT_CUTOFF)

    def test_unsupported_mode_is_rejected(self):
        with self.assertRaisesRegex(ValueError, "unsupported"):
            mode_range("custom")

    def test_dataset_actual_end_at_cutoff_is_rejected(self):
        with self.assertRaisesRegex(ValueError, "INVALID_CONTAMINATED"):
            build_cloud_summary("smoke", dataset(actual_end=DEVELOPMENT_CUTOFF))

    def test_dataset_requested_end_after_cutoff_is_rejected(self):
        contaminated = dataset(requested_end="2026-02-01T00:00:00+00:00")
        with self.assertRaisesRegex(ValueError, "requested end"):
            build_cloud_summary("smoke", contaminated)

    def test_dataset_must_match_fixed_mode_range(self):
        wrong_range = dataset(requested_end="2024-03-01T00:00:00+00:00")
        with self.assertRaisesRegex(ValueError, "fixed cloud mode range"):
            build_cloud_summary("smoke", wrong_range)

    def test_smoke_uses_fake_builder_and_never_runs_experiment(self):
        calls = []

        def builder(root, start, end):
            calls.append((Path(root), start, end))
            return dataset()

        def forbidden_experiment(root, dataset_id):
            raise AssertionError("smoke must not run development experiments")

        with tempfile.TemporaryDirectory() as directory:
            output = Path(directory) / "cloud_results" / "brian_cloud_summary.json"
            summary = run_cloud("smoke", Path(directory) / "research_data", output,
                                dataset_builder=builder, experiment_runner=forbidden_experiment)
            self.assertTrue(output.exists())
        self.assertEqual(calls[0][1:], mode_range("smoke"))
        self.assertEqual(summary["holdout"]["status"], HOLDOUT_STATUS)
        self.assertFalse(summary["holdout"]["evaluation_allowed"])
        self.assertEqual(summary["execution_declaration"], EXECUTION_DECLARATION)
        self.assertNotIn("phase25_experiment_id", summary)

    def test_full_requires_untouched_holdout_declaration(self):
        with self.assertRaisesRegex(ValueError, "holdout declaration"):
            build_cloud_summary("full-development", replace(
                dataset(),
                requested_start="2020-01-01T00:00:00+00:00",
                requested_end="2026-01-01T00:00:00+00:00",
            ), {"declaration": "missing"})

    def test_full_summary_preserves_negative_candidate_decisions(self):
        full_dataset = replace(
            dataset(),
            requested_start="2020-01-01T00:00:00+00:00",
            requested_end="2026-01-01T00:00:00+00:00",
        )
        candidates = {"logistic_regression": {"status": "INSUFFICIENT_EVIDENCE"}}
        experiment = {
            "declaration": FINAL_HOLDOUT_DECLARATION,
            "experiment_id": "experiment-id",
            "date_range": {"max_observed_timestamp": DEVELOPMENT_CUTOFF - 0.001},
            "candidate_decisions": candidates,
        }
        summary = build_cloud_summary("full-development", full_dataset, experiment)
        self.assertEqual(summary["candidate_decisions"], candidates)
        self.assertEqual(summary["final_holdout_declaration"], FINAL_HOLDOUT_DECLARATION)

    def _development_dataset(self):
        return replace(dataset(), requested_start="2020-01-01T00:00:00+00:00",
                       requested_end="2026-01-01T00:00:00+00:00")

    def _experiment(self, experiment_id):
        return {"declaration": FINAL_HOLDOUT_DECLARATION, "experiment_id": experiment_id,
                "date_range": {"max_observed_timestamp": DEVELOPMENT_CUTOFF - .001},
                "candidate_decision": {"status": "INSUFFICIENT_EVIDENCE"}}

    def test_phase27_mode_uses_fixed_range_and_preserves_declarations(self):
        summary = build_cloud_summary("phase27-development", self._development_dataset(), self._experiment("p27"))
        self.assertEqual(summary["phase27_experiment_id"], "p27")
        self.assertEqual(summary["execution_declaration"], EXECUTION_DECLARATION)
        self.assertEqual(summary["holdout"]["status"], HOLDOUT_STATUS)

    def test_phase28_mode_is_fixed_shadow_development_only(self):
        summary = build_cloud_summary("phase28-development", self._development_dataset(), self._experiment("p28"))
        self.assertEqual(summary["phase28_experiment_id"], "p28")
        self.assertEqual(summary["candidate_decisions"]["status"], "INSUFFICIENT_EVIDENCE")
        self.assertEqual(summary["requested_range"]["end_exclusive"], DEVELOPMENT_END.isoformat())
        self.assertFalse(summary["holdout"]["evaluation_allowed"])

    def test_phase29_mode_is_fixed_shadow_development_only(self):
        summary = build_cloud_summary("phase29-development", self._development_dataset(), self._experiment("p29"))
        self.assertEqual(summary["phase29_experiment_id"], "p29")
        self.assertEqual(summary["candidate_decisions"]["status"], "INSUFFICIENT_EVIDENCE")
        self.assertEqual(summary["requested_range"]["end_exclusive"], DEVELOPMENT_END.isoformat())
        self.assertEqual(summary["execution_declaration"], EXECUTION_DECLARATION)
        self.assertEqual(summary["holdout"]["status"], HOLDOUT_STATUS)
        self.assertFalse(summary["holdout"]["evaluation_allowed"])

    def test_no_exchange_execution_surface_is_introduced(self):
        engine = BrianEngine({"shadow_only": False})
        self.assertTrue(engine.shadow_only)
        forbidden = {"buy", "sell", "place_order", "create_order", "execute_order", "submit_order"}
        self.assertFalse(forbidden.intersection(dir(engine)))

    def test_cached_dataset_manifest_reuses_scientific_identity(self):
        original = dataset()
        restored = replace(original, creation_timestamp=original.creation_timestamp + 60)
        with tempfile.TemporaryDirectory() as directory:
            first = original.write(directory)
            second = restored.write(directory)
            self.assertEqual(first, second)
            self.assertEqual(original.dataset_id, restored.dataset_id)

    def test_cached_dataset_manifest_rejects_identity_mismatch(self):
        original = dataset()
        with tempfile.TemporaryDirectory() as directory:
            target = original.write(directory)
            target.write_text(target.read_text().replace('"quality_status":"READY"',
                                                         '"quality_status":"BROKEN"'))
            with self.assertRaisesRegex(FileExistsError, "immutable"):
                replace(original, creation_timestamp=original.creation_timestamp + 60).write(directory)


if __name__ == "__main__":
    unittest.main()
