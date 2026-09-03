from __future__ import annotations

from pathlib import Path
import tempfile
import unittest

from brian2026.intelligence_store import IntelligenceCapture, IntelligenceStore


class IntelligenceStoreTests(unittest.TestCase):
    def capture(self, **overrides):
        values = dict(
            provider="binance_public",
            record_type="announcement",
            observed_at=1_700_000_001.0,
            captured_at=1_700_000_002.0,
            payload={"asset": "ABC", "headline": "ABC listing"},
            provenance_uri="https://example.invalid/a",
            provider_record_id="a-1",
        )
        values.update(overrides)
        return IntelligenceCapture(**values)

    def test_capture_is_content_addressed_and_deterministic(self):
        first = self.capture()
        second = self.capture()
        self.assertEqual(first.capture_id, second.capture_id)
        self.assertEqual(first.payload_hash, second.payload_hash)

    def test_capture_cannot_precede_observation(self):
        with self.assertRaisesRegex(ValueError, "cannot precede"):
            self.capture(captured_at=1_700_000_000.0)

    def test_store_is_idempotent_for_identical_capture(self):
        with tempfile.TemporaryDirectory() as directory:
            store = IntelligenceStore(directory)
            row = self.capture()
            first = store.put(row)
            second = store.put(row)
            self.assertEqual(first, second)
            self.assertTrue(first.exists())
            loaded = store.read(row.capture_id, provider=row.provider, record_type=row.record_type)
            self.assertEqual(loaded["capture_id"], row.capture_id)
            self.assertEqual(loaded["payload_hash"], row.payload_hash)

    def test_payload_changes_create_new_scientific_capture(self):
        first = self.capture()
        second = self.capture(payload={"asset": "ABC", "headline": "Different"})
        self.assertNotEqual(first.capture_id, second.capture_id)
        self.assertNotEqual(first.payload_hash, second.payload_hash)

    def test_non_finite_payload_is_rejected(self):
        with self.assertRaisesRegex(ValueError, "NaN or infinity"):
            self.capture(payload={"score": float("nan")})

    def test_store_path_is_provider_and_record_type_scoped(self):
        with tempfile.TemporaryDirectory() as directory:
            row = self.capture(provider="arkham", record_type="whale_flow")
            path = IntelligenceStore(directory).put(row)
            self.assertIn(str(Path("captures") / "arkham" / "whale_flow"), str(path))


if __name__ == "__main__":
    unittest.main()
