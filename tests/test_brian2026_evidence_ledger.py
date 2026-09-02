from __future__ import annotations

import unittest

from brian2026.evidence_ledger import (
    EvidenceConflictError,
    EvidenceLedger,
    EvidenceRecord,
    canonical_json,
    content_hash,
)
from brian2026.portfolio import DEVELOPMENT_CUTOFF


class EvidenceLedgerTests(unittest.TestCase):
    def record(self, **overrides) -> EvidenceRecord:
        kwargs = dict(
            phase="phase29-development",
            logical_experiment_id="phase29-logical-001",
            dataset_id="dataset-2020-2025",
            code_commit="abc123",
            scope="development",
            max_data_timestamp=DEVELOPMENT_CUTOFF - 300,
            metrics={"net_pnl": -12.5, "profit_factor": 0.91},
            gates={"all_required_gates_passed": False, "robust": False},
            decision="INSUFFICIENT_EVIDENCE",
        )
        kwargs.update(overrides)
        return EvidenceRecord(**kwargs)

    def test_canonical_hash_is_independent_of_mapping_order(self):
        left = {"b": 2, "a": {"y": 2, "x": 1}}
        right = {"a": {"x": 1, "y": 2}, "b": 2}
        self.assertEqual(canonical_json(left), canonical_json(right))
        self.assertEqual(content_hash(left), content_hash(right))

    def test_record_identity_is_deterministic(self):
        first = self.record()
        second = self.record(metrics={"profit_factor": 0.91, "net_pnl": -12.5})
        self.assertEqual(first.evidence_id, second.evidence_id)
        self.assertEqual(first.manifest(), second.manifest())

    def test_2026_data_is_forbidden(self):
        with self.assertRaisesRegex(ValueError, "INVALID_CONTAMINATED"):
            self.record(max_data_timestamp=DEVELOPMENT_CUTOFF)

    def test_final_holdout_evaluation_is_forbidden(self):
        with self.assertRaisesRegex(ValueError, "no pristine final holdout"):
            self.record(scope="final_holdout")
        with self.assertRaisesRegex(ValueError, "must not be evaluated"):
            self.record(final_holdout_evaluated=True)

    def test_shadow_and_no_auto_promotion_are_hard_requirements(self):
        with self.assertRaisesRegex(ValueError, "SHADOW_RESEARCH_ONLY"):
            self.record(shadow_only=False)
        with self.assertRaisesRegex(ValueError, "automatic model promotion"):
            self.record(automatic_promotion=True)

    def test_shadow_candidate_requires_all_gates(self):
        with self.assertRaisesRegex(ValueError, "requires all required scientific gates"):
            self.record(decision="SHADOW_CANDIDATE")
        accepted = self.record(
            decision="SHADOW_CANDIDATE",
            gates={"all_required_gates_passed": True, "robust": True},
        )
        self.assertEqual(accepted.decision, "SHADOW_CANDIDATE")

    def test_duplicate_append_is_idempotent(self):
        ledger = EvidenceLedger()
        record = self.record()
        first = ledger.append(record)
        second = ledger.append(self.record())
        self.assertFalse(first.duplicate)
        self.assertTrue(second.duplicate)
        self.assertEqual(first.ordinal, second.ordinal)
        self.assertEqual(len(ledger.records), 1)

    def test_same_logical_experiment_cannot_be_rewritten(self):
        ledger = EvidenceLedger([self.record()])
        with self.assertRaisesRegex(EvidenceConflictError, "rewriting history"):
            ledger.append(self.record(metrics={"net_pnl": 999.0, "profit_factor": 9.0}))

    def test_lineage_requires_known_parent(self):
        child = self.record(
            logical_experiment_id="phase30-child",
            parent_evidence_ids=("missing-parent",),
        )
        with self.assertRaisesRegex(EvidenceConflictError, "unknown parent"):
            EvidenceLedger().append(child)

        parent = self.record()
        ledger = EvidenceLedger([parent])
        valid_child = self.record(
            logical_experiment_id="phase30-child",
            phase="phase30-development",
            parent_evidence_ids=(parent.evidence_id,),
        )
        receipt = ledger.append(valid_child)
        self.assertFalse(receipt.duplicate)
        self.assertEqual(len(ledger.records), 2)

    def test_ledger_hash_changes_if_evidence_changes(self):
        one = EvidenceLedger([self.record()]).manifest()["ledger_hash"]
        parent = self.record()
        child = self.record(
            logical_experiment_id="phase30-child",
            phase="phase30-development",
            parent_evidence_ids=(parent.evidence_id,),
        )
        two = EvidenceLedger([parent, child]).manifest()["ledger_hash"]
        self.assertNotEqual(one, two)


if __name__ == "__main__":
    unittest.main()
