from __future__ import annotations

import unittest

from brian2026.intelligence_memory import (
    OpportunityMemory,
    OpportunityOutcome,
    SourceOutcome,
    SourceReputationMemory,
)


T0 = 1_700_000_000.0


class IntelligenceMemoryTests(unittest.TestCase):
    def source(self, **overrides):
        values = dict(
            source_id="official-feed", event_kind="listing", observed_at=T0,
            resolution_at=T0 + 3600, predicted_direction=1.0, realized_return=0.03,
            truth_confirmed=True, manipulation_confirmed=False,
        )
        values.update(overrides)
        return SourceOutcome(**values)

    def test_source_outcome_cannot_learn_before_resolution(self):
        memory = SourceReputationMemory()
        with self.assertRaisesRegex(ValueError, "before its resolution"):
            memory.learn(self.source(), current_timestamp=T0 + 30)

    def test_source_reputation_as_of_excludes_future_resolved_outcomes(self):
        memory = SourceReputationMemory()
        first = self.source(resolution_at=T0 + 100)
        second = self.source(resolution_at=T0 + 200, realized_return=-0.03, truth_confirmed=False)
        memory.learn(first, current_timestamp=T0 + 100)
        memory.learn(second, current_timestamp=T0 + 200)
        early = memory.reputation("official-feed", event_kind="listing", as_of=T0 + 150)
        late = memory.reputation("official-feed", event_kind="listing", as_of=T0 + 250)
        self.assertEqual(early.resolved_samples, 1)
        self.assertEqual(late.resolved_samples, 2)
        self.assertGreater(early.directional_accuracy, late.directional_accuracy)

    def test_one_lucky_sample_has_low_confidence(self):
        memory = SourceReputationMemory()
        row = self.source()
        memory.learn(row, current_timestamp=row.resolution_at)
        rep = memory.reputation("official-feed", event_kind="listing", as_of=row.resolution_at)
        self.assertLess(rep.confidence, 0.10)
        self.assertLess(rep.reliability, 0.60)

    def test_repeated_clean_source_builds_reputation(self):
        memory = SourceReputationMemory()
        for i in range(40):
            row = self.source(observed_at=T0 + i * 7200, resolution_at=T0 + i * 7200 + 3600)
            memory.learn(row, current_timestamp=row.resolution_at)
        rep = memory.reputation("official-feed", event_kind="listing", as_of=T0 + 400_000)
        self.assertGreater(rep.confidence, 0.60)
        self.assertGreater(rep.reliability, 0.80)

    def test_chronological_learning_is_required(self):
        memory = SourceReputationMemory()
        later = self.source(observed_at=T0 + 200, resolution_at=T0 + 300)
        earlier = self.source(observed_at=T0 + 100, resolution_at=T0 + 150)
        memory.learn(later, current_timestamp=T0 + 300)
        with self.assertRaisesRegex(ValueError, "chronological"):
            memory.learn(earlier, current_timestamp=T0 + 150)

    def test_opportunity_memory_cannot_use_unresolved_outcome(self):
        memory = OpportunityMemory()
        row = OpportunityOutcome("listing+smartmoney", T0, T0 + 600, 0.02, 0.01, 0.9)
        with self.assertRaisesRegex(ValueError, "before resolution"):
            memory.learn(row, current_timestamp=T0 + 100)

    def test_opportunity_experience_is_point_in_time(self):
        memory = OpportunityMemory()
        first = OpportunityOutcome("listing+smartmoney", T0, T0 + 100, 0.03, 0.01, 0.9)
        second = OpportunityOutcome("listing+smartmoney", T0 + 200, T0 + 300, -0.02, 0.03, 0.8)
        memory.learn(first, current_timestamp=T0 + 100)
        memory.learn(second, current_timestamp=T0 + 300)
        early = memory.experience("listing+smartmoney", as_of=T0 + 150)
        late = memory.experience("listing+smartmoney", as_of=T0 + 350)
        self.assertEqual(early.samples, 1)
        self.assertEqual(late.samples, 2)
        self.assertGreater(early.mean_net_return, late.mean_net_return)

    def test_unknown_context_abstains_with_zero_confidence(self):
        row = OpportunityMemory().experience("never-seen", as_of=T0)
        self.assertEqual(row.samples, 0)
        self.assertEqual(row.conservative_edge, 0.0)
        self.assertEqual(row.confidence, 0.0)


if __name__ == "__main__":
    unittest.main()
