from __future__ import annotations

import unittest

from brian2026.intelligence_fabric import (
    IntelEvent,
    SocialBurst,
    WhaleObservation,
    assert_historical_replay_safe,
    assess_truth,
    assess_whale,
    fuse_asset_intelligence,
    rank_opportunities,
    social_authenticity,
)
from brian2026.intelligence_providers import (
    PROVIDER_REGISTRY,
    enabled_without_secrets,
    historical_replay_allowlist,
    provider_map,
)


T0 = 1_700_000_000.0


def event(**overrides):
    base = dict(
        asset="ABC",
        event_kind="listing",
        source_kind="exchange",
        source_id="binance",
        published_at=T0,
        observed_at=T0 + 1,
        claim="Binance will list ABC",
        direction=0.8,
        magnitude=0.9,
        trust_class="verified_official",
        entity_confidence=1.0,
        content_fingerprint="official-abc-listing",
        corroboration_key="ABC:listing:2023-11-14",
        provenance_uri="https://example.invalid/official",
        historical_timestamp_verified=True,
    )
    base.update(overrides)
    return IntelEvent(**base)


class IntelligenceFabricTests(unittest.TestCase):
    def test_event_cannot_be_observed_before_publication(self):
        with self.assertRaisesRegex(ValueError, "observed before"):
            event(observed_at=T0 - 1)

    def test_verified_official_plus_independent_market_source_scores_high(self):
        rows = [
            event(),
            event(
                source_kind="market",
                source_id="binance-market-stream",
                claim="ABC volume and price react after listing publication",
                trust_class="verified_provider",
                direction=0.6,
                magnitude=0.7,
                content_fingerprint="market-reaction",
            ),
        ]
        truth = assess_truth(rows)
        self.assertTrue(truth.official_confirmation)
        self.assertEqual(truth.independent_sources, 2)
        self.assertGreater(truth.truth_score, 0.80)
        self.assertLess(truth.manipulation_risk, 0.20)

    def test_copied_social_posts_do_not_become_independent_truth(self):
        rows = [
            event(
                event_kind="social_surge", source_kind="social", source_id=f"bot-{i}",
                claim="ABC to the moon", direction=1.0, magnitude=0.9,
                trust_class="unknown", content_fingerprint="same-copy",
                corroboration_key="ABC:social-rumor",
            )
            for i in range(8)
        ]
        truth = assess_truth(rows)
        self.assertGreater(truth.duplicate_ratio, 0.80)
        self.assertLess(truth.truth_score, 0.25)
        self.assertGreater(truth.manipulation_risk, 0.70)

    def test_social_authenticity_penalizes_bot_copy_burst(self):
        organic = SocialBurst("ABC", T0, 1000, 700, 0.85, 800, 0.08, 0.10, 3)
        fake = SocialBurst("ABC", T0, 1000, 30, 0.05, 5, 0.00, 0.95, 1)
        self.assertGreater(social_authenticity(organic), social_authenticity(fake))
        self.assertLess(social_authenticity(fake), 0.10)

    def test_internal_whale_transfer_has_no_directional_meaning(self):
        row = WhaleObservation(
            "ABC", T0, "fund-x", "arkham", "verified_provider", 0.95,
            "internal_transfer", 5_000_000, tx_hash="0x1", historical_timestamp_verified=True,
        )
        result = assess_whale(row)
        self.assertEqual(result.economic_direction, 0.0)
        self.assertIn("internal transfer", result.reasons[0])

    def test_user_generated_whale_label_is_unresolved(self):
        row = WhaleObservation(
            "ABC", T0, "mystery-wallet", "social-user", "user_generated", 0.90,
            "dex_buy", 2_000_000, tx_hash="0x2", historical_timestamp_verified=True,
        )
        result = assess_whale(row)
        self.assertTrue(result.suspicious_or_unresolved)
        self.assertLess(result.confidence, 0.40)
        self.assertLess(result.economic_direction, 0.40)

    def test_verified_dex_buy_can_add_bullish_whale_context(self):
        row = WhaleObservation(
            "ABC", T0, "smart-money-1", "nansen", "verified_provider", 0.95,
            "dex_buy", 1_000_000, tx_hash="0x3", historical_timestamp_verified=True,
        )
        result = assess_whale(row)
        self.assertFalse(result.suspicious_or_unresolved)
        self.assertGreater(result.economic_direction, 0.80)

    def test_social_only_inorganic_claim_is_vetoed(self):
        rows = [
            event(
                event_kind="social_surge", source_kind="social", source_id="x",
                claim="ABC rumor", direction=1.0, magnitude=1.0, trust_class="unknown",
                content_fingerprint="rumor", corroboration_key="ABC:rumor",
            )
        ]
        fake = SocialBurst("ABC", T0 + 2, 5000, 40, 0.03, 3, 0.0, 0.98, 1)
        intelligence = fuse_asset_intelligence(rows, social=fake, market_confirmation=0.6)
        self.assertTrue(intelligence.veto_reasons)
        self.assertLess(intelligence.opportunity_priority, 0.05)
        self.assertTrue(intelligence.shadow_only)

    def test_cross_source_verified_event_can_rank_above_social_noise(self):
        strong = fuse_asset_intelligence([
            event(),
            event(
                source_kind="market", source_id="market", trust_class="verified_provider",
                claim="ABC volume confirms event", direction=0.7, magnitude=0.8,
                content_fingerprint="market-confirm", corroboration_key="ABC:listing:2023-11-14",
            ),
        ], market_confirmation=0.8)
        weak = fuse_asset_intelligence([
            event(
                asset="XYZ", event_kind="social_surge", source_kind="social", source_id="x",
                claim="XYZ moon", trust_class="unknown", direction=1.0, magnitude=1.0,
                content_fingerprint="spam", corroboration_key="XYZ:rumor",
            )
        ], social=SocialBurst("XYZ", T0, 1000, 20, 0.02, 2, 0.0, 0.99, 1),
           market_confirmation=0.8)
        ranked = rank_opportunities((weak, strong), top_n=2)
        self.assertEqual(ranked[0].asset, "ABC")

    def test_historical_replay_rejects_hindsight_event_label(self):
        unsafe = event(historical_timestamp_verified=False)
        with self.assertRaisesRegex(ValueError, "prospective shadow capture"):
            assert_historical_replay_safe([unsafe])

    def test_historical_replay_rejects_hindsight_whale_attribution(self):
        whale = WhaleObservation(
            "ABC", T0, "entity", "provider", "verified_provider", 1.0,
            "exchange_deposit", 1_000_000, historical_timestamp_verified=False,
        )
        with self.assertRaisesRegex(ValueError, "prospective shadow capture"):
            assert_historical_replay_safe([event()], [whale])

    def test_provider_registry_keeps_secret_sources_disabled_by_default(self):
        rows = provider_map()
        self.assertIn("arkham", rows)
        self.assertIn("nansen", rows)
        self.assertIn("lunarcrush", rows)
        self.assertIn("x_api", rows)
        self.assertTrue(all(not row.default_enabled for row in PROVIDER_REGISTRY if row.requires_secret))
        self.assertEqual(enabled_without_secrets(), ("binance_public",))

    def test_prospective_social_api_is_not_historical_allowlist(self):
        allowed = historical_replay_allowlist()
        self.assertIn("binance_public", allowed)
        self.assertIn("macro_official", allowed)
        self.assertNotIn("x_api", allowed)
        self.assertNotIn("reddit_api", allowed)


if __name__ == "__main__":
    unittest.main()
