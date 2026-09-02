from dataclasses import replace
import unittest

from brian2026.engine import BrianEngine
from brian2026.market_structure import StructureCandle,StructureConfig,compute_market_structure,join_completed_timeframes
from brian2026.portfolio import DEVELOPMENT_CUTOFF,PortfolioBar,PortfolioConfig,simulate_portfolio
from brian2026.structure_exit import StructureExitConfig,apply_structure_exit
from brian2026.phase27_specialists import dip_specialist,market_structure_specialist

def candles(highs,lows=None):
    lows=lows or [x-2 for x in highs]
    return tuple(StructureCandle(1_600_000_000+i*300,(h+l)/2,h,l,(h+l)/2,100+i) for i,(h,l) in enumerate(zip(highs,lows)))

class Phase27PointInTimeTests(unittest.TestCase):
    def setUp(self):self.cfg=StructureConfig(left_bars=1,right_bars=2,atr_period=2,volume_period=2)

    def test_swing_waits_for_right_confirmation(self):
        rows=candles([10,15,12,11])
        features=compute_market_structure(rows,self.cfg)
        self.assertEqual(features[1].confirmed_swings,())
        self.assertEqual(features[2].confirmed_swings,())
        self.assertEqual(features[3].confirmed_swings[-1].pivot_index,1)
        self.assertEqual(features[3].confirmed_swings[-1].confirmation_index,3)

    def test_future_edits_cannot_change_past_features(self):
        original=candles([10,15,12,11,13,9,12])
        changed=original[:5]+tuple(replace(x,high=x.high+20,close=x.close+10) for x in original[5:])
        left=compute_market_structure(original,self.cfg);right=compute_market_structure(changed,self.cfg)
        self.assertEqual(left[:5],right[:5])

    def test_bos_and_choch_cannot_use_unconfirmed_swing(self):
        features=compute_market_structure(candles([10,15,12]),self.cfg)
        self.assertFalse(any(x.bullish_bos or x.bearish_bos or x.bullish_choch or x.bearish_choch for x in features))

    def test_divergence_not_backfilled_to_pivot(self):
        rows=candles([14,12,15,11,14,10,13],[12,9,12,8,11,7,10])
        features=compute_market_structure(rows,self.cfg)
        for index,feature in enumerate(features):
            if feature.bullish_rsi_divergence or feature.bearish_rsi_divergence:
                self.assertEqual(feature.confirmed_swings[-1].confirmation_index,index)

    def test_htf_never_joins_before_close(self):
        primary=compute_market_structure(candles([10,11,12,13]),self.cfg)
        htf=(replace(primary[-1],timestamp=primary[-1].timestamp+300),)
        joined=join_completed_timeframes(primary,htf,htf)
        self.assertTrue(all(row["structure_15m"] is None for row in joined))

    def test_exit_disabled_preserves_phase25_actions(self):
        features=compute_market_structure(candles([10,15,12,11]),self.cfg)
        actions=("WAIT","BUY","BUY","SELL")
        self.assertEqual(apply_structure_exit(actions,features,StructureExitConfig("fixed")),actions)
        bars=tuple(PortfolioBar(x.close_timestamp,x.open,x.high,x.low,x.close) for x in candles([10,15,12,11]))
        config=PortfolioConfig(stop_loss_pct=50,take_profit_pct=50,max_holding_bars=10)
        self.assertEqual(simulate_portfolio(bars,actions,config),
                         simulate_portfolio(bars,apply_structure_exit(actions,features),config))

    def test_early_atr_and_volume_features_are_explicitly_unavailable(self):
        first=compute_market_structure(candles([10,11]),self.cfg)[0]
        self.assertIsNone(first.atr);self.assertIsNone(first.relative_volume)
        self.assertIsNone(first.displacement);self.assertIsNone(first.selling_exhaustion_proxy)

    def test_cutoff_and_historical_book_boundary(self):
        with self.assertRaisesRegex(ValueError,"INVALID_CONTAMINATED"):
            StructureCandle(DEVELOPMENT_CUTOFF,1,2,1,1.5,1)
        self.assertFalse(hasattr(compute_market_structure(candles([10,11]),self.cfg)[-1],"order_book"))

    def test_hard_shadow_only(self):
        engine=BrianEngine({"shadow_only":False});self.assertTrue(engine.shadow_only)
        self.assertFalse({"create_order","execute_order","place_order"}&set(dir(engine)))

    def test_phase27_specialists_abstain_without_critical_features(self):
        feature=compute_market_structure(candles([10,11]),self.cfg)[0]
        self.assertEqual(market_structure_specialist(feature).action,"WAIT")
        self.assertEqual(dip_specialist(feature).action,"WAIT")

    def test_bos_appears_only_after_level_confirmation(self):
        rows=(
            StructureCandle(1_600_000_000,10,11,9,10,100),
            StructureCandle(1_600_000_300,11,15,10,12,100),
            StructureCandle(1_600_000_600,11,12,10,11,100),
            StructureCandle(1_600_000_900,15,17,14,16.5,100),
        )
        features=compute_market_structure(rows,StructureConfig(left_bars=1,right_bars=1,atr_period=2,volume_period=2))
        self.assertFalse(features[1].bullish_bos)
        self.assertFalse(features[2].bullish_bos)
        self.assertTrue(features[3].bullish_bos)

if __name__=="__main__":unittest.main()
