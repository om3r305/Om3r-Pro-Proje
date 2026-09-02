import unittest

from brian2026.engine import BrianEngine
from brian2026.portfolio import ChronologicalOutcomeQueue, DEVELOPMENT_CUTOFF, PortfolioBar, PortfolioConfig, StatefulPortfolioSimulator, simulate_portfolio
from brian2026.robustness import EvidencePolicy, assert_development_only, development_candidate, purged_temporal_yearly_splits
from brian2026.phase25_experiment import HORIZON, _samples


def bar(i, close=100.0, high=None, low=None):
    return PortfolioBar(1_600_000_000+i*300, close, high if high is not None else close+0.2,
                        low if low is not None else close-0.2, close)


class StatefulPortfolioTests(unittest.TestCase):
    def cfg(self, **updates):
        values=dict(starting_equity=10_000,fixed_notional=1_000,max_position_notional=1_000,
                    max_equity_fraction=.2,stop_loss_pct=10,take_profit_pct=10,max_holding_bars=100,
                    cooldown_bars=0,fee_bps=10,assumed_spread_bps=2,slippage_bps=1)
        values.update(updates);return PortfolioConfig(**values)

    def test_repeated_buy_and_sell_do_not_stack(self):
        for action,side in (("BUY","LONG"),("SELL","SHORT")):
            sim=StatefulPortfolioSimulator(self.cfg())
            sim.step(bar(0),action);first=sim.position
            sim.step(bar(1),action);sim.step(bar(2),action)
            self.assertIs(sim.position,first);self.assertEqual(sim.state,side);self.assertEqual(sim.duplicates,2)

    def test_reversal_closes_before_reopening(self):
        sim=StatefulPortfolioSimulator(self.cfg(reversal_enabled=True,cooldown_bars=0))
        sim.step(bar(0),"BUY");sim.step(bar(1,101),"SELL")
        self.assertEqual(sim.trades[0].reason,"REVERSAL");self.assertEqual(sim.position.side,"SHORT")
        self.assertEqual(len(sim.trades),1)

    def test_fees_only_on_transitions(self):
        sim=StatefulPortfolioSimulator(self.cfg())
        sim.step(bar(0),"WAIT");self.assertEqual(sim.fees,0)
        sim.step(bar(1),"BUY");entry_fees=sim.fees
        sim.step(bar(2),"BUY");sim.step(bar(3),"WAIT");self.assertEqual(sim.fees,entry_fees)
        sim.finish(bar(3));self.assertGreater(sim.fees,entry_fees)

    def test_chronology_and_max_one_position(self):
        sim=StatefulPortfolioSimulator(self.cfg());sim.step(bar(1),"BUY")
        with self.assertRaises(ValueError):sim.step(bar(1),"SELL")
        self.assertIsNotNone(sim.position)

    def test_sizing_obeys_cash_and_equity_bounds(self):
        sim=StatefulPortfolioSimulator(self.cfg(fixed_notional=50_000,max_position_notional=8_000,max_equity_fraction=.1))
        sim.step(bar(0),"BUY");self.assertLessEqual(sim.position.notional,1_000);self.assertGreaterEqual(sim.cash,0)

    def test_cooldown_blocks_full_next_bar(self):
        sim=StatefulPortfolioSimulator(self.cfg(cooldown_bars=1,max_holding_bars=1))
        sim.step(bar(0),"BUY");sim.step(bar(1),"BUY")
        self.assertIsNone(sim.position);self.assertEqual(len(sim.trades),1)
        sim.step(bar(2),"BUY");self.assertIsNone(sim.position)
        sim.step(bar(3),"BUY");self.assertIsNotNone(sim.position)

    def test_force_close_and_signal_trade_count_difference(self):
        bars=[bar(i,100+i*.1) for i in range(5)];result=simulate_portfolio(bars,["BUY"]*5,self.cfg())
        self.assertEqual(result.trades[-1].reason,"FORCED_END");self.assertEqual(result.signals,5)
        self.assertEqual(result.entries,1);self.assertGreater(result.signals,result.entries)

    def test_quality_segment_break_forces_flat(self):
        sim=StatefulPortfolioSimulator(self.cfg());sim.step(bar(0),"BUY");sim.break_segment(bar(0))
        self.assertIsNone(sim.position);self.assertEqual(sim.trades[-1].reason,"QUALITY_EXCLUSION")

    def test_equity_curve_and_drawdown_are_sequential(self):
        bars=[bar(0,100),bar(1,95),bar(2,90)];result=simulate_portfolio(bars,["BUY","WAIT","WAIT"],self.cfg())
        stamps=[p.timestamp for p in result.equity_curve];self.assertEqual(stamps,sorted(stamps))
        self.assertGreater(result.max_drawdown,0);self.assertAlmostEqual(result.ending_equity-result.starting_equity,result.net_pnl)

    def test_2026_bar_is_rejected(self):
        with self.assertRaisesRegex(ValueError,"INVALID_CONTAMINATED"):
            PortfolioBar(DEVELOPMENT_CUTOFF,100,101,99,100)

    def test_adaptive_outcome_is_unavailable_before_resolution(self):
        queue=ChronologicalOutcomeQueue();queue.schedule(1_600_001_000,{"won":True})
        self.assertEqual(queue.release(1_600_000_999),())
        self.assertEqual(queue.release(1_600_001_000),({"won":True},))


class RobustnessAndEvidenceTests(unittest.TestCase):
    def test_sample_target_must_exist_before_construction(self):
        import numpy as np
        times=np.arange(10,dtype=float);features={"x":times};labels=np.zeros(10,dtype=int);future=np.zeros(10)
        with self.assertRaises(IndexError):_samples(np.asarray([len(times)-HORIZON]),labels,future,features,times,"d",("x",))

    def test_purged_temporal_yearly_robustness_purge_and_embargo(self):
        times=[1_600_000_000+i*3600 for i in range(12)];groups=[i//3 for i in range(12)]
        splits=purged_temporal_yearly_splits(times,groups,purge_seconds=3600,embargo_seconds=3600)
        self.assertEqual(len({s.split_id for s in splits}),len(splits))
        for split in splits:
            self.assertFalse(set(split.train_indices)&set(split.test_indices))
            lo=min(times[i] for i in split.test_indices);hi=max(times[i] for i in split.test_indices)
            self.assertTrue(all(not lo-3600<=times[i]<=hi+3600 for i in split.train_indices))

    def test_purged_temporal_yearly_robustness_rejects_2026(self):
        with self.assertRaisesRegex(ValueError,"INVALID_CONTAMINATED"):
            purged_temporal_yearly_splits([1_600_000_000,DEVELOPMENT_CUTOFF],[0,1])

    def test_every_development_context_rejects_2026(self):
        for context in ("training","validation","development test","calibration","ablation","purged temporal yearly robustness","portfolio replay","champion evaluation"):
            with self.subTest(context=context),self.assertRaisesRegex(ValueError,"INVALID_CONTAMINATED"):
                assert_development_only([DEVELOPMENT_CUTOFF],context)

    def test_tiny_lucky_and_low_coverage_are_rejected(self):
        folds=[{"trades":2,"expectancy":10,"profit_factor":9,"max_drawdown_pct":1}]*3
        result=development_candidate(folds,coverage=.001,calibration_samples=10,stress_net_pnl=100)
        self.assertEqual(result["status"],"INSUFFICIENT_EVIDENCE")
        self.assertIn("minimum total trades not met",result["reasons"]);self.assertIn("insufficient coverage",result["reasons"])

    def test_cost_stress_can_reject(self):
        p=EvidencePolicy(min_total_trades=1,min_trades_per_fold=1,min_coverage=.01,max_wait_rate=.99,min_positive_expectancy_folds=1,min_profit_factor=1,max_drawdown_pct=100,min_calibration_samples=1)
        folds=[{"trades":100,"expectancy":1,"profit_factor":1.2,"max_drawdown_pct":2}]
        result=development_candidate(folds,coverage=.5,calibration_samples=100,stress_net_pnl=-1,policy=p)
        self.assertIn("cost-stress survivability failed",result["reasons"])

    def test_hard_shadow_and_no_execution_surface(self):
        import tempfile
        with tempfile.TemporaryDirectory() as td:
            engine=BrianEngine({"shadow_only":False},runtime_root=td);self.assertTrue(engine.shadow_only)
            forbidden={"buy","sell","place_order","create_order","execute_order","submit_order"}
            self.assertFalse(forbidden&set(dir(engine)))


if __name__=="__main__":unittest.main()
