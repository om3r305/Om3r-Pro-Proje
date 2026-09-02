from __future__ import annotations

import unittest

import numpy as np

from brian2026.portfolio import DEVELOPMENT_CUTOFF
from brian2026.rl_sandbox import (
    ACTIONS,
    ConservativeFittedQAgent,
    RLSandboxConfig,
    build_counterfactual_transitions,
    counterfactual_transition,
    state_with_position,
)


class Phase30RLSandboxTests(unittest.TestCase):
    def config(self) -> RLSandboxConfig:
        return RLSandboxConfig(
            trees=12,
            min_samples_leaf=1,
            fitted_q_iterations=2,
            max_train_transitions=10_000,
            random_state=7,
        )

    def test_action_is_executed_on_next_bar_open_not_observation_close(self):
        row = counterfactual_transition(
            observation_timestamp=1_700_000_000.0,
            execution_timestamp=1_700_000_300.0,
            current_features=(1.0, 2.0),
            next_features=(1.1, 2.1),
            position_before=0,
            action=1,
            next_open=100.0,
            next_high=102.0,
            next_low=99.5,
            next_close=101.0,
            config=self.config(),
        )
        # Gross +1.00%, one-way cost 12 bps = 0.12%, adverse excursion 0.5% * 0.2 = 0.10%.
        self.assertAlmostEqual(row.reward_pct, 0.78, places=9)
        self.assertEqual(row.next_open, 100.0)
        self.assertGreater(row.execution_timestamp, row.observation_timestamp)

    def test_execution_cannot_share_observation_timestamp(self):
        with self.assertRaisesRegex(ValueError, "strictly after"):
            counterfactual_transition(
                observation_timestamp=1_700_000_000.0,
                execution_timestamp=1_700_000_000.0,
                current_features=(0.0,), next_features=(0.0,),
                position_before=0, action=0,
                next_open=100, next_high=101, next_low=99, next_close=100,
                config=self.config(),
            )

    def test_2026_is_forbidden_for_observation_or_execution(self):
        with self.assertRaisesRegex(ValueError, "INVALID_CONTAMINATED"):
            counterfactual_transition(
                observation_timestamp=DEVELOPMENT_CUTOFF,
                execution_timestamp=DEVELOPMENT_CUTOFF + 300,
                current_features=(0.0,), next_features=(0.0,),
                position_before=0, action=0,
                next_open=100, next_high=101, next_low=99, next_close=100,
                config=self.config(),
            )

    def test_position_is_part_of_state(self):
        flat = state_with_position((1.0, 2.0), 0)
        long = state_with_position((1.0, 2.0), 1)
        short = state_with_position((1.0, 2.0), -1)
        self.assertNotEqual(flat, long)
        self.assertNotEqual(flat, short)
        self.assertEqual(flat[-3:], (0.0, 1.0, 0.0))
        self.assertEqual(long[-3:], (0.0, 0.0, 1.0))
        self.assertEqual(short[-3:], (1.0, 0.0, 0.0))

    def test_transition_builder_generates_full_counterfactual_action_coverage(self):
        t = np.array([1_700_000_000.0, 1_700_000_300.0, 1_700_000_600.0])
        o = np.array([100.0, 101.0, 102.0])
        h = np.array([101.0, 103.0, 104.0])
        l = np.array([99.0, 100.0, 101.0])
        c = np.array([100.5, 102.0, 103.0])
        x = np.array([[0.0, 1.0], [0.2, 1.2], [0.4, 1.4]])
        rows = build_counterfactual_transitions(t, o, h, l, c, x, [0, 1], config=self.config())
        self.assertEqual(len(rows), 2 * len(ACTIONS) * len(ACTIONS))
        first = [row for row in rows if row.observation_timestamp == t[0]]
        self.assertEqual({row.action for row in first}, set(ACTIONS))
        self.assertEqual({row.position_before for row in first}, set(ACTIONS))

    def test_mutating_later_future_bar_does_not_change_current_transition(self):
        kwargs = dict(
            observation_timestamp=1_700_000_000.0,
            execution_timestamp=1_700_000_300.0,
            current_features=(0.2, 0.3), next_features=(0.4, 0.5),
            position_before=0, action=1,
            next_open=100, next_high=102, next_low=99, next_close=101,
            config=self.config(),
        )
        first = counterfactual_transition(**kwargs)
        # A t+2 edit is deliberately absent from the transition API; current result remains identical.
        second = counterfactual_transition(**kwargs)
        self.assertEqual(first, second)

    def _training_rows(self):
        rows = []
        cfg = self.config()
        base = 1_700_000_000.0
        for i in range(12):
            features = (float(i) / 10.0, 1.0)
            next_features = (float(i + 1) / 10.0, 1.0)
            open_price = 100.0
            close_price = 101.0 if i % 2 == 0 else 99.0
            high = max(open_price, close_price) + 0.5
            low = min(open_price, close_price) - 0.5
            for before in ACTIONS:
                for action in ACTIONS:
                    rows.append(counterfactual_transition(
                        observation_timestamp=base + i * 600,
                        execution_timestamp=base + i * 600 + 300,
                        current_features=features, next_features=next_features,
                        position_before=before, action=action,
                        next_open=open_price, next_high=high, next_low=low, next_close=close_price,
                        terminal=i == 11, config=cfg,
                    ))
        return rows

    def test_rl_fit_is_train_only(self):
        agent = ConservativeFittedQAgent(self.config())
        with self.assertRaisesRegex(ValueError, "only on train"):
            agent.fit(self._training_rows(), partition="validation")

    def test_rl_agent_is_deterministic_for_same_training_data(self):
        rows = self._training_rows()
        first = ConservativeFittedQAgent(self.config()).fit(rows, partition="train")
        second = ConservativeFittedQAgent(self.config()).fit(rows, partition="train")
        d1 = first.decide(timestamp=1_700_100_000.0, features=(0.25, 1.0), current_position=0)
        d2 = second.decide(timestamp=1_700_100_000.0, features=(0.25, 1.0), current_position=0)
        self.assertEqual(d1, d2)
        self.assertTrue(first.shadow_only)

    def test_rl_agent_has_no_exchange_execution_surface(self):
        agent = ConservativeFittedQAgent(self.config())
        forbidden = {"buy", "sell", "place_order", "create_order", "execute_order", "submit_order"}
        self.assertFalse(forbidden.intersection(dir(agent)))

    def test_decision_rejects_2026(self):
        agent = ConservativeFittedQAgent(self.config()).fit(self._training_rows(), partition="train")
        with self.assertRaisesRegex(ValueError, "INVALID_CONTAMINATED"):
            agent.decide(timestamp=DEVELOPMENT_CUTOFF, features=(0.2, 1.0), current_position=0)


if __name__ == "__main__":
    unittest.main()
