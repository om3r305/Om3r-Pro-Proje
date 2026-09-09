import copy
import unittest
from review import review


class ReviewTests(unittest.TestCase):
    def setUp(self):
        self.data = dict(start=60000, end=120000, as_of=2100000,
                         complete_session_history=True, symbol='ETHUSDT', market_source='BINANCE_USDM_PERP',
                         decisions=[dict(session_id='s', episode_id='e', decision_at=61000,
                                         symbol='ETHUSDT', direction='UP', entry_price=100,
                                         evidence=dict(cost_bps=22, market_source='BINANCE_USDM_PERP'))],
                         bars=[dict(t=t, ct=t+59999, o=100, h=102, l=99, c=101) for t in range(60000, 2100000, 60000)])

    def test_formation_bar_excluded_and_direction(self):
        self.data['bars'][0]['h'] = 900
        row = review(self.data)['episodes'][0]['horizons']['5']
        self.assertAlmostEqual(row['favorable_excursion_bps'], 200)
        self.assertAlmostEqual(row['adverse_excursion_bps'], 100)
        self.assertTrue(row['close_exceeds_cost'])
        self.data['decisions'][0]['direction'] = 'DOWN'
        row = review(self.data)['episodes'][0]['horizons']['5']
        self.assertAlmostEqual(row['favorable_excursion_bps'], 100)
        self.assertAlmostEqual(row['adverse_excursion_bps'], 200)
        self.assertFalse(row['close_exceeds_cost'])

    def test_missing_and_pending(self):
        self.data['bars'].pop(2)
        self.data['as_of'] = 600000
        rows = review(self.data)['episodes'][0]['horizons']
        self.assertEqual(rows['5']['status'], 'MISSING_DATA')
        self.assertEqual(rows['15']['status'], 'PENDING')

    def test_prior_episode_excluded(self):
        old = copy.deepcopy(self.data['decisions'][0])
        old['decision_at'] = 1000
        self.data['decisions'].append(old)
        self.assertEqual(review(self.data)['episodes'], [])

    def test_future_resolution_hidden(self):
        self.data['decisions'][0].update(hit=True, resolved_at=2200000)
        self.assertIsNone(review(self.data)['episodes'][0]['recorded_hit'])

    def test_wrong_market_rejected(self):
        self.data['market_source'] = 'SPOT'
        with self.assertRaises(ValueError):
            review(self.data)

    def test_duplicate_bars_rejected(self):
        self.data['bars'].insert(1, self.data['bars'][0])
        with self.assertRaises(ValueError):
            review(self.data)


if __name__ == '__main__':
    unittest.main()
