"""Offline observation report. Inputs: exported decisions + Binance USD-M 1m bars.
No DB/network writes, no trade decisions. python review.py input.json > report.json
All timestamps are UTC milliseconds or ISO strings; bars use t, ct, o, h, l, c.
"""
import json
import math
import sys
from datetime import datetime


def ms(value):
    return int(value) if isinstance(value, (int, float)) else int(datetime.fromisoformat(value.replace('Z', '+00:00')).timestamp() * 1000)


def review(data):
    start, end, asof = (ms(data[k]) for k in ('start', 'end', 'as_of'))
    if not start < end <= asof:
        raise ValueError('Invalid report window')
    # Export must include earlier session decisions to identify first occurrences.
    if data.get('complete_session_history') is not True:
        raise ValueError('Complete session history is required for episode deduplication')
    bars = data['bars']
    for i, b in enumerate(bars):
        if not all(math.isfinite(float(b[k])) for k in ('t', 'ct', 'o', 'h', 'l', 'c')):
            raise ValueError('Invalid bar')
        if b['t'] % 60000 or b['ct'] != b['t'] + 59999 or b['l'] <= 0 or b['h'] < max(b['o'], b['c']) or b['l'] > min(b['o'], b['c']):
            raise ValueError('Invalid OHLC or timeframe')
        if i and b['t'] <= bars[i-1]['t']:
            raise ValueError('Bars must be unique and ordered')
    if data.get('symbol') != 'ETHUSDT' or data.get('market_source') != 'BINANCE_USDM_PERP':
        raise ValueError('Market identity must match DIP')
    seen, output = set(), []
    for d in sorted(data['decisions'], key=lambda x: ms(x['decision_at'])):
        key = (d['session_id'], d['episode_id'])
        if key in seen:
            continue
        seen.add(key)
        at = ms(d['decision_at'])
        if not start <= at < end:
            continue
        if d['symbol'] != data['symbol'] or d['evidence'].get('market_source') != data['market_source']:
            raise ValueError('Decision market mismatch')
        if d['direction'] not in ('UP', 'DOWN'):
            continue
        entry, cost = float(d['entry_price']), float(d['evidence']['cost_bps'])
        if not math.isfinite(entry) or entry <= 0 or not math.isfinite(cost) or cost < 0:
            raise ValueError('Invalid entry or cost')
        sign = 1 if d['direction'] == 'UP' else -1
        known = d.get('resolved_at') and ms(d['resolved_at']) <= asof
        row = {'episode_id': d['episode_id'], 'decision_at': d['decision_at'],
               'direction': d['direction'], 'veto': d['evidence'].get('veto', []),
               'recorded_hit': d.get('hit') if known else None,
               'target_history': d['evidence'].get('level_observation'), 'horizons': {}}
        # Exclude the formation candle: its pre-decision high/low are unknowable.
        first = (at // 60000 + 1) * 60000
        for minutes in (5, 15, 30):
            last = first + minutes * 60000
            selected = [b for b in bars if first <= b['t'] < last and b['ct'] < asof]
            if last > asof:
                result = {'status': 'PENDING'}
            elif [b['t'] for b in selected] != list(range(first, last, 60000)):
                result = {'status': 'MISSING_DATA'}
            else:
                high, low = max(b['h'] for b in selected), min(b['l'] for b in selected)
                ret = sign * (selected[-1]['c'] / entry - 1) * 10000
                result = {'status': 'OBSERVED', 'window_start': first, 'window_end': last,
                          'directional_close_bps': ret,
                          'favorable_excursion_bps': max(0, sign * ((high if sign == 1 else low) / entry - 1) * 10000),
                          'adverse_excursion_bps': max(0, -sign * ((low if sign == 1 else high) / entry - 1) * 10000),
                          'cost_hurdle_bps': cost, 'close_exceeds_cost': ret > cost}
            row['horizons'][str(minutes)] = result
        output.append(row)
    return {'mode': 'OBSERVATION_ONLY', 'episodes': output,
            'limitations': ['Closed full bars after decision; formation minute excluded.',
                           'Excursions do not establish intrabar event order or executable fills.',
                           'Cost comparison is diagnostic, not realized P&L; entry may already include slippage.',
                           'No automatic veto override, target replacement or fee adjustment.']}


if __name__ == '__main__':
    with open(sys.argv[1]) as source:
        print(json.dumps(review(json.load(source)), ensure_ascii=False, indent=2))
