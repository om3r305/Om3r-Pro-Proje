// Operator-owned acceptance tests. The model never receives this file.
const { test } = require('node:test');
const assert = require('node:assert/strict');
const { assessEvidence } = require(process.env.BRIAN_CANARY_CANDIDATE || '/candidate/candidate.cjs');
const now = Date.parse('2026-10-19T00:00:00Z');
const row = (asset_id, age, price = 100, cost_bps = 3) =>
  ({ asset_id, observed_at: new Date(now - age).toISOString(), price, cost_bps });
test('returns current finite evidence with measured age', () => {
  assert.deepEqual(assessEvidence([row('BTC', 1000)], now), [{ ...row('BTC', 1000), age_ms: 1000 }]);
});
test('exact age boundary is inclusive; future and stale observations are excluded', () => {
  assert.deepEqual(assessEvidence([row('A', 120000), row('B', 120001), row('C', -1)], now).map(x => x.asset_id), ['A']);
});
test('cost is required and must be finite, numeric and nonnegative', () => {
  const invalid = [null, undefined, NaN, Infinity, -1, '3'];
  assert.deepEqual(assessEvidence(invalid.map((value, i) => ({ ...row(String(i), 0), cost_bps: value })), now), []);
  assert.equal(assessEvidence([row('A', 0, 100, 0)], now)[0].cost_bps, 0);
});
test('invalid prices and dates cannot become evidence', () => {
  const rows = [0, -1, NaN, Infinity, '100', null].map((v, i) => row(String(i), 0, v));
  rows.push({ ...row('bad-date', 0), observed_at: 'invalid' });
  assert.deepEqual(assessEvidence(rows, now), []);
});
test('newest valid observation wins; output is sorted by asset', () => {
  assert.deepEqual(assessEvidence([row('Z', 10), row('A', 200), row('A', 5, 42), row('A', 1, -1)], now),
    [{ ...row('A', 5, 42), age_ms: 5 }, { ...row('Z', 10), age_ms: 10 }]);
});
test('equal-time duplicates preserve first valid input and do not mutate input', () => {
  const rows = [row('B', 5, 41), row('B', 5, 99)];
  const before = structuredClone(rows);
  rows.forEach(Object.freeze); Object.freeze(rows);
  assert.equal(assessEvidence(rows, now)[0].price, 41);
  assert.deepEqual(rows, before);
});
test('invalid container, decision time and age bound fail closed', () => {
  for (const rows of [null, {}, 'rows']) assert.deepEqual(assessEvidence(rows, now), []);
  for (const at of [null, '2026-10-19', NaN, Infinity]) assert.deepEqual(assessEvidence([row('A', 0)], at), []);
  for (const age of [-1, NaN, Infinity, '10']) assert.deepEqual(assessEvidence([row('A', 0)], now, age), []);
});
test('malformed records are ignored; explicit zero maximum age is respected', () => {
  assert.deepEqual(assessEvidence([null, {}, { ...row('', 0) }, row('A', 1), row('B', 0)], now, 0),
    [{ ...row('B', 0), age_ms: 0 }]);
});
