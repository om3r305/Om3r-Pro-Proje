#!/usr/bin/env node
'use strict';
const { spawnSync } = require('node:child_process');
const fs = require('node:fs');
const path = require('node:path');

// Owned by the operator, not by the candidate being evaluated.
const baseline = [
  'tests/open-world-cycle.test.cjs', 'tests/open-world-research.test.cjs',
  'tests/alpha-price-path-bridge.test.cjs', 'tests/alpha-audit-cost.test.cjs',
  'tests/operator-state.test.cjs', 'tests/research-scheduler.test.cjs',
  'tests/derivatives-universe.test.cjs', 'tests/direct-wire-fallback.test.cjs',
  'tests/main-multiasset-pipeline.test.cjs', 'tests/frontier-reliability.test.cjs',
  'tests/shadow-scorecard.test.cjs', 'tests/shadow-scorecard-page-scope.test.cjs',
  'tests/engineer-node-tests.test.cjs',
  'tests/engineering-room-model.test.cjs',
  'tests/engineer-provider-canary.test.cjs',
  'tests/engineer-preflight.test.cjs',
  'tests/engineer-provider-mode.test.cjs',
  'tests/engineer-evaluator-boundary.test.cjs',
  'tests/engineer-checkpoint.test.cjs',
  'tests/cron-auth-contract.test.cjs', 'tests/world-dossier.test.cjs',
];

function selectTests(changed, required = baseline) {
  return [...new Set([...required, ...changed.filter(file => /\.(test|spec)\.(cjs|mjs|js)$/.test(file))])].sort();
}

function runTests(files, cwd = process.cwd()) {
  const root = fs.realpathSync(cwd);
  let passed = 0;
  for (const file of files) {
    const absolute = path.resolve(root, file);
    if (!absolute.startsWith(root + path.sep) || fs.lstatSync(absolute).isSymbolicLink() ||
        !fs.realpathSync(absolute).startsWith(root + path.sep)) throw Error(`Unsafe test path: ${file}`);
    const result = spawnSync(process.execPath, ['--test', '--test-reporter=tap', absolute],
      { cwd: root, encoding: 'utf8', timeout: 120_000, maxBuffer: 16 * 1024 * 1024 });
    process.stdout.write(result.stdout || '');
    process.stderr.write(result.stderr || '');
    if (result.error || result.signal || result.status !== 0) throw Error(`Node test failed: ${file}`);
    const count = Number((result.stdout.match(/^# pass (\d+)\s*$/m) || [])[1]);
    if (!(count > 0)) throw Error(`No passing tests executed: ${file}`);
    passed += count;
  }
  return { files: files.length, passed };
}

function main() {
  const base = process.argv[2];
  if (base === '--baseline') {
    console.log(`BRIAN_NODE_TESTS=${JSON.stringify(runTests(selectTests([])))}`);
    return;
  }
  if (!/^[a-f0-9]{40}$/i.test(base || '')) throw Error('Exact base commit SHA required');
  const diff = spawnSync('git', ['diff', '--name-only', '--diff-filter=ACMR', '-z', `${base}...HEAD`, '--'],
    { encoding: 'utf8', maxBuffer: 1024 * 1024 });
  if (diff.error || diff.status !== 0) throw Error('Cannot enumerate candidate tests');
  const result = runTests(selectTests(diff.stdout.split('\0').filter(Boolean)));
  console.log(`BRIAN_NODE_TESTS=${JSON.stringify(result)}`);
}
if (require.main === module) {
  try { main(); } catch (error) { console.error(String(error)); process.exitCode = 1; }
}
module.exports = { selectTests, runTests, baseline };
