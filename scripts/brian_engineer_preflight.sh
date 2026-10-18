#!/usr/bin/env bash
set -euo pipefail

# An exact operator-dispatched request uses the existing manual claim path.
# It does not open the autonomous queue; claim still verifies OIDC, concurrency,
# request eligibility, shadow-only scope and the approval invariant server-side.
if [[ -n "${REQUEST_ID:-}" ]]; then
  [[ "$REQUEST_ID" =~ ^[a-f0-9]{64}$ ]] || { echo 'Invalid exact request id' >&2; exit 1; }
  echo 'should_run=true' >> "$GITHUB_OUTPUT"
  echo 'Brian Engineer: exact manual request; server-side claim gates still apply.' >> "$GITHUB_STEP_SUMMARY"
  exit 0
fi

defer() {
  echo 'should_run=false' >> "$GITHUB_OUTPUT"
  echo 'Brian Engineer preflight unavailable; no task claimed.' >> "$GITHUB_STEP_SUMMARY"
  exit 0
}
oidc_json="$(curl -fsS --max-time 20 -H "Authorization: bearer $ACTIONS_ID_TOKEN_REQUEST_TOKEN" "${ACTIONS_ID_TOKEN_REQUEST_URL}&audience=${ENGINEERING_AUDIENCE}")" || defer
token="$(printf '%s' "$oidc_json" | jq -er '.value')" || defer
response="$(curl -fsS --max-time 20 -X POST "$ENGINEERING_GATEWAY" -H "Authorization: Bearer $token" -H 'content-type: application/json' --data '{"action":"preflight"}')" || defer
should_run="$(printf '%s' "$response" | jq -er '.should_run | if type=="boolean" then tostring else error("invalid flag") end')" || defer
echo "should_run=$should_run" >> "$GITHUB_OUTPUT"
echo "Brian Engineer automatic preflight: should_run=$should_run" >> "$GITHUB_STEP_SUMMARY"
