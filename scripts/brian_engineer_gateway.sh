#!/usr/bin/env bash
set -euo pipefail

: "${ENGINEERING_GATEWAY:?ENGINEERING_GATEWAY is required}"
: "${ENGINEERING_AUDIENCE:?ENGINEERING_AUDIENCE is required}"
: "${ACTIONS_ID_TOKEN_REQUEST_TOKEN:?GitHub OIDC request token is required}"
: "${ACTIONS_ID_TOKEN_REQUEST_URL:?GitHub OIDC request URL is required}"

curl_retry() {
  curl -fsS \
    --retry 5 \
    --retry-delay 2 \
    --retry-max-time 35 \
    --retry-all-errors \
    --connect-timeout 10 \
    --max-time 45 \
    "$@"
}

mode="${1:-}"
oidc_json="$(curl_retry -H "Authorization: bearer $ACTIONS_ID_TOKEN_REQUEST_TOKEN" "${ACTIONS_ID_TOKEN_REQUEST_URL}&audience=${ENGINEERING_AUDIENCE}")"
token="$(printf '%s' "$oidc_json" | jq -er '.value')"

case "$mode" in
  event)
    run_id="${2:?run_id required}"
    event_kind="${3:?event_kind required}"
    phase="${4:?phase required}"
    commit_sha="${5:-}"
    payload="${6:-}"
    if [ -z "$payload" ]; then payload='{}'; fi
    printf '%s' "$payload" | jq -e 'type=="object"' >/dev/null
    body="$(jq -nc \
      --arg action event \
      --arg run_id "$run_id" \
      --arg event_kind "$event_kind" \
      --arg phase "$phase" \
      --arg commit_sha "$commit_sha" \
      --argjson payload "$payload" \
      '{action:$action,run_id:$run_id,event_kind:$event_kind,phase:$phase,commit_sha:(if $commit_sha=="" then null else $commit_sha end),payload:$payload}')"
    response="$(curl_retry -X POST "$ENGINEERING_GATEWAY" -H "Authorization: Bearer $token" -H 'content-type: application/json' --data "$body")"
    printf '%s\n' "$response"
    test "$(printf '%s' "$response" | jq -r '.status')" = 'RECORDED'
    ;;
  measure)
    run_id="${2:?run_id required}"
    commit_sha="${3:?commit_sha required}"
    measurement="${4:?measurement JSON required}"
    printf '%s' "$measurement" | jq -e 'type=="object"' >/dev/null
    body="$(jq -nc \
      --arg action measure \
      --arg run_id "$run_id" \
      --arg commit_sha "$commit_sha" \
      --argjson measurement "$measurement" \
      '{action:$action,run_id:$run_id,commit_sha:$commit_sha,measurement:$measurement}')"
    response="$(curl_retry -X POST "$ENGINEERING_GATEWAY" -H "Authorization: Bearer $token" -H 'content-type: application/json' --data "$body")"
    printf '%s\n' "$response"
    test "$(printf '%s' "$response" | jq -r '.status')" = 'MEASURED'
    ;;
  *)
    echo 'usage: brian_engineer_gateway.sh event <run_id> <event_kind> <phase> [commit_sha] [payload_json]' >&2
    echo '   or: brian_engineer_gateway.sh measure <run_id> <commit_sha> <measurement_json>' >&2
    exit 2
    ;;
esac
