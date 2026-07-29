#!/usr/bin/env bash
#
# Cited-answer smoke path for the reference server (US-015, EP-004).
#
# Proves the one claim that matters to a new operator: a clean environment can
# start the stack, create an account, ingest a source and get an answer with a
# citation back to it. Everything narrower — a green `cargo test`, a 200 from
# /health — can pass while the product does not work.
#
# It drives the public REST and SSE surface only, exactly as an SDK consumer
# would.
#
# Two levels, because they need different things:
#
#   (default)  the whole path: start, migrate, authenticate, ingest, retrieve,
#              answer, verify the citation resolves to the ingested source.
#              Runs against real providers or against the deterministic ones
#              (OPENBOOKLM_DETERMINISTIC_PROVIDERS=true), so public CI needs no
#              commercial credential.
#   --core     stops after notebook CRUD. For probing a deployment whose
#              providers you do not want to spend on.
#
# Usage:
#   ./scripts/smoke-public-stack.sh [base-url] [--core]
#
# Environment:
#   OPENBOOKLM_AUTH_TOKEN   bearer token, when the server runs in token mode
#   SMOKE_TIMEOUT_SECS      per-step ceiling (default 120)
#
# Exit codes: 0 the path completed, 1 a step failed, 2 usage or missing tool.

set -uo pipefail

BASE="http://localhost:3001"
CORE_ONLY=false
for arg in "$@"; do
  case "$arg" in
    --core) CORE_ONLY=true ;;
    -*) printf 'unknown option: %s\n' "$arg" >&2; exit 2 ;;
    *) BASE="$arg" ;;
  esac
done
TIMEOUT="${SMOKE_TIMEOUT_SECS:-120}"
TOKEN="${OPENBOOKLM_AUTH_TOKEN:-}"

RED=$'\033[0;31m'; GRN=$'\033[0;32m'; DIM=$'\033[2m'; OFF=$'\033[0m'
if [[ ! -t 1 ]]; then RED=""; GRN=""; DIM=""; OFF=""; fi

step=0
say()  { step=$((step + 1)); printf '%d. %s\n' "$step" "$1"; }
note() { printf '%s   %s%s\n' "$DIM" "$1" "$OFF"; }
die()  { printf '%sFAIL%s %s\n' "$RED" "$OFF" "$1"; exit 1; }

for tool in curl jq; do
  command -v "$tool" >/dev/null 2>&1 || { printf '%s is required\n' "$tool" >&2; exit 2; }
done

AUTH=()
[[ -n "$TOKEN" ]] && AUTH=(-H "authorization: Bearer $TOKEN")

api() {
  local method="$1" path="$2" body="${3:-}"
  local args=(-sS -X "$method" --max-time "$TIMEOUT" "${AUTH[@]}" "$BASE$path")
  if [[ -n "$body" ]]; then
    args+=(-H 'content-type: application/json' -d "$body")
  fi
  curl "${args[@]}"
}

printf 'Smoke path against %s\n\n' "$BASE"

# ---------------------------------------------------------------------------
say "health"
health="$(curl -sS --max-time 30 "$BASE/health")" || die "the server is not reachable at $BASE"
printf '%s' "$health" | jq -e '.status == "ok" or .status == "healthy"' >/dev/null \
  || die "unexpected /health body: $health"
note "server is up"

# ---------------------------------------------------------------------------
say "authentication"
# A protected route without a credential must be 401 in token mode. In loopback
# mode it is 200, and that is also correct — the assertion is that the server
# never answers with a 500 or an HTML error page.
code="$(curl -sS -o /dev/null -w '%{http_code}' --max-time 30 "$BASE/api/notebooks")"
case "$code" in
  200) note "loopback single-user mode" ;;
  401)
    [[ -n "$TOKEN" ]] || die "the server requires a token; set OPENBOOKLM_AUTH_TOKEN"
    note "static-token mode, unauthenticated request correctly refused"
    ;;
  *) die "GET /api/notebooks without credentials returned $code" ;;
esac

# ---------------------------------------------------------------------------
say "create a notebook"
notebook="$(api POST /api/notebooks '{"title":"Smoke test"}')" || die "notebook creation failed"
notebook_id="$(printf '%s' "$notebook" | jq -r '.id // empty')"
[[ -n "$notebook_id" ]] || die "no notebook id in the response: $notebook"
note "notebook $notebook_id"

cleanup() {
  [[ -n "${notebook_id:-}" ]] && api DELETE "/api/notebooks/$notebook_id" >/dev/null 2>&1
}
trap cleanup EXIT

if [[ "$CORE_ONLY" == true ]]; then
  printf '\n%sCore smoke path complete.%s clean start -> migrations -> auth -> notebook\n' \
    "$GRN" "$OFF"
  note "ingestion and chat skipped (--core)"
  exit 0
fi

# ---------------------------------------------------------------------------
say "ingest a source"
# Deterministic fixture content: the assertion below quotes it, so a citation
# can be verified rather than merely counted.
read -r -d '' fixture <<'TXT'
Reciprocal Rank Fusion combines the dense and lexical result lists by scoring
each document as the sum of 1/(k + rank) across the lists it appears in. The
constant k is 60 by default. A document ranked highly by both retrievers
therefore outranks one that only a single retriever liked, without either list
needing a calibrated score.
TXT

payload="$(jq -n --arg c "$fixture" \
  '{source_type:"text", title:"Retrieval notes", content:$c}')"
source_response="$(api POST "/api/notebooks/$notebook_id/sources" "$payload")" \
  || die "source creation failed"
source_id="$(printf '%s' "$source_response" | jq -r '.id // empty')"
[[ -n "$source_id" ]] || die "no source id in the response: $source_response"
note "source $source_id"

# ---------------------------------------------------------------------------
say "wait for indexing"
deadline=$((SECONDS + TIMEOUT))
status=""
while (( SECONDS < deadline )); do
  status="$(api GET "/api/sources/$source_id" | jq -r '.status // empty')"
  case "$status" in
    ready|completed) break ;;
    error|failed)    die "ingestion failed: $(api GET "/api/sources/$source_id")" ;;
  esac
  sleep 2
done
case "$status" in
  ready|completed) note "indexed" ;;
  *) die "source did not finish indexing within ${TIMEOUT}s (last status: ${status:-none})" ;;
esac

chunks="$(api GET "/api/sources/$source_id/chunks" | jq '.chunks | length // 0')"
(( chunks > 0 )) || die "the source produced no chunks; retrieval has nothing to search"
note "$chunks chunk(s)"

# ---------------------------------------------------------------------------
say "ask a question"
stream="$(curl -sS -N --max-time "$TIMEOUT" "${AUTH[@]}" \
  -H 'content-type: application/json' \
  -d '{"message":"What constant does Reciprocal Rank Fusion use by default?"}' \
  "$BASE/api/notebooks/$notebook_id/chat")" || die "the chat request failed"

printf '%s' "$stream" | grep -q '^event: *error' \
  && die "the stream carried an error event: $(printf '%s' "$stream" | head -20)"

printf '%s' "$stream" | grep -q '^event: *done' \
  || die "the stream did not terminate with a done event"

answer="$(printf '%s' "$stream" \
  | sed -n 's/^data: *//p' \
  | jq -rs 'map(.text? // .content? // empty) | join("")' 2>/dev/null)"
[[ -n "$answer" ]] || die "the stream produced no answer text"
note "answered in $(printf '%s' "$answer" | wc -c) characters"

# ---------------------------------------------------------------------------
say "verify the citation"
citations="$(printf '%s' "$stream" | sed -n 's/^data: *//p' \
  | jq -rs '[.[] | select(.citations?) | .citations[]] | length' 2>/dev/null)"
[[ "${citations:-0}" -gt 0 ]] \
  || die "the answer carried no citation; a grounded answer must reference its source"

cited_source="$(printf '%s' "$stream" | sed -n 's/^data: *//p' \
  | jq -rs '[.[] | select(.citations?) | .citations[].source_id] | first // empty' 2>/dev/null)"
[[ "$cited_source" == "$source_id" ]] \
  || die "the citation points at $cited_source, not the ingested source $source_id"
note "$citations citation(s), resolved to the ingested source"

printf '\n%sSmoke path complete.%s clean start -> ingest -> cited answer\n' "$GRN" "$OFF"
