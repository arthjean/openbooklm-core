#!/usr/bin/env bash
# Verify the generated contract artifacts match their Rust sources (US-010).
#
#   ./scripts/check-contracts.sh
#
# Four things must hold, in this order:
#
#   1. contracts/*.json regenerate byte-identically from backend/src.
#   2. Regenerating twice produces the same bytes (determinism, not luck).
#   3. packages/sdk-ts/src/generated/* regenerates byte-identically from those
#      artifacts.
#   4. The SDK typechecks and its contract tests pass against the shared golden
#      fixtures.
#
# Nothing here writes to the working tree except through the generators
# themselves, and every generated file is restored before the script exits, so a
# failing run leaves no half-updated artifact behind.

set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT"

RED=$'\e[31m'; GRN=$'\e[32m'; YEL=$'\e[33m'; OFF=$'\e[0m'
failures=0

fail() { printf '%sFAIL%s %s\n' "$RED" "$OFF" "$1"; failures=$((failures + 1)); }
ok()   { printf '%s ok %s %s\n' "$GRN" "$OFF" "$1"; }
note() { printf '%sNOTE%s %s\n' "$YEL" "$OFF" "$1"; }

GENERATED=(
  contracts/openapi.json
  contracts/core-constants.json
  packages/sdk-ts/src/generated/openapi.ts
  packages/sdk-ts/src/generated/catalog.ts
)

SNAPSHOT="$(mktemp -d)"
restore() {
  for f in "${GENERATED[@]}"; do
    if [[ -f "$SNAPSHOT/$(basename "$f")" ]]; then
      cp "$SNAPSHOT/$(basename "$f")" "$f"
    fi
  done
  rm -rf "$SNAPSHOT"
}
trap restore EXIT

for f in "${GENERATED[@]}"; do
  [[ -f "$f" ]] || { fail "$f is missing — run the generators and commit it"; exit 1; }
  cp "$f" "$SNAPSHOT/$(basename "$f")"
done

printf 'Contract check\n\n'

# --- 1 + 2: Rust artifacts, generated twice ---------------------------------

printf 'Generating Rust contract artifacts...\n'
(cd backend && cargo run --quiet --bin contracts) >/dev/null

for f in contracts/openapi.json contracts/core-constants.json; do
  if diff -q "$SNAPSHOT/$(basename "$f")" "$f" >/dev/null; then
    ok "$f matches backend/src"
  else
    fail "$f is stale. Run 'cd backend && cargo run --bin contracts' and commit the diff."
    diff -u "$SNAPSHOT/$(basename "$f")" "$f" | head -40 || true
  fi
done

FIRST_RUN="$(mktemp -d)"
cp contracts/openapi.json "$FIRST_RUN/openapi.json"
cp contracts/core-constants.json "$FIRST_RUN/core-constants.json"
(cd backend && cargo run --quiet --bin contracts) >/dev/null
if diff -q "$FIRST_RUN/openapi.json" contracts/openapi.json >/dev/null \
  && diff -q "$FIRST_RUN/core-constants.json" contracts/core-constants.json >/dev/null; then
  ok "generation is deterministic across two consecutive runs"
else
  fail "generation is not deterministic — two consecutive runs differ"
fi
rm -rf "$FIRST_RUN"

# --- 3: SDK generated sources ------------------------------------------------

if ! command -v bun >/dev/null 2>&1; then
  note "bun is not installed — skipping the SDK half of the check"
else
  printf '\nGenerating SDK sources...\n'
  (cd packages/sdk-ts && bun install --frozen-lockfile >/dev/null 2>&1 || bun install >/dev/null 2>&1)
  (cd packages/sdk-ts && bun run generate) >/dev/null

  for f in packages/sdk-ts/src/generated/openapi.ts packages/sdk-ts/src/generated/catalog.ts; do
    if diff -q "$SNAPSHOT/$(basename "$f")" "$f" >/dev/null; then
      ok "$f matches contracts/"
    else
      fail "$f is stale. Run 'cd packages/sdk-ts && bun run generate' and commit the diff."
      diff -u "$SNAPSHOT/$(basename "$f")" "$f" | head -40 || true
    fi
  done

  # --- 4: the SDK compiles and agrees with the golden fixtures ---------------

  printf '\nChecking the SDK...\n'
  if (cd packages/sdk-ts && bun run typecheck) >/dev/null 2>&1; then
    ok "packages/sdk-ts typechecks"
  else
    fail "packages/sdk-ts does not typecheck"
    (cd packages/sdk-ts && bun run typecheck) 2>&1 | head -20 || true
  fi

  if (cd packages/sdk-ts && bun run test) >/dev/null 2>&1; then
    ok "packages/sdk-ts contract tests pass"
  else
    fail "packages/sdk-ts contract tests fail"
    (cd packages/sdk-ts && bun run test) 2>&1 | tail -30 || true
  fi
fi

printf '\n'
if [[ $failures -gt 0 ]]; then
  printf '%s%d contract check(s) failed.%s\n' "$RED" "$failures" "$OFF"
  exit 1
fi
printf '%sContracts are in sync.%s\n' "$GRN" "$OFF"
