#!/usr/bin/env bash
#
# Build the public repository snapshot (US-014, EP-004).
#
# The public repository is created from a reviewed export, never by flipping
# this repository's visibility: its history contains private product material
# and secret-shaped fixtures, and history cannot be un-published.
#
# The export is a pure function of the boundary classification:
#
#   ./scripts/check-open-core-boundary.sh --list   ->  class<TAB>path
#
# Only `public` files are copied. A few are renamed on the way out, because a
# file cannot be both the private repository's manifest and the public one's
# (see RENAMES). Everything else keeps its path, so a public issue's file
# reference is valid here and vice versa.
#
# Three gates run before the snapshot is declared usable, and any of them
# aborts the export:
#
#   1. the boundary check itself must pass, with zero transitional rules;
#   2. no exported file may contain a private marker (Clerk, Stripe, PostHog,
#      Resend, the production domain, a private path);
#   3. the exported tree must contain every file the public build needs.
#
# Usage:
#   ./scripts/export-public-repo.sh <target-dir> [--force]
#
# The target directory must not exist unless --force is given, in which case it
# is emptied first. This script never touches a remote: publishing the snapshot
# is a separate, deliberate act.
#
# Exit codes: 0 clean, 1 export blocked, 2 usage error.

set -uo pipefail

cd "$(dirname "${BASH_SOURCE[0]}")/.." || exit 2

RED=$'\033[0;31m'; YEL=$'\033[0;33m'; GRN=$'\033[0;32m'; DIM=$'\033[2m'; OFF=$'\033[0m'
if [[ ! -t 1 ]]; then RED=""; YEL=""; GRN=""; DIM=""; OFF=""; fi

failures=0
fail() { printf '%sFAIL%s %s\n' "$RED" "$OFF" "$1"; failures=$((failures + 1)); }
note() { printf '%s     %s%s\n' "$DIM" "$1" "$OFF"; }
info() { printf '%s\n' "$1"; }

TARGET="${1:-}"
FORCE="${2:-}"

if [[ -z "$TARGET" || "$TARGET" == -* ]]; then
  printf 'usage: %s <target-dir> [--force]\n' "$0" >&2
  exit 2
fi

if [[ -e "$TARGET" ]]; then
  if [[ "$FORCE" != "--force" ]]; then
    printf 'refusing to write into existing %s (pass --force to replace it)\n' "$TARGET" >&2
    exit 2
  fi
  if [[ ! -d "$TARGET" ]]; then
    printf '%s exists and is not a directory\n' "$TARGET" >&2
    exit 2
  fi
  # Emptied rather than removed: the caller may have created it with the
  # permissions and ownership they want.
  find "$TARGET" -mindepth 1 -maxdepth 1 -exec rm -rf {} + || exit 2
fi

# =============================================================================
# Path rewrites
#
# "<source>|<destination>" — a file whose public name differs from its private
# one. Each exists because one repository's root file cannot serve both.
# =============================================================================

RENAMES=(
  "backend/Cargo.public.toml|backend/Cargo.toml"
  "docs/public/README.md|README.md"
  "docs/self-hosting/env.core.example|.env.example"
  "docs/public/gitignore|.gitignore"
)

rename_of() {
  local src="$1" entry
  for entry in "${RENAMES[@]}"; do
    if [[ "${entry%%|*}" == "$src" ]]; then
      printf '%s' "${entry#*|}"
      return
    fi
  done
  printf '%s' "$src"
}

# =============================================================================
# Files the public build cannot start without
#
# A boundary rule can be deleted by accident; this list turns that into an
# export failure instead of a public repository that does not compile.
# =============================================================================

REQUIRED=(
  "backend/Cargo.toml"
  "backend/src/lib.rs"
  "backend/src/bin/openbooklm-server.rs"
  "backend/src/core/router.rs"
  "backend/src/core/identity.rs"
  "backend/migration-core/Cargo.toml"
  "backend/migration-core/src/lib.rs"
  "contracts/openapi.json"
  "packages/sdk-ts/package.json"
  "LICENSE"
  "README.md"
  "CONTRIBUTING.md"
  "SECURITY.md"
  ".env.example"
)

# =============================================================================
# Private markers
#
# Grepped over every exported file. `docs/open-core-boundary.md` and the
# governance documents legitimately name the vendors they exclude, so prose is
# exempted by path rather than by weakening the pattern.
# =============================================================================

MARKERS=(
  "clerk_rs@@Clerk SDK"
  "CLERK_SECRET_KEY@@Clerk credential"
  "STRIPE_SECRET_KEY@@Stripe credential"
  "async-stripe@@Stripe SDK"
  "POSTHOG_API_KEY@@PostHog credential"
  "RESEND_API_KEY@@Resend credential"
  "openbooklm\.fr@@production domain"
  "crate::saas@@private module"
  "openbooklm::saas@@private module"
  "sk_live@@live secret key"
  "sk_test_@@Stripe test key"
  "whsec_@@webhook secret"
)

# Paths whose subject *is* the boundary, so naming a vendor is the point. The
# governance documents describe what is excluded, and the checks themselves are
# the list of things to exclude.
MARKER_EXEMPT=(
  "scripts/check-open-core-boundary.sh"
  "scripts/export-public-repo.sh"
  "scripts/check-public-manifest.py"
  "scripts/secret-scan-allowlist.json"
  "scripts/scan-git-history-secrets.py"
  "docs/open-core-boundary.md"
  "docs/security/publication-audit.md"
  "CONTRIBUTING.md"
  "SECURITY.md"
  "README.md"
  "docs/migrations.md"
  "docs/upgrading.md"
  "docs/contracts/known-drift.md"
  "docs/contracts/sdk-replacement-map.md"
  "docs/contracts/event-protocol-v1.md"
)

is_exempt() {
  local file="$1" entry
  for entry in "${MARKER_EXEMPT[@]}"; do
    [[ "$file" == "$entry" ]] && return 0
  done
  return 1
}

# =============================================================================
# Gate 1 — the boundary must be clean, with no transitional rule left
# =============================================================================

info "Public export -> $TARGET"
printf '\n1. boundary\n'

boundary_output="$(./scripts/check-open-core-boundary.sh 2>&1)"
boundary_status=$?
if [[ $boundary_status -ne 0 ]]; then
  fail "boundary check failed; the classification is not export-ready"
  printf '%s\n' "$boundary_output"
  exit 1
fi

transitional_rules="$(printf '%s\n' "$boundary_output" \
  | sed -n 's/.*transitional *[0-9]* files (\([0-9]*\) rules).*/\1/p')"
if [[ -z "$transitional_rules" ]]; then
  fail "could not read the transitional rule count from the boundary check"
  exit 1
fi
if [[ "$transitional_rules" -ne 0 ]]; then
  fail "$transitional_rules transitional rule(s) remain"
  note "US-014 cannot complete until every transitional rule is resolved"
  exit 1
fi
note "boundary clean, 0 transitional rules"

# =============================================================================
# Copy the public files
# =============================================================================

printf '\n2. copy\n'

mapfile -t PUBLIC_FILES < <(./scripts/check-open-core-boundary.sh --list \
  | awk -F'\t' '$1 == "public" { print $2 }' | sort)

if [[ ${#PUBLIC_FILES[@]} -eq 0 ]]; then
  fail "the boundary check classified no file as public"
  exit 1
fi

copied=0
for src in "${PUBLIC_FILES[@]}"; do
  [[ -f "$src" ]] || continue
  dest="$(rename_of "$src")"
  mkdir -p "$TARGET/$(dirname "$dest")" || exit 2
  cp -p "$src" "$TARGET/$dest" || exit 2
  copied=$((copied + 1))
done
note "$copied file(s) copied"

# The private manifest must never reach the public tree under its own name.
rm -f "$TARGET/backend/Cargo.public.toml"

# Cargo.lock is seeded, not exported. The private lock is the hosted
# resolution: same versions, more crates. Copying it and letting Cargo resolve
# against the public manifest prunes every hosted entry while pinning the
# shared ones to the versions this repository actually tests — a public repo
# that resolved from scratch would drift from what was verified here.
#
# The pruned lock still goes through the private-marker gate below, so a
# failure to prune blocks the export rather than shipping `async-stripe` in it.
if cp -p backend/Cargo.lock "$TARGET/backend/Cargo.lock" 2>/dev/null; then
  if (cd "$TARGET/backend" && cargo metadata --format-version 1 --offline >/dev/null 2>&1); then
    note "Cargo.lock seeded from the private resolution and pruned"
  else
    rm -f "$TARGET/backend/Cargo.lock"
    note "could not prune Cargo.lock offline; the public repository will resolve its own"
  fi
fi

# =============================================================================
# Gate 2 — no private marker survived
# =============================================================================

printf '\n3. private markers\n'

marker_hits=0
while IFS= read -r -d '' file; do
  rel="${file#"$TARGET"/}"
  is_exempt "$rel" && continue
  # Skip binaries and fonts: a false positive on random bytes helps no one.
  grep -Iq . "$file" 2>/dev/null || continue
  # A mention in prose is not a dependency: Rust line comments are stripped
  # before matching. `//` preceded by `:` is a URL scheme, not a comment —
  # stripping there would hide `https://<production domain>` from the very gate
  # that exists to catch it.
  if [[ "$rel" == *.rs ]]; then
    scan_source() { sed -E 's@(^|[^:])//.*@\1@' "$1"; }
  else
    scan_source() { cat "$1"; }
  fi
  for entry in "${MARKERS[@]}"; do
    pattern="${entry%%@@*}"
    label="${entry#*@@}"
    if scan_source "$file" | grep -Eq "$pattern"; then
      fail "private marker in exported file: $rel"
      note "matched \`$pattern\` ($label)"
      marker_hits=$((marker_hits + 1))
    fi
  done
done < <(find "$TARGET" -type f -print0)

if [[ $marker_hits -eq 0 ]]; then
  note "no private marker found"
fi

# =============================================================================
# Gate 3 — the export is complete
# =============================================================================

printf '\n4. completeness\n'

for required in "${REQUIRED[@]}"; do
  if [[ ! -f "$TARGET/$required" ]]; then
    fail "the export is missing $required"
    note "check its rule in scripts/check-open-core-boundary.sh"
  fi
done

# The hosted crates must not be here at all.
for forbidden_path in "backend/src/main.rs" "backend/src/saas" "backend/src/auth.rs" \
                      "backend/src/config.rs" "backend/src/app_state.rs" \
                      "backend/migration" "frontend" "tasks"; do
  if [[ -e "$TARGET/$forbidden_path" ]]; then
    fail "hosted path reached the export: $forbidden_path"
  fi
done

if [[ $failures -eq 0 ]]; then
  note "every required file is present, no hosted path leaked"
fi

# =============================================================================
# Report
# =============================================================================

if [[ $failures -gt 0 ]]; then
  printf '\n%sPublic export blocked by private content.%s %d problem(s).\n' "$RED" "$OFF" "$failures"
  printf 'The snapshot in %s is NOT safe to publish.\n' "$TARGET"
  exit 1
fi

printf '\n%sExport clean.%s %d files in %s\n' "$GRN" "$OFF" "$copied" "$TARGET"
printf '\n%sNext, by hand:%s\n' "$YEL" "$OFF"
printf '  1. cd %s && cargo test && cargo clippy --all-targets -- -D warnings\n' "$TARGET/backend"
printf '  2. review the diff against the previous public release\n'
printf '  3. git init, commit as a reviewed snapshot, and only then create the remote\n'
exit 0
