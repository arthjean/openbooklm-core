#!/usr/bin/env bash
#
# Open-core boundary check (US-003, EP-001).
#
# Enforces docs/open-core-boundary.md mechanically:
#
#   1. Every tracked file matches exactly one classification rule.
#      Zero matches -> unclassified, fails. Two or more conflicting matches ->
#      ambiguous, fails. The PRD forbids resolving ambiguity implicitly.
#   2. No PUBLIC Rust module imports Clerk, Stripe, PostHog, Resend, a private
#      module or a private generated artifact.
#   3. Every TRANSITIONAL rule carries a blocking story ID, and the count is
#      reported. It must reach zero before US-014 can complete.
#
# Usage:
#   ./scripts/check-open-core-boundary.sh            # check
#   ./scripts/check-open-core-boundary.sh --list     # print the classification
#
# Exit codes: 0 clean, 1 violation, 2 script error.

set -uo pipefail

cd "$(dirname "${BASH_SOURCE[0]}")/.." || exit 2

RED=$'\033[0;31m'; YEL=$'\033[0;33m'; GRN=$'\033[0;32m'; DIM=$'\033[2m'; OFF=$'\033[0m'
if [[ ! -t 1 ]]; then RED=""; YEL=""; GRN=""; DIM=""; OFF=""; fi

failures=0
fail() { printf '%sFAIL%s %s\n' "$RED" "$OFF" "$1"; failures=$((failures + 1)); }
note() { printf '%s     %s%s\n' "$DIM" "$1" "$OFF"; }

# =============================================================================
# Classification rules
#
# Each entry is "<glob>|<class>|<note>". Order does not matter: every rule is
# evaluated against every file so that conflicts are detected rather than
# shadowed by the first match.
#
# Classes:
#   public       exported to the public repository
#   private      never exported
#   transitional exported only after the named story removes its SaaS coupling
#
# A transitional note MUST start with a US-0NN story ID.
# =============================================================================

RULES=(
  # --- backend: crate-level -------------------------------------------------
  "backend/Cargo.toml|private|the hosted manifest; Cargo.public.toml is exported in its place"
  "backend/Cargo.public.toml|public|exported as backend/Cargo.toml"
  "backend/Cargo.lock|private|the hosted dependency graph; the public repository resolves and commits its own"
  "backend/deny.toml|public|"
  "backend/rustfmt.toml|public|"
  "backend/.cargo/*|public|"
  "backend/assets/demo/*|public|synthetic demo content, no PII"
  "backend/Dockerfile|private|the hosted image; docker/Dockerfile builds the reference server"
  "backend/entrypoint.sh|private|the hosted entrypoint; docker/entrypoint.sh is the public one"
  "docker/*|public|reference image, entrypoint and Compose stack"

  # --- backend: runtime roots -----------------------------------------------
  "backend/src/lib.rs|public|"
  "backend/src/db.rs|public|"
  "backend/src/types.rs|public|"
  "backend/src/validation.rs|public|"
  "backend/src/xml.rs|public|"
  "backend/src/error.rs|public|"
  "backend/src/main.rs|private|SaaS composition root; US-013 adds the public reference binary"
  "backend/src/bin/contracts.rs|public|"
  "backend/src/bin/openbooklm-server.rs|public|the reference self-hosted binary"
  "backend/src/config.rs|private|SaaS configuration wrapper around the public CoreConfig"
  "backend/src/app_state.rs|private|SaaS state wrapper around the public CoreState"
  "backend/src/auth.rs|private|Clerk JWT verification; injects the public Principal"

  # --- backend: the open-source core ----------------------------------------
  "backend/src/core/*|public|"

  # --- backend: private SaaS adapters ---------------------------------------

  # --- backend: api ---------------------------------------------------------
  "backend/src/api/mod.rs|public|"
  "backend/src/saas/*|private|Clerk, Stripe, PostHog and Resend adapters plus every hosted route, entity and repository"
  "backend/src/api/common.rs|public|"
  "backend/src/api/health.rs|public|"
  "backend/src/api/suggestions.rs|public|"
  "backend/src/api/rag_logs.rs|public|"
  "backend/src/api/notebooks.rs|public|"
  "backend/src/api/sources.rs|public|"
  "backend/src/api/notes.rs|public|"
  "backend/src/api/openapi.rs|public|"
  "backend/src/api/memory.rs|public|"
  "backend/src/api/settings.rs|public|"
  "backend/src/api/chat/*|public|"

  # --- backend: clients -----------------------------------------------------
  "backend/src/clients/mod.rs|public|"
  "backend/src/clients/anthropic.rs|public|"
  "backend/src/clients/mistral.rs|public|"
  "backend/src/clients/mistral_ocr.rs|public|"
  "backend/src/clients/openai.rs|public|"
  "backend/src/clients/openai_compat.rs|public|"
  "backend/src/clients/voyage.rs|public|"
  "backend/src/clients/voyage_rate_limiter.rs|public|"
  "backend/src/clients/firecrawl.rs|public|"
  "backend/src/clients/youtube.rs|public|"
  "backend/src/clients/models.rs|public|"
  "backend/src/clients/llm_router.rs|public|"
  "backend/src/clients/circuit_breaker.rs|public|"
  "backend/src/clients/retry.rs|public|"
  "backend/src/clients/resilience.rs|public|"
  "backend/src/clients/metrics.rs|public|"

  # --- backend: entities ----------------------------------------------------
  "backend/src/entities/mod.rs|public|"
  "backend/src/entities/notebook.rs|public|"
  "backend/src/entities/source.rs|public|"
  "backend/src/entities/chunk.rs|public|"
  "backend/src/entities/note.rs|public|"
  "backend/src/entities/chat_message.rs|public|"
  "backend/src/entities/notebook_memory.rs|public|"
  "backend/src/entities/rag_log.rs|public|"
  "backend/src/entities/ocr_cache.rs|public|"
  "backend/src/entities/account.rs|public|"
  "backend/src/entities/account_settings.rs|public|"
  "backend/src/repositories/account.rs|public|"

  # --- backend: llm, middleware --------------------------------------------
  "backend/src/llm/*|public|"
  "backend/src/middleware/*|public|"

  # --- backend: repositories -----------------------------------------------
  "backend/src/repositories/mod.rs|public|"
  "backend/src/repositories/traits.rs|public|"
  "backend/src/repositories/notebook.rs|public|"
  "backend/src/repositories/source.rs|public|"
  "backend/src/repositories/chunk.rs|public|"
  "backend/src/repositories/note.rs|public|"
  "backend/src/repositories/chat.rs|public|"
  "backend/src/repositories/memory.rs|public|"
  "backend/src/repositories/search.rs|public|"
  "backend/src/repositories/rag_log.rs|public|"
  "backend/src/repositories/ocr_cache.rs|public|"

  # --- backend: services ----------------------------------------------------
  "backend/src/services/mod.rs|public|"
  "backend/src/services/rag/*|public|"
  "backend/src/services/memory/*|public|"
  "backend/src/services/chat/*|public|"
  "backend/src/services/notebooks.rs|public|"
  "backend/src/services/sources.rs|public|"
  "backend/src/services/processor.rs|public|"
  "backend/src/services/content_cleaning.rs|public|"
  "backend/src/services/embeddings.rs|public|"
  "backend/src/services/suggestions.rs|public|"
  "backend/src/services/source_events.rs|public|"
  "backend/src/services/source_processing.rs|public|"

  # --- backend: tests -------------------------------------------------------
  "backend/tests/contract_baseline.rs|public|"
  "backend/tests/latency_baseline.rs|public|"
  "backend/tests/common/*|private|dead helper: no test declares mod common, and it imports the hosted openbooklm::config"
  "backend/tests/p0_integration.rs|public|"
  "backend/tests/ocr_integration.rs|public|"
  "backend/tests/reference_server.rs|public|"
  "backend/tests/saas_ocr_limits.rs|private|hosted OCR allowance"
  "backend/tests/saas_p0_integration.rs|private|hosted provisioning and webhook limits"

  # --- migrations -----------------------------------------------------------
  "backend/migration-core/Cargo.toml|public|the public core schema crate"
  "backend/migration-core/src/*|public|"
  "backend/migration/src/saas_track/*|private|"
  "backend/migration/Cargo.toml|private|hosted migration crate; the public one is backend/migration-core"
  "backend/migration/src/main.rs|private|SaaS migration composition root, like backend/src/main.rs"
  "backend/migration/src/lib.rs|private|composes all three tracks; the public crate owns the core track alone"
  "backend/migration/src/m20*.rs|private|legacy applied history, immutable; US-012 adds the public core baseline"

  # --- contracts and public governance --------------------------------------
  "contracts/*|public|"
  "scripts/*|public|"
  "docs/contracts/*|public|"
  "docs/migrations.md|public|"
  "docs/upgrading.md|public|"
  "docs/public/*|public|public README, release notes and root dotfiles"
  "docs/self-hosting/*|public|exported as .env.example and the self-hosting guide"
  "docs/open-core-boundary.md|public|"
  "docs/security/publication-audit.md|public|"
  "LICENSE|public|"
  "CONTRIBUTING.md|public|"
  "SECURITY.md|public|"
  "CODE_OF_CONDUCT.md|public|"
  "TRADEMARK.md|public|"
  # --- packages: the public TypeScript SDK (US-010) --------------------------
  "packages/sdk-ts/*|public|"

  "README.md|private|the hosted README; docs/public/README.md is exported in its place"
  ".codex/*|private|local coding-agent state and hooks"
  ".editorconfig|public|"
  ".gitignore|private|the hosted ignore rules, which cover the Next.js app; docs/public/gitignore is exported in its place"
  ".dockerignore|public|"

  # --- private: frontend and SaaS operations --------------------------------
  "frontend/*|private|"
  "tasks/*|private|"
  ".github/workflows/ci.yml|private|the hosted pipeline"
  ".github/workflows/public-ci.yml|public|"
  ".github/workflows/public-release.yml|public|"
  ".env.example|private|hosted sample; docs/self-hosting/env.core.example is exported as the public one"
  "docker-compose.yml|private|the hosted stack; docker/docker-compose.yml is the public one"
  "Caddyfile|private|"
  "lefthook.yml|private|"
  "package.json|private|"
  "CLAUDE.md|private|"
  "IMPORTANT_FUTUR.md|private|"
  "OpenbookLM-FAQ.md|private|"
  "PRODUCT_HUNT_STRATEGY.md|private|"
)

# =============================================================================
# Forbidden imports in public Rust modules
#
# "<extended regex>@@<what it is>" — `@@` because the regexes contain `|`.
# =============================================================================

FORBIDDEN=(
  "clerk_rs@@Clerk identity SDK"
  "(crate|openbooklm)::saas@@private SaaS adapter module"
  "(crate|openbooklm)::auth@@Clerk auth module"
  "(crate|openbooklm)::(config|app_state)@@hosted config or state wrapper"
  "\bstripe\b@@Stripe billing SDK"
  "(crate|openbooklm)::services::billing@@billing policy module"
  "crate::clients::posthog@@PostHog analytics client"
  "crate::clients::resend@@Resend email client"
  "crate::services::email@@email service"
  "crate::services::onboarding@@SaaS onboarding service"
  "crate::api::(billing|webhooks|feedback|micro_feedback|newsletter|stats)@@private API module"
  "crate::entities::(subscription|usage|feedback|micro_feedback|newsletter_subscriber|processed_event)@@private entity"
  "crate::repositories::(Subscription|Usage|SeaOrmSubscription|SeaOrmUsage)@@private repository"
)

# =============================================================================
# Pass 1 — classify every tracked file
# =============================================================================

declare -A CLASS_OF
transitional_count=0
unclassified=()
ambiguous=()

classify() {
  local file="$1" matched_class="" matched_glob="" conflict=""
  local rule glob class
  for rule in "${RULES[@]}"; do
    glob="${rule%%|*}"
    class="${rule#*|}"; class="${class%%|*}"
    # shellcheck disable=SC2053  # glob matching is intentional
    if [[ "$file" == $glob ]]; then
      if [[ -n "$matched_class" && "$matched_class" != "$class" ]]; then
        conflict="$matched_glob ($matched_class) vs $glob ($class)"
      fi
      matched_class="$class"
      matched_glob="$glob"
    fi
  done
  if [[ -n "$conflict" ]]; then
    ambiguous+=("$file: $conflict")
    return
  fi
  if [[ -z "$matched_class" ]]; then
    unclassified+=("$file")
    return
  fi
  CLASS_OF["$file"]="$matched_class"
}

# Tracked files plus new, non-ignored files. A file must be classified before it
# is committed, not after: an unclassified new file is exactly the case that
# leaks private material into the public export.
mapfile -t TRACKED < <(git ls-files --cached --others --exclude-standard | sort -u)
if [[ ${#TRACKED[@]} -eq 0 ]]; then
  printf 'no tracked files; run from inside the repository\n' >&2
  exit 2
fi

for file in "${TRACKED[@]}"; do
  classify "$file"
done

if [[ "${1:-}" == "--list" ]]; then
  for file in "${!CLASS_OF[@]}"; do
    printf '%s\t%s\n' "${CLASS_OF[$file]}" "$file"
  done | sort
  exit 0
fi

printf 'Open-core boundary check\n'
printf '%s%d tracked files, %d rules%s\n\n' "$DIM" "${#TRACKED[@]}" "${#RULES[@]}" "$OFF"

for entry in "${ambiguous[@]}"; do
  fail "ambiguous classification: $entry"
  note "a file must match exactly one class; make the intended rule explicit"
done

for file in "${unclassified[@]}"; do
  fail "unclassified: $file"
  note "add a rule to scripts/check-open-core-boundary.sh and document it in docs/open-core-boundary.md"
done

# =============================================================================
# Pass 2 — forbidden imports in public modules
# =============================================================================

for file in "${!CLASS_OF[@]}"; do
  [[ "${CLASS_OF[$file]}" == "public" ]] || continue
  [[ "$file" == *.rs ]] || continue
  [[ -f "$file" ]] || continue
  for entry in "${FORBIDDEN[@]}"; do
    pattern="${entry%%@@*}"
    label="${entry#*@@}"
    # Strip comments and doc comments: a mention in prose is not a dependency.
    if sed 's://.*::' "$file" | grep -Eq "$pattern"; then
      fail "public module depends on private code: $file"
      note "matched \`$pattern\` ($label)"
    fi
  done
done

# =============================================================================
# Pass 3 — transitional exceptions must be justified and must shrink
# =============================================================================

declare -A SEEN_GLOB
for rule in "${RULES[@]}"; do
  glob="${rule%%|*}"
  rest="${rule#*|}"
  class="${rest%%|*}"
  reason="${rest#*|}"

  if [[ -n "${SEEN_GLOB[$glob]:-}" ]]; then
    fail "duplicate rule for glob: $glob"
  fi
  SEEN_GLOB["$glob"]=1

  case "$class" in
    public | private) ;;
    transitional)
      transitional_count=$((transitional_count + 1))
      if [[ ! "$reason" =~ ^US-[0-9]{3} ]]; then
        fail "transitional rule without a blocking story ID: $glob"
        note "format: US-0NN <what removes the coupling>"
      fi
      ;;
    *)
      fail "unknown class \`$class\` for glob: $glob"
      ;;
  esac
done

# =============================================================================
# Report
# =============================================================================

public_count=0; private_count=0; trans_files=0
for file in "${!CLASS_OF[@]}"; do
  case "${CLASS_OF[$file]}" in
    public) public_count=$((public_count + 1)) ;;
    private) private_count=$((private_count + 1)) ;;
    transitional) trans_files=$((trans_files + 1)) ;;
  esac
done

printf '\n  public       %4d files\n' "$public_count"
printf '  private      %4d files\n' "$private_count"
printf '  transitional %4d files (%d rules)\n' "$trans_files" "$transitional_count"

if [[ $transitional_count -gt 0 ]]; then
  printf '\n%sNOTE%s %d transitional rule(s) remain. US-014 cannot complete until this reaches 0.\n' \
    "$YEL" "$OFF" "$transitional_count"
fi

if [[ $failures -gt 0 ]]; then
  printf '\n%s%d boundary violation(s).%s See docs/open-core-boundary.md.\n' "$RED" "$failures" "$OFF"
  exit 1
fi

printf '\n%sBoundary clean.%s\n' "$GRN" "$OFF"
exit 0
