# Open-core boundary

**Stories:** US-003 (EP-001) defined it; US-004 to US-008 (EP-002) moved 8 rules
out of `transitional`; US-009 to US-012 (EP-003) moved 9 more and added the
public SDK and the public migration track; US-013 to US-015 (EP-004) removed
the last 13 and added the reference server, the public manifest, the export and
the public workflows. **`transitional` is now empty.** See
`tasks/prd-open-core.md`.
**Enforced by:** `scripts/check-open-core-boundary.sh` (must exit 0 on every pull request)
**Machine-readable source:** the `RULES` array in that script. This document is
the rationale; the script is the authority. They are changed together.

Every tracked file has exactly one class. A file with no rule fails the check. A
file matching two rules with different classes fails the check. The PRD forbids
resolving ambiguity implicitly, so the check reports the conflicting globs and
refuses to pick.

## Classes

| Class | Meaning |
|---|---|
| **public** | Exported to the public repository. May not reference any private module. |
| **private** | Never exported. Free to depend on public code. |
| **transitional** | Belongs in public but still has a SaaS dependency. Carries a blocking story ID. Must reach zero before US-014. |

Current state after EP-004 (run the check for live numbers):

```
public        200 files
private       429 files
transitional    0 files (0 rules)
```

The private count is dominated by the Next.js application, which is proprietary
in full and is not part of this split's ambition.

`transitional` reaching zero is what makes the export possible: every remaining
file has a permanent home, so `scripts/export-public-repo.sh` can copy the
`public` set and refuse when anything else appears.

The check classifies tracked files **and** new non-ignored files
(`git ls-files --cached --others --exclude-standard`). A file must be classified
before it is committed: an unclassified new file is exactly the case that leaks
private material into a public export.

## The one-way dependency rule

Public code may not import private code. The reverse is fine and is the whole
point: the private SaaS composes the public core.

`scripts/check-open-core-boundary.sh` greps every **public** `.rs` file, with
comments stripped so a mention in prose is not treated as a dependency, for:

| Pattern | What it is |
|---|---|
| `clerk_rs` | Clerk identity SDK |
| `crate::auth` | Clerk JWT middleware and `AuthUser` |
| `\bstripe\b` | Stripe billing SDK |
| `crate::services::billing` | plan, limit and usage policy |
| `crate::clients::posthog` | analytics client |
| `crate::clients::resend` | transactional email client |
| `crate::services::email`, `crate::services::onboarding` | SaaS side effects |
| `crate::saas` | private adapter module (Clerk, Stripe, PostHog, Resend) |
| `crate::api::(billing\|webhooks\|feedback\|micro_feedback\|newsletter\|stats)` | private API modules |
| `crate::entities::(subscription\|usage\|feedback\|micro_feedback\|newsletter_subscriber\|processed_event)` | private tables |
| `crate::repositories::(Subscription\|Usage\|…)` | private repositories |

**This rule already found two misclassifications.** `api/notes.rs` and
`api/rag_logs.rs` were classified public on the strength of their domain, but
both extracted `AuthUser` from `crate::auth`. They were transitional until
US-005 replaced `AuthUser` with `Principal`; both are public now. That is the
check earning its keep on the day it was written.

### The `AppState` coupling, resolved

Public handlers used to take `State<AppState>`, which lives in the private
`app_state.rs`. EP-001 through EP-003 tracked that here rather than in the
check, because adding `crate::AppState` to `FORBIDDEN` would have pushed every
core handler back to `transitional` without changing a line of behaviour.

US-020 removed the last one of the same kind. `CoreState.clients.voyage` was
typed `Option<VoyageClient>`, so the public core's state struct named a
commercial vendor and retrieval could not run without that vendor's key. It is
now `Option<Arc<dyn EmbeddingProvider>>` plus `Option<Arc<dyn Reranker>>`, and
the shipped `DeterministicEmbedder` / `DeterministicLlm` let the whole
ingest-retrieve-cite path run with no provider account — which is what makes
the public CI smoke path secret-free.

US-013 removed it. Every core handler now takes `State<CoreState>`, and
`core::router::build_core_router` composes them into a `Router<CoreState>` with
no state applied and no identity middleware attached. The hosted binary layers
Clerk over it; the reference binary layers the static-token or loopback adapter.
One router, two identity models, no shared state type.

### The second mechanism: the `saas` feature

Classification says which files are exported. It cannot say whether the
remaining ones *compile* on their own — and a public repository that does not
build is worse than one that leaks a filename.

US-013 added the compile-time half. `openbooklm` has one feature, `saas`,
default-on privately and absent from `Cargo.public.toml`. `lib.rs` gates
`app_state`, `auth`, `config` and `saas` behind it, and Clerk and Stripe are
optional dependencies it enables. So:

```bash
cd backend && cargo check --no-default-features --all-targets
```

is a local proof that the core stands alone. A core file that acquires a hosted
dependency fails there, in one command, instead of in the public repository's
first CI run. `scripts/check-public-manifest.py` keeps the two manifests from
drifting apart, and `public-ci.yml` runs the core-only build on every pull
request.

## Public allowlist

**RAG and retrieval** — `services/rag/**` (chunking, embedding cache, HyDE, query
reformulation, vector store, `search/**`, RAG logging), `services/embeddings.rs`,
`repositories/{search,chunk,rag_log}.rs`.

**Document processing** — `services/processor.rs`, `services/content_cleaning.rs`,
`services/sources.rs`, `services/source_events.rs`.

**LLM providers** — `clients/{anthropic,mistral,mistral_ocr,openai,openai_compat,
voyage,voyage_rate_limiter,firecrawl,youtube,models,llm_router}.rs` and the
resilience layer `clients/{circuit_breaker,retry,resilience,metrics}.rs`.

**LLM abstraction** — all of `llm/**`: provider trait, types, prompts, citations,
token budget, SSE parsing.

**Core entities and repositories** — `entities/{notebook,source,chunk,note,
chat_message,notebook_memory,rag_log,ocr_cache}.rs` and their repositories.

**The core seams** — all of `src/core/**`: `CoreConfig`, `CoreState`,
`Principal`, `EntitlementPolicy` with its unrestricted adapter, and `EventSink`
with its no-op and tracing adapters. This is the module a private or
self-hosted composition injects into.

**Core REST and SSE handlers** — `api/{health,suggestions,common,notes,rag_logs,
sources,memory}.rs`. The notebook, settings and chat handlers join once their
remaining transitional couplings are removed.

**The public contract** — `backend/src/api/openapi.rs` (the `#[utoipa::path]`
document), `backend/src/core/catalog.rs` (limits, teaching modes, source types,
provider capabilities) and `backend/src/bin/contracts.rs` (the generator that
writes both artifacts).

**The TypeScript SDK** — all of `packages/sdk-ts/**`: generated REST types,
generated constants, the typed client, the SSE unions and their parsers, and the
contract test that checks them against the shared golden fixtures.

**The public migration track** — `backend/migration/src/core_track/**`
(`CoreMigrator` and the core baseline) and `backend/migration/src/validate.rs`
(migration-state classification). `validate.rs` takes the expected version lists
as an argument rather than importing the migrators, which is what keeps it free
of any dependency on the private SaaS track.

**Core account ownership** — `entities/{account,account_settings}.rs` and
`repositories/account.rs`: the generic account the core owns, with no email,
identity subject or plan.

**The reference server** — `backend/src/bin/openbooklm-server.rs`,
`backend/src/core/router.rs` and `backend/src/core/identity.rs`: the composition
root, the router builder and the two self-hosted identity modes.

**Self-host assets** — `docker/{Dockerfile,entrypoint.sh,docker-compose.yml}`
and `docs/self-hosting/env.core.example`, which the export publishes as
`.env.example`.

**Public governance and CI** — `docs/public/**` (the public README and release
notes), `docs/upgrading.md`, `.github/workflows/public-ci.yml` and
`.github/workflows/public-release.yml`.

**Memory** — all of `services/memory/**`.

**Middleware** — `middleware/**`: request id, security headers, rate limit,
graceful shutdown.

**Runtime roots** — `lib.rs`, `db.rs`, `types.rs`, `validation.rs`, `error.rs`.

**Contracts, tests and governance** — `contracts/**`, `backend/tests/**`,
`scripts/**`, `docs/contracts/**`, `docs/security/publication-audit.md`,
`docs/open-core-boundary.md`, `LICENSE`, `CONTRIBUTING.md`, `SECURITY.md`,
`CODE_OF_CONDUCT.md`, `TRADEMARK.md`.

**Self-host assets** — `backend/{Cargo.toml,Cargo.lock,deny.toml,rustfmt.toml}`,
`backend/.cargo/**`, `backend/assets/demo/**` (synthetic, no PII).

## Private allowlist

**The Next.js application** — all of `frontend/**`. Proprietary UI, Clerk
components, billing screens, marketing pages, blog content, email templates,
locale files. Not split, not partially exported.

**Identity** — `auth.rs`, and every Clerk claim, JWKS and session concern.

**Monetization** — `api/billing.rs`, `api/webhooks.rs`, `services/billing/**`,
`entities/{subscription,usage,processed_event}.rs`,
`repositories/{subscription,usage}.rs`.

**Analytics and email** — `clients/{posthog,resend}.rs`, `services/email.rs`,
`services/email_templates/**`, `services/onboarding.rs`.

**The SaaS adapters** — all of `src/saas/**`: the Stripe-backed
`EntitlementPolicy`, the PostHog/Resend `EventSink` consumer and the upgrade
nudge tracker. Plus `config.rs` and `app_state.rs`, which wrap the public
`CoreConfig` and `CoreState` with the commercial fields.

**Commercial feedback and growth** — `api/{feedback,micro_feedback,newsletter,
stats}.rs` and their entities.

**SaaS composition and deployment** — `main.rs`, `Caddyfile`, `lefthook.yml`,
root `package.json`.

**Product and internal context** — `CLAUDE.md`, `tasks/**`, `IMPORTANT_FUTUR.md`,
`OpenbookLM-FAQ.md`, `PRODUCT_HUNT_STRATEGY.md`.

**Legacy migrations** — `backend/migration/src/m20*.rs`. These stay private and
byte-for-byte unchanged: they are the applied history of the hosted database.
US-012 added a separate public core baseline rather than rewriting them.

**The SaaS migration track** — `backend/migration/src/saas_track/**`: the SaaS
baseline and the legacy bridge. Plus `backend/migration/{Cargo.toml,src/lib.rs,
src/main.rs}`, which compose all three tracks. They are private for the same
reason `backend/src/main.rs` is: a composition root belongs to whoever composes.
US-014 authors the public migration crate from `core_track/` alone, exactly as
US-013 authors the public binary.

**Identity and SaaS settings** — `entities/{user,user_settings,identity,
saas_account_settings}.rs`, `repositories/{user_settings,identity}.rs` and
`saas/settings.rs`: Clerk subjects, email addresses, lifecycle email state and
onboarding progress.

## Transitional exceptions

**None.** The class still exists in the check, and a new rule may use it, but it
must name the story that removes it and the export refuses to run while any
remain.

The last 13 were removed by EP-004:

| Path | Was blocked by | How it was resolved |
|---|---|---|
| `src/api/mod.rs` | US-013 | hosted handlers moved to `src/saas/api/`; the core router is `core/router.rs` |
| `src/api/notebooks.rs` | US-013 | the onboarding hook became `saas::onboarding::demo_index_middleware`, layered over the core router by the hosted composition |
| `src/services/mod.rs` | US-013 | billing, email and onboarding moved to `src/saas/` |
| `src/clients/mod.rs` | US-013 | PostHog and Resend moved to `src/saas/clients/` |
| `src/entities/mod.rs`, `src/repositories/mod.rs`, `src/repositories/traits.rs` | US-013 | hosted entities, repositories and traits moved to `src/saas/{entities,repositories}/` |
| `backend/{Dockerfile,entrypoint.sh}`, `docker-compose.yml` | US-015 | the hosted assets stayed private; `docker/**` is the public stack |
| `.github/**` | US-015 | classified per file: `ci.yml` private, `public-*.yml` public |
| `.env.example` | US-013 | stayed private; `docs/self-hosting/env.core.example` is exported as the public one |
| `README.md` | US-014 | stayed private; `docs/public/README.md` is exported as the public one |

**0 transitional rules (30 before EP-002, 22 before EP-003, 13 before EP-004).**

US-009 removed the two chat rules: `services/chat/**` no longer imports
`axum::response::sse` or the `api::chat` SSE helpers, and `api/chat/**` is now
purely the transport adapter — which is a legitimate role for a public Axum
library, and never was the coupling that mattered.

US-011 removed four: `api/settings.rs` is public now that onboarding lives in
`saas/settings.rs`, and `entities/{user,user_settings}.rs` plus
`repositories/user_settings.rs` are simply private, because the core reads
`accounts` and `account_settings` instead.

US-012 removed three by classifying the migration crate's manifest and roots as
private composition, and adding the public `core_track/`. US-013 went further
and made the core track its own crate, `backend/migration-core`: the private
`migration` crate now depends on it and re-exports `core_track` and `validate`
under their historical paths, so the hosted call sites are unchanged while the
crate boundary matches the repository boundary.

**Three rules moved from US-011 to US-013:** `entities/mod.rs`,
`repositories/mod.rs` and `repositories/traits.rs`. Their coupling is not about
data ownership — US-011 finished that — but about *module tree* organisation:
they declare and re-export private entities and traits. That is the same
coupling `services/mod.rs` and `clients/mod.rs` already carry, and US-013 owns
it for all five. Splitting them under US-011 would have moved private entities
into `saas/` for one story's benefit while leaving the identical pattern in two
other modules.
The check prints the count on every run, so the number is visible rather than
tracked in a spreadsheet.

## Frontend contract surface

`frontend/**` is private in full, so it needs no per-file classification. Two
files are nevertheless **contract** surfaces and are replaced, not exported, by
US-010 and US-018:

| File | Replaced by |
|---|---|
| `src/types/core.ts` | generated `packages/sdk-ts` REST types |
| `src/lib/api/*.ts` | generated SDK client methods |
| `src/lib/sse.ts`, the `ChatStreamEvent` union in `src/lib/api/chat.ts` | generated SDK SSE unions and parsers |

Since US-010 each of those files carries a `TRANSITIONAL` header naming its
replacement, and the substitution is enumerated type by type and method by method
in `docs/contracts/sdk-replacement-map.md`. US-018 executes it.

Their remaining divergence from the Rust definitions is catalogued in
`docs/contracts/known-drift.md` and pinned by
`frontend/src/lib/__tests__/contract-drift.test.ts`.

## Root operational assets

| Asset | Class |
|---|---|
| `LICENSE`, `CONTRIBUTING.md`, `SECURITY.md`, `CODE_OF_CONDUCT.md`, `TRADEMARK.md` | public |
| `.editorconfig`, `.gitignore`, `.dockerignore` | public |
| `.codex/**` | private local tooling state |
| `README.md`, `docker-compose.yml`, `.env.example` | private; public replacements are exported |
| `.github/workflows/ci.yml` | private |
| `.github/workflows/public-ci.yml`, `.github/workflows/public-release.yml` | public |
| `Caddyfile`, `lefthook.yml`, `package.json`, `CLAUDE.md`, `tasks/**` | private |
| `IMPORTANT_FUTUR.md`, `OpenbookLM-FAQ.md`, `PRODUCT_HUNT_STRATEGY.md` | private |

## Baseline metrics

For the PRD success table. "Core" means everything outside `services/billing/`,
`api/{billing,webhooks,feedback,micro_feedback,newsletter,stats}.rs` and
`src/saas/`, which are private by construction.

| Metric | At `4c3b06b` | After EP-002 | After EP-003 | After EP-004 |
|---|---|---|---|---|
| Core files referencing `services::billing`, `SubscriptionRepository` or `UsageRepository` | 15 | 0 | 0 | 0 |
| Core files importing PostHog, Resend or the email service | 2 | 0 | 0 | 0 |
| Public modules with a forbidden import, after classification | 0 | 0 | 0 | 0 |
| Known REST/SSE contract drift cases | 11 | 11 | 4, all frontend-side | 4, all frontend-side |
| Core tables carrying identity or campaign data | 2 | 2 | 0 | 0 |
| Transitional rules | 30 | 22 | 13 | **0** |
| Core modules naming a provider vendor by type | 11 files | 11 | 11 | **0** (US-020) |
| Commercial variables required to start the public server | 8 | 8 | 8 | **0** |
| Core edition builds and tests without the hosted feature | no | no | no | **yes, 701 tests** |

The first two reached 0 through US-006, US-007 and US-008. Drift fell through
US-009 (D-001 to D-007) and US-010 (D-008 to D-011 at the source); the four that
remain are the frontend's handwritten copies, which US-018 deletes. Identity and
campaign data left the core tables in US-011. Transitional rules must reach 0
before US-014.

## Adding a file

1. Add a rule to `RULES` in `scripts/check-open-core-boundary.sh`.
2. If it is `transitional`, start the note with the blocking `US-0NN`.
3. Add it to the matching section of this document.
4. Run `./scripts/check-open-core-boundary.sh`.

`./scripts/check-open-core-boundary.sh --list` prints the full
`class<TAB>path` classification, which is what `scripts/export-public-repo.sh`
consumes.

## Producing the export

```bash
./scripts/export-public-repo.sh /tmp/openbooklm-public
```

Three gates, any of which aborts before the snapshot is usable:

1. the boundary check passes and **zero** transitional rules remain;
2. no exported file contains a private marker — Clerk, Stripe, PostHog, Resend,
   the production domain, a private module path or a secret-shaped string.
   Rust comments are stripped first, because a mention in prose is not a
   dependency;
3. every file the public build needs is present, and no hosted path leaked.

Four files are renamed on the way out, because one repository's root file
cannot serve both:

| Private | Public |
|---|---|
| `backend/Cargo.public.toml` | `backend/Cargo.toml` |
| `docs/public/README.md` | `README.md` |
| `docs/self-hosting/env.core.example` | `.env.example` |

`backend/Cargo.lock` is seeded rather than exported: the private lock is the
hosted resolution, so the export copies it and lets Cargo resolve against the
public manifest, which prunes every hosted entry while pinning the shared ones
to the versions this repository actually tests. The pruned lock then goes
through the marker gate like everything else.

The script never touches a remote. Creating the public repository is a separate,
deliberate act, performed after the exported tree has been built, tested and
reviewed.
