# Pre-publication legal and secret audit

**Story:** US-001 (EP-001, `tasks/prd-open-core.md`)
**Auditor:** Arthur Jean
**Audit date:** 2026-07-28
**Repository state:** `arthjean/openbooklm` @ `4c3b06b66ef98c1823446fb88b1e446689416cd8` (private)
**Gate verdict:** **PASS** for secrets and licenses. One deferred blocker for the
first public release is recorded in [§5](#5-deferred-release-blockers).

This document is the reviewed publication gate. It records what was scanned, what
was found, how each finding was classified, and what remains open. It is
regenerated, not amended silently: re-running the scan and updating this file is
part of the release checklist.

---

## 1. License and package metadata

| Item | Before | After | Evidence |
|---|---|---|---|
| Root license file | absent | `LICENSE` — canonical Apache License 2.0, 201 lines, appendix included, copyright line filled | `LICENSE` |
| `backend` crate SPDX | `UNLICENSED` | `Apache-2.0` | `backend/Cargo.toml` |
| `migration` crate SPDX | absent | `Apache-2.0` | `backend/migration/Cargo.toml` |
| Contribution terms | absent | DCO 1.1 sign-off, no CLA | `CONTRIBUTING.md` |
| Vulnerability disclosure | absent | private reporting, 90-day coordinated disclosure | `SECURITY.md` |
| Conduct | absent | Contributor Covenant 2.1 | `CODE_OF_CONDUCT.md` |
| Trademark use | absent | marks excluded from the Apache grant, fork and hosting rules stated | `TRADEMARK.md` |

The `frontend/` package remains `"private": true` and is not part of the public
export; its license metadata is intentionally unchanged.

## 2. Full-history secret scan

**Tool:** `scripts/scan-git-history-secrets.py` v1.0.0 (this repository)
**Underlying VCS:** git 2.55.0
**Executed:** 2026-07-28T16:32:48Z

A vendor scanner was not used. The gate must run on a clean machine and inside
public CI without installing a scanner and without transmitting repository
contents to a third party, so the scanner is self-contained and reviewable.

**Scope**

| Dimension | Value |
|---|---|
| Commits reachable from all refs | 267 |
| Objects enumerated (`git rev-list --objects --all`) | 5,274 |
| Blobs among them | 3,399 |
| Blobs scanned | 3,354 |
| Blobs skipped | 45 (binary content, known binary extensions, or > 2 MiB) |
| Detection rules | 18 (17 vendor patterns + 1 entropy heuristic) |
| Wall-clock | ~13 s |

**Method.** Every reachable blob is decoded as UTF-8 and matched against
provider-specific credential formats (Stripe, Clerk, Anthropic, OpenAI, Voyage,
Firecrawl, Resend, PostHog, GitHub, AWS, Google, Slack, PEM private keys,
PostgreSQL URLs with inline passwords, JWTs). A second pass flags any string of
24+ characters with Shannon entropy ≥ 3.6 bits/char assigned to an identifier
whose own name contains `SECRET`, `TOKEN`, `API_KEY`, `PASSWORD`, `CREDENTIAL` or
`PRIVATE_KEY`. Matches are reported with the path, a line hint, the introducing
commit (resolved with `git log --find-object`) and a redacted preview: length, a
truncated SHA-256, and the first four characters **only for values of 16
characters or more**, because on a short value four characters plus the length is
most of the secret. **The scanner never prints a secret, and the allowlist stores
fingerprints rather than values.**

A fingerprint is not a safe place for a real credential: SHA-256 of a
low-entropy value is brute-forceable, which is exactly how F-001 below was
confirmed. The allowlist is therefore restricted to placeholders, synthetic
fixtures and public upstream material. A real credential is rotated, never
classified.

**Skipped-blob risk.** The 45 skipped blobs are fonts, images and generated
bundles. They are enumerated by the same pass and their exclusion is by extension
or by NUL-byte detection, not by path allowlist, so a text credential cannot hide
behind a directory rule. Oversized text blobs (> 2 MiB) would be skipped; none
exist in this history.

### Findings

Three findings. Zero unresolved. Zero real or possibly-real credentials.

#### F-001 — `.env.example` sample database password

| Field | Value |
|---|---|
| Rule | `postgres_url_password` (HIGH severity pattern) |
| Path | `.env.example:11` |
| Introduced | `31bbe143069ecc1ed983c726d0985131a1b92988` |
| Fingerprint | `5e884898da28047151d0e56f8dc6292773603d0d6aabbdd62a11ef721d1542d8` |
| Classification | **PLACEHOLDER** |

The matched value is the literal string `password` in a sample `DATABASE_URL`.
Its SHA-256 is the well-known digest of that word, which confirms the
classification without exposing anything. Never issued, never live, no rotation
applicable.

#### F-002 — Svix webhook signing secret in unit tests

| Field | Value |
|---|---|
| Rule | `stripe_webhook_secret` (HIGH severity pattern) |
| Path | `backend/src/api/webhooks.rs:597` |
| Introduced | `0f4e5f3c9372a80a1d4d3428b719871ab4d6ea7b` |
| Fingerprint | `06ef15097dc88af9dfa1a59d4f5bcfc5286c75ce514fa1e3966598b9f9217eaf` |
| Classification | **SYNTHETIC_TEST_FIXTURE** |

`TEST_SECRET` in the `verify_clerk_signature` unit tests. The value is
`whsec_` followed by the base64 encoding of the ASCII string
`testsecretforunittests`, chosen so the HMAC tests have a decodable key. It was
never issued by Clerk or Stripe and grants nothing. It stays in the public
repository: the tests need a deterministic key, and the value is self-evidently
synthetic. No rotation applicable.

#### F-003 — YouTube InnerTube ANDROID client key

| Field | Value |
|---|---|
| Rule | `google_api_key` (MEDIUM severity pattern) |
| Path | `backend/src/clients/youtube.rs` (constant `DEFAULT_INNERTUBE_API_KEY`) |
| Introduced | `d69e47dd6dc868786ea4610e5f07fa7046b0e2d1`, also in `2a2344515e6d7fd5f09b1b07d124b7b1ddd93e99` |
| Fingerprint | `31043aae8be22df7c8159c11f0b029a8a0da56817f00ea996a37cc48db57482c` |
| Classification | **PUBLIC_UPSTREAM_MATERIAL** |

**Decision: keep, classified, with a documented override.**

Rationale. This is not a StriveX credential and there is no StriveX account
behind it. It is the InnerTube key that ships inside the public YouTube Android
application, is recoverable from YouTube's public JavaScript, and is hardcoded
identically by the widely used transcript libraries this client's request shape
follows. It authenticates a client type, not a principal: it grants no account
access, carries no billable quota belonging to this project, and cannot be
rotated by us.

Removal was considered and rejected: without a default the YouTube source type
stops working out of the box for every self-hosted operator, and each operator
would have to rediscover the same public value. The stated risk is upstream
invalidation, not disclosure.

Override mechanism. `YOUTUBE_INNERTUBE_API_KEY` replaces the compiled default at
startup (`backend/src/core/config.rs`, consumed in
`backend/src/clients/youtube.rs`). When Google changes the upstream value, an
operator sets the env var and restarts. No redeploy, no rebuild, no fork. The
classification and this mechanism are documented at the constant's definition
site so the next reader does not have to re-derive them.

**Post-publication outcome.** GitHub secret scanning raised this key as alert #1
nine minutes after the repository went public on 2026-07-29, reporting
`validity: unknown` — it could not establish that the key is live. It was the
only alert on the repository, which is the result this audit predicted: the one
detectable pattern, already classified. Resolved as `wont_fix` with a comment
pointing here, rather than `false_positive`: the detection is correct, it is the
"exposed secret" conclusion that does not apply. Push protection was enabled on
the repository at the same time, so a genuine credential pushed by mistake is
blocked at the source rather than audited afterwards.

#### F-004 — `openbooklm` Postgres password in the public sample and CI

| Field | Value |
|---|---|
| Rule | `postgres_url_password` (HIGH severity pattern) |
| Paths | `docs/self-hosting/env.core.example` (exported as `.env.example`), `.github/workflows/public-ci.yml`, `.github/workflows/public-release.yml` |
| Fingerprint | `8d91f87a34f4881c56238ad455824e62d602fe07706d54aca448471978f4904a` |
| Classification | **PLACEHOLDER** (sample) / **SYNTHETIC_TEST_FIXTURE** (CI) |

**Decision: keep, classified, with the sample documented as a development default.**

Introduced by EP-004, after the original audit ran, which is why it appears as
UNRESOLVED on a first re-scan rather than in the table above. One value in three
places, none of them a credential.

In the two workflows it is the password of the throwaway `pgvector/pgvector:pg16`
service container, written in plain text seven lines above the URL that uses it
and reachable only from the job that starts it. Replacing it with a secret would
add a secret to a pipeline whose entire point is to need none.

In the public sample it is a development default. The Compose stack binds
Postgres to `127.0.0.1`, so nothing off-host reaches it, and the file now says
plainly that the value is published, therefore known, and must change before the
database is reachable by anything else. Removing the default instead was
considered and rejected: it breaks the one-command start the PRD treats as the
operator-DX mitigation, and it trades a documented weak default for a worse
failure mode, an operator inventing their own `DATABASE_URL` with no working
example to copy.

### Credential rotation

**No rotation was required.** Zero findings were classified real or
possibly-real, so the AC "every credential classified as real or possibly real is
rotated" is satisfied vacuously and provably: the classification table above has
no such entry.

Live credentials for the hosted product live only in untracked `.env` files,
GitHub Actions secrets, the Vercel project and the VPS environment. None of these
are in Git history and none enter the public export. They are governed by the
private repository, not by this gate.

## 3. Third-party asset and dependency licenses

`cargo deny check licenses bans advisories` executed 2026-07-28 against
`backend/`:

- **licenses: ok** — every crate in the dependency graph resolves to a license
  allowed by `backend/deny.toml`. No unclear or missing license.
- **bans: ok** — no banned crate, no disallowed duplicate.
- **advisories: FAILED** — see [§5](#5-deferred-release-blockers).

Non-Rust assets: the local `.woff2` font files and the proprietary UI live under
`frontend/`, which is classified private and is not part of the public export
(see `docs/open-core-boundary.md`). Their licensing is therefore out of scope for
the public publication gate and is re-examined only if a font is ever moved into
the public tree.

## 4. Documentation claims corrected before export

The README asserted a BYOK subsystem that does not exist in the source. Publishing
an unproven security claim is itself a security defect, so the claims were removed
rather than softened.

| Claim | Reality | Correction |
|---|---|---|
| "Les clés sont chiffrées au repos avec AES-256-GCM" | No encryption code, no `aes-gcm` dependency, no key endpoint | Section rewritten: BYOK is not implemented; provider keys are server-side environment configuration only |
| Stack table row `Chiffrement · aes-gcm 0.10` | Not a dependency of the crate | Row removed |
| Security section bullet "AES-256-GCM pour les clés BYOK stockées" | Same | Replaced with the actual property: keys never persisted |
| Project tree listing `services/encryption.rs` | File does not exist | Entry removed |
| "Multi-LLM ... avec fallback automatique et BYOK" | BYOK absent; OpenAI provider present but unlisted | Corrected to the providers actually implemented |

`backend/src/entities/user_settings.rs` also documented `providers` as "encrypted
API keys per provider". The column is dead and holds no secret; the doc comment
now says so, because a false claim in source ships to every reader of the public
crate.

## 5. Deferred release blockers

These do not block US-001, whose gate covers legal terms, secrets and license
clarity. They block the first **public release** (US-015, and the
"0 high or critical dependency advisories" non-functional requirement).

**RUSTSEC advisories: 13 open, pre-existing.**

`RUSTSEC-2024-0370`, `RUSTSEC-2026-0097`, `RUSTSEC-2026-0098`,
`RUSTSEC-2026-0099`, `RUSTSEC-2026-0104`, `RUSTSEC-2026-0173`,
`RUSTSEC-2026-0187`, `RUSTSEC-2026-0190`, `RUSTSEC-2026-0192`,
`RUSTSEC-2026-0194`, `RUSTSEC-2026-0195`, `RUSTSEC-2026-0204`,
`RUSTSEC-2026-0206`.

Breakdown: 8 vulnerabilities, 2 unsoundness, 4 unmaintained (counted per
dependency path). The material ones for this product are the two `lopdf` stack
overflows on deeply nested PDF objects, which are reachable from untrusted
uploaded PDFs, and the `quick-xml` denial-of-service pair. The `rustls-webpki`
name-constraint and CRL issues arrive through `clerk-rs`, which is private-side
only. `ttf-parser`/`rustybuzz`/`printpdf` arrive through PDF export.

One yanked crate is also flagged: `spin 0.9.8` via `multer` → `axum`.

Owner: US-015. Required action before the first tag: upgrade or vendor the
affected paths, or record an explicit, time-boxed `deny.toml` exception per
advisory with a stated reason. Silent suppression is not acceptable.

## 6. Reproducing this audit

```bash
# Full history, human-readable, exits 1 if any finding is unresolved
python3 scripts/scan-git-history-secrets.py

# Machine-readable, for CI
python3 scripts/scan-git-history-secrets.py --json

# Tracked files at HEAD only (fast pre-commit variant)
python3 scripts/scan-git-history-secrets.py --working-tree

# Stricter: block on LOW-severity findings too
python3 scripts/scan-git-history-secrets.py --fail-on LOW

# Dependency licenses and advisories
cd backend && cargo deny check licenses bans advisories
```

The scan exits `1` and prints `PUBLICATION BLOCKED` when any finding at or above
the configured severity lacks a classification in
`scripts/secret-scan-allowlist.json`. The blocking output names the rule, the path
with a line hint, and the introducing commit, and prints only the redacted
preview.

Adding an allowlist entry is a publication decision. Each entry must carry a
classification, a reason, and an `audit_ref` pointing at a section of this
document. Reviewers should treat a new entry as they would a change to
authentication code.
