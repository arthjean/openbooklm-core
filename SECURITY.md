# Security Policy

## Reporting a vulnerability

Report privately. Do not open a public issue, pull request or discussion for a
suspected vulnerability.

- **Preferred:** GitHub private vulnerability reporting on this repository
  (Security → Report a vulnerability).
- **Email:** arthur.jean@strivex.fr

Include the affected version or commit, the impact you believe it has, and the
smallest reproduction you have. If you cannot avoid including a credential or
personal data in the report, say so explicitly at the top so it can be handled
and purged.

## Response targets

| Stage | Target |
|---|---|
| Acknowledgement | 3 business days |
| Initial assessment and severity | 10 business days |
| Fix or documented mitigation for high and critical severity | 90 days |

These are targets for a small maintainer team, not a contractual SLA. You will
receive a status update even when the answer is "still investigating".

## Disclosure

Coordinated disclosure. Please give 90 days from acknowledgement, or until a fix
ships, whichever comes first. Reporters are credited in release notes unless they
ask not to be. There is no bug bounty.

## Supported versions

Before `1.0.0`, only the latest tagged release receives security fixes. Backports
to older tags are not provided.

## Scope

**In scope**

- The public core library, its REST and SSE handlers, and the reference server
- Core migrations and the migration validation path
- The published TypeScript SDK
- Published container images and release artifacts

**Out of scope**

- The hosted OpenbookLM SaaS, its private frontend, billing and integrations —
  report those to arthur.jean@strivex.fr, not through this repository
- Third-party model providers, PostgreSQL, and other upstream dependencies;
  report those upstream, and tell us if this project's use makes them exploitable
- Findings that require a self-hosted operator to have already misconfigured the
  deployment in a way the documentation warns against
- Automated scanner output without a demonstrated impact

## Operator security expectations

The public core assumes the operator provides these; the project does not:

- **Transport security.** The reference server speaks plain HTTP. Terminate TLS
  in front of it.
- **Network placement.** PostgreSQL and the server are not meant to be exposed
  directly to the public internet without a reverse proxy.
- **Credential handling.** Model-provider keys are read from the environment and
  never persisted. There is no per-user key storage and no key encryption
  subsystem: earlier documentation claiming an AES-256-GCM BYOK implementation
  was inaccurate and has been corrected.
- **Identity.** Single-user loopback mode is for local use. Any non-loopback bind
  requires a static token, and a static token is not a substitute for a real
  identity provider on a shared deployment.
- **Backups.** Migration rollback assumes a restorable database backup; the
  project never performs destructive down migrations for you.

## Publication and supply chain

- Every reachable commit is scanned for credentials by
  `scripts/scan-git-history-secrets.py`; unresolved high-confidence findings block
  publication. Classified exceptions are recorded in
  `docs/security/publication-audit.md` and fingerprinted, never stored in
  plaintext, in `scripts/secret-scan-allowlist.json`.
- Dependency licenses and advisories are gated by `cargo deny check licenses bans advisories`.
- Released binaries and container images carry an SBOM and a build attestation.
  An attestation proves provenance. It is not a security certification.
