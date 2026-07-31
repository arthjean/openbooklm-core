# Repository Agent Guidance

`AGENTS.md` is the canonical instruction source for this repository.
`CLAUDE.md` only imports it for Claude Code.

## Critical boundaries

- Keep this repository deployable as the standalone OpenbookLM core. Do not add
  hosted-only identity vendors, billing, analytics, lifecycle email, the
  proprietary web UI, or commercial operations. When core behavior must
  interact with a host, use or extend the injection seams described in
  [README.md](README.md) and [docs/open-core-boundary.md](docs/open-core-boundary.md).
- Treat `.codex/` as local agent state. Do not edit or commit it.
- Never commit credentials, production-derived data, real account identifiers,
  or real email addresses. Fixtures and demo content must remain synthetic.
- This is already the exported public repository. The origin-only checks
  `scripts/check-open-core-boundary.sh` and
  `scripts/check-public-manifest.py` apply only when
  `backend/Cargo.public.toml` exists. Do not run them in this checkout.

## Rust commands

Run Cargo commands from `backend/`. Validate the public edition explicitly:

```bash
cargo check --no-default-features --all-targets
cargo clippy --no-default-features --all-targets -- -D warnings
cargo test --no-default-features
cargo deny check licenses bans advisories
```

Do not use `cargo fmt --check` here. Rustfmt follows module declarations without
evaluating `cfg` and looks for hosted modules that this repository does not
contain. Use the same file-level check as CI:

```bash
cd backend
git ls-files '*.rs' | grep -v '^src/lib\.rs$' \
  | xargs rustfmt --check --edition 2024
```

Tests must remain offline and deterministic. Database tests stay ignored by
default and use `TEST_DATABASE_URL` when run deliberately.

## Contracts and generated files

Edit each authoritative source, then regenerate its dependents:

| Generated artifact | Authoritative source |
|---|---|
| `contracts/openapi.json` | Rust DTOs and API annotations, assembled by `backend/src/api/openapi.rs` |
| `contracts/core-constants.json` | `backend/src/core/catalog.rs` |
| `packages/sdk-ts/src/generated/openapi.ts` | `contracts/openapi.json` |
| `packages/sdk-ts/src/generated/catalog.ts` | `contracts/core-constants.json` |

Never edit those four generated artifacts directly. Regenerate and verify them:

```bash
(cd backend && cargo run --bin contracts)
(cd packages/sdk-ts && bun run generate)
./scripts/check-contracts.sh
```

REST and SSE changes are public compatibility changes. When SSE event names,
payloads, ordering, or terminal behavior change, update
`docs/contracts/sse-protocol-v1.md` and the relevant
`contracts/baseline/sse/` fixture in the same change.

## Migrations

- Never edit, rename, or reorder an applied migration. Add a new timestamped
  module under `backend/migration-core/src/core_track/` and append it to
  `CoreMigrator::migrations()` in that directory's `mod.rs`.
- Evolve the schema additively. Expand, backfill, deploy compatible readers,
  and contract only in a later release.
- Never run destructive `down`, `fresh`, or `refresh` migration operations.
  Validate first, then apply `up`; rollback means redeploying the previous
  binary or deliberately restoring a backup.

See [docs/migrations.md](docs/migrations.md) and
[docs/upgrading.md](docs/upgrading.md) before changing schema behavior.

## TypeScript SDK and delivery

Use Bun in `packages/sdk-ts/` and preserve `bun.lock`:

```bash
bun install --frozen-lockfile
bun run typecheck
bun test
bun run build
```

If asked to commit, every commit must carry the DCO sign-off added by
`git commit -s`. Use `.github/workflows/public-ci.yml` as the executable
reference for public CI gates.
