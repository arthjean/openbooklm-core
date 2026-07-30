# OpenbookLM 0.0.2

This patch release hardens the public distribution path without changing the
core API or database contract.

## Supply-chain changes

- GitHub release immutability locks the release tag and attached assets after
  publication.
- The release workflow assembles every source asset in a draft before
  publishing and verifies that GitHub reports the release as immutable.
- npm publishing uses the repository's scoped Trusted Publisher through OIDC.
  No long-lived npm token is stored in GitHub.

The source archive, TypeScript SDK and container image remain contract-checked,
SBOM-backed and attested by GitHub Actions.
