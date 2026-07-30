# OpenbookLM 0.0.3

This patch release corrects the public provider capability contract.

## Model catalog

- Mistral now exposes `mistral-small-latest` and
  `mistral-large-latest`, including display names, descriptions and context
  windows, through the generated core catalog and TypeScript SDK.
- Runtime model discovery remains available and can return additional Mistral
  models when credentials and network access are present.

There are no REST, SSE or database schema changes in this release.
