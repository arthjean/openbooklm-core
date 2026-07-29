# The OpenbookLM reference server (US-015, EP-004).
#
# Build from the repository root:
#
#   docker build -f docker/Dockerfile -t openbooklm .
#
# The image contains two binaries: the server, and the core migration CLI for
# operators who want the schema change and the code change to be separate steps.
# The server applies pending core migrations itself on start, so the CLI is a
# choice rather than a requirement.

# ============================================================================
# Stage 1: cargo-chef, for a dependency layer that survives a source change
# ============================================================================
FROM rust:1.88-bookworm AS chef
RUN cargo install cargo-chef --locked
WORKDIR /app

# ============================================================================
# Stage 2: dependency recipe
# ============================================================================
FROM chef AS planner
COPY backend/ backend/
WORKDIR /app/backend
RUN cargo chef prepare --recipe-path recipe.json

# ============================================================================
# Stage 3: build
# ============================================================================
FROM chef AS builder

RUN apt-get update && apt-get install -y --no-install-recommends \
    libpq-dev \
    pkg-config \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app/backend

# Only rebuilds when the manifests change.
COPY --from=planner /app/backend/recipe.json recipe.json
RUN cargo chef cook --release --recipe-path recipe.json

COPY backend/ .
RUN cargo build --release --locked \
      --bin openbooklm-server -p openbooklm \
      --bin openbooklm-migrate -p openbooklm-migration-core \
 && strip target/release/openbooklm-server target/release/openbooklm-migrate

# ============================================================================
# Stage 4: runtime
# ============================================================================
FROM debian:bookworm-slim AS runtime

RUN apt-get update && apt-get install -y --no-install-recommends \
    libpq5 \
    ca-certificates \
    curl \
    && rm -rf /var/lib/apt/lists/*

# Non-root. The server writes nothing to disk, so the filesystem can be
# mounted read-only (see docker/docker-compose.yml).
RUN groupadd --gid 1001 app && useradd --uid 1001 --gid app --create-home app

COPY --from=builder /app/backend/target/release/openbooklm-server /usr/local/bin/openbooklm-server
COPY --from=builder /app/backend/target/release/openbooklm-migrate /usr/local/bin/openbooklm-migrate
COPY docker/entrypoint.sh /usr/local/bin/entrypoint.sh
RUN chmod +x /usr/local/bin/entrypoint.sh

USER app
EXPOSE 3001

# A container is not on loopback from its own point of view, so the default
# bind must be the wildcard. That means the server refuses to start without
# OPENBOOKLM_AUTH_TOKEN — which is the intended behaviour: a container reachable
# from a network must be authenticated.
ENV OPENBOOKLM_BIND=0.0.0.0

HEALTHCHECK --interval=10s --timeout=3s --start-period=20s --retries=3 \
    CMD curl -fsS http://localhost:3001/health || exit 1

ENTRYPOINT ["/usr/local/bin/entrypoint.sh"]
