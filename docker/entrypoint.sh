#!/bin/sh
set -e

# Reference server entrypoint (US-015).
#
# The server validates the migration state and applies pending core migrations
# itself, under a Postgres advisory lock, so there is nothing to sequence here.
# What this script adds is a *readable failure* for the one configuration
# mistake a container makes easy.
#
# A container's own address is not loopback, so single-user mode cannot apply:
# anything that can reach the published port is on a network. The server would
# refuse to start anyway — this check just says why before the Rust error does,
# and points at the fix.

if [ -z "${OPENBOOKLM_AUTH_TOKEN:-}" ]; then
  echo "OPENBOOKLM_AUTH_TOKEN is not set." >&2
  echo >&2
  echo "A container is reachable from a network, so requests must be" >&2
  echo "authenticated. Loopback single-user mode is for a local binary only." >&2
  echo >&2
  echo "  openssl rand -hex 32" >&2
  echo >&2
  echo "Put the result in .env as OPENBOOKLM_AUTH_TOKEN and restart." >&2
  exit 1
fi

exec /usr/local/bin/openbooklm-server
