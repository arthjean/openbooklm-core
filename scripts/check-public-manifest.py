#!/usr/bin/env python3
"""Keep backend/Cargo.public.toml honest (US-014, EP-004).

The public repository ships `Cargo.public.toml` as its `backend/Cargo.toml`.
Two manifests means two places to add a dependency, and a manifest that drifts
is discovered at release time, in someone else's CI. This check makes the drift
a local failure instead:

  1. no forbidden crate appears in the public manifest;
  2. every public dependency exists in the private manifest at the same
     version requirement — a version bump on one side must happen on both;
  3. the public manifest declares no hosted feature set and no hosted target;
  4. every non-optional private dependency is present in the public manifest,
     unless it is one of the hosted optional crates.

Rule 4 is the one that catches the common mistake: adding a dependency the core
needs and forgetting the public manifest, which fails only when someone
actually builds the public repository.

Exit codes: 0 clean, 1 drift, 2 script error.
"""

from __future__ import annotations

import pathlib
import sys
import tomllib

ROOT = pathlib.Path(__file__).resolve().parent.parent
PRIVATE = ROOT / "backend" / "Cargo.toml"
PUBLIC = ROOT / "backend" / "Cargo.public.toml"

# Crates the public edition must never link. Matched as a substring of the
# dependency name, so `async-stripe-billing` is caught by `stripe`.
FORBIDDEN = ("clerk", "stripe", "posthog", "resend")


def version_of(spec: object) -> str | None:
    """The version requirement of a dependency, however it is spelled."""
    if isinstance(spec, str):
        return spec
    if isinstance(spec, dict):
        if "workspace" in spec:
            return "<workspace>"
        if "path" in spec and "version" not in spec:
            return "<path>"
        version = spec.get("version")
        return version if isinstance(version, str) else None
    return None


def main() -> int:
    for path in (PRIVATE, PUBLIC):
        if not path.is_file():
            print(f"error: {path} not found", file=sys.stderr)
            return 2

    private = tomllib.loads(PRIVATE.read_text())
    public_text = PUBLIC.read_text()
    public = tomllib.loads(public_text)

    failures: list[str] = []

    priv_deps: dict = private.get("dependencies", {})
    pub_deps: dict = public.get("dependencies", {})

    # 1. no forbidden crate
    for name in pub_deps:
        lowered = name.lower()
        if any(word in lowered for word in FORBIDDEN):
            failures.append(f"public manifest depends on `{name}`, which is hosted-only")

    # A commercial crate can also arrive through a feature list or a comment
    # that survived the derivation.
    for word in FORBIDDEN:
        for line_no, line in enumerate(public_text.splitlines(), 1):
            stripped = line.strip()
            if stripped.startswith("#"):
                continue
            if word in stripped.lower():
                failures.append(
                    f"public manifest line {line_no} mentions `{word}`: {stripped}"
                )

    # 2. shared dependencies must agree on version
    for name, spec in pub_deps.items():
        if name not in priv_deps:
            failures.append(
                f"`{name}` is in the public manifest but not the private one; "
                "the private build would not exercise it"
            )
            continue
        pub_version = version_of(spec)
        priv_version = version_of(priv_deps[name])
        if pub_version != priv_version:
            failures.append(
                f"`{name}` version differs: public `{pub_version}` vs private `{priv_version}`"
            )

    # 3. the only feature the public manifest may declare is an empty `saas`,
    #    which exists so the shared source tree's `#[cfg(feature = "saas")]`
    #    attributes are a known cfg rather than a warning. It must stay empty:
    #    a non-empty one would claim the public repository can build the hosted
    #    edition, and there is nothing there to build.
    features = public.get("features", {})
    for name, enables in features.items():
        if name != "saas":
            failures.append(
                f"public manifest declares feature `{name}`; only an empty `saas` is allowed"
            )
        elif enables:
            failures.append(f"public feature `saas` must be empty, got {enables}")
    if "default" in features:
        failures.append("public manifest declares a default feature set")
    for target_kind in ("bin", "test", "bench"):
        for target in public.get(target_kind, []):
            if target.get("required-features"):
                failures.append(
                    f"public {target_kind} `{target.get('name')}` has required-features; "
                    "hosted targets do not belong in the public manifest"
                )
            if str(target.get("name", "")).startswith("saas"):
                failures.append(f"public {target_kind} `{target['name']}` is hosted-only")

    # 4. every mandatory private dependency reaches the public manifest
    for name, spec in priv_deps.items():
        optional = isinstance(spec, dict) and spec.get("optional") is True
        if optional or name in pub_deps:
            continue
        failures.append(
            f"`{name}` is a non-optional private dependency missing from the public "
            "manifest; either add it there or mark it optional behind `saas`"
        )

    # The public workspace must not build the private migration crate.
    members = public.get("workspace", {}).get("members", [])
    if "migration" in members:
        failures.append("public workspace includes the private `migration` crate")

    if failures:
        print("Public manifest check")
        for failure in failures:
            print(f"FAIL {failure}")
        print(f"\n{len(failures)} drift(s). See backend/Cargo.public.toml.")
        return 1

    print(
        f"Public manifest clean: {len(pub_deps)} dependencies, "
        f"{len(priv_deps) - len(pub_deps)} hosted-only."
    )
    return 0


if __name__ == "__main__":
    sys.exit(main())
