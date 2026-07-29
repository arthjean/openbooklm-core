#!/usr/bin/env python3
"""Reject an incompatible REST contract change (US-010).

    ./scripts/check-openapi-compat.py <baseline.json> [candidate.json]
    ./scripts/check-openapi-compat.py <baseline.json> --breaking

`baseline` is the contract of the last released version; `candidate` defaults to
`contracts/openapi.json`. Exit 0 means the candidate is a compatible successor
under SemVer for a pre-1.0 public API, where compatible means: an existing
client keeps working.

Three classes of change are rejected:

  1. **Removal** — a path, operation, response, schema, or a property a client
     relies on receiving.
  2. **Required-field addition** — a request gains a field an existing client
     does not send.
  3. **Incompatible type change** — a property's type, format or enum narrows.

`--breaking` acknowledges the release is declared breaking and turns the same
findings into a printed inventory with exit 0, so the changes are still
reviewed rather than merely permitted.

Additive changes pass silently: new paths, new optional request fields, new
response properties, new enum values in a *response*.

Only the standard library is used: this runs in public CI with no commercial
credentials and no extra install step.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path
from typing import Any

RED = "\033[31m"
GRN = "\033[32m"
YEL = "\033[33m"
OFF = "\033[0m"


def load(path: Path) -> dict[str, Any]:
    try:
        return json.loads(path.read_text())
    except FileNotFoundError:
        sys.exit(f"{RED}baseline not found: {path}{OFF}")
    except json.JSONDecodeError as exc:
        sys.exit(f"{RED}{path} is not valid JSON: {exc}{OFF}")


def schemas(doc: dict[str, Any]) -> dict[str, Any]:
    return doc.get("components", {}).get("schemas", {}) or {}


def resolve(node: Any, doc: dict[str, Any], seen: frozenset[str] = frozenset()) -> dict[str, Any]:
    """Follow one `$ref` into `components/schemas`, guarding against cycles."""
    if not isinstance(node, dict):
        return {}
    ref = node.get("$ref")
    if not isinstance(ref, str) or not ref.startswith("#/components/schemas/"):
        return node
    name = ref.rsplit("/", 1)[-1]
    if name in seen:
        return {}
    return resolve(schemas(doc).get(name, {}), doc, seen | {name})


def type_of(node: dict[str, Any]) -> str:
    """A comparable type signature: type, format and enum, order-insensitive."""
    parts: list[str] = []
    raw_type = node.get("type")
    if isinstance(raw_type, list):
        parts.append("|".join(sorted(str(t) for t in raw_type)))
    elif raw_type is not None:
        parts.append(str(raw_type))
    if node.get("format"):
        parts.append(f"format={node['format']}")
    if isinstance(node.get("enum"), list):
        parts.append("enum=" + ",".join(sorted(str(v) for v in node["enum"])))
    if node.get("$ref"):
        parts.append(f"ref={node['$ref']}")
    return " ".join(parts) or "any"


def required_of(node: dict[str, Any]) -> set[str]:
    value = node.get("required")
    return set(value) if isinstance(value, list) else set()


def properties_of(node: dict[str, Any]) -> dict[str, Any]:
    value = node.get("properties")
    return value if isinstance(value, dict) else {}


def operations(doc: dict[str, Any]) -> dict[tuple[str, str], dict[str, Any]]:
    """Flatten paths into `(path, method) -> operation`."""
    methods = {"get", "put", "post", "delete", "options", "head", "patch", "trace"}
    flat: dict[tuple[str, str], dict[str, Any]] = {}
    for path, item in (doc.get("paths") or {}).items():
        if not isinstance(item, dict):
            continue
        for method, op in item.items():
            if method in methods and isinstance(op, dict):
                flat[(path, method)] = op
    return flat


def body_schema(op: dict[str, Any], doc: dict[str, Any]) -> dict[str, Any]:
    content = (op.get("requestBody") or {}).get("content") or {}
    media = content.get("application/json") or {}
    return resolve(media.get("schema") or {}, doc)


def response_schema(op: dict[str, Any], doc: dict[str, Any], status: str) -> dict[str, Any]:
    content = ((op.get("responses") or {}).get(status) or {}).get("content") or {}
    for media in content.values():
        if isinstance(media, dict) and media.get("schema"):
            return resolve(media["schema"], doc)
    return {}


def compare_object(
    where: str,
    old: dict[str, Any],
    new: dict[str, Any],
    old_doc: dict[str, Any],
    new_doc: dict[str, Any],
    *,
    direction: str,
    findings: list[str],
) -> None:
    """Compare one object schema.

    `direction` is `"request"` (the client sends it) or `"response"` (the client
    receives it). It decides which side a removal or a new requirement hurts.
    """
    old_props, new_props = properties_of(old), properties_of(new)

    for name, old_prop in old_props.items():
        if name not in new_props:
            if direction == "response":
                findings.append(f"{where}: response property `{name}` was removed")
            elif name in required_of(old):
                findings.append(f"{where}: request property `{name}` was removed")
            continue
        old_res = resolve(old_prop, old_doc)
        new_res = resolve(new_props[name], new_doc)
        old_type, new_type = type_of(old_res), type_of(new_res)
        if old_type != new_type:
            findings.append(
                f"{where}: property `{name}` changed type from `{old_type}` to `{new_type}`"
            )

    if direction == "request":
        added_required = required_of(new) - required_of(old)
        for name in sorted(added_required):
            findings.append(
                f"{where}: request property `{name}` became required; existing clients do not send it"
            )
    else:
        # A response field that used to be guaranteed and is now optional
        # breaks a client that reads it unconditionally.
        for name in sorted(required_of(old) - required_of(new)):
            if name in new_props:
                findings.append(
                    f"{where}: response property `{name}` is no longer guaranteed to be present"
                )


def check(old_doc: dict[str, Any], new_doc: dict[str, Any]) -> list[str]:
    findings: list[str] = []
    old_ops, new_ops = operations(old_doc), operations(new_doc)

    for key, old_op in old_ops.items():
        path, method = key
        where = f"{method.upper()} {path}"
        if key not in new_ops:
            findings.append(f"{where}: operation was removed")
            continue
        new_op = new_ops[key]

        old_responses = set((old_op.get("responses") or {}).keys())
        new_responses = set((new_op.get("responses") or {}).keys())
        for status in sorted(old_responses - new_responses):
            findings.append(f"{where}: response {status} was removed")

        compare_object(
            f"{where} request",
            body_schema(old_op, old_doc),
            body_schema(new_op, new_doc),
            old_doc,
            new_doc,
            direction="request",
            findings=findings,
        )
        for status in sorted(old_responses & new_responses):
            if not status.startswith("2"):
                continue
            compare_object(
                f"{where} response {status}",
                response_schema(old_op, old_doc, status),
                response_schema(new_op, new_doc, status),
                old_doc,
                new_doc,
                direction="response",
                findings=findings,
            )

        old_params = {
            (p.get("name"), p.get("in")) for p in old_op.get("parameters") or [] if isinstance(p, dict)
        }
        new_required_params = {
            (p.get("name"), p.get("in"))
            for p in new_op.get("parameters") or []
            if isinstance(p, dict) and p.get("required")
        }
        for name, location in sorted(new_required_params - old_params, key=lambda t: str(t)):
            findings.append(f"{where}: new required {location} parameter `{name}`")

    for name in sorted(set(schemas(old_doc)) - set(schemas(new_doc))):
        findings.append(f"components/schemas: `{name}` was removed")

    return findings


def main() -> int:
    args = [a for a in sys.argv[1:] if a != "--breaking"]
    breaking = "--breaking" in sys.argv[1:]
    if not args:
        print(__doc__)
        return 2

    baseline = Path(args[0])
    candidate = Path(args[1]) if len(args) > 1 else Path("contracts/openapi.json")

    findings = check(load(baseline), load(candidate))

    print(f"OpenAPI compatibility: {baseline} -> {candidate}\n")
    if not findings:
        print(f"{GRN}No incompatible change. This is a compatible release.{OFF}")
        return 0

    for finding in findings:
        marker = f"{YEL}BREAKING{OFF}" if breaking else f"{RED}FAIL{OFF}"
        print(f"{marker} {finding}")

    print()
    if breaking:
        print(
            f"{YEL}{len(findings)} breaking change(s), acknowledged by --breaking.{OFF}\n"
            "Bump the major version (or the minor, pre-1.0) and record them in the release notes."
        )
        return 0
    print(
        f"{RED}{len(findings)} incompatible change(s).{OFF}\n"
        "Either keep the change additive, or declare the release breaking and rerun with --breaking."
    )
    return 1


if __name__ == "__main__":
    sys.exit(main())
