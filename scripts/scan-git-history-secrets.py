#!/usr/bin/env python3
"""Full-history secret scanner for the OpenbookLM publication gate (US-001).

Walks every blob reachable from every ref, applies provider-specific credential
patterns plus a Shannon-entropy heuristic, and reports redacted findings with the
originating path and commit. Findings can be classified as accepted only through
``scripts/secret-scan-allowlist.json``, which stores the SHA-256 of the matched
value rather than the value itself.

The scanner is intentionally self-contained: the publication gate must be
runnable on a clean machine and inside public CI without installing a scanner or
sending repository contents to a third party.

Usage::

    python3 scripts/scan-git-history-secrets.py                 # human report
    python3 scripts/scan-git-history-secrets.py --json          # machine report
    python3 scripts/scan-git-history-secrets.py --working-tree  # tracked files only

Exit codes::

    0  no unresolved finding at or above the blocking severity
    1  publication blocked: at least one unresolved finding
    2  scanner or repository error
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import re
import subprocess
import sys
from collections import OrderedDict
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path

TOOL_NAME = "openbooklm-secret-scan"
TOOL_VERSION = "1.0.0"

REPO_ROOT = Path(__file__).resolve().parent.parent
ALLOWLIST_PATH = REPO_ROOT / "scripts" / "secret-scan-allowlist.json"

# Blobs above this size are skipped: lockfiles, bundled assets and vendored data
# dominate scan time while credentials live in source, config and docs.
MAX_BLOB_BYTES = 2 * 1024 * 1024

# Binary-ish extensions never carry reviewable credentials in this repository.
SKIPPED_SUFFIXES = frozenset(
    {
        ".png", ".jpg", ".jpeg", ".gif", ".webp", ".avif", ".ico", ".bmp",
        ".woff", ".woff2", ".ttf", ".otf", ".eot",
        ".pdf", ".zip", ".gz", ".tar", ".bz2", ".xz", ".7z",
        ".mp3", ".mp4", ".webm", ".mov", ".wav",
        ".glb", ".gltf", ".hdr", ".exr",
    }
)

SEVERITY_ORDER = {"HIGH": 3, "MEDIUM": 2, "LOW": 1}


@dataclass(frozen=True)
class Rule:
    """A single credential detection rule."""

    rule_id: str
    severity: str
    description: str
    pattern: re.Pattern[str]
    group: int = 0


def _rx(pattern: str, flags: int = 0) -> re.Pattern[str]:
    return re.compile(pattern, flags)


# Provider-specific rules. Prefixes come from each vendor's documented key
# format, so a match is high-confidence rather than heuristic.
RULES: tuple[Rule, ...] = (
    Rule("stripe_live_secret", "HIGH", "Stripe live secret or restricted key",
         _rx(r"\b(?:sk|rk)_live_[0-9A-Za-z]{16,}")),
    Rule("stripe_test_secret", "MEDIUM", "Stripe test secret or restricted key",
         _rx(r"\b(?:sk|rk)_test_[0-9A-Za-z]{16,}")),
    Rule("stripe_webhook_secret", "HIGH", "Stripe webhook signing secret",
         _rx(r"\bwhsec_[0-9A-Za-z]{16,}")),
    Rule("clerk_secret", "HIGH", "Clerk backend secret key",
         _rx(r"\bsk_(?:live|test)_[0-9A-Za-z]{20,}")),
    Rule("anthropic_key", "HIGH", "Anthropic API key",
         _rx(r"\bsk-ant-[0-9A-Za-z_\-]{24,}")),
    Rule("openai_key", "HIGH", "OpenAI API key",
         _rx(r"\bsk-(?:proj-)?[0-9A-Za-z_\-]{32,}")),
    Rule("voyage_key", "HIGH", "Voyage AI API key",
         _rx(r"\bpa-[0-9A-Za-z_\-]{32,}")),
    Rule("firecrawl_key", "HIGH", "Firecrawl API key",
         _rx(r"\bfc-[0-9a-f]{32}\b")),
    Rule("resend_key", "HIGH", "Resend API key",
         _rx(r"\bre_[0-9A-Za-z]{8}_[0-9A-Za-z]{16,}")),
    Rule("posthog_personal_key", "HIGH", "PostHog personal API key",
         _rx(r"\bphx_[0-9A-Za-z_\-]{32,}")),
    Rule("github_token", "HIGH", "GitHub token",
         _rx(r"\b(?:ghp|gho|ghu|ghs|ghr)_[0-9A-Za-z]{36,}|\bgithub_pat_[0-9A-Za-z_]{40,}")),
    Rule("aws_access_key_id", "HIGH", "AWS access key id",
         _rx(r"\b(?:AKIA|ASIA)[0-9A-Z]{16}\b")),
    Rule("google_api_key", "MEDIUM", "Google-format API key",
         _rx(r"\bAIza[0-9A-Za-z_\-]{35}\b")),
    Rule("slack_token", "HIGH", "Slack token",
         _rx(r"\bxox[abposr]-[0-9A-Za-z\-]{10,}")),
    Rule("private_key_block", "HIGH", "PEM private key block",
         _rx(r"-----BEGIN (?:RSA |EC |DSA |OPENSSH |PGP )?PRIVATE KEY-----")),
    Rule("postgres_url_password", "HIGH", "PostgreSQL URL with inline password",
         _rx(r"postgres(?:ql)?://[^\s:@/\"']+:([^\s:@/\"']{6,})@"), group=1),
    Rule("jwt_token", "MEDIUM", "JWT-shaped token",
         _rx(r"\beyJ[0-9A-Za-z_\-]{10,}\.eyJ[0-9A-Za-z_\-]{10,}\.[0-9A-Za-z_\-]{10,}")),
)

# Generic assignment heuristic: only fires when the identifier itself names a
# credential, which keeps hashes, UUIDs and base64 fixtures out of the report.
SECRET_ASSIGNMENT = _rx(
    r"(?i)\b([A-Z0-9_]*(?:SECRET|TOKEN|API[_-]?KEY|APIKEY|PASSWORD|PASSWD|CREDENTIAL|PRIVATE[_-]?KEY)[A-Z0-9_]*)\b"
    r"\s*[:=]\s*[\"']([^\"'\s]{20,200})[\"']"
)

# Values that look like placeholders rather than live credentials.
PLACEHOLDER = _rx(
    r"(?i)^(?:x{4,}|\.{3,}|<[^>]+>|\$\{[^}]+\}|change[_-]?me|your[_-]|placeholder|example|dummy|redacted|test[_-]?only|"
    r"sk-\.\.\.|null|none|undefined|todo)"
)

ENTROPY_MIN_LENGTH = 24
ENTROPY_THRESHOLD = 3.6


def shannon_entropy(value: str) -> float:
    """Return Shannon entropy in bits per character."""
    if not value:
        return 0.0
    counts: dict[str, int] = {}
    for char in value:
        counts[char] = counts.get(char, 0) + 1
    length = len(value)
    return -sum((c / length) * math.log2(c / length) for c in counts.values())


def redact(value: str) -> str:
    """Return a preview that locates a finding without disclosing it.

    The prefix is shown only for values long enough that four characters do not
    materially narrow a guess. A short value gets no prefix at all: for an
    eight-character credential, "pass…[8 chars]" is most of the secret.
    """
    digest = hashlib.sha256(value.encode("utf-8", "replace")).hexdigest()
    head = f"{value[:4]}…" if len(value) >= 16 else ""
    return f"{head}[{len(value)} chars, sha256:{digest[:12]}]"


def fingerprint(value: str) -> str:
    return hashlib.sha256(value.encode("utf-8", "replace")).hexdigest()


@dataclass
class Finding:
    rule_id: str
    severity: str
    description: str
    path: str
    redacted: str
    fingerprint: str
    blobs: set[str] = field(default_factory=set)
    line_hint: int = 0
    classification: str | None = None
    reason: str | None = None
    commits: list[str] = field(default_factory=list)

    @property
    def key(self) -> tuple[str, str, str]:
        return (self.rule_id, self.path, self.fingerprint)


def run_git(args: list[str], binary: bool = False) -> bytes | str:
    result = subprocess.run(
        ["git", *args],
        cwd=REPO_ROOT,
        check=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
    )
    return result.stdout if binary else result.stdout.decode("utf-8", "replace")


def git_version() -> str:
    return str(run_git(["--version"])).strip()


def enumerate_blobs(working_tree_only: bool) -> "OrderedDict[str, str]":
    """Return an ordered mapping of blob sha to the first path it appeared under."""
    blobs: OrderedDict[str, str] = OrderedDict()
    if working_tree_only:
        listing = str(run_git(["ls-files", "-s"]))
        for line in listing.splitlines():
            # <mode> <sha> <stage>\t<path>
            meta, _, path = line.partition("\t")
            parts = meta.split()
            if len(parts) >= 2:
                blobs.setdefault(parts[1], path)
        return blobs

    listing = str(run_git(["rev-list", "--objects", "--all"]))
    for line in listing.splitlines():
        sha, _, path = line.partition(" ")
        if path:
            blobs.setdefault(sha, path)
    return blobs


def blob_sizes(shas: list[str]) -> dict[str, tuple[str, int]]:
    """Batch-resolve object type and size for the given shas."""
    payload = "\n".join(shas).encode()
    result = subprocess.run(
        ["git", "cat-file", "--batch-check=%(objectname) %(objecttype) %(objectsize)"],
        cwd=REPO_ROOT,
        input=payload,
        check=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
    )
    sizes: dict[str, tuple[str, int]] = {}
    for line in result.stdout.decode("utf-8", "replace").splitlines():
        parts = line.split()
        if len(parts) == 3 and parts[1] != "missing":
            sizes[parts[0]] = (parts[1], int(parts[2]))
    return sizes


def read_blob(sha: str) -> str | None:
    raw = run_git(["cat-file", "blob", sha], binary=True)
    assert isinstance(raw, bytes)
    if b"\0" in raw[:8192]:
        return None
    return raw.decode("utf-8", "replace")


def line_of(text: str, index: int) -> int:
    return text.count("\n", 0, index) + 1


def scan_text(text: str, path: str, blob: str) -> list[Finding]:
    findings: list[Finding] = []

    for rule in RULES:
        for match in rule.pattern.finditer(text):
            value = match.group(rule.group)
            if not value or PLACEHOLDER.match(value):
                continue
            findings.append(
                Finding(
                    rule_id=rule.rule_id,
                    severity=rule.severity,
                    description=rule.description,
                    path=path,
                    redacted=redact(value),
                    fingerprint=fingerprint(value),
                    blobs={blob},
                    line_hint=line_of(text, match.start()),
                )
            )

    for match in SECRET_ASSIGNMENT.finditer(text):
        identifier, value = match.group(1), match.group(2)
        if PLACEHOLDER.match(value) or len(value) < ENTROPY_MIN_LENGTH:
            continue
        if shannon_entropy(value) < ENTROPY_THRESHOLD:
            continue
        findings.append(
            Finding(
                rule_id="entropy_assignment",
                severity="MEDIUM",
                description=f"High-entropy value assigned to `{identifier}`",
                path=path,
                redacted=redact(value),
                fingerprint=fingerprint(value),
                blobs={blob},
                line_hint=line_of(text, match.start()),
            )
        )

    return findings


def load_allowlist() -> dict[str, dict[str, str]]:
    if not ALLOWLIST_PATH.exists():
        return {}
    data = json.loads(ALLOWLIST_PATH.read_text(encoding="utf-8"))
    entries = {}
    for entry in data.get("accepted", []):
        entries[entry["fingerprint"]] = entry
    return entries


def commits_for_blob(sha: str, limit: int = 3) -> list[str]:
    """Return commits that introduced or removed the given blob."""
    try:
        output = str(run_git(["log", "--all", "--format=%H", f"--find-object={sha}"]))
    except subprocess.CalledProcessError:
        return []
    return output.split()[:limit]


def scan(working_tree_only: bool) -> tuple[list[Finding], dict[str, object]]:
    blobs = enumerate_blobs(working_tree_only)
    sizes = blob_sizes(list(blobs))

    merged: dict[tuple[str, str, str], Finding] = {}
    scanned = skipped = blob_total = 0

    for sha, path in blobs.items():
        kind_size = sizes.get(sha)
        if not kind_size or kind_size[0] != "blob":
            continue
        blob_total += 1
        if Path(path).suffix.lower() in SKIPPED_SUFFIXES or kind_size[1] > MAX_BLOB_BYTES:
            skipped += 1
            continue
        text = read_blob(sha)
        if text is None:
            skipped += 1
            continue
        scanned += 1
        for finding in scan_text(text, path, sha):
            existing = merged.get(finding.key)
            if existing:
                existing.blobs.update(finding.blobs)
            else:
                merged[finding.key] = finding

    allowlist = load_allowlist()
    findings = sorted(
        merged.values(),
        key=lambda f: (-SEVERITY_ORDER[f.severity], f.rule_id, f.path),
    )
    for finding in findings:
        entry = allowlist.get(finding.fingerprint)
        if entry:
            finding.classification = entry.get("classification", "accepted")
            finding.reason = entry.get("reason")
        finding.commits = commits_for_blob(sorted(finding.blobs)[0])

    scope = "tracked working tree" if working_tree_only else "all blobs reachable from all refs"
    metadata = {
        "tool": TOOL_NAME,
        "tool_version": TOOL_VERSION,
        "git_version": git_version(),
        "scanned_at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "scope": scope,
        "head": str(run_git(["rev-parse", "HEAD"])).strip(),
        "reachable_commits": int(str(run_git(["rev-list", "--all", "--count"])).strip()),
        "objects_enumerated": len(blobs),
        "blobs_total": blob_total,
        "blobs_scanned": scanned,
        "blobs_skipped": skipped,
        "rules": len(RULES) + 1,
    }
    return findings, metadata


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--json", action="store_true", help="emit a machine-readable report")
    parser.add_argument(
        "--working-tree",
        action="store_true",
        help="scan only tracked files at HEAD instead of the full history",
    )
    parser.add_argument(
        "--fail-on",
        choices=("HIGH", "MEDIUM", "LOW"),
        default="MEDIUM",
        help="lowest unresolved severity that blocks publication (default: MEDIUM)",
    )
    args = parser.parse_args()

    try:
        findings, metadata = scan(args.working_tree)
    except subprocess.CalledProcessError as exc:
        stderr = exc.stderr.decode("utf-8", "replace") if exc.stderr else ""
        print(f"scanner error: git failed: {stderr.strip()}", file=sys.stderr)
        return 2

    threshold = SEVERITY_ORDER[args.fail_on]
    unresolved = [
        f
        for f in findings
        if f.classification is None and SEVERITY_ORDER[f.severity] >= threshold
    ]

    if args.json:
        print(
            json.dumps(
                {
                    "metadata": metadata,
                    "findings": [
                        {
                            "rule": f.rule_id,
                            "severity": f.severity,
                            "description": f.description,
                            "path": f.path,
                            "line_hint": f.line_hint,
                            "redacted": f.redacted,
                            "fingerprint": f.fingerprint,
                            "commits": f.commits,
                            "classification": f.classification or "UNRESOLVED",
                            "reason": f.reason,
                        }
                        for f in findings
                    ],
                    "unresolved": len(unresolved),
                    "blocked": bool(unresolved),
                },
                indent=2,
            )
        )
    else:
        print(f"{TOOL_NAME} {TOOL_VERSION} ({metadata['git_version']})")
        print(f"scanned_at : {metadata['scanned_at']}")
        print(f"scope      : {metadata['scope']}")
        print(f"head       : {metadata['head']}")
        print(
            f"coverage   : {metadata['reachable_commits']} commits, "
            f"{metadata['blobs_scanned']}/{metadata['blobs_total']} blobs scanned, "
            f"{metadata['blobs_skipped']} skipped (binary or oversized)"
        )
        print(f"findings   : {len(findings)} ({len(unresolved)} unresolved)")
        print()
        for f in findings:
            status = f.classification or "UNRESOLVED"
            print(f"[{f.severity}] {f.rule_id} — {f.description}")
            print(f"  path       : {f.path}:{f.line_hint}")
            print(f"  value      : {f.redacted}")
            print(f"  fingerprint: {f.fingerprint}")
            print(f"  commits    : {', '.join(f.commits) or 'unknown'}")
            print(f"  status     : {status}" + (f" — {f.reason}" if f.reason else ""))
            print()

    if unresolved:
        print(
            f"PUBLICATION BLOCKED: {len(unresolved)} unresolved finding(s) at or above {args.fail_on}.",
            file=sys.stderr,
        )
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())
