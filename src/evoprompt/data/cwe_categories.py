"""CWE major category utilities.

Provides a mapping from CWE IDs (e.g., "CWE-120") to broader CWE major categories
that we will use as classification targets. This is a pragmatic subset designed
for PrimeVul experiments and can be extended as needed.
"""

from typing import List, Optional
import re

# Canonical major categories used as output labels
CWE_MAJOR_CATEGORIES = [
    "Benign",  # Special label for non-vulnerable samples
    "Buffer Errors",
    "Injection",
    "Memory Management",
    "Pointer Dereference",
    "Integer Errors",
    "Concurrency Issues",
    "Path Traversal",
    "Cryptography Issues",
    "Information Exposure",
    "Other",
]

# Map common CWE IDs to major categories
# Note: extend this mapping as needed for better coverage
_CWE_ID_TO_MAJOR = {
    # Buffer/Bounds related
    119: "Buffer Errors",  # Improper Restriction of Operations within the Bounds of a Memory Buffer
    120: "Buffer Errors",  # Classic Buffer Overflow
    121: "Buffer Errors",  # Stack-based Buffer Overflow
    122: "Buffer Errors",  # Heap-based Buffer Overflow
    125: "Buffer Errors",  # Out-of-bounds Read
    131: "Buffer Errors",  # Incorrect Calculation of Buffer Size
    787: "Buffer Errors",  # Out-of-bounds Write
    805: "Buffer Errors",  # Buffer Access with Incorrect Length Value

    # Injection
    74:  "Injection",      # Injection
    77:  "Injection",      # Command Injection
    78:  "Injection",      # OS Command Injection
    79:  "Injection",      # Cross-site Scripting (XSS)
    89:  "Injection",      # SQL Injection

    # Memory Management
    416: "Memory Management",  # Use After Free
    415: "Memory Management",  # Double Free
    401: "Memory Management",  # Missing Release of Memory after Effective Lifetime

    # Pointer deref / null deref
    476: "Pointer Dereference",  # NULL Pointer Dereference

    # Integer issues
    190: "Integer Errors",  # Integer Overflow or Wraparound
    191: "Integer Errors",  # Integer Underflow (Wrap or Wraparound)

    # Concurrency / Race conditions
    362: "Concurrency Issues",  # Concurrent Execution using Shared Resource with Improper Synchronization ('Race Condition')

    # Path traversal
    22:  "Path Traversal",  # Improper Limitation of a Pathname to a Restricted Directory ('Path Traversal')

    # Crypto
    326: "Cryptography Issues",  # Inadequate Encryption Strength
    327: "Cryptography Issues",  # Use of a Broken or Risky Cryptographic Algorithm

    # Information exposure
    200: "Information Exposure",  # Information Exposure
}

_CWE_ID_REGEX = re.compile(r"CWE-(\d+)")


def _extract_cwe_ids(cwe_codes: List[str]) -> List[int]:
    ids: List[int] = []
    for code in cwe_codes:
        if not code:
            continue
        m = _CWE_ID_REGEX.search(str(code))
        if m:
            try:
                ids.append(int(m.group(1)))
            except ValueError:
                continue
    return ids


essential_major_fallback_order = [
    # Order in which we prefer to pick a category when multiple CWEs exist
    "Buffer Errors",
    "Injection",
    "Memory Management",
    "Pointer Dereference",
    "Integer Errors",
    "Concurrency Issues",
    "Path Traversal",
    "Cryptography Issues",
    "Information Exposure",
]


def map_cwe_to_major(cwe_codes: List[str]) -> str:
    """Map a list of CWE codes like ["CWE-120", "CWE-787"] to a single major category.

    If multiple CWE IDs are present, pick by a sensible priority order.
    If none are recognized, return "Other".
    """
    ids = _extract_cwe_ids(cwe_codes)
    if not ids:
        return "Other"

    majors = { _CWE_ID_TO_MAJOR[i] for i in ids if i in _CWE_ID_TO_MAJOR }
    if not majors:
        return "Other"

    # Choose by priority order
    for cat in essential_major_fallback_order:
        if cat in majors:
            return cat

    # Otherwise arbitrary stable pick
    return sorted(majors)[0]


def canonicalize_category(text: str) -> Optional[str]:
    """Normalize model output to one of CWE_MAJOR_CATEGORIES if possible.

    - Exact case-insensitive match to category name
    - If contains a CWE ID, map to major
    - Common synonyms handling can be added here if needed
    """
    normalized = text.strip().lower()
    if not normalized:
        return None

    # Try exact category match (case-insensitive)
    for cat in CWE_MAJOR_CATEGORIES:
        if normalized == cat.lower():
            return cat

    # Try to detect any CWE ID and map
    ids = _extract_cwe_ids([text])
    if ids:
        # Prefer the first detected id
        major = _CWE_ID_TO_MAJOR.get(ids[0])
        if major:
            return major

    # Simple keyword helps
    if "buffer" in normalized or "out-of-bounds" in normalized:
        return "Buffer Errors"
    if "inject" in normalized or "sql" in normalized or "command" in normalized:
        return "Injection"
    if "use after free" in normalized or "double free" in normalized or "memory leak" in normalized:
        return "Memory Management"
    if "null pointer" in normalized or "pointer deref" in normalized:
        return "Pointer Dereference"
    if "integer" in normalized or "overflow" in normalized or "underflow" in normalized:
        return "Integer Errors"
    if "race" in normalized or "concurrency" in normalized:
        return "Concurrency Issues"
    if "path traversal" in normalized or "directory traversal" in normalized:
        return "Path Traversal"
    if "crypto" in normalized or "encryption" in normalized or "cipher" in normalized:
        return "Cryptography Issues"
    if "information exposure" in normalized or "leak" in normalized:
        return "Information Exposure"

    if "benign" in normalized or "no vulnerability" in normalized or "no vuln" in normalized:
        return "Benign"

    return None