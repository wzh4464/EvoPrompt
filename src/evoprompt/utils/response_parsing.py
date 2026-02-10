"""Unified response parsing for LLM outputs in vulnerability detection.

This module provides a centralized ResponseParser class that handles all
response parsing logic, including:
- Text normalization
- Binary vulnerability label extraction
- CWE major category classification
- Keyword matching with proper negation handling
"""

from __future__ import annotations

import re
from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple

from ..data.cwe_categories import CWE_MAJOR_CATEGORIES, _CWE_ID_REGEX, _CWE_ID_TO_MAJOR


@dataclass
class ParsedResponse:
    """Structured result from parsing an LLM response.

    Attributes:
        raw: Original response text
        normalized: Lowercased and stripped text
        is_vulnerable: Whether the response indicates a vulnerability
        vulnerability_label: Binary label ("0" or "1")
        cwe_category: One of CWE_MAJOR_CATEGORIES
    """
    raw: str
    normalized: str
    is_vulnerable: bool
    vulnerability_label: str
    cwe_category: str


class ResponseParser:
    """Unified LLM response parser for vulnerability detection.

    Centralizes all parsing logic to ensure consistency across the codebase.
    All methods are class methods for easy use without instantiation.

    Example:
        >>> result = ResponseParser.parse("This code has a buffer overflow vulnerability")
        >>> result.is_vulnerable
        True
        >>> result.cwe_category
        'Buffer Errors'

        >>> ResponseParser.extract_vulnerability_label("The code is benign")
        '0'
    """

    # Patterns that negate vulnerability detection
    NEGATION_PATTERNS: Tuple[str, ...] = (
        "no vulnerab",
        "not vulnerab",
        "without vulnerab",
        "free of vulnerab",
        "no security issue",
        "no issue",
    )

    # Patterns that indicate benign/safe code
    BENIGN_PATTERNS: Tuple[str, ...] = (
        "benign",
        "safe",
        "secure",
    )

    # Keywords mapped to CWE categories (order matters for priority)
    CATEGORY_KEYWORDS: Dict[str, List[str]] = {
        "Buffer Errors": ["buffer", "out-of-bounds", "oob"],
        "Injection": ["inject", "sql", "command", "xss"],
        "Memory Management": ["use after free", "double free", "memory leak", "uaf"],
        "Pointer Dereference": ["null pointer", "nullptr", "pointer deref"],
        "Integer Errors": ["integer overflow", "integer underflow", "int overflow"],
        "Concurrency Issues": ["race condition", "race", "concurrency", "deadlock"],
        "Path Traversal": ["path traversal", "directory traversal"],
        "Cryptography Issues": ["crypto", "encryption", "cipher", "weak hash"],
        "Information Exposure": ["information exposure", "info leak", "data leak"],
    }

    # CWE ID regex (reuse from cwe_categories)
    _CWE_PATTERN = _CWE_ID_REGEX

    @classmethod
    def normalize(cls, text: Optional[str]) -> str:
        """Normalize response text for parsing.

        Args:
            text: Raw response text, may be None

        Returns:
            Stripped text, or empty string if None
        """
        if not text:
            return ""
        return text.strip()

    @classmethod
    def _check_benign(cls, text: str) -> bool:
        """Check if text indicates benign/safe code."""
        lower = text.lower()
        return any(pattern in lower for pattern in cls.BENIGN_PATTERNS)

    @classmethod
    def _check_negated_vulnerability(cls, text: str) -> bool:
        """Check if text contains negated vulnerability patterns."""
        lower = text.lower()
        return any(pattern in lower for pattern in cls.NEGATION_PATTERNS)

    @classmethod
    def _check_vulnerable(cls, text: str) -> bool:
        """Check if text indicates vulnerability."""
        lower = text.lower()
        # "vulnerab" matches both "vulnerable" and "vulnerability"
        return "vulnerab" in lower

    @classmethod
    def _check_positive_response(cls, text: str) -> bool:
        """Check for positive indicators (yes, 1, etc.)."""
        lower = text.lower().strip()
        if lower.startswith("1") or lower.startswith("yes"):
            return True
        if "yes" in lower:
            return True
        return False

    @classmethod
    def _extract_cwe_id(cls, text: str) -> Optional[int]:
        """Extract first CWE ID from text."""
        match = cls._CWE_PATTERN.search(text)
        if match:
            try:
                return int(match.group(1))
            except ValueError:
                pass
        return None

    @classmethod
    def _match_category_keywords(cls, text: str) -> Optional[str]:
        """Match text against category keywords.

        Returns the first matching category, or None if no match.
        """
        lower = text.lower()

        # Check each category's keywords
        for category, keywords in cls.CATEGORY_KEYWORDS.items():
            if any(kw in lower for kw in keywords):
                return category

        # Special case: "overflow" alone could be integer or buffer
        # We default to Integer Errors for standalone "overflow"
        if "overflow" in lower and "buffer" not in lower:
            return "Integer Errors"
        if "underflow" in lower:
            return "Integer Errors"

        # Check for benign indicators in category context
        if cls._check_benign(text) or "no vuln" in lower:
            return "Benign"

        return None

    @classmethod
    def extract_vulnerability_label(cls, response: Optional[str]) -> str:
        """Extract binary vulnerability label from LLM response.

        Args:
            response: Raw LLM response text

        Returns:
            "1" if vulnerable, "0" if benign/safe
        """
        text = cls.normalize(response)
        if not text:
            return "0"

        lower = text.lower()

        # Priority 1: Explicit benign indicators
        if cls._check_benign(lower):
            return "0"

        # Priority 2: Negated vulnerability patterns
        if cls._check_negated_vulnerability(lower):
            return "0"

        # Priority 3: Vulnerability indicators
        if cls._check_vulnerable(lower):
            return "1"

        # Priority 4: Positive response patterns
        if cls._check_positive_response(lower):
            return "1"

        # Default: benign
        return "0"

    @classmethod
    def extract_cwe_category(cls, response: Optional[str]) -> str:
        """Extract CWE major category from LLM response.

        Args:
            response: Raw LLM response text

        Returns:
            One of CWE_MAJOR_CATEGORIES
        """
        text = cls.normalize(response)
        if not text:
            return "Other"

        lower = text.lower()

        # Priority 1: Exact category name match
        for cat in CWE_MAJOR_CATEGORIES:
            if lower == cat.lower():
                return cat

        # Priority 2: CWE ID extraction
        cwe_id = cls._extract_cwe_id(text)
        if cwe_id and cwe_id in _CWE_ID_TO_MAJOR:
            return _CWE_ID_TO_MAJOR[cwe_id]

        # Priority 3: Keyword matching
        category = cls._match_category_keywords(text)
        if category:
            return category

        return "Other"

    @classmethod
    def parse(cls, response: Optional[str]) -> ParsedResponse:
        """Parse LLM response into structured result.

        This is the main entry point for full response parsing.

        Args:
            response: Raw LLM response text

        Returns:
            ParsedResponse with all extracted information
        """
        raw = response or ""
        normalized = cls.normalize(response).lower()
        vulnerability_label = cls.extract_vulnerability_label(response)
        is_vulnerable = vulnerability_label == "1"
        cwe_category = cls.extract_cwe_category(response)

        return ParsedResponse(
            raw=raw,
            normalized=normalized,
            is_vulnerable=is_vulnerable,
            vulnerability_label=vulnerability_label,
            cwe_category=cwe_category,
        )


# =============================================================================
# Backward-compatible function aliases
# =============================================================================

def normalize_text(text: Optional[str]) -> str:
    """Normalize response text for downstream parsing.

    .. deprecated::
        Use ``ResponseParser.normalize()`` instead.
    """
    return ResponseParser.normalize(text)


def extract_vulnerability_label(response: Optional[str]) -> str:
    """Map free-form LLM output to binary vulnerability labels.

    .. deprecated::
        Use ``ResponseParser.extract_vulnerability_label()`` instead.
    """
    return ResponseParser.extract_vulnerability_label(response)


def extract_cwe_major(response: Optional[str]) -> str:
    """Parse CWE major category from LLM output.

    .. deprecated::
        Use ``ResponseParser.extract_cwe_category()`` instead.
    """
    return ResponseParser.extract_cwe_category(response)
