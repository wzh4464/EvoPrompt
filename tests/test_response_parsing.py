"""Tests for response parsing utilities.

Covers:
- ResponseParser class (unified parsing interface)
- Vulnerability label extraction (binary classification)
- CWE major category extraction
- Edge cases and real API response formats
- Backward-compatible function aliases
"""
import os
import pytest

from evoprompt.utils.response_parsing import (
    ResponseParser,
    ParsedResponse,
    # Backward-compatible aliases
    extract_cwe_major,
    extract_vulnerability_label,
    normalize_text,
)
from evoprompt.data.cwe_categories import (
    CWE_MAJOR_CATEGORIES,
    canonicalize_category,
    map_cwe_to_major,
)


pytestmark = pytest.mark.skipif(
    os.getenv("RUN_RESPONSE_PARSING_TESTS") != "1",
    reason="响应解析测试默认关闭，设置 RUN_RESPONSE_PARSING_TESTS=1 后再运行",
)


# =============================================================================
# ResponseParser.normalize tests
# =============================================================================
class TestNormalize:
    @pytest.mark.parametrize(
        "input_text,expected",
        [
            ("  hello world  ", "hello world"),
            ("\n\ntest\n\n", "test"),
            ("", ""),
            (None, ""),
            ("no change", "no change"),
        ],
    )
    def test_normalize(self, input_text, expected):
        assert ResponseParser.normalize(input_text) == expected

    def test_backward_compat_normalize_text(self):
        """Test backward-compatible function alias."""
        assert normalize_text("  test  ") == "test"
        assert normalize_text(None) == ""


# =============================================================================
# ResponseParser.extract_vulnerability_label tests
# =============================================================================
class TestExtractVulnerabilityLabel:
    """Test binary vulnerability classification from LLM responses."""

    @pytest.mark.parametrize(
        "response,expected",
        [
            # Explicit "vulnerable" keyword
            ("The code is vulnerable.\nRecommend patching.", "1"),
            ("This function is VULNERABLE to buffer overflow", "1"),
            ("vulnerable", "1"),
            ("VULNERABLE", "1"),
            # "vulnerability" keyword (bug fix)
            ("This code has a vulnerability", "1"),
            ("buffer overflow vulnerability detected", "1"),
            # Explicit "benign" keyword
            ("BENIGN sample with no issues detected", "0"),
            ("The code appears benign", "0"),
            ("benign", "0"),
            ("BENIGN", "0"),
            # Yes/No patterns
            ("Yes, clearly a vulnerability", "1"),
            ("yes", "1"),
            ("YES", "1"),
            ("1", "1"),
            # Edge cases
            ("", "0"),
            (None, "0"),
            ("   ", "0"),
            # Ambiguous responses default to 0
            ("I'm not sure about this code", "0"),
            ("Need more context", "0"),
        ],
    )
    def test_basic_labels(self, response, expected):
        assert ResponseParser.extract_vulnerability_label(response) == expected

    @pytest.mark.parametrize(
        "response,expected",
        [
            # Negated vulnerability patterns
            ("no vulnerability found", "0"),
            ("not vulnerable to attacks", "0"),
            ("This code is free of vulnerabilities", "0"),
            ("no security issues detected", "0"),
            # Real API response formats (multi-line)
            (
                "Analysis: This code contains a buffer overflow vulnerability.\n"
                "The vulnerable line is at line 42 where memcpy is used without bounds checking.",
                "1",
            ),
            (
                "After careful analysis, I conclude this code is benign.\n"
                "No security issues were detected.",
                "0",
            ),
            # JSON-like responses
            ('{"result": "vulnerable", "confidence": 0.95}', "1"),
            ('{"classification": "benign"}', "0"),
            # Markdown formatted responses
            ("**Verdict**: Vulnerable\n\n- CWE-120: Buffer Overflow", "1"),
            ("## Analysis\n\nThe code is **benign**.", "0"),
        ],
    )
    def test_real_api_formats(self, response, expected):
        assert ResponseParser.extract_vulnerability_label(response) == expected

    def test_backward_compat_function(self):
        """Test backward-compatible function alias."""
        assert extract_vulnerability_label("vulnerable") == "1"
        assert extract_vulnerability_label("benign") == "0"


# =============================================================================
# ResponseParser.extract_cwe_category tests
# =============================================================================
class TestExtractCweCategory:
    """Test CWE major category classification."""

    @pytest.mark.parametrize(
        "response,expected",
        [
            # Direct category names
            ("Benign", "Benign"),
            ("Buffer Errors", "Buffer Errors"),
            ("Injection", "Injection"),
            ("Memory Management", "Memory Management"),
            ("Pointer Dereference", "Pointer Dereference"),
            ("Integer Errors", "Integer Errors"),
            ("Concurrency Issues", "Concurrency Issues"),
            ("Path Traversal", "Path Traversal"),
            ("Cryptography Issues", "Cryptography Issues"),
            ("Information Exposure", "Information Exposure"),
            ("Other", "Other"),
            # Case insensitive
            ("buffer errors", "Buffer Errors"),
            ("INJECTION", "Injection"),
            ("BENIGN", "Benign"),
        ],
    )
    def test_direct_category_names(self, response, expected):
        assert ResponseParser.extract_cwe_category(response) == expected

    @pytest.mark.parametrize(
        "response,expected",
        [
            # CWE ID patterns
            ("CWE-120 classic overflow", "Buffer Errors"),
            ("CWE-119: Buffer overflow", "Buffer Errors"),
            ("CWE-787 out-of-bounds write", "Buffer Errors"),
            ("CWE-125", "Buffer Errors"),
            ("CWE-89 SQL Injection", "Injection"),
            ("CWE-78 OS Command Injection", "Injection"),
            ("CWE-79 XSS", "Injection"),
            ("CWE-416 Use After Free", "Memory Management"),
            ("CWE-415 Double Free", "Memory Management"),
            ("CWE-476 NULL Pointer Dereference", "Pointer Dereference"),
            ("CWE-190 Integer Overflow", "Integer Errors"),
            ("CWE-362 Race Condition", "Concurrency Issues"),
            ("CWE-22 Path Traversal", "Path Traversal"),
            ("CWE-327 Broken Crypto", "Cryptography Issues"),
            ("CWE-200 Information Exposure", "Information Exposure"),
        ],
    )
    def test_cwe_id_parsing(self, response, expected):
        assert ResponseParser.extract_cwe_category(response) == expected

    @pytest.mark.parametrize(
        "response,expected",
        [
            # Keyword-based detection
            ("buffer overflow detected", "Buffer Errors"),
            ("out-of-bounds read", "Buffer Errors"),
            ("SQL injection detected", "Injection"),
            ("command injection vulnerability", "Injection"),
            ("use after free bug", "Memory Management"),
            ("double free vulnerability", "Memory Management"),
            ("memory leak detected", "Memory Management"),
            ("null pointer dereference", "Pointer Dereference"),
            ("integer overflow possible", "Integer Errors"),
            ("integer underflow", "Integer Errors"),
            ("race condition", "Concurrency Issues"),
            ("concurrency bug", "Concurrency Issues"),
            ("path traversal attack", "Path Traversal"),
            ("directory traversal", "Path Traversal"),
            ("weak cryptography", "Cryptography Issues"),
            ("broken encryption", "Cryptography Issues"),
            ("information exposure", "Information Exposure"),
            ("data leak", "Information Exposure"),
            ("no vulnerability detected", "Benign"),
            ("no vuln", "Benign"),
        ],
    )
    def test_keyword_detection(self, response, expected):
        assert ResponseParser.extract_cwe_category(response) == expected

    @pytest.mark.parametrize(
        "response,expected",
        [
            # Unknown/unrecognized patterns
            ("Unknown text", "Other"),
            ("something random", "Other"),
            ("", "Other"),
            ("   ", "Other"),
        ],
    )
    def test_fallback_to_other(self, response, expected):
        assert ResponseParser.extract_cwe_category(response) == expected

    def test_backward_compat_function(self):
        """Test backward-compatible function alias."""
        assert extract_cwe_major("CWE-120") == "Buffer Errors"
        assert extract_cwe_major("benign") == "Benign"


# =============================================================================
# ResponseParser.parse tests
# =============================================================================
class TestParsedResponse:
    """Test the unified parse() method."""

    def test_parse_vulnerable_with_category(self):
        response = "This code has a CWE-120 buffer overflow vulnerability"
        result = ResponseParser.parse(response)

        assert isinstance(result, ParsedResponse)
        assert result.raw == response
        assert result.is_vulnerable is True
        assert result.vulnerability_label == "1"
        assert result.cwe_category == "Buffer Errors"

    def test_parse_benign(self):
        response = "The code is benign and safe"
        result = ResponseParser.parse(response)

        assert result.is_vulnerable is False
        assert result.vulnerability_label == "0"
        assert result.cwe_category == "Benign"

    def test_parse_empty(self):
        result = ResponseParser.parse("")
        assert result.is_vulnerable is False
        assert result.vulnerability_label == "0"
        assert result.cwe_category == "Other"

    def test_parse_none(self):
        result = ResponseParser.parse(None)
        assert result.raw == ""
        assert result.is_vulnerable is False


# =============================================================================
# CWE category data tests
# =============================================================================
class TestCweCategories:
    """Test CWE ID to major category mapping."""

    def test_all_major_categories_defined(self):
        """Verify all expected categories are in the list."""
        expected = [
            "Benign",
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
        assert CWE_MAJOR_CATEGORIES == expected

    @pytest.mark.parametrize(
        "cwe_codes,expected",
        [
            (["CWE-120"], "Buffer Errors"),
            (["CWE-89"], "Injection"),
            (["CWE-416"], "Memory Management"),
            (["CWE-476"], "Pointer Dereference"),
            (["CWE-190"], "Integer Errors"),
            (["CWE-362"], "Concurrency Issues"),
            (["CWE-22"], "Path Traversal"),
            (["CWE-327"], "Cryptography Issues"),
            (["CWE-200"], "Information Exposure"),
            # Unknown CWE
            (["CWE-9999"], "Other"),
            ([], "Other"),
            # Multiple CWEs - priority order
            (["CWE-89", "CWE-120"], "Buffer Errors"),  # Buffer > Injection
            (["CWE-89", "CWE-416"], "Injection"),  # Injection > Memory
        ],
    )
    def test_map_cwe_to_major(self, cwe_codes, expected):
        assert map_cwe_to_major(cwe_codes) == expected

    @pytest.mark.parametrize(
        "text,expected",
        [
            # Exact category names
            ("Buffer Errors", "Buffer Errors"),
            ("injection", "Injection"),
            # CWE IDs
            ("CWE-120", "Buffer Errors"),
            ("CWE-89", "Injection"),
            # None cases (keywords now handled by ResponseParser)
            ("", None),
            ("random gibberish xyz", None),
            ("buffer overflow", None),  # keyword matching removed from canonicalize
        ],
    )
    def test_canonicalize_category(self, text, expected):
        """Test canonicalize_category (now only exact match + CWE ID)."""
        assert canonicalize_category(text) == expected


# =============================================================================
# Integration tests with real-world response formats
# =============================================================================
class TestRealWorldResponses:
    """Test parsing of realistic LLM response formats."""

    @pytest.mark.parametrize(
        "response,expected_label,expected_category",
        [
            # Typical structured response
            (
                "This code contains a CWE-120 buffer overflow vulnerability. "
                "The memcpy call at line 15 does not check buffer bounds.",
                "1",
                "Buffer Errors",
            ),
            # Response with reasoning
            (
                "After analyzing the code, I found no security vulnerabilities. "
                "The input validation is properly implemented. Verdict: Benign.",
                "0",
                "Benign",
            ),
            # Response with CWE reference
            (
                "Vulnerable: Yes\n"
                "Category: CWE-89 (SQL Injection)\n"
                "The query is constructed using string concatenation.",
                "1",
                "Injection",
            ),
            # Minimal response
            ("CWE-416", "0", "Memory Management"),
            ("vulnerable", "1", "Other"),
        ],
    )
    def test_combined_extraction(self, response, expected_label, expected_category):
        assert ResponseParser.extract_vulnerability_label(response) == expected_label
        assert ResponseParser.extract_cwe_category(response) == expected_category

    def test_full_parse_integration(self):
        """Test full parse() with realistic response."""
        response = """
        ## Security Analysis

        After reviewing the code, I identified a **buffer overflow vulnerability**.

        - **CWE**: CWE-120 (Classic Buffer Overflow)
        - **Location**: Line 42, memcpy() call
        - **Severity**: High

        The code does not validate the input length before copying to a fixed-size buffer.
        """
        result = ResponseParser.parse(response)

        assert result.is_vulnerable is True
        assert result.vulnerability_label == "1"
        assert result.cwe_category == "Buffer Errors"
