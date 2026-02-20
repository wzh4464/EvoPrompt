"""Prompt template with fixed structure and learnable rules."""

from typing import List

# Initial rules based on known confusion patterns from prior experiments
DEFAULT_INITIAL_RULES = [
    "1. If the code uses memcpy/strcpy/strcat without explicit size bounds checking, classify as CWE-120 (Classic Buffer Overflow), NOT CWE-119 or CWE-787.",
    "2. If the code accesses an array with an index that could exceed allocated bounds (read path), classify as CWE-125 (Out-of-bounds Read). If the access is on a write path, classify as CWE-787 (Out-of-bounds Write).",
    "3. If the code calls free() on a pointer and later dereferences that same pointer, classify as CWE-416 (Use After Free), NOT CWE-476.",
    "4. If the code dereferences a pointer that was never checked for NULL after allocation or function return, classify as CWE-476 (NULL Pointer Dereference).",
    "5. If the code performs arithmetic (especially multiplication or addition) on integers without overflow checks before using the result as a size/index, classify as CWE-190 (Integer Overflow).",
    "6. If there is no clear vulnerability pattern (no unsafe memory ops, no missing checks, no injection), classify as Benign. Do NOT guess a CWE when the code looks safe.",
    "7. For cryptographic code: using deprecated algorithms (DES, MD5, SHA1 for security) is CWE-327; using rand()/time() for random numbers is CWE-330; storing passwords in plaintext is CWE-312.",
    "8. If user-controlled input flows into system()/exec()/popen() without sanitization, classify as CWE-78 (OS Command Injection).",
    "9. If the code has a time-of-check-to-time-of-use pattern (check then use without lock), classify as CWE-362 (Race Condition).",
    "10. When multiple vulnerability types seem present, choose the most specific one that matches the primary security impact of the code.",
]


CLASSIFY_TEMPLATE = """You are an expert C/C++ security auditor performing vulnerability classification.

## Task
Analyze the code below. Determine if it is Benign (no vulnerability) or contains a specific CWE vulnerability.

## Candidate CWE Types (from code similarity analysis)
{cwe_candidates}

## Classification Rules
{learnable_rules}

## Code
```c
{code}
```

Output a single JSON object (no other text):
{{"label": "Benign" or "CWE-XXX", "confidence": 0.0-1.0, "reason": "one sentence explaining your decision"}}"""


class PromptTemplate:
    """Manages the classification prompt with learnable rules."""

    def __init__(self, initial_rules: List[str] = None):
        self.rules = list(initial_rules or DEFAULT_INITIAL_RULES)

    def render(self, code: str, cwe_candidates_text: str) -> str:
        """Render the full classification prompt."""
        rules_text = "\n".join(self.rules)
        return CLASSIFY_TEMPLATE.format(
            cwe_candidates=cwe_candidates_text,
            learnable_rules=rules_text,
            code=code,
        )

    def update_rules(self, new_rules: List[str]):
        """Replace learnable rules with updated ones from meta-prompter."""
        self.rules = list(new_rules)

    def get_rules_text(self) -> str:
        return "\n".join(self.rules)

    def num_rules(self) -> int:
        return len(self.rules)
