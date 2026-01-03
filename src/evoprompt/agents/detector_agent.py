"""Category-specific Detector Agents.

Implements Stage II from method.md Algorithm 3:
1. Retrieve type-specific evidence from category subset K_m
2. Detect vulnerability type with confidence and evidence
3. Return structured DetectionResult
"""

import json
import re
from typing import Dict, Optional, List

from .base import DetectionResult

# Default prompts for each category
CATEGORY_PROMPTS = {
    "Memory": """You are a memory safety expert. Analyze this code for memory vulnerabilities.

## Category-Specific Evidence:
{evidence}

## Code:
```
{code}
```

## Detect these vulnerability types:
- Buffer Overflow (CWE-120, CWE-119)
- Use After Free (CWE-416)
- Null Pointer Dereference (CWE-476)
- Integer Overflow (CWE-190)
- Out-of-bounds Read/Write (CWE-125, CWE-787)

## Output (JSON):
{{
  "prediction": "CWE-120" or "Benign",
  "confidence": 0.85,
  "evidence": "Line 15: strcpy without bounds check",
  "subcategory": "Buffer Overflow"
}}""",

    "Injection": """You are an injection vulnerability expert. Analyze this code for injection flaws.

## Category-Specific Evidence:
{evidence}

## Code:
```
{code}
```

## Detect these vulnerability types:
- SQL Injection (CWE-89)
- Command Injection (CWE-78)
- Code Injection (CWE-94)
- XSS (CWE-79)
- LDAP Injection (CWE-90)

## Output (JSON):
{{
  "prediction": "CWE-89" or "Benign",
  "confidence": 0.85,
  "evidence": "Line 23: User input directly in SQL query",
  "subcategory": "SQL Injection"
}}""",

    "Logic": """You are a logic vulnerability expert. Analyze this code for logic flaws.

## Category-Specific Evidence:
{evidence}

## Code:
```
{code}
```

## Detect these vulnerability types:
- Race Condition (CWE-362)
- Improper Access Control (CWE-284)
- Information Exposure (CWE-200)
- Missing Authorization (CWE-862)

## Output (JSON):
{{
  "prediction": "CWE-362" or "Benign",
  "confidence": 0.85,
  "evidence": "Line 45: TOCTOU race condition",
  "subcategory": "Race Condition"
}}""",

    "Input": """You are an input validation expert. Analyze this code for input handling flaws.

## Category-Specific Evidence:
{evidence}

## Code:
```
{code}
```

## Detect these vulnerability types:
- Path Traversal (CWE-22)
- Improper Input Validation (CWE-20)
- Uncontrolled Format String (CWE-134)

## Output (JSON):
{{
  "prediction": "CWE-22" or "Benign",
  "confidence": 0.85,
  "evidence": "Line 12: User-controlled path without sanitization",
  "subcategory": "Path Traversal"
}}""",

    "Crypto": """You are a cryptography expert. Analyze this code for cryptographic weaknesses.

## Category-Specific Evidence:
{evidence}

## Code:
```
{code}
```

## Detect these vulnerability types:
- Use of Weak Crypto (CWE-327)
- Improper Certificate Validation (CWE-295)
- Hard-coded Credentials (CWE-798)
- Insufficient Entropy (CWE-331)

## Output (JSON):
{{
  "prediction": "CWE-327" or "Benign",
  "confidence": 0.85,
  "evidence": "Line 8: MD5 used for password hashing",
  "subcategory": "Weak Cryptography"
}}""",
}


class DetectorAgent:
    """Category-specific vulnerability detector.

    Each detector specializes in one vulnerability category
    and retrieves evidence from its category-specific subset K_m.
    """

    def __init__(
        self,
        category: str,
        llm_client,
        retriever=None,
        prompt: str = None,
    ):
        self.category = category
        self.llm_client = llm_client
        self.retriever = retriever
        self.prompt = prompt or CATEGORY_PROMPTS.get(category, CATEGORY_PROMPTS["Logic"])
        self.agent_id = f"detector_{category.lower()}"

    def detect(self, code: str) -> DetectionResult:
        """Detect vulnerability in code.

        Args:
            code: Source code to analyze

        Returns:
            DetectionResult with prediction, confidence, and evidence
        """
        # 1. Retrieve category-specific evidence
        evidence = self._retrieve_evidence(code)

        # 2. Format prompt
        evidence_text = self._format_evidence(evidence)
        prompt = self.prompt.format(code=code[:4000], evidence=evidence_text)

        # 3. Query LLM
        response = self.llm_client.generate(prompt)

        # 4. Parse response
        return self._parse_response(response)

    def _retrieve_evidence(self, code: str, top_k: int = 3) -> List[Dict]:
        """Retrieve evidence from category-specific subset."""
        if not self.retriever:
            return []

        return self.retriever.retrieve_from_category(
            code, self.category, top_k=top_k
        )

    def _format_evidence(self, evidence: List[Dict]) -> str:
        """Format evidence for prompt."""
        if not evidence:
            return "No category-specific evidence available."

        lines = []
        for i, sample in enumerate(evidence[:3], 1):
            vuln_type = sample.get("cwe", sample.get("type", "Unknown"))
            snippet = sample.get("code", "")[:300]
            lines.append(f"Example {i} [{vuln_type}]:\n```\n{snippet}\n```")

        return "\n\n".join(lines)

    def _parse_response(self, response: str) -> DetectionResult:
        """Parse LLM response to DetectionResult."""
        # Try JSON parsing
        try:
            json_match = re.search(r'\{[\s\S]*?\}', response)
            if json_match:
                data = json.loads(json_match.group())
                return DetectionResult(
                    prediction=data.get("prediction", "Unknown"),
                    confidence=float(data.get("confidence", 0.5)),
                    evidence=data.get("evidence", ""),
                    category=self.category,
                    subcategory=data.get("subcategory", ""),
                    cwe=data.get("prediction", "") if data.get("prediction", "").startswith("CWE") else "",
                    agent_id=self.agent_id,
                    raw_response=response,
                )
        except (json.JSONDecodeError, ValueError):
            pass

        # Fallback parsing
        return self._fallback_parse(response)

    def _fallback_parse(self, response: str) -> DetectionResult:
        """Fallback parsing using keyword matching."""
        response_lower = response.lower()

        # Check for benign indicators
        if any(word in response_lower for word in ["benign", "safe", "no vulnerability", "not vulnerable"]):
            return DetectionResult(
                prediction="Benign",
                confidence=0.6,
                evidence="No clear vulnerability pattern detected",
                category=self.category,
                agent_id=self.agent_id,
                raw_response=response,
            )

        # Extract CWE if mentioned
        cwe_match = re.search(r'CWE-(\d+)', response, re.IGNORECASE)
        cwe = f"CWE-{cwe_match.group(1)}" if cwe_match else ""

        return DetectionResult(
            prediction=cwe or self.category,
            confidence=0.5,
            evidence=response[:200],
            category=self.category,
            cwe=cwe,
            agent_id=self.agent_id,
            raw_response=response,
        )


class DetectorAgentFactory:
    """Factory for creating category-specific detector agents."""

    @staticmethod
    def create_all_detectors(
        llm_client,
        retriever=None,
        prompts: Dict[str, str] = None,
        categories: List[str] = None,
    ) -> Dict[str, DetectorAgent]:
        """Create detector agents for all categories.

        Args:
            llm_client: LLM client for queries
            retriever: Optional retriever for evidence
            prompts: Optional custom prompts per category
            categories: Categories to create detectors for

        Returns:
            Dict mapping category name to DetectorAgent
        """
        categories = categories or list(CATEGORY_PROMPTS.keys())
        prompts = prompts or {}

        return {
            category: DetectorAgent(
                category=category,
                llm_client=llm_client,
                retriever=retriever,
                prompt=prompts.get(category),
            )
            for category in categories
        }
