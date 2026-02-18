"""Router Agent for top-k category routing.

Implements Stage I from method.md Algorithm 3:
1. Convert code to SCALE representation T(x)
2. Retrieve cross-type contrastive evidence from K
3. Predict top-k categories with confidence scores
"""

import json
import re
from typing import List, Tuple, Optional, Dict, Any

from .base import RoutingResult

# Major categories aligned with CWE taxonomy
MAJOR_CATEGORIES = ["Memory", "Injection", "Logic", "Input", "Crypto", "Benign"]

DEFAULT_ROUTER_PROMPT = """You are a security expert routing code to specialized vulnerability detectors.

Analyze the code and predict the TOP-3 most likely vulnerability categories.

## Categories:
- Memory: Buffer overflow, use-after-free, null pointer, integer overflow
- Injection: SQL injection, command injection, code injection, XSS
- Logic: Race condition, improper access control, information exposure
- Input: Path traversal, improper input validation
- Crypto: Weak cryptography, improper certificate validation
- Benign: No vulnerability detected

## Contrastive Evidence:
{evidence}

## Code to Analyze:
```
{code}
```

## Output Format (JSON):
{{
  "predictions": [
    {{"category": "Memory", "confidence": 0.85, "reason": "Buffer operations without bounds checking"}},
    {{"category": "Benign", "confidence": 0.10, "reason": "Could be safe if inputs are validated"}},
    {{"category": "Injection", "confidence": 0.05, "reason": "No obvious injection points"}}
  ]
}}

Respond with ONLY the JSON object."""


class RouterAgent:
    """Routes code to top-k most relevant Detector Agents.

    Uses cross-type contrastive retrieval to provide evidence
    spanning multiple vulnerability categories.
    """

    def __init__(
        self,
        llm_client,
        retriever=None,
        prompt: str = DEFAULT_ROUTER_PROMPT,
        k: int = 3,
        categories: List[str] = None,
    ):
        self.llm_client = llm_client
        self.retriever = retriever
        self.prompt = prompt
        self.k = k
        self.categories = categories or MAJOR_CATEGORIES

    def route(self, code: str) -> RoutingResult:
        """Route code to top-k categories.

        Args:
            code: Source code to analyze

        Returns:
            RoutingResult with top-k categories and confidence scores
        """
        # 1. Retrieve cross-type contrastive evidence
        evidence = self._retrieve_contrastive_evidence(code)

        # 2. Format prompt
        evidence_text = self._format_evidence(evidence)
        prompt = self.prompt.format(code=code[:4000], evidence=evidence_text)

        # 3. Query LLM
        response = self.llm_client.generate(prompt)

        # 4. Parse response
        categories = self._parse_response(response)

        return RoutingResult(
            categories=categories[:self.k],
            evidence_used=evidence,
            raw_response=response,
        )

    def _retrieve_contrastive_evidence(self, code: str, n_per_category: int = 2) -> List[Dict]:
        """Retrieve samples from each category for contrastive evidence."""
        if not self.retriever:
            return []

        evidence = []
        for category in self.categories:
            if category == "Benign":
                continue
            samples = self.retriever.retrieve_from_category(
                code, category, top_k=n_per_category
            )
            evidence.extend(samples)

        return evidence

    def _format_evidence(self, evidence: List[Dict]) -> str:
        """Format evidence for prompt injection."""
        if not evidence:
            return "No contrastive evidence available."

        lines = []
        for i, sample in enumerate(evidence[:6], 1):
            category = sample.get("category", "Unknown")
            snippet = sample.get("code", "")[:200]
            lines.append(f"Example {i} [{category}]: {snippet}...")

        return "\n".join(lines)

    def _parse_response(self, response: str) -> List[Tuple[str, float]]:
        """Parse LLM response to extract categories and confidence."""
        # Try JSON parsing
        try:
            # Extract JSON from response
            json_match = re.search(r'\{[\s\S]*\}', response)
            if json_match:
                data = json.loads(json_match.group())
                predictions = data.get("predictions", [])
                return [
                    (p.get("category", "Unknown"), float(p.get("confidence", 0.5)))
                    for p in predictions
                ]
        except (json.JSONDecodeError, ValueError):
            pass

        # Fallback: keyword matching
        return self._fallback_parse(response)

    def _fallback_parse(self, response: str) -> List[Tuple[str, float]]:
        """Fallback parsing using keyword matching."""
        response_lower = response.lower()
        scores = []

        for category in self.categories:
            cat_lower = category.lower()
            if cat_lower in response_lower:
                # Higher score if mentioned first
                pos = response_lower.find(cat_lower)
                score = 1.0 - (pos / max(len(response_lower), 1))
                scores.append((category, min(score, 0.9)))

        # Sort by score descending
        scores.sort(key=lambda x: x[1], reverse=True)

        # Ensure we have at least one prediction
        if not scores:
            scores = [("Benign", 0.5)]

        return scores
