"""Meta-prompter: uses LLM to generate improved classification rules."""

import json
import re
import time
from typing import List, Optional

from openai import OpenAI


META_PROMPT_TEMPLATE = """You are a meta-learning system that improves vulnerability classification rules.

## Current Classification Rules
{current_rules}

## Error Analysis
{error_analysis}

## Your Task
Based on the error analysis above, generate an improved set of classification rules. Focus on:
1. Fix the TOP 3-5 confusion pairs shown above
2. Keep rules that are working well (classes with correct predictions)
3. Add specific distinguishing criteria for commonly confused CWE pairs
4. Each rule should be actionable: "If [specific pattern], classify as [CWE-XXX], NOT [CWE-YYY]"
5. Maximum {max_rules} rules total

## Output Format
Return ONLY a JSON array of rule strings, numbered sequentially:
[
  "1. If the code ..., classify as CWE-XXX, NOT CWE-YYY.",
  "2. ...",
  ...
]

Return ONLY the JSON array, no other text."""


class MetaPrompter:
    """Uses an LLM to generate improved classification rules based on error analysis."""

    def __init__(
        self,
        api_base: str,
        api_key: str,
        model: str = "gpt-4o",
        temperature: float = 0.3,
        max_tokens: int = 4000,
    ):
        self.client = OpenAI(base_url=api_base, api_key=api_key)
        self.model = model
        self.temperature = temperature
        self.max_tokens = max_tokens
        self.update_history: List[dict] = []

    def generate_improved_rules(
        self,
        current_rules: List[str],
        error_analysis_text: str,
        max_rules: int = 30,
    ) -> List[str]:
        """Generate improved rules based on current rules and error analysis."""
        current_rules_text = "\n".join(current_rules)

        prompt = META_PROMPT_TEMPLATE.format(
            current_rules=current_rules_text,
            error_analysis=error_analysis_text,
            max_rules=max_rules,
        )

        for attempt in range(3):
            try:
                t0 = time.time()
                resp = self.client.chat.completions.create(
                    model=self.model,
                    messages=[{"role": "user", "content": prompt}],
                    temperature=self.temperature,
                    max_tokens=self.max_tokens,
                )
                content = resp.choices[0].message.content.strip()
                elapsed = time.time() - t0

                new_rules = self._parse_rules(content)
                if new_rules and len(new_rules) >= 3:
                    self.update_history.append({
                        "timestamp": time.time(),
                        "n_rules_before": len(current_rules),
                        "n_rules_after": len(new_rules),
                        "elapsed": elapsed,
                    })
                    print(f"  MetaPrompter: {len(current_rules)} -> {len(new_rules)} rules ({elapsed:.1f}s)")
                    return new_rules[:max_rules]

                print(f"  MetaPrompter: parse failed (got {len(new_rules)} rules), retrying...")
            except Exception as e:
                print(f"  MetaPrompter error (attempt {attempt+1}): {e}")
                if attempt < 2:
                    time.sleep(2 ** (attempt + 1))

        print("  MetaPrompter: all attempts failed, keeping current rules")
        return current_rules

    def _parse_rules(self, content: str) -> List[str]:
        """Parse rules from LLM response."""
        # Try JSON parse first
        try:
            # Find JSON array in response
            m = re.search(r'\[.*\]', content, re.DOTALL)
            if m:
                rules = json.loads(m.group(0))
                if isinstance(rules, list) and all(isinstance(r, str) for r in rules):
                    return [r.strip() for r in rules if r.strip()]
        except (json.JSONDecodeError, ValueError):
            pass

        # Fallback: parse numbered list
        lines = content.split("\n")
        rules = []
        for line in lines:
            line = line.strip()
            if not line:
                continue
            # Match lines starting with a number
            m = re.match(r'^\d+[\.\)]\s*(.+)', line)
            if m:
                rules.append(line)
            elif line.startswith('"') and line.endswith('"'):
                rules.append(line.strip('"'))
            elif line.startswith("- "):
                rules.append(line[2:])

        return rules
