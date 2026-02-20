"""CWE Knowledge Base: load CWE descriptions, code examples, and category info."""

import json
import re
from collections import defaultdict
from pathlib import Path
from typing import Dict, List, Optional

import sys
sys.path.insert(0, str(Path(__file__).resolve().parents[3] / "src"))

from evoprompt.data.cwe_hierarchy import CWE_TO_MIDDLE, MIDDLE_TO_MAJOR, CWE_DESCRIPTIONS


class CWEKnowledgeBase:
    """Loads and serves CWE knowledge for prompt construction."""

    def __init__(self, kb_path: str):
        with open(kb_path) as f:
            self.kb = json.load(f)

        # CWE descriptions: cwe_id (int) -> description string
        self.descriptions: Dict[int, str] = {}
        for cwe_key, info in self.kb.get("descriptions", {}).items():
            m = re.search(r"(\d+)", str(cwe_key))
            if m:
                self.descriptions[int(m.group(1))] = info

        # Code examples: cwe_id (int) -> list of code strings
        self.examples: Dict[int, List[str]] = defaultdict(list)
        for cwe_key, examples in self.kb.get("examples", {}).items():
            m = re.search(r"(\d+)", str(cwe_key))
            if m:
                cwe_id = int(m.group(1))
                for ex in examples:
                    code = ex if isinstance(ex, str) else ex.get("code", "")
                    if code:
                        self.examples[cwe_id].append(code)

        # Category info from hierarchy
        self.cwe_to_middle = CWE_TO_MIDDLE
        self.middle_to_major = MIDDLE_TO_MAJOR
        self.cwe_short_desc = CWE_DESCRIPTIONS

        # All known CWE IDs in the KB
        self.all_cwe_ids = sorted(
            set(self.descriptions.keys()) | set(self.examples.keys()) | set(CWE_TO_MIDDLE.keys())
        )

    def get_cwe_context(self, cwe_id: int, max_example_lines: int = 20) -> str:
        """Get formatted context for a single CWE for inclusion in prompts."""
        parts = [f"CWE-{cwe_id}"]

        # Description
        desc = self.descriptions.get(cwe_id) or self.cwe_short_desc.get(cwe_id, "")
        if desc:
            parts.append(f"  Description: {desc}")

        # Category
        middle = self.cwe_to_middle.get(cwe_id, "Other")
        major = self.middle_to_major.get(middle, "Logic")
        parts.append(f"  Category: {major} > {middle}")

        # One example (truncated)
        examples = self.examples.get(cwe_id, [])
        if examples:
            ex = examples[0]
            lines = ex.split("\n")
            if len(lines) > max_example_lines:
                ex = "\n".join(lines[:max_example_lines]) + "\n  // ... truncated"
            parts.append(f"  Example:\n  ```c\n  {ex}\n  ```")

        return "\n".join(parts)

    def get_candidates_context(self, cwe_ids: List[int], max_example_lines: int = 15) -> str:
        """Get formatted context for multiple candidate CWEs."""
        sections = []
        for cwe_id in cwe_ids:
            sections.append(self.get_cwe_context(cwe_id, max_example_lines))
        return "\n\n".join(sections)

    @staticmethod
    def build(
        knowledge_cwe_path: str,
        train_data_path: str,
        output_path: str,
        max_examples_per_cwe: int = 5,
        max_example_chars: int = 3000,
    ) -> "CWEKnowledgeBase":
        """Build knowledge base from raw data sources."""
        print("Building CWE Knowledge Base...")

        # 1. Load CWE descriptions from knowledge-cwe.jsonl
        descriptions = {}
        with open(knowledge_cwe_path) as f:
            for line in f:
                entry = json.loads(line)
                cwe_id_str = entry.get("cwe_id", "")
                m = re.search(r"(\d+)", str(cwe_id_str))
                if not m:
                    continue
                cwe_id = int(m.group(1))
                name = entry.get("name", "")
                definition = entry.get("definition", "")
                desc = f"{name}: {definition}" if definition else name
                descriptions[cwe_id] = desc

        # Merge with built-in short descriptions
        for cwe_id, short_desc in CWE_DESCRIPTIONS.items():
            if cwe_id not in descriptions:
                descriptions[cwe_id] = short_desc

        print(f"  Loaded {len(descriptions)} CWE descriptions")

        # 2. Collect code examples from training data (vulnerable only)
        examples_by_cwe: Dict[int, List[dict]] = defaultdict(list)
        with open(train_data_path) as f:
            for line in f:
                item = json.loads(line)
                if int(item.get("target", 0)) == 0:
                    continue
                cwe_codes = item.get("cwe", [])
                if isinstance(cwe_codes, str):
                    cwe_codes = [cwe_codes]
                if not cwe_codes:
                    continue
                cwe_str = cwe_codes[0]
                m = re.search(r"(\d+)", str(cwe_str))
                if not m:
                    continue
                cwe_id = int(m.group(1))
                code = item.get("func", "")
                if not code or len(code) < 20:
                    continue
                examples_by_cwe[cwe_id].append({
                    "code": code,
                    "len": len(code),
                    "cve": item.get("cve", ""),
                })

        # Pick shortest examples (most focused) per CWE
        final_examples: Dict[str, List[dict]] = {}
        for cwe_id, exs in examples_by_cwe.items():
            exs.sort(key=lambda x: x["len"])
            selected = []
            for ex in exs[:max_examples_per_cwe]:
                code = ex["code"]
                if len(code) > max_example_chars:
                    code = code[:max_example_chars] + "\n// ... truncated"
                selected.append({"code": code, "cve": ex["cve"]})
            final_examples[f"CWE-{cwe_id}"] = selected

        print(f"  Collected examples for {len(final_examples)} CWEs")

        # 3. Save KB
        kb_data = {
            "descriptions": {f"CWE-{k}": v for k, v in descriptions.items()},
            "examples": final_examples,
            "cwe_to_middle": {f"CWE-{k}": v for k, v in CWE_TO_MIDDLE.items()},
            "middle_to_major": MIDDLE_TO_MAJOR,
        }

        output_path = Path(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        with open(output_path, "w") as f:
            json.dump(kb_data, f, indent=2, ensure_ascii=False)

        print(f"  Saved KB to {output_path}")
        print(f"  Total: {len(descriptions)} descriptions, {sum(len(v) for v in final_examples.values())} examples")

        return CWEKnowledgeBase(str(output_path))
