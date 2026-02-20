#!/usr/bin/env python3
"""Evolutionary optimization of MulVul prompts for CWE-130 Macro-F1.

Uses error-guided LLM mutation to evolve Router and Detector prompts.
Tracks evolution path statistics (per-CWE F1, coverage, confusion patterns)
and feeds them into the mutation process.

Usage:
    uv run python scripts/rebuttal/evolve_cwe130.py \
        --data outputs/rebuttal/cwe130/eval_subset_300.jsonl \
        --kb outputs/knowledge_base_hierarchical.json \
        --model gpt-4o \
        --generations 15 \
        --pop-size 6 \
        --workers 16 \
        --output-dir outputs/rebuttal/cwe130_evolution
"""

import os
import sys
import json
import copy
import time
import re
import random
import argparse
import hashlib
from pathlib import Path
from datetime import datetime
from collections import Counter, defaultdict
from concurrent.futures import ThreadPoolExecutor, as_completed
from typing import Dict, List, Optional, Tuple, Any
from dataclasses import dataclass, field, asdict

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "src"))

from evoprompt.llm.client import create_llm_client, load_env_vars
from evoprompt.agents.base import DetectionResult, RoutingResult
from evoprompt.agents.router_agent import RouterAgent
from evoprompt.agents.detector_agent import DetectorAgent, DetectorAgentFactory
from evoprompt.agents.aggregator import DecisionAggregator
from evoprompt.agents.mulvul import MulVulDetector
from evoprompt.rag.retriever import MulVulRetriever
from evoprompt.data.cwe_hierarchy import (
    CWE_TO_MIDDLE, MIDDLE_TO_MAJOR, CWE_DESCRIPTIONS,
    cwe_to_major, cwe_to_middle, extract_cwe_id,
    get_cwes_for_major,
)

# ============================================================================
# CWE descriptions (extended beyond cwe_hierarchy.py)
# ============================================================================
FULL_CWE_DESCRIPTIONS = {
    **CWE_DESCRIPTIONS,
    121: "Stack-based buffer overflow - overwriting stack memory",
    122: "Heap-based buffer overflow - overwriting heap memory",
    131: "Incorrect buffer size calculation - wrong allocation size",
    805: "Buffer access with incorrect length value",
    401: "Memory leak - missing release of memory after use",
    772: "Resource leak - missing release of resource after use",
    415: "Double free - freeing memory pointer twice",
    617: "Reachable assertion - assertion failure reachable by attacker",
    191: "Integer underflow - arithmetic below minimum integer",
    189: "Numeric errors - general arithmetic/numeric issues",
    369: "Divide by zero - division with unchecked zero divisor",
    74: "Injection - general injection of untrusted data",
    77: "Command injection - injecting commands into execution",
    79: "Cross-site scripting (XSS) - injecting client-side scripts",
    94: "Code injection - injecting executable code",
    667: "Improper locking - incorrect synchronization",
    59: "Improper link resolution - symlink following",
    310: "Cryptographic issues - general crypto weaknesses",
    311: "Missing encryption of sensitive data",
    312: "Cleartext storage of sensitive information",
    326: "Inadequate encryption strength",
    327: "Broken or risky cryptographic algorithm",
    330: "Insufficiently random values",
    254: "Security features - general security mechanism issues",
    209: "Info leak via error message - sensitive data in errors",
    399: "Resource management errors - general resource issues",
    400: "Uncontrolled resource consumption (DoS)",
    770: "Resource allocation without limits",
    835: "Infinite loop - loop with unreachable exit",
    264: "Permissions/privileges - improper access controls",
    284: "Improper access control - missing authorization checks",
    269: "Improper privilege management",
    703: "Improper exception handling - unchecked error conditions",
}


# ============================================================================
# Enriched base prompts (Generation 0 seed)
# ============================================================================

def build_cwe_list_for_major(major: str) -> str:
    """Build a formatted CWE list for a major category."""
    cwes = get_cwes_for_major(major)
    lines = []
    for cwe_id in sorted(cwes):
        desc = FULL_CWE_DESCRIPTIONS.get(cwe_id, "")
        middle = CWE_TO_MIDDLE.get(cwe_id, "Other")
        lines.append(f"  - CWE-{cwe_id}: {desc} [{middle}]")
    return "\n".join(lines)


ENRICHED_ROUTER_PROMPT = """You are a security expert routing source code to specialized vulnerability detectors.

Analyze the code and predict the TOP-3 most likely vulnerability categories with confidence scores.

## Categories and Their Coverage:
- **Memory**: Buffer overflow (CWE-119/120/121/122/125/787), use-after-free (CWE-416), double free (CWE-415), memory leak (CWE-401), null pointer (CWE-476), integer overflow (CWE-190/189/369), assertion (CWE-617)
- **Injection**: SQL injection (CWE-89), command injection (CWE-78), code injection (CWE-94), XSS (CWE-79), general injection (CWE-74)
- **Logic**: Race condition (CWE-362), improper locking (CWE-667), information exposure (CWE-200/209), resource management (CWE-399/400/770/835), access control (CWE-264/284/269)
- **Input**: Path traversal (CWE-22/59), improper input validation (CWE-20), improper exception handling (CWE-703)
- **Crypto**: Weak crypto (CWE-310/327), missing encryption (CWE-311), cleartext storage (CWE-312), weak randomness (CWE-330)
- **Benign**: No vulnerability detected

## Key Routing Rules:
- Error handling / exception checking issues -> Input (CWE-703)
- Divide-by-zero -> Memory (CWE-369)
- Resource leaks / infinite loops -> Logic (CWE-399/835)
- Permission/privilege issues -> Logic (CWE-264/284)

{evidence}

## Code to Analyze:
```
{code}
```

## Output Format (JSON):
{{
  "predictions": [
    {{"category": "<category>", "confidence": <0.0-1.0>, "reason": "<brief reason>"}},
    {{"category": "<category>", "confidence": <0.0-1.0>, "reason": "<brief reason>"}},
    {{"category": "<category>", "confidence": <0.0-1.0>, "reason": "<brief reason>"}}
  ]
}}

Respond with ONLY the JSON object."""


def make_enriched_detector_prompt(major: str) -> str:
    """Build enriched detector prompt with all CWEs for a category."""
    cwe_list = build_cwe_list_for_major(major)

    return f"""You are a {major.lower()} vulnerability expert. Analyze the code and determine the SPECIFIC CWE vulnerability type.

## IMPORTANT: You must output a SPECIFIC CWE-ID, not a general category.

## Possible CWE Types for {major}:
{cwe_list}

## Contrastive Evidence from Knowledge Base:
{{evidence}}

## Code to Analyze:
```
{{code}}
```

## Analysis Steps:
1. Check if the code has any vulnerability patterns
2. If vulnerable, identify the MOST SPECIFIC matching CWE from the list above
3. Compare with the evidence examples for confirmation
4. If no vulnerability pattern matches, output "Benign"

## CRITICAL: Distinguish between similar CWEs:
- CWE-119 (general buffer overflow) vs CWE-125 (OOB read) vs CWE-787 (OOB write) vs CWE-120 (classic overflow)
- CWE-416 (use-after-free) vs CWE-415 (double free) vs CWE-401 (memory leak)
- CWE-476 (null pointer) vs CWE-617 (reachable assertion)
- CWE-190 (integer overflow) vs CWE-369 (divide by zero) vs CWE-189 (numeric error)

## Output (JSON):
{{{{
  "prediction": "CWE-XXX" or "Benign",
  "confidence": 0.0-1.0,
  "evidence": "Specific code pattern that matches this CWE",
  "subcategory": "vulnerability type name"
}}}}"""


ENRICHED_DETECTOR_PROMPTS = {
    cat: make_enriched_detector_prompt(cat)
    for cat in ["Memory", "Injection", "Logic", "Input", "Crypto"]
}


# ============================================================================
# Data structures
# ============================================================================

@dataclass
class PromptSet:
    """A complete set of prompts for MulVul (1 router + 5 detectors)."""
    router_prompt: str
    detector_prompts: Dict[str, str]  # {category: prompt}
    generation: int = 0
    parent_id: str = ""
    mutation_type: str = "seed"

    @property
    def id(self) -> str:
        h = hashlib.md5(
            (self.router_prompt + "".join(sorted(self.detector_prompts.values()))).encode()
        ).hexdigest()[:8]
        return f"gen{self.generation}_{h}"

    def to_dict(self) -> dict:
        return {
            "id": self.id,
            "generation": self.generation,
            "parent_id": self.parent_id,
            "mutation_type": self.mutation_type,
            "router_prompt": self.router_prompt,
            "detector_prompts": self.detector_prompts,
        }


@dataclass
class EvalResult:
    """Evaluation result for one PromptSet."""
    prompt_id: str
    generation: int
    macro_f1: float
    weighted_f1: float
    accuracy: float
    cwe_coverage: int  # number of distinct CWEs correctly predicted
    total_classes: int
    per_class_f1: Dict[str, float]
    binary_f1: float
    n_samples: int
    n_errors: int
    confusion_summary: Dict[str, Any]
    eval_time: float

    @property
    def fitness(self) -> float:
        """Combined fitness: Macro-F1 + coverage bonus."""
        coverage_bonus = self.cwe_coverage / max(self.total_classes, 1) * 0.05
        return self.macro_f1 + coverage_bonus

    def to_dict(self) -> dict:
        return {
            "prompt_id": self.prompt_id,
            "generation": self.generation,
            "fitness": round(self.fitness, 4),
            "macro_f1": round(self.macro_f1, 4),
            "weighted_f1": round(self.weighted_f1, 4),
            "accuracy": round(self.accuracy, 4),
            "binary_f1": round(self.binary_f1, 4),
            "cwe_coverage": self.cwe_coverage,
            "total_classes": self.total_classes,
            "n_samples": self.n_samples,
            "n_errors": self.n_errors,
            "eval_time": round(self.eval_time, 1),
            "per_class_f1_nonzero": {
                k: round(v, 4)
                for k, v in sorted(self.per_class_f1.items(), key=lambda x: x[1], reverse=True)
                if v > 0
            },
        }


# ============================================================================
# Evaluator
# ============================================================================

class PromptEvaluator:
    """Evaluates a PromptSet on a dataset."""

    def __init__(self, kb_path: str, model: str, max_workers: int = 16, k: int = 3):
        self.kb_path = kb_path
        self.model = model
        self.max_workers = max_workers
        self.k = k
        self.retriever = None
        if kb_path and os.path.exists(kb_path):
            self.retriever = MulVulRetriever(knowledge_base_path=kb_path)

    def evaluate(self, prompt_set: PromptSet, data: List[dict]) -> EvalResult:
        """Run MulVul with given prompts and compute CWE Macro-F1."""
        start = time.time()
        predictions = []  # (pred_cwe, gt_cwe)
        errors = 0

        def eval_one(item):
            client = create_llm_client(model_name=self.model)
            # Build detector with custom prompts
            router = RouterAgent(
                llm_client=client,
                retriever=self.retriever,
                prompt=prompt_set.router_prompt,
                k=self.k,
            )
            detectors = {}
            for cat, prompt in prompt_set.detector_prompts.items():
                detectors[cat] = DetectorAgent(
                    category=cat,
                    llm_client=client,
                    retriever=self.retriever,
                    prompt=prompt,
                )
            detector = MulVulDetector(
                router=router,
                detectors=detectors,
                aggregator=DecisionAggregator(),
                parallel=False,
                k=self.k,
            )

            code = item.get("func", "")
            gt_cwe = self._get_gt_cwe(item)

            try:
                result = detector.detect(code)
                pred_cwe = self._extract_pred_cwe(result)
                return (pred_cwe, gt_cwe, None)
            except Exception as e:
                return ("Error", gt_cwe, str(e))

        with ThreadPoolExecutor(max_workers=self.max_workers) as executor:
            futures = {executor.submit(eval_one, item): item for item in data}
            for future in as_completed(futures):
                try:
                    pred, gt, err = future.result()
                    predictions.append((pred, gt))
                    if err:
                        errors += 1
                except Exception as e:
                    item = futures[future]
                    gt = self._get_gt_cwe(item)
                    predictions.append(("Error", gt))
                    errors += 1

        elapsed = time.time() - start

        # Compute metrics
        return self._compute_eval_result(
            prompt_set, predictions, errors, elapsed
        )

    def _get_gt_cwe(self, item: dict) -> str:
        target = int(item.get("target", 0))
        if target == 0:
            return "Benign"
        cwe_codes = item.get("cwe", [])
        if isinstance(cwe_codes, str):
            cwe_codes = [cwe_codes] if cwe_codes else []
        if not cwe_codes:
            return "Unknown"
        cwe = cwe_codes[0]
        if not cwe.startswith("CWE-"):
            m = re.search(r"(\d+)", str(cwe))
            if m:
                cwe = f"CWE-{m.group(1)}"
        return cwe

    def _extract_pred_cwe(self, result: DetectionResult) -> str:
        if not result.is_vulnerable():
            return "Benign"
        if result.cwe and result.cwe.startswith("CWE"):
            return result.cwe
        if result.prediction.startswith("CWE"):
            return result.prediction
        m = re.search(r"CWE-(\d+)", result.raw_response or "")
        if m:
            return f"CWE-{m.group(1)}"
        return "Vulnerable-Unknown"

    def _compute_eval_result(
        self, prompt_set: PromptSet, predictions: List[Tuple[str, str]],
        errors: int, elapsed: float
    ) -> EvalResult:
        from evoprompt.evaluators.multiclass_metrics import MultiClassMetrics

        metrics = MultiClassMetrics()
        for pred, gt in predictions:
            metrics.add_prediction(pred, gt)

        # Per-class F1 (GT classes only)
        gt_classes = set(gt for _, gt in predictions)
        per_class_f1 = {}
        for cls in gt_classes:
            if cls in metrics.class_metrics:
                per_class_f1[cls] = metrics.class_metrics[cls].f1_score

        # CWE coverage: classes with F1 > 0
        cwe_coverage = sum(1 for v in per_class_f1.values() if v > 0)

        # GT-only Macro-F1 (like sklearn)
        gt_f1s = [per_class_f1.get(c, 0.0) for c in gt_classes]
        macro_f1 = sum(gt_f1s) / len(gt_f1s) if gt_f1s else 0.0

        # Binary F1
        bin_metrics = MultiClassMetrics()
        for pred, gt in predictions:
            p = "Benign" if pred == "Benign" else "Vulnerable"
            g = "Benign" if gt == "Benign" else "Vulnerable"
            bin_metrics.add_prediction(p, g)

        # Confusion summary (top confusions)
        confusion = defaultdict(lambda: Counter())
        for pred, gt in predictions:
            if pred != gt:
                confusion[gt][pred] += 1
        top_confusions = {}
        for gt_cls, preds in sorted(confusion.items(), key=lambda x: sum(x[1].values()), reverse=True)[:10]:
            top_confusions[gt_cls] = dict(preds.most_common(3))

        return EvalResult(
            prompt_id=prompt_set.id,
            generation=prompt_set.generation,
            macro_f1=macro_f1,
            weighted_f1=metrics.compute_weighted_f1(),
            accuracy=metrics.accuracy,
            cwe_coverage=cwe_coverage,
            total_classes=len(gt_classes),
            per_class_f1=per_class_f1,
            binary_f1=bin_metrics.compute_macro_f1(),
            n_samples=len(predictions),
            n_errors=errors,
            confusion_summary=top_confusions,
            eval_time=elapsed,
        )


# ============================================================================
# Mutator
# ============================================================================

class ErrorGuidedMutator:
    """Generates prompt mutations guided by evaluation error analysis."""

    def __init__(self, model: str):
        self.model = model
        self.mutation_history: List[dict] = []
        self.successful_patterns: List[str] = []
        self.failed_patterns: List[str] = []

    def mutate(
        self,
        prompt_set: PromptSet,
        eval_result: EvalResult,
        mutation_type: str = "error_guided",
        target_component: str = None,
    ) -> PromptSet:
        """Generate a mutated PromptSet based on error analysis."""
        client = create_llm_client(model_name=self.model)

        if target_component == "router":
            return self._mutate_router(client, prompt_set, eval_result)
        elif target_component and target_component in prompt_set.detector_prompts:
            return self._mutate_detector(client, prompt_set, eval_result, target_component)
        else:
            # Pick the weakest component to mutate
            target = self._pick_weakest_component(eval_result)
            if target == "router":
                return self._mutate_router(client, prompt_set, eval_result)
            else:
                return self._mutate_detector(client, prompt_set, eval_result, target)

    def _pick_weakest_component(self, eval_result: EvalResult) -> str:
        """Pick the component most in need of improvement."""
        # Check which major category has worst performance
        major_f1 = defaultdict(list)
        for cwe_str, f1 in eval_result.per_class_f1.items():
            if cwe_str == "Benign":
                continue
            cwe_id = extract_cwe_id(cwe_str)
            if cwe_id and cwe_id in CWE_TO_MIDDLE:
                major = MIDDLE_TO_MAJOR.get(CWE_TO_MIDDLE[cwe_id], "Logic")
                major_f1[major].append(f1)

        if not major_f1:
            return random.choice(["Memory", "Logic", "Input"])

        # Return the category with lowest average F1
        avg_f1 = {cat: sum(f1s) / len(f1s) for cat, f1s in major_f1.items()}
        worst = min(avg_f1, key=avg_f1.get)
        return worst

    def _mutate_router(
        self, client, prompt_set: PromptSet, eval_result: EvalResult
    ) -> PromptSet:
        """Mutate the router prompt."""
        # Identify routing failures from confusion
        routing_issues = []
        for gt_cwe, pred_counts in eval_result.confusion_summary.items():
            if gt_cwe == "Benign":
                continue
            cwe_id = extract_cwe_id(gt_cwe)
            if cwe_id:
                expected_major = cwe_to_major([gt_cwe])
                for pred_cwe in pred_counts:
                    if pred_cwe == "Benign":
                        routing_issues.append(f"{gt_cwe} ({expected_major}) often misclassified as Benign")
                    elif pred_cwe.startswith("CWE-"):
                        pred_id = extract_cwe_id(pred_cwe)
                        if pred_id:
                            pred_major = cwe_to_major([pred_cwe])
                            if pred_major != expected_major:
                                routing_issues.append(
                                    f"{gt_cwe} should route to {expected_major} but gets routed to {pred_major}"
                                )

        history_context = ""
        if self.successful_patterns:
            history_context += f"\nPreviously successful changes: {'; '.join(self.successful_patterns[-3:])}"
        if self.failed_patterns:
            history_context += f"\nPreviously failed changes (avoid these): {'; '.join(self.failed_patterns[-3:])}"

        mutation_prompt = f"""You are optimizing a vulnerability detection router prompt. The router routes code to specialized detectors.

Current router prompt performance:
- Macro-F1: {eval_result.macro_f1:.4f}
- CWE coverage: {eval_result.cwe_coverage}/{eval_result.total_classes}
- Binary F1: {eval_result.binary_f1:.4f}

Routing issues identified:
{chr(10).join(f'- {issue}' for issue in routing_issues[:8]) if routing_issues else '- No specific routing issues identified'}
{history_context}

Current router prompt:
---
{prompt_set.router_prompt}
---

Generate an IMPROVED router prompt that:
1. Better distinguishes between the 6 categories
2. Fixes the routing issues above
3. Keeps the same JSON output format
4. Is concise but precise about category boundaries

Output ONLY the improved prompt (no explanation). Keep the {{evidence}} and {{code}} placeholders."""

        try:
            new_router = client.generate(mutation_prompt, temperature=0.7, max_tokens=2000)
            # Validate placeholders
            if "{code}" not in new_router or "{evidence}" not in new_router:
                new_router = prompt_set.router_prompt  # fallback
        except Exception:
            new_router = prompt_set.router_prompt

        new_set = PromptSet(
            router_prompt=new_router,
            detector_prompts=dict(prompt_set.detector_prompts),
            generation=prompt_set.generation + 1,
            parent_id=prompt_set.id,
            mutation_type="mutate_router",
        )
        return new_set

    def _mutate_detector(
        self, client, prompt_set: PromptSet, eval_result: EvalResult, category: str
    ) -> PromptSet:
        """Mutate a specific detector prompt."""
        # Identify CWE-level issues for this category
        cat_cwes = get_cwes_for_major(category)
        missed_cwes = []
        confused_cwes = []

        for cwe_id in cat_cwes:
            cwe_str = f"CWE-{cwe_id}"
            f1 = eval_result.per_class_f1.get(cwe_str, 0.0)
            if f1 == 0.0:
                desc = FULL_CWE_DESCRIPTIONS.get(cwe_id, "")
                missed_cwes.append(f"CWE-{cwe_id}: {desc}")

            if cwe_str in eval_result.confusion_summary:
                for pred, count in eval_result.confusion_summary[cwe_str].items():
                    if pred.startswith("CWE-"):
                        confused_cwes.append(f"CWE-{cwe_id} confused with {pred} ({count} times)")

        # Non-zero F1 CWEs (working well)
        good_cwes = [
            f"CWE-{cwe_id}: F1={eval_result.per_class_f1.get(f'CWE-{cwe_id}', 0):.3f}"
            for cwe_id in cat_cwes
            if eval_result.per_class_f1.get(f"CWE-{cwe_id}", 0) > 0
        ]

        history_context = ""
        if self.successful_patterns:
            history_context += f"\nPreviously successful changes: {'; '.join(self.successful_patterns[-3:])}"
        if self.failed_patterns:
            history_context += f"\nPreviously failed changes (avoid): {'; '.join(self.failed_patterns[-3:])}"

        mutation_prompt = f"""You are optimizing a {category} vulnerability detector prompt for CWE-level classification.

Performance stats:
- Overall Macro-F1: {eval_result.macro_f1:.4f}
- This category's CWEs that WORK well: {', '.join(good_cwes[:5]) if good_cwes else 'None'}
- CWEs MISSED (never predicted, need improvement): {', '.join(missed_cwes[:8]) if missed_cwes else 'None'}
- CWE CONFUSIONS: {', '.join(confused_cwes[:5]) if confused_cwes else 'None'}
{history_context}

Current {category} detector prompt:
---
{prompt_set.detector_prompts[category]}
---

Generate an IMPROVED {category} detector prompt that:
1. Better identifies the SPECIFIC CWE types listed above, especially the missed ones
2. Adds clearer distinguishing criteria between confused CWEs
3. Maintains the JSON output format with "prediction": "CWE-XXX" or "Benign"
4. Preserves what already works well
5. Keeps {{evidence}} and {{code}} placeholders

Output ONLY the improved prompt (no explanation)."""

        try:
            new_prompt = client.generate(mutation_prompt, temperature=0.7, max_tokens=2000)
            if "{code}" not in new_prompt:
                new_prompt = prompt_set.detector_prompts[category]
        except Exception:
            new_prompt = prompt_set.detector_prompts[category]

        new_detectors = dict(prompt_set.detector_prompts)
        new_detectors[category] = new_prompt

        return PromptSet(
            router_prompt=prompt_set.router_prompt,
            detector_prompts=new_detectors,
            generation=prompt_set.generation + 1,
            parent_id=prompt_set.id,
            mutation_type=f"mutate_{category}",
        )

    def crossover(self, parent1: PromptSet, parent2: PromptSet) -> PromptSet:
        """Crossover: take router from parent1, mix detectors from both."""
        # Randomly pick router from one parent
        router = random.choice([parent1.router_prompt, parent2.router_prompt])

        # For each detector, pick from the better-performing parent
        new_detectors = {}
        for cat in parent1.detector_prompts:
            if random.random() < 0.5:
                new_detectors[cat] = parent1.detector_prompts[cat]
            else:
                new_detectors[cat] = parent2.detector_prompts.get(cat, parent1.detector_prompts[cat])

        return PromptSet(
            router_prompt=router,
            detector_prompts=new_detectors,
            generation=max(parent1.generation, parent2.generation) + 1,
            parent_id=f"{parent1.id}x{parent2.id}",
            mutation_type="crossover",
        )

    def record_mutation_result(self, mutation_type: str, delta_f1: float):
        """Track mutation success/failure for future guidance."""
        entry = {"type": mutation_type, "delta_f1": round(delta_f1, 4)}
        self.mutation_history.append(entry)

        if delta_f1 > 0.005:
            self.successful_patterns.append(f"{mutation_type} (+{delta_f1:.4f})")
        elif delta_f1 < -0.005:
            self.failed_patterns.append(f"{mutation_type} ({delta_f1:.4f})")


# ============================================================================
# Evolution Engine
# ============================================================================

class CWE130Evolution:
    """Main evolution engine."""

    def __init__(
        self,
        eval_data: List[dict],
        kb_path: str,
        model: str,
        output_dir: str,
        pop_size: int = 6,
        max_workers: int = 16,
        k: int = 3,
    ):
        self.eval_data = eval_data
        self.output_dir = output_dir
        self.pop_size = pop_size

        self.evaluator = PromptEvaluator(kb_path, model, max_workers, k)
        self.mutator = ErrorGuidedMutator(model)

        self.population: List[PromptSet] = []
        self.eval_cache: Dict[str, EvalResult] = {}
        self.generation_log: List[dict] = []
        self.best_ever: Optional[Tuple[PromptSet, EvalResult]] = None

        os.makedirs(output_dir, exist_ok=True)

    def initialize_population(self):
        """Create initial population from enriched base prompts + variants."""
        print("\n=== Initializing Population ===")

        # Seed 0: Enriched base prompts
        base = PromptSet(
            router_prompt=ENRICHED_ROUTER_PROMPT,
            detector_prompts=dict(ENRICHED_DETECTOR_PROMPTS),
            generation=0,
            mutation_type="enriched_seed",
        )
        self.population.append(base)

        # Seed 1: Base prompts with more emphasis on rare CWEs
        rare_emphasis = PromptSet(
            router_prompt=ENRICHED_ROUTER_PROMPT,
            detector_prompts=dict(ENRICHED_DETECTOR_PROMPTS),
            generation=0,
            mutation_type="rare_emphasis_seed",
        )
        # Modify detector prompts to emphasize rare CWEs
        for cat in rare_emphasis.detector_prompts:
            rare_emphasis.detector_prompts[cat] = rare_emphasis.detector_prompts[cat].replace(
                "## Analysis Steps:",
                "## IMPORTANT: Pay special attention to RARE vulnerability types that are often overlooked:\n"
                "- CWE-703 (improper exception handling), CWE-617 (reachable assertion)\n"
                "- CWE-369 (divide by zero), CWE-415 (double free), CWE-401 (memory leak)\n"
                "- CWE-362 (race condition), CWE-835 (infinite loop)\n\n## Analysis Steps:",
            )
        self.population.append(rare_emphasis)

        # Generate remaining individuals via mutation from base
        print(f"  Generating {self.pop_size - 2} initial mutations...")
        base_eval = self._evaluate_individual(base)
        self.eval_cache[base.id] = base_eval

        categories = ["Memory", "Logic", "Input", "router", "Injection"]
        for i in range(min(self.pop_size - 2, len(categories))):
            target = categories[i % len(categories)]
            if target == "router":
                mutant = self.mutator.mutate(base, base_eval, target_component="router")
            else:
                mutant = self.mutator.mutate(base, base_eval, target_component=target)
            mutant.generation = 0
            self.population.append(mutant)

        print(f"  Population size: {len(self.population)}")

    def evolve(self, n_generations: int):
        """Run the evolution loop."""
        print(f"\n{'='*70}")
        print(f"Starting Evolution: {n_generations} generations, pop_size={self.pop_size}")
        print(f"Eval data: {len(self.eval_data)} samples")
        print(f"{'='*70}")

        for gen in range(n_generations):
            gen_start = time.time()
            print(f"\n--- Generation {gen} ---")

            # 1. Evaluate all individuals
            eval_results = []
            for i, individual in enumerate(self.population):
                if individual.id in self.eval_cache:
                    result = self.eval_cache[individual.id]
                    print(f"  [{i}] {individual.id} (cached): F1={result.macro_f1:.4f}, coverage={result.cwe_coverage}")
                else:
                    print(f"  [{i}] Evaluating {individual.id} ({individual.mutation_type})...")
                    result = self._evaluate_individual(individual)
                    self.eval_cache[individual.id] = result
                    print(f"       F1={result.macro_f1:.4f}, coverage={result.cwe_coverage}, binary={result.binary_f1:.4f}")
                eval_results.append((individual, result))

            # 2. Sort by fitness
            eval_results.sort(key=lambda x: x[1].fitness, reverse=True)
            best_individual, best_result = eval_results[0]

            # Update best ever
            if self.best_ever is None or best_result.fitness > self.best_ever[1].fitness:
                self.best_ever = (best_individual, best_result)
                self._save_best(best_individual, best_result)

            # 3. Log generation stats
            gen_stats = {
                "generation": gen,
                "best_fitness": round(best_result.fitness, 4),
                "best_macro_f1": round(best_result.macro_f1, 4),
                "best_coverage": best_result.cwe_coverage,
                "best_binary_f1": round(best_result.binary_f1, 4),
                "best_id": best_individual.id,
                "best_mutation_type": best_individual.mutation_type,
                "population_fitness": [round(r.fitness, 4) for _, r in eval_results],
                "avg_fitness": round(sum(r.fitness for _, r in eval_results) / len(eval_results), 4),
                "elapsed": round(time.time() - gen_start, 1),
                "best_ever_f1": round(self.best_ever[1].macro_f1, 4),
            }
            self.generation_log.append(gen_stats)
            self._save_generation_log()

            print(f"\n  Gen {gen} Summary:")
            print(f"    Best: {best_individual.id} ({best_individual.mutation_type})")
            print(f"    Macro-F1: {best_result.macro_f1:.4f} | Coverage: {best_result.cwe_coverage}/{best_result.total_classes}")
            print(f"    Binary-F1: {best_result.binary_f1:.4f} | Accuracy: {best_result.accuracy:.4f}")
            print(f"    Best-ever: {self.best_ever[1].macro_f1:.4f}")
            print(f"    Population fitness: {gen_stats['population_fitness']}")

            # 4. Early stopping check
            if gen >= 3:
                recent_best = [log["best_macro_f1"] for log in self.generation_log[-3:]]
                if max(recent_best) - min(recent_best) < 0.002:
                    print(f"\n  Convergence detected (variance < 0.002 over 3 gens)")
                    # Don't stop, but increase mutation temperature
                    print(f"  Increasing mutation diversity...")

            # 5. Generate next generation (unless last)
            if gen < n_generations - 1:
                self.population = self._generate_next_generation(eval_results)

        # Final summary
        self._print_final_summary()

    def _evaluate_individual(self, individual: PromptSet) -> EvalResult:
        """Evaluate one individual on the eval data."""
        return self.evaluator.evaluate(individual, self.eval_data)

    def _generate_next_generation(
        self, eval_results: List[Tuple[PromptSet, EvalResult]]
    ) -> List[PromptSet]:
        """Generate next generation via selection, mutation, and crossover."""
        next_gen = []

        # Elitism: keep top 2
        for individual, result in eval_results[:2]:
            elite = PromptSet(
                router_prompt=individual.router_prompt,
                detector_prompts=dict(individual.detector_prompts),
                generation=individual.generation + 1,
                parent_id=individual.id,
                mutation_type="elite",
            )
            next_gen.append(elite)

        best_individual, best_result = eval_results[0]
        second_individual, second_result = eval_results[1]

        # Record mutation outcomes
        for individual, result in eval_results:
            if individual.parent_id and individual.parent_id in self.eval_cache:
                parent_result = self.eval_cache[individual.parent_id]
                delta = result.fitness - parent_result.fitness
                self.mutator.record_mutation_result(individual.mutation_type, delta)

        # Mutation: create offspring from best individuals
        categories = ["Memory", "Logic", "Input", "Injection", "Crypto", "router"]
        for i in range(self.pop_size - 3):
            target = categories[i % len(categories)]
            parent = eval_results[i % 2][0]  # alternate between top 2
            parent_result = eval_results[i % 2][1]

            if target == "router":
                mutant = self.mutator.mutate(parent, parent_result, target_component="router")
            else:
                mutant = self.mutator.mutate(parent, parent_result, target_component=target)
            next_gen.append(mutant)

        # Crossover: combine top 2
        crossover = self.mutator.crossover(best_individual, second_individual)
        next_gen.append(crossover)

        return next_gen[:self.pop_size]

    def _save_best(self, individual: PromptSet, result: EvalResult):
        """Save the best-ever individual."""
        path = Path(self.output_dir) / "best_prompts.json"
        data = {
            "prompts": individual.to_dict(),
            "eval": result.to_dict(),
            "timestamp": datetime.now().isoformat(),
        }
        with open(path, "w") as f:
            json.dump(data, f, indent=2, ensure_ascii=False)

    def _save_generation_log(self):
        """Save evolution log."""
        path = Path(self.output_dir) / "evolution_log.json"
        with open(path, "w") as f:
            json.dump(self.generation_log, f, indent=2, ensure_ascii=False)

    def _print_final_summary(self):
        """Print final evolution summary."""
        print(f"\n{'='*70}")
        print("EVOLUTION COMPLETE")
        print(f"{'='*70}")

        if self.best_ever:
            best_prompts, best_result = self.best_ever
            print(f"  Best Macro-F1: {best_result.macro_f1:.4f} ({best_result.macro_f1*100:.2f}%)")
            print(f"  Best Coverage: {best_result.cwe_coverage}/{best_result.total_classes}")
            print(f"  Best Binary-F1: {best_result.binary_f1:.4f}")
            print(f"  Best ID: {best_prompts.id}")
            print(f"  From generation: {best_prompts.generation}")
            print(f"  Mutation type: {best_prompts.mutation_type}")

        print(f"\n  Evolution path:")
        for log in self.generation_log:
            print(
                f"    Gen {log['generation']:2d}: "
                f"F1={log['best_macro_f1']:.4f} "
                f"(avg={log['avg_fitness']:.4f}) "
                f"coverage={log['best_coverage']} "
                f"[{log['best_mutation_type']}]"
            )

        print(f"\n  Mutation history:")
        success = [m for m in self.mutator.mutation_history if m["delta_f1"] > 0.005]
        fail = [m for m in self.mutator.mutation_history if m["delta_f1"] < -0.005]
        print(f"    Successful: {len(success)}, Failed: {len(fail)}, Neutral: {len(self.mutator.mutation_history) - len(success) - len(fail)}")

        if success:
            print(f"    Best mutations: {sorted(success, key=lambda x: x['delta_f1'], reverse=True)[:5]}")

        print(f"\n  Saved to: {self.output_dir}")
        print(f"    - best_prompts.json")
        print(f"    - evolution_log.json")
        print(f"{'='*70}")


# ============================================================================
# Main
# ============================================================================

def main():
    parser = argparse.ArgumentParser(description="Evolve MulVul prompts for CWE-130 Macro-F1")
    parser.add_argument("--data", required=True, help="Evaluation subset JSONL")
    parser.add_argument("--kb", required=True, help="Knowledge base JSON")
    parser.add_argument("--model", default=None, help="Model name")
    parser.add_argument("--generations", type=int, default=15, help="Number of generations")
    parser.add_argument("--pop-size", type=int, default=6, help="Population size")
    parser.add_argument("--workers", type=int, default=16, help="Eval workers")
    parser.add_argument("--k", type=int, default=3, help="Top-k routing")
    parser.add_argument("--output-dir", default="outputs/rebuttal/cwe130_evolution", help="Output directory")
    parser.add_argument("--seed", type=int, default=42, help="Random seed")

    args = parser.parse_args()

    load_env_vars()
    random.seed(args.seed)
    model = args.model or os.getenv("MODEL_NAME", "gpt-4o")

    # Load eval data
    print(f"Loading eval data: {args.data}")
    data = []
    with open(args.data) as f:
        for line in f:
            if line.strip():
                data.append(json.loads(line))
    print(f"  Loaded {len(data)} samples")

    # Create evolution engine
    engine = CWE130Evolution(
        eval_data=data,
        kb_path=args.kb,
        model=model,
        output_dir=args.output_dir,
        pop_size=args.pop_size,
        max_workers=args.workers,
        k=args.k,
    )

    # Initialize and run
    engine.initialize_population()
    engine.evolve(args.generations)

    return 0


if __name__ == "__main__":
    sys.exit(main())
