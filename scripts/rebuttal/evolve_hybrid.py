#!/usr/bin/env python3
"""Evolutionary optimization of Hybrid LLM+kNN for CWE-130 classification.

Evolves:
1. Binary classification prompt (for better vulnerability recall)
2. kNN parameters (k value)
3. Optional LLM refinement prompt for confused CWE pairs

Tracks per-CWE F1, coverage, and confusion patterns across generations.
"""

import os
import sys
import json
import time
import re
import random
import argparse
import hashlib
from pathlib import Path
from collections import Counter, defaultdict
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass, field

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "src"))

from evoprompt.llm.client import create_llm_client, load_env_vars


# ============================================================================
# kNN Classifier
# ============================================================================

class VulnKNN:
    """kNN on vulnerable-only training data."""
    def __init__(self, train_path: str):
        self.data = []
        self.tokens = []
        with open(train_path) as f:
            for line in f:
                item = json.loads(line)
                if int(item.get("target", 0)) == 0:
                    continue
                cwe_codes = item.get("cwe", [])
                if isinstance(cwe_codes, str):
                    cwe_codes = [cwe_codes]
                if not cwe_codes:
                    continue
                cwe = cwe_codes[0]
                if not cwe.startswith("CWE-"):
                    m = re.search(r"(\d+)", str(cwe))
                    cwe = f"CWE-{m.group(1)}" if m else cwe
                code = item.get("func", "")
                self.data.append({"code": code, "cwe": cwe})
                self.tokens.append(set(re.findall(r'[a-zA-Z_]\w*', code)))
        print(f"  kNN: {len(self.data)} vuln samples, {len(Counter(d['cwe'] for d in self.data))} CWEs")

    def predict(self, code: str, k: int = 3) -> tuple:
        """Return (predicted_cwe, top_k_cwes, top_k_sims)."""
        tokens = set(re.findall(r'[a-zA-Z_]\w*', code))
        sims = []
        for i, tt in enumerate(self.tokens):
            inter = len(tokens & tt)
            union = len(tokens | tt)
            sims.append((i, inter / union if union > 0 else 0))
        sims.sort(key=lambda x: x[1], reverse=True)
        top_k = sims[:k]
        votes = defaultdict(float)
        for idx, sim in top_k:
            votes[self.data[idx]["cwe"]] += sim
        pred = max(votes, key=votes.get) if votes else "Unknown"
        top_cwes = [self.data[idx]["cwe"] for idx, _ in top_k]
        top_sims = [sim for _, sim in top_k]
        return pred, top_cwes, top_sims


# ============================================================================
# Prompt variants for evolution
# ============================================================================

SEED_PROMPTS = [
    # Variant 0: Simple
    """Analyze this C/C++ code for security vulnerabilities.

```c
{code}
```

Is this code vulnerable? Output JSON:
{{"vulnerable": true/false, "confidence": 0.0-1.0, "reason": "brief"}}""",

    # Variant 1: Detailed analysis steps
    """You are a security auditor reviewing C/C++ code.

## Code:
```c
{code}
```

## Analysis Checklist:
1. Check for buffer operations without bounds checking
2. Check for NULL pointer dereferences
3. Check for use-after-free or double-free patterns
4. Check for integer overflow/underflow
5. Check for missing error handling
6. Check for race conditions
7. Check for input validation issues
8. Check for path traversal
9. Check for information exposure
10. Check for cryptographic weaknesses

## Output (JSON):
{{"vulnerable": true/false, "confidence": 0.0-1.0, "reason": "brief"}}""",

    # Variant 2: Assume vulnerable (high recall)
    """Carefully analyze this C/C++ code. Most real-world code contains subtle vulnerabilities. Look for ANY security weakness, even minor ones.

```c
{code}
```

Common vulnerability patterns to check:
- Buffer overflow, out-of-bounds access
- NULL pointer dereference, use-after-free
- Integer overflow, divide by zero
- Missing error/return value checks
- Race conditions, improper locking
- Path traversal, input validation
- Information leakage

Even if the code looks mostly safe, check edge cases. Report ANY potential vulnerability.

Output JSON: {{"vulnerable": true/false, "confidence": 0.0-1.0, "reason": "brief"}}""",

    # Variant 3: Contrastive
    """Compare this code against known vulnerability patterns. Be thorough - many vulnerabilities are subtle.

```c
{code}
```

Check these patterns:
- Unchecked buffer sizes (CWE-119/120/787) - array access without bounds check
- NULL deref (CWE-476) - using pointer without NULL check
- Use-after-free (CWE-416) - accessing freed memory
- Missing error handling (CWE-703) - unchecked return values
- Integer issues (CWE-190/369) - overflow or division by zero
- Memory leaks (CWE-401) - allocation without matching free

Output JSON: {{"vulnerable": true/false, "confidence": 0.0-1.0, "reason": "brief"}}""",
]


# ============================================================================
# Evolution data structures
# ============================================================================

@dataclass
class Individual:
    prompt: str
    knn_k: int = 3
    generation: int = 0
    parent_id: str = ""
    mutation_type: str = "seed"

    @property
    def id(self) -> str:
        h = hashlib.md5(self.prompt.encode()).hexdigest()[:8]
        return f"g{self.generation}_{h}"


@dataclass
class EvalResult:
    macro_f1: float
    weighted_f1: float
    binary_f1: float
    binary_recall: float
    binary_precision: float
    coverage: int
    total_classes: int
    accuracy: float
    per_class_f1: dict
    n_samples: int
    eval_time: float

    @property
    def fitness(self):
        """Fitness = Macro-F1 + coverage bonus + recall bonus."""
        cov_bonus = self.coverage / max(self.total_classes, 1) * 0.03
        rec_bonus = max(0, self.binary_recall - 0.8) * 0.1  # bonus for recall > 80%
        return self.macro_f1 + cov_bonus + rec_bonus


# ============================================================================
# Evaluator
# ============================================================================

class HybridEvaluator:
    def __init__(self, knn: VulnKNN, model: str, max_workers: int = 16):
        self.knn = knn
        self.model = model
        self.max_workers = max_workers

    def evaluate(self, individual: Individual, data: list) -> EvalResult:
        start = time.time()
        results = []

        def eval_one(item):
            client = create_llm_client(model_name=self.model)
            code = item.get("func", "")
            prompt = individual.prompt.format(code=code[:8000])
            try:
                resp = client.generate(prompt, max_tokens=150, temperature=0.0)
                is_vuln = self._parse_binary(resp)
            except:
                is_vuln = False

            gt = self._get_gt(item)
            if not is_vuln:
                return {"gt_cwe": gt, "pred_cwe": "Benign"}

            pred, _, _ = self.knn.predict(code, k=individual.knn_k)
            return {"gt_cwe": gt, "pred_cwe": pred}

        with ThreadPoolExecutor(max_workers=self.max_workers) as executor:
            futures = {executor.submit(eval_one, item): i for i, item in enumerate(data)}
            for future in as_completed(futures):
                try:
                    results.append(future.result())
                except:
                    i = futures[future]
                    gt = self._get_gt(data[i])
                    results.append({"gt_cwe": gt, "pred_cwe": "Error"})

        return self._compute_metrics(results, time.time() - start)

    def _parse_binary(self, response):
        try:
            m = re.search(r'\{[^{}]*\}', response, re.DOTALL)
            if m:
                return bool(json.loads(m.group(0)).get("vulnerable", False))
        except:
            pass
        return "true" in response.lower()[:200]

    def _get_gt(self, item):
        if int(item.get("target", 0)) == 0:
            return "Benign"
        cwe_codes = item.get("cwe", [])
        if isinstance(cwe_codes, str):
            cwe_codes = [cwe_codes]
        if not cwe_codes:
            return "Unknown"
        cwe = cwe_codes[0]
        if not cwe.startswith("CWE-"):
            m = re.search(r"(\d+)", str(cwe))
            cwe = f"CWE-{m.group(1)}" if m else cwe
        return cwe

    def _compute_metrics(self, results, elapsed):
        class_tp = defaultdict(int)
        class_fp = defaultdict(int)
        class_fn = defaultdict(int)

        for r in results:
            gt, pred = r["gt_cwe"], r["pred_cwe"]
            if gt == pred:
                class_tp[gt] += 1
            else:
                class_fn[gt] += 1
                class_fp[pred] += 1

        gt_classes = set(r["gt_cwe"] for r in results)
        gt_counts = Counter(r["gt_cwe"] for r in results)

        class_f1 = {}
        for cls in gt_classes:
            tp, fp, fn = class_tp[cls], class_fp[cls], class_fn[cls]
            prec = tp / (tp + fp) if (tp + fp) > 0 else 0
            rec = tp / (tp + fn) if (tp + fn) > 0 else 0
            f1 = 2 * prec * rec / (prec + rec) if (prec + rec) > 0 else 0
            class_f1[cls] = f1

        f1s = list(class_f1.values())
        macro_f1 = sum(f1s) / len(f1s) if f1s else 0
        total = sum(gt_counts.values())
        weighted_f1 = sum(class_f1.get(c, 0) * gt_counts[c] / total for c in gt_classes)
        coverage = sum(1 for v in class_f1.values() if v > 0)

        bin_tp = sum(1 for r in results if r["gt_cwe"] != "Benign" and r["pred_cwe"] != "Benign")
        bin_fp = sum(1 for r in results if r["gt_cwe"] == "Benign" and r["pred_cwe"] != "Benign")
        bin_fn = sum(1 for r in results if r["gt_cwe"] != "Benign" and r["pred_cwe"] == "Benign")
        bin_prec = bin_tp / (bin_tp + bin_fp) if (bin_tp + bin_fp) > 0 else 0
        bin_rec = bin_tp / (bin_tp + bin_fn) if (bin_tp + bin_fn) > 0 else 0
        bin_f1 = 2 * bin_prec * bin_rec / (bin_prec + bin_rec) if (bin_prec + bin_rec) > 0 else 0

        return EvalResult(
            macro_f1=macro_f1,
            weighted_f1=weighted_f1,
            binary_f1=bin_f1,
            binary_recall=bin_rec,
            binary_precision=bin_prec,
            coverage=coverage,
            total_classes=len(gt_classes),
            accuracy=sum(1 for r in results if r["gt_cwe"] == r["pred_cwe"]) / len(results),
            per_class_f1={k: round(v, 4) for k, v in class_f1.items()},
            n_samples=len(results),
            eval_time=elapsed,
        )


# ============================================================================
# Mutator
# ============================================================================

class PromptMutator:
    def __init__(self, model: str):
        self.model = model
        self.mutation_history = []

    def mutate(self, parent: Individual, result: EvalResult) -> Individual:
        """Use LLM to mutate the binary classification prompt."""
        client = create_llm_client(model_name=self.model)

        # Identify weaknesses
        zero_f1_classes = [cls for cls, f1 in result.per_class_f1.items()
                           if f1 == 0 and cls != "Benign"]
        low_recall_msg = f"Binary recall is {result.binary_recall*100:.1f}% - {(1-result.binary_recall)*100:.1f}% of vulnerabilities are missed."
        coverage_msg = f"CWE coverage: {result.coverage}/{result.total_classes} classes. {len(zero_f1_classes)} classes have zero F1."

        mutation_prompt = f"""You are optimizing a prompt for vulnerability detection in C/C++ code.

## Current Prompt:
{parent.prompt}

## Current Performance:
- Macro-F1: {result.macro_f1*100:.2f}%
- Binary recall: {result.binary_recall*100:.1f}% (CRITICAL: higher recall = more vulnerabilities caught)
- Binary precision: {result.binary_precision*100:.1f}%
- {low_recall_msg}
- {coverage_msg}

## Goal:
Improve the prompt to catch MORE vulnerabilities (increase recall) while maintaining reasonable precision.
The prompt should make the model more sensitive to subtle vulnerability patterns.

## Rules:
1. Keep the {{code}} placeholder for code insertion
2. Keep the JSON output format: {{"vulnerable": true/false, "confidence": 0.0-1.0, "reason": "brief"}}
3. The prompt should be comprehensive but concise (under 500 words)
4. Focus on improving recall - it's better to have false positives than miss real vulnerabilities

## Output the improved prompt ONLY (no explanation):"""

        try:
            new_prompt = client.generate(mutation_prompt, max_tokens=1000, temperature=0.7)
            # Clean up: remove markdown code blocks if present
            new_prompt = re.sub(r'^```\w*\n', '', new_prompt)
            new_prompt = re.sub(r'\n```$', '', new_prompt)
            new_prompt = new_prompt.strip()

            # Validate it has the required placeholders
            if "{code}" not in new_prompt:
                new_prompt = parent.prompt  # fallback
            if '"vulnerable"' not in new_prompt and "'vulnerable'" not in new_prompt:
                new_prompt = parent.prompt  # fallback
        except:
            new_prompt = parent.prompt

        # Also try different k values
        knn_k = random.choice([1, 3, 5, 7])

        return Individual(
            prompt=new_prompt,
            knn_k=knn_k,
            generation=parent.generation + 1,
            parent_id=parent.id,
            mutation_type="llm_mutate",
        )

    def crossover(self, parent1: Individual, parent2: Individual) -> Individual:
        """Combine parts of two prompts."""
        # Simple: take the better prompt's structure but vary k
        knn_k = random.choice([parent1.knn_k, parent2.knn_k, 3, 5])
        # Pick random parent's prompt
        prompt = random.choice([parent1.prompt, parent2.prompt])
        return Individual(
            prompt=prompt,
            knn_k=knn_k,
            generation=max(parent1.generation, parent2.generation) + 1,
            parent_id=f"{parent1.id}x{parent2.id}",
            mutation_type="crossover",
        )

    def record(self, mutation_type: str, delta_f1: float):
        self.mutation_history.append({"type": mutation_type, "delta": delta_f1})


# ============================================================================
# Evolution loop
# ============================================================================

class HybridEvolution:
    def __init__(self, evaluator, mutator, data, pop_size, output_dir):
        self.evaluator = evaluator
        self.mutator = mutator
        self.data = data
        self.pop_size = pop_size
        self.output_dir = output_dir
        self.population = []
        self.eval_cache = {}
        self.generation_log = []
        self.best_ever = None

        os.makedirs(output_dir, exist_ok=True)

    def initialize(self):
        """Create initial population from seed prompts."""
        print("Initializing population...")
        for i, prompt in enumerate(SEED_PROMPTS[:self.pop_size]):
            for k in [3, 5]:
                ind = Individual(prompt=prompt, knn_k=k, generation=0, mutation_type=f"seed_v{i}_k{k}")
                self.population.append(ind)
                if len(self.population) >= self.pop_size:
                    break
            if len(self.population) >= self.pop_size:
                break

        # Fill remaining with k variants
        while len(self.population) < self.pop_size:
            base = SEED_PROMPTS[len(self.population) % len(SEED_PROMPTS)]
            k = random.choice([1, 3, 5, 7])
            self.population.append(Individual(prompt=base, knn_k=k, generation=0, mutation_type=f"seed_k{k}"))

        print(f"  Population: {len(self.population)} individuals")

    def evolve(self, n_generations: int):
        print(f"\n{'='*70}")
        print(f"Hybrid Evolution: {n_generations} gens, pop={self.pop_size}")
        print(f"{'='*70}")

        for gen in range(n_generations):
            gen_start = time.time()
            print(f"\n--- Generation {gen} ---", flush=True)

            # Evaluate
            eval_results = []
            for i, ind in enumerate(self.population):
                if ind.id in self.eval_cache:
                    result = self.eval_cache[ind.id]
                    print(f"  [{i}] {ind.id} (cached): F1={result.macro_f1*100:.2f}%, rec={result.binary_recall*100:.1f}%")
                else:
                    print(f"  [{i}] Evaluating {ind.id} (k={ind.knn_k}, {ind.mutation_type})...", flush=True)
                    result = self.evaluator.evaluate(ind, self.data)
                    self.eval_cache[ind.id] = result
                    print(f"       F1={result.macro_f1*100:.2f}%, cov={result.coverage}, rec={result.binary_recall*100:.1f}%, prec={result.binary_precision*100:.1f}%")
                eval_results.append((ind, result))

            # Sort by fitness
            eval_results.sort(key=lambda x: x[1].fitness, reverse=True)
            best_ind, best_result = eval_results[0]

            if self.best_ever is None or best_result.fitness > self.best_ever[1].fitness:
                self.best_ever = (best_ind, best_result)
                self._save_best()

            # Log
            gen_time = time.time() - gen_start
            stats = {
                "generation": gen,
                "best_macro_f1": round(best_result.macro_f1, 4),
                "best_coverage": best_result.coverage,
                "best_binary_f1": round(best_result.binary_f1, 4),
                "best_binary_recall": round(best_result.binary_recall, 4),
                "best_knn_k": best_ind.knn_k,
                "best_mutation": best_ind.mutation_type,
                "avg_fitness": round(sum(r.fitness for _, r in eval_results) / len(eval_results), 4),
                "pop_fitness": [round(r.fitness, 4) for _, r in eval_results],
                "elapsed": round(gen_time, 1),
                "best_ever_f1": round(self.best_ever[1].macro_f1, 4),
            }
            self.generation_log.append(stats)
            self._save_log()

            print(f"\n  Gen {gen}: F1={best_result.macro_f1*100:.2f}%, cov={best_result.coverage}, "
                  f"rec={best_result.binary_recall*100:.1f}%, k={best_ind.knn_k} [{best_ind.mutation_type}]")
            print(f"  Best ever: {self.best_ever[1].macro_f1*100:.2f}%")

            # Generate next generation
            if gen < n_generations - 1:
                self.population = self._next_gen(eval_results)

        self._print_summary()

    def _next_gen(self, eval_results):
        next_gen = []

        # Elitism: top 2
        for ind, _ in eval_results[:2]:
            elite = Individual(
                prompt=ind.prompt,
                knn_k=ind.knn_k,
                generation=ind.generation + 1,
                parent_id=ind.id,
                mutation_type="elite",
            )
            next_gen.append(elite)

        # Mutations from top individuals
        for i in range(self.pop_size - 3):
            parent, parent_result = eval_results[i % min(3, len(eval_results))]
            mutant = self.mutator.mutate(parent, parent_result)
            next_gen.append(mutant)

        # Crossover
        if len(eval_results) >= 2:
            cross = self.mutator.crossover(eval_results[0][0], eval_results[1][0])
            next_gen.append(cross)

        return next_gen[:self.pop_size]

    def _save_best(self):
        path = Path(self.output_dir) / "best_individual.json"
        ind, result = self.best_ever
        data = {
            "prompt": ind.prompt,
            "knn_k": ind.knn_k,
            "generation": ind.generation,
            "mutation_type": ind.mutation_type,
            "macro_f1": round(result.macro_f1, 4),
            "coverage": result.coverage,
            "binary_f1": round(result.binary_f1, 4),
            "binary_recall": round(result.binary_recall, 4),
        }
        with open(path, "w") as f:
            json.dump(data, f, indent=2, ensure_ascii=False)

    def _save_log(self):
        path = Path(self.output_dir) / "evolution_log.json"
        with open(path, "w") as f:
            json.dump(self.generation_log, f, indent=2, ensure_ascii=False)

    def _print_summary(self):
        print(f"\n{'='*70}")
        print("EVOLUTION COMPLETE")
        print(f"{'='*70}")
        if self.best_ever:
            ind, result = self.best_ever
            print(f"  Best Macro-F1: {result.macro_f1*100:.2f}%")
            print(f"  Coverage: {result.coverage}/{result.total_classes}")
            print(f"  Binary: F1={result.binary_f1*100:.2f}%, Rec={result.binary_recall*100:.1f}%, Prec={result.binary_precision*100:.1f}%")
            print(f"  kNN k: {ind.knn_k}")
            print(f"  Mutation: {ind.mutation_type}")
        print(f"\n  Evolution path:")
        for log in self.generation_log:
            print(f"    Gen {log['generation']:2d}: F1={log['best_macro_f1']*100:.2f}% cov={log['best_coverage']} "
                  f"rec={log['best_binary_recall']*100:.1f}% k={log['best_knn_k']} [{log['best_mutation']}]")
        print(f"\n  Saved to: {self.output_dir}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data", required=True, help="Eval data JSONL")
    parser.add_argument("--train", required=True, help="Training data for kNN")
    parser.add_argument("--model", default="gpt-4o")
    parser.add_argument("--generations", type=int, default=10)
    parser.add_argument("--pop-size", type=int, default=6)
    parser.add_argument("--workers", type=int, default=16)
    parser.add_argument("--output-dir", default="outputs/rebuttal/cwe130/hybrid_evolution")
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    random.seed(args.seed)
    load_env_vars()

    # Load eval data
    with open(args.data) as f:
        data = [json.loads(l) for l in f]
    print(f"Eval data: {len(data)} samples")

    # Build kNN
    knn = VulnKNN(args.train)

    # Create components
    evaluator = HybridEvaluator(knn, args.model, args.workers)
    mutator = PromptMutator(args.model)

    # Run evolution
    evo = HybridEvolution(evaluator, mutator, data, args.pop_size, args.output_dir)
    evo.initialize()
    evo.evolve(args.generations)


if __name__ == "__main__":
    main()
