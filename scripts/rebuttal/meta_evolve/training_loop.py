"""Main meta-evolution training loop: PyTorch-style batched training with meta-learning."""

import json
import os
import random
import re
import time
from collections import Counter, defaultdict
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from typing import List, Optional

from dotenv import load_dotenv
from openai import OpenAI

try:
    from .config import MetaEvolveConfig
    from .cwe_knowledge import CWEKnowledgeBase
    from .error_analyzer import ErrorAnalyzer
    from .knn_retriever import CWERetriever
    from .meta_prompter import MetaPrompter
    from .prompt_template import PromptTemplate
except ImportError:
    from config import MetaEvolveConfig
    from cwe_knowledge import CWEKnowledgeBase
    from error_analyzer import ErrorAnalyzer
    from knn_retriever import CWERetriever
    from meta_prompter import MetaPrompter
    from prompt_template import PromptTemplate


def parse_response(response: str) -> dict:
    """Parse LLM classification response into structured result."""
    try:
        m = re.search(r'\{[^{}]*\}', response, re.DOTALL)
        if m:
            data = json.loads(m.group(0))
            label = data.get("label", "")
            confidence = float(data.get("confidence", 0.5))
            reason = data.get("reason", "")

            if label.lower() in ("benign", "safe", "none", "no vulnerability"):
                return {"label": "Benign", "confidence": confidence, "reason": reason}

            m2 = re.search(r"CWE-(\d+)", label)
            if m2:
                return {"label": f"CWE-{m2.group(1)}", "confidence": confidence, "reason": reason}
    except (json.JSONDecodeError, ValueError, TypeError):
        pass

    # Fallback: search for CWE pattern
    m = re.search(r"CWE-(\d+)", response)
    if m:
        return {"label": f"CWE-{m.group(1)}", "confidence": 0.5, "reason": ""}

    if any(w in response.lower() for w in ["benign", "no vulnerability", "not vulnerable"]):
        return {"label": "Benign", "confidence": 0.5, "reason": ""}

    return {"label": "Unknown", "confidence": 0.0, "reason": ""}


def get_gt_cwe(item: dict) -> str:
    """Extract ground truth CWE from data item."""
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


class MetaEvolveTrainer:
    """Main training loop for meta-learning prompt evolution."""

    def __init__(self, config: MetaEvolveConfig):
        self.config = config
        self.output_dir = Path(config.resolve_path(config.output_dir))
        self.output_dir.mkdir(parents=True, exist_ok=True)

        # Load env
        env_path = config.project_root / ".env"
        if env_path.exists():
            load_dotenv(env_path)

        api_base = os.getenv("API_BASE_URL", "https://api.chatanywhere.tech/v1")
        api_key = os.getenv("API_KEY", "")

        # Components
        print("Initializing components...", flush=True)

        # CWE Knowledge Base
        kb_path = config.resolve_path(config.kb_output)
        if not kb_path.exists():
            print("  KB not found, building...", flush=True)
            self.kb = CWEKnowledgeBase.build(
                str(config.resolve_path(config.knowledge_cwe)),
                str(config.resolve_path(config.train_data)),
                str(kb_path),
            )
        else:
            self.kb = CWEKnowledgeBase(str(kb_path))

        # kNN Retriever
        self.knn = CWERetriever(str(config.resolve_path(config.train_data)))

        # Prompt Template
        initial_rules = config.initial_rules if config.initial_rules else None
        self.template = PromptTemplate(initial_rules)

        # Error Analyzer
        self.analyzer = ErrorAnalyzer(ema_alpha=config.ema_alpha, max_failures=config.max_failures)

        # Meta Prompter
        meta_base = os.getenv("META_API_BASE_URL", api_base)
        meta_key = os.getenv("META_API_KEY", api_key)
        self.meta_prompter = MetaPrompter(
            api_base=meta_base,
            api_key=meta_key,
            model=config.meta_model,
            temperature=config.meta_temperature,
            max_tokens=config.meta_max_tokens,
        )

        # LLM client params for detection
        self.detect_api_base = api_base
        self.detect_api_key = api_key

        # Load test data
        print("Loading test data...", flush=True)
        with open(config.resolve_path(config.test_data)) as f:
            self.test_data = [json.loads(line) for line in f]
        print(f"  {len(self.test_data)} test samples", flush=True)

        # State
        self.epoch = 0
        self.global_batch = 0
        self.best_macro_f1 = 0.0
        self.best_rules: List[str] = list(self.template.rules)

    def _make_stratified_batches(self, data: List[dict], batch_size: int, seed: int = 42) -> List[List[dict]]:
        """Create batches with ~50% benign and ~50% vulnerable samples each.

        This ensures the meta-prompter sees both classes in every batch,
        preventing bias from data ordering.
        """
        rng = random.Random(seed)

        benign = [d for d in data if int(d.get("target", 0)) == 0]
        vuln = [d for d in data if int(d.get("target", 0)) == 1]
        rng.shuffle(benign)
        rng.shuffle(vuln)

        half = batch_size // 2
        batches = []
        bi, vi = 0, 0

        while bi < len(benign) or vi < len(vuln):
            batch = []
            # Take up to half benign
            b_take = min(half, len(benign) - bi)
            batch.extend(benign[bi:bi + b_take])
            bi += b_take
            # Take up to half vuln
            v_take = min(half, len(vuln) - vi)
            batch.extend(vuln[vi:vi + v_take])
            vi += v_take
            # Fill remainder if one side exhausted
            remaining = batch_size - len(batch)
            if remaining > 0 and bi < len(benign):
                extra = min(remaining, len(benign) - bi)
                batch.extend(benign[bi:bi + extra])
                bi += extra
            elif remaining > 0 and vi < len(vuln):
                extra = min(remaining, len(vuln) - vi)
                batch.extend(vuln[vi:vi + extra])
                vi += extra

            if batch:
                rng.shuffle(batch)  # shuffle within batch
                batches.append(batch)

        print(f"  Stratified batching: {len(benign)} benign + {len(vuln)} vuln -> {len(batches)} batches", flush=True)
        return batches

    def _classify_sample(self, item: dict) -> dict:
        """Classify a single sample using current prompt template."""
        code = item.get("func", "")
        gt_cwe = get_gt_cwe(item)

        # Get kNN candidates
        candidates = self.knn.get_candidates(
            code, k=self.config.knn_k, max_cwes=self.config.candidate_cwes
        )
        candidate_ids = [cwe_id for cwe_id, _ in candidates]

        # Add "Benign" context
        cwe_context = self.kb.get_candidates_context(candidate_ids, max_example_lines=12)
        if not cwe_context:
            cwe_context = "(No similar vulnerable code found - consider Benign)"

        # Render prompt
        truncated_code = code[:self.config.max_code_chars]
        prompt = self.template.render(truncated_code, cwe_context)

        # Call LLM
        client = OpenAI(base_url=self.detect_api_base, api_key=self.detect_api_key)
        resp = client.chat.completions.create(
            model=self.config.detect_model,
            messages=[{"role": "user", "content": prompt}],
            temperature=self.config.detect_temperature,
            max_tokens=self.config.detect_max_tokens,
        )
        response_text = resp.choices[0].message.content.strip()

        # Parse
        parsed = parse_response(response_text)

        return {
            "gt_cwe": gt_cwe,
            "pred_cwe": parsed["label"],
            "confidence": parsed["confidence"],
            "reason": parsed["reason"],
            "code": code[:2000],
            "candidates": [f"CWE-{cid}" for cid in candidate_ids],
        }

    def _process_batch(self, batch: List[dict], batch_idx: int) -> List[dict]:
        """Process a batch of samples with concurrent LLM calls."""
        results = []

        def classify_one(item):
            try:
                return self._classify_sample(item)
            except Exception as e:
                gt = get_gt_cwe(item)
                return {
                    "gt_cwe": gt,
                    "pred_cwe": "Error",
                    "confidence": 0.0,
                    "reason": str(e)[:100],
                }

        with ThreadPoolExecutor(max_workers=self.config.max_workers) as executor:
            futures = {executor.submit(classify_one, item): i for i, item in enumerate(batch)}
            for future in as_completed(futures):
                results.append(future.result())

        return results

    def _compute_epoch_metrics(self, all_results: List[dict]) -> dict:
        """Compute full metrics for an epoch."""
        class_tp = defaultdict(int)
        class_fp = defaultdict(int)
        class_fn = defaultdict(int)

        for r in all_results:
            gt, pred = r["gt_cwe"], r["pred_cwe"]
            if gt == pred:
                class_tp[gt] += 1
            else:
                class_fn[gt] += 1
                class_fp[pred] += 1

        gt_classes = set(r["gt_cwe"] for r in all_results)
        gt_counts = Counter(r["gt_cwe"] for r in all_results)

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
        accuracy = sum(1 for r in all_results if r["gt_cwe"] == r["pred_cwe"]) / len(all_results)

        # Confusion matrix (top pairs)
        confusion = defaultdict(int)
        for r in all_results:
            if r["gt_cwe"] != r["pred_cwe"]:
                confusion[(r["gt_cwe"], r["pred_cwe"])] += 1
        top_confusion = sorted(confusion.items(), key=lambda x: x[1], reverse=True)[:15]

        return {
            "macro_f1": macro_f1,
            "weighted_f1": weighted_f1,
            "coverage": coverage,
            "total_classes": len(gt_classes),
            "accuracy": accuracy,
            "class_f1": {k: round(v, 4) for k, v in sorted(class_f1.items(), key=lambda x: x[1], reverse=True)},
            "top_confusion": [{"gt": g, "pred": p, "count": c} for (g, p), c in top_confusion],
        }

    def _save_checkpoint(self, epoch: int, batch_idx: int, all_results: List[dict], metrics: dict):
        """Save checkpoint with current state."""
        ckpt_dir = self.output_dir / f"epoch_{epoch}"
        ckpt_dir.mkdir(parents=True, exist_ok=True)

        # Save rules
        rules_file = ckpt_dir / f"rules_batch_{batch_idx}.json"
        with open(rules_file, "w") as f:
            json.dump({
                "epoch": epoch,
                "batch": batch_idx,
                "global_batch": self.global_batch,
                "rules": self.template.rules,
                "n_rules": len(self.template.rules),
            }, f, indent=2)

        # Save results
        results_file = ckpt_dir / f"results_batch_{batch_idx}.jsonl"
        with open(results_file, "w") as f:
            for r in all_results:
                # Don't save full code in results
                r_save = {k: v for k, v in r.items() if k != "code"}
                f.write(json.dumps(r_save, ensure_ascii=False) + "\n")

        # Save metrics
        metrics_file = ckpt_dir / f"metrics_batch_{batch_idx}.json"
        with open(metrics_file, "w") as f:
            json.dump(metrics, f, indent=2)

    def _save_best(self, epoch: int, metrics: dict):
        """Save best model."""
        best_dir = self.output_dir / "best"
        best_dir.mkdir(parents=True, exist_ok=True)

        with open(best_dir / "rules.json", "w") as f:
            json.dump({
                "epoch": epoch,
                "macro_f1": metrics["macro_f1"],
                "rules": self.best_rules,
            }, f, indent=2)

        with open(best_dir / "metrics.json", "w") as f:
            json.dump(metrics, f, indent=2)

    def run(self):
        """Run the full training loop."""
        print(f"\n{'='*70}")
        print(f"Meta-Evolution Training")
        print(f"{'='*70}")
        print(f"  Epochs: {self.config.epochs}")
        print(f"  Batch size: {self.config.batch_size}")
        print(f"  Meta update freq: every {self.config.meta_update_freq} batches")
        print(f"  Model: {self.config.detect_model}")
        print(f"  Workers: {self.config.max_workers}")
        print(f"  Initial rules: {self.template.num_rules()}")
        print(f"  Test samples: {len(self.test_data)}")
        print(f"  Output: {self.output_dir}")
        print(flush=True)

        total_start = time.time()

        for epoch in range(self.config.epochs):
            self.epoch = epoch
            epoch_start = time.time()
            all_results: List[dict] = []

            # Stratified batches: ~50% benign, ~50% vulnerable per batch
            batches = self._make_stratified_batches(
                self.test_data, self.config.batch_size, seed=42 + epoch
            )
            n_batches = len(batches)

            print(f"\n{'='*70}")
            print(f"Epoch {epoch + 1}/{self.config.epochs} ({n_batches} batches)")
            print(f"{'='*70}")
            print(f"  Rules: {self.template.num_rules()}")
            print(flush=True)

            for batch_idx, batch in enumerate(batches):
                self.global_batch += 1
                batch_start_time = time.time()

                # Process batch
                batch_results = self._process_batch(batch, batch_idx)
                all_results.extend(batch_results)

                # Update error analyzer
                self.analyzer.update_batch(batch_results)

                # Batch stats
                batch_correct = sum(1 for r in batch_results if r["gt_cwe"] == r["pred_cwe"])
                batch_errors = sum(1 for r in batch_results if r["pred_cwe"] == "Error")
                batch_elapsed = time.time() - batch_start_time
                total_correct = sum(1 for r in all_results if r["gt_cwe"] == r["pred_cwe"])

                print(
                    f"  Batch {batch_idx+1}/{n_batches}: "
                    f"{batch_correct}/{len(batch_results)} correct, "
                    f"{batch_errors} errors, "
                    f"{len(batch_results)/batch_elapsed:.1f} samples/sec, "
                    f"running acc={total_correct}/{len(all_results)} "
                    f"({total_correct/len(all_results):.1%})",
                    flush=True,
                )

                # Meta-update every N batches
                if self.global_batch % self.config.meta_update_freq == 0:
                    print(f"\n  --- Meta-update at batch {self.global_batch} ---", flush=True)
                    error_text = self.analyzer.format_for_meta_prompter()
                    new_rules = self.meta_prompter.generate_improved_rules(
                        self.template.rules,
                        error_text,
                        max_rules=self.config.max_rules,
                    )
                    self.template.update_rules(new_rules)
                    print(f"  Rules updated: {len(new_rules)} rules", flush=True)

                    # Checkpoint after meta-update
                    interim_metrics = self._compute_epoch_metrics(all_results)
                    self._save_checkpoint(epoch, batch_idx, all_results, interim_metrics)

            # End of epoch
            epoch_elapsed = time.time() - epoch_start
            epoch_metrics = self._compute_epoch_metrics(all_results)

            print(f"\n{'='*70}")
            print(f"Epoch {epoch + 1} Complete ({epoch_elapsed:.0f}s)")
            print(f"{'='*70}")
            print(f"  Macro-F1: {epoch_metrics['macro_f1']*100:.2f}%")
            print(f"  Weighted-F1: {epoch_metrics['weighted_f1']*100:.2f}%")
            print(f"  Accuracy: {epoch_metrics['accuracy']*100:.2f}%")
            print(f"  Coverage: {epoch_metrics['coverage']}/{epoch_metrics['total_classes']}")
            print(f"  Rules: {self.template.num_rules()}")

            print(f"\n  Top-10 Confusions:")
            for conf in epoch_metrics["top_confusion"][:10]:
                print(f"    {conf['gt']:15s} -> {conf['pred']:15s}: {conf['count']}")

            # Save epoch checkpoint
            self._save_checkpoint(epoch, n_batches, all_results, epoch_metrics)

            # Track best
            if epoch_metrics["macro_f1"] > self.best_macro_f1:
                self.best_macro_f1 = epoch_metrics["macro_f1"]
                self.best_rules = list(self.template.rules)
                self._save_best(epoch, epoch_metrics)
                print(f"\n  New best Macro-F1: {self.best_macro_f1*100:.2f}%")

            # Save epoch summary
            summary_file = self.output_dir / f"epoch_{epoch}" / "summary.json"
            with open(summary_file, "w") as f:
                json.dump({
                    "epoch": epoch,
                    "elapsed_sec": epoch_elapsed,
                    "n_samples": len(all_results),
                    "metrics": epoch_metrics,
                    "rules": self.template.rules,
                    "analyzer_summary": self.analyzer.get_summary(),
                }, f, indent=2, ensure_ascii=False)

        # Training complete
        total_elapsed = time.time() - total_start
        print(f"\n{'='*70}")
        print(f"Training Complete ({total_elapsed:.0f}s)")
        print(f"{'='*70}")
        print(f"  Best Macro-F1: {self.best_macro_f1*100:.2f}%")
        print(f"  Best rules saved to: {self.output_dir / 'best'}")
        print(flush=True)

        # Save final summary
        with open(self.output_dir / "training_summary.json", "w") as f:
            json.dump({
                "epochs": self.config.epochs,
                "batch_size": self.config.batch_size,
                "total_batches": self.global_batch,
                "total_elapsed_sec": total_elapsed,
                "best_macro_f1": self.best_macro_f1,
                "best_rules": self.best_rules,
                "f1_trend": self.analyzer.macro_f1_history,
                "config": {
                    "detect_model": self.config.detect_model,
                    "meta_model": self.config.meta_model,
                    "knn_k": self.config.knn_k,
                    "candidate_cwes": self.config.candidate_cwes,
                    "meta_update_freq": self.config.meta_update_freq,
                    "max_rules": self.config.max_rules,
                    "ema_alpha": self.config.ema_alpha,
                    "max_workers": self.config.max_workers,
                },
            }, f, indent=2, ensure_ascii=False)

        return self.best_macro_f1, self.best_rules
