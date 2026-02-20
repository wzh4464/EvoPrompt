"""Error analysis with EMA-decayed confusion matrix and failure case tracking."""

import re
from collections import defaultdict
from typing import Dict, List, Optional, Tuple


class ErrorAnalyzer:
    """Tracks classification errors with EMA-decayed confusion matrix.

    Maintains a rolling confusion matrix where recent batches are weighted more
    heavily, plus a rotating buffer of representative failure cases.
    """

    def __init__(self, ema_alpha: float = 0.9, max_failures: int = 30):
        self.ema_alpha = ema_alpha
        self.max_failures = max_failures

        # EMA confusion matrix: (gt, pred) -> decayed count
        self.confusion: Dict[Tuple[str, str], float] = defaultdict(float)

        # Per-class cumulative stats
        self.class_tp: Dict[str, float] = defaultdict(float)
        self.class_fp: Dict[str, float] = defaultdict(float)
        self.class_fn: Dict[str, float] = defaultdict(float)

        # Rotating failure buffer
        self.failures: List[dict] = []

        # Batch-level tracking
        self.batch_count = 0
        self.macro_f1_history: List[float] = []
        self.total_samples = 0
        self.total_correct = 0

    def update_batch(self, results: List[dict]):
        """Update with a batch of results.

        Each result: {"gt_cwe": str, "pred_cwe": str, "code": str (optional)}
        """
        self.batch_count += 1

        # Decay existing counts
        alpha = self.ema_alpha
        for key in list(self.confusion.keys()):
            self.confusion[key] *= alpha
        for key in list(self.class_tp.keys()):
            self.class_tp[key] *= alpha
        for key in list(self.class_fp.keys()):
            self.class_fp[key] *= alpha
        for key in list(self.class_fn.keys()):
            self.class_fn[key] *= alpha

        # Add new batch
        for r in results:
            gt = r["gt_cwe"]
            pred = r["pred_cwe"]
            self.confusion[(gt, pred)] += 1.0
            self.total_samples += 1

            if gt == pred:
                self.class_tp[gt] += 1.0
                self.total_correct += 1
            else:
                self.class_fn[gt] += 1.0
                self.class_fp[pred] += 1.0

                # Add to failure buffer
                failure = {
                    "gt_cwe": gt,
                    "pred_cwe": pred,
                    "batch": self.batch_count,
                }
                if "code" in r:
                    # Store truncated code snippet
                    code = r["code"]
                    lines = code.split("\n")
                    if len(lines) > 30:
                        code = "\n".join(lines[:30]) + "\n// ... truncated"
                    failure["code_snippet"] = code[:2000]
                if "reason" in r:
                    failure["model_reason"] = r["reason"][:200]

                self.failures.append(failure)
                # Rotate buffer
                if len(self.failures) > self.max_failures:
                    self.failures = self.failures[-self.max_failures:]

        # Track macro-F1
        macro_f1 = self._compute_macro_f1()
        self.macro_f1_history.append(macro_f1)

    def _compute_macro_f1(self) -> float:
        """Compute current macro-F1 from EMA stats."""
        all_classes = set(self.class_tp.keys()) | set(self.class_fn.keys())
        if not all_classes:
            return 0.0

        f1s = []
        for cls in all_classes:
            tp = self.class_tp.get(cls, 0)
            fp = self.class_fp.get(cls, 0)
            fn = self.class_fn.get(cls, 0)
            prec = tp / (tp + fp) if (tp + fp) > 0 else 0
            rec = tp / (tp + fn) if (tp + fn) > 0 else 0
            f1 = 2 * prec * rec / (prec + rec) if (prec + rec) > 0 else 0
            f1s.append(f1)

        return sum(f1s) / len(f1s) if f1s else 0.0

    def get_top_confusions(self, n: int = 10) -> List[Tuple[str, str, float]]:
        """Get top-N confusion pairs (gt, pred, count) excluding correct predictions."""
        pairs = [
            (gt, pred, count)
            for (gt, pred), count in self.confusion.items()
            if gt != pred and count > 0.1
        ]
        pairs.sort(key=lambda x: x[2], reverse=True)
        return pairs[:n]

    def get_improved_classes(self) -> List[str]:
        """Classes whose F1 improved in recent batches."""
        # Simplified: classes with tp > 0
        return [cls for cls, tp in self.class_tp.items() if tp > 0.5]

    def get_degraded_classes(self) -> List[str]:
        """Classes with high FN rate."""
        degraded = []
        for cls in set(self.class_fn.keys()):
            fn = self.class_fn.get(cls, 0)
            tp = self.class_tp.get(cls, 0)
            if fn > tp and fn > 0.5:
                degraded.append(cls)
        return degraded

    def get_failure_examples(self, n: int = 10, focus_pairs: Optional[List[Tuple[str, str]]] = None) -> List[dict]:
        """Get representative failure examples, optionally focused on specific confusion pairs."""
        if focus_pairs:
            focused = [
                f for f in self.failures
                if (f["gt_cwe"], f["pred_cwe"]) in focus_pairs
            ]
            if len(focused) >= n:
                return focused[-n:]
            # Fill rest from general failures
            rest = [f for f in self.failures if f not in focused]
            return (focused + rest[-max(0, n - len(focused)):])[: n]
        return self.failures[-n:]

    def get_summary(self) -> dict:
        """Get summary for meta-prompter input."""
        top_confusions = self.get_top_confusions(10)
        return {
            "batch_count": self.batch_count,
            "total_samples": self.total_samples,
            "total_correct": self.total_correct,
            "accuracy": self.total_correct / max(self.total_samples, 1),
            "current_macro_f1": self.macro_f1_history[-1] if self.macro_f1_history else 0.0,
            "macro_f1_trend": self.macro_f1_history[-5:],
            "top_confusions": [
                {"gt": gt, "pred": pred, "count": round(count, 1)}
                for gt, pred, count in top_confusions
            ],
            "improved_classes": self.get_improved_classes()[:10],
            "degraded_classes": self.get_degraded_classes()[:10],
        }

    def format_for_meta_prompter(self) -> str:
        """Format error analysis as text for the meta-prompter LLM."""
        summary = self.get_summary()
        top_confusions = self.get_top_confusions(10)
        failures = self.get_failure_examples(10)

        parts = [
            f"## Error Analysis (after {summary['batch_count']} batches, {summary['total_samples']} samples)",
            f"Accuracy: {summary['accuracy']:.1%}",
            f"EMA Macro-F1: {summary['current_macro_f1']:.4f}",
            f"F1 Trend (last 5 batches): {[round(x, 4) for x in summary['macro_f1_trend']]}",
            "",
            "## Top Confusion Pairs (GT -> Predicted, EMA count):",
        ]

        for gt, pred, count in top_confusions:
            parts.append(f"  {gt} -> {pred}: {count:.1f}")

        if summary["degraded_classes"]:
            parts.append(f"\nDegraded classes (high miss rate): {', '.join(summary['degraded_classes'][:5])}")

        if failures:
            parts.append("\n## Representative Failure Cases:")
            for i, f in enumerate(failures[:8]):
                parts.append(f"\n### Failure {i+1}: GT={f['gt_cwe']}, Predicted={f['pred_cwe']}")
                if "model_reason" in f:
                    parts.append(f"  Model's reason: {f['model_reason']}")
                if "code_snippet" in f:
                    code = f["code_snippet"]
                    if len(code) > 500:
                        code = code[:500] + "\n  // ... truncated"
                    parts.append(f"  Code:\n  ```c\n  {code}\n  ```")

        return "\n".join(parts)
