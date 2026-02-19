#!/usr/bin/env python3
"""MulVul 130-class CWE evaluation on full PrimeVul test set.

Reproduces MulVul paper's 34.79% Macro-F1 (130 CWE classes).

Pipeline per sample:
  Code -> Router (top-k majors) -> Detectors (CWE prediction) -> Aggregator -> final CWE

Ground truth classes: specific CWE-IDs for vulnerable, "Benign" for benign.
Prediction classes: result.cwe for vulnerable, "Benign" otherwise.

Supports sharding for parallel execution:
  --shard-id N --num-shards M
"""

import os
import sys
import json
import time
import re
import argparse
from pathlib import Path
from datetime import datetime
from concurrent.futures import ThreadPoolExecutor, as_completed

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "src"))

from evoprompt.llm.client import create_llm_client, load_env_vars
from evoprompt.agents import MulVulDetector
from evoprompt.rag.retriever import MulVulRetriever


def load_jsonl(path: str) -> list:
    samples = []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                try:
                    samples.append(json.loads(line))
                except json.JSONDecodeError:
                    continue
    return samples


def get_ground_truth_cwe(item: dict) -> str:
    """Return ground truth CWE label for 130-class evaluation.

    Benign samples -> "Benign"
    Vulnerable samples -> first CWE code (e.g., "CWE-119")
    """
    target = int(item.get("target", 0))
    if target == 0:
        return "Benign"

    cwe_codes = item.get("cwe", [])
    if isinstance(cwe_codes, str):
        cwe_codes = [cwe_codes] if cwe_codes else []

    if not cwe_codes:
        return "Unknown"

    cwe = cwe_codes[0]
    # Normalize to CWE-XXX format
    if not cwe.startswith("CWE-"):
        m = re.search(r"(\d+)", str(cwe))
        if m:
            cwe = f"CWE-{m.group(1)}"
    return cwe


def extract_predicted_cwe(result) -> str:
    """Extract predicted CWE from MulVul DetectionResult.

    If result indicates vulnerable and has a CWE, return it.
    Otherwise return "Benign".
    """
    if not result.is_vulnerable():
        return "Benign"

    # Use cwe field if present
    if result.cwe and result.cwe.startswith("CWE"):
        return result.cwe

    # Try to extract CWE from prediction
    if result.prediction.startswith("CWE"):
        return result.prediction

    # Try raw response
    m = re.search(r"CWE-(\d+)", result.raw_response or "")
    if m:
        return f"CWE-{m.group(1)}"

    # Vulnerable but no specific CWE
    return "Vulnerable-Unknown"


def evaluate_sample(item: dict, detector: MulVulDetector) -> dict:
    """Evaluate a single sample and return prediction details."""
    code = item.get("func", "")
    gt_cwe = get_ground_truth_cwe(item)
    idx = item.get("idx", -1)

    try:
        details = detector.detect_with_details(code)
        final = details["final"]
        routing = details["routing"]

        # Build DetectionResult-like from dict for extract
        from evoprompt.agents.base import DetectionResult
        result_obj = DetectionResult(
            prediction=final.get("prediction", "Unknown"),
            confidence=final.get("confidence", 0.0),
            evidence=final.get("evidence", ""),
            category=final.get("category", ""),
            subcategory=final.get("subcategory", ""),
            cwe=final.get("cwe", ""),
            agent_id=final.get("agent_id", ""),
            raw_response="",
        )

        pred_cwe = extract_predicted_cwe(result_obj)

        return {
            "idx": idx,
            "gt_cwe": gt_cwe,
            "pred_cwe": pred_cwe,
            "pred_category": final.get("category", ""),
            "pred_confidence": final.get("confidence", 0.0),
            "routing_top_k": routing.get("top_k", []),
            "detectors": details.get("detectors", []),
            "error": None,
        }

    except Exception as e:
        return {
            "idx": idx,
            "gt_cwe": gt_cwe,
            "pred_cwe": "Error",
            "pred_category": "Error",
            "pred_confidence": 0.0,
            "routing_top_k": [],
            "detectors": [],
            "error": str(e),
        }


def run_shard(
    data_file: str,
    kb_path: str,
    model: str,
    output_file: str,
    shard_id: int = 0,
    num_shards: int = 1,
    max_samples: int = None,
    max_workers: int = 16,
    k: int = 3,
):
    """Run evaluation on one shard of the data."""
    load_env_vars()

    print("=" * 70)
    print(f"MulVul 130-class CWE Evaluation - Shard {shard_id}/{num_shards}")
    print("=" * 70)

    # Load data
    print(f"Loading data: {data_file}")
    all_samples = load_jsonl(data_file)
    print(f"  Total samples in file: {len(all_samples)}")

    # Shard the data
    shard_samples = [s for i, s in enumerate(all_samples) if i % num_shards == shard_id]
    print(f"  Shard {shard_id} samples: {len(shard_samples)}")

    if max_samples:
        shard_samples = shard_samples[:max_samples]
        print(f"  Limited to: {len(shard_samples)}")

    # Load knowledge base
    retriever = None
    if kb_path and os.path.exists(kb_path):
        retriever = MulVulRetriever(knowledge_base_path=kb_path)
    else:
        print(f"  WARNING: Knowledge base not found at {kb_path}, running without RAG")

    print(f"  Model: {model}")
    print(f"  Workers: {max_workers}")
    print(f"  Top-k routing: {k}")

    # Count GT distribution
    from collections import Counter
    gt_dist = Counter(get_ground_truth_cwe(s) for s in shard_samples)
    n_benign = gt_dist.get("Benign", 0)
    n_vuln = len(shard_samples) - n_benign
    n_unique_cwe = len([c for c in gt_dist if c != "Benign"])
    print(f"\n  GT distribution: {n_benign} benign, {n_vuln} vulnerable, {n_unique_cwe} unique CWEs")

    # Evaluate with thread pool
    os.makedirs(os.path.dirname(output_file) or ".", exist_ok=True)

    start_time = time.time()
    results = []
    errors = 0
    completed = 0

    def eval_one(item):
        """Create a fresh client+detector per thread."""
        client = create_llm_client(model_name=model)
        detector = MulVulDetector.create_default(
            llm_client=client, retriever=retriever, k=k, parallel=False
        )
        return evaluate_sample(item, detector)

    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        futures = {executor.submit(eval_one, item): item for item in shard_samples}

        for future in as_completed(futures):
            try:
                result = future.result()
                results.append(result)
                if result["error"]:
                    errors += 1
            except Exception as e:
                item = futures[future]
                results.append({
                    "idx": item.get("idx", -1),
                    "gt_cwe": get_ground_truth_cwe(item),
                    "pred_cwe": "Error",
                    "error": str(e),
                })
                errors += 1

            completed += 1
            if completed % 50 == 0 or completed == len(shard_samples):
                elapsed = time.time() - start_time
                rate = completed / elapsed if elapsed > 0 else 0
                correct = sum(1 for r in results if r.get("gt_cwe") == r.get("pred_cwe"))
                print(
                    f"  [{completed}/{len(shard_samples)}] "
                    f"{rate:.1f} samples/sec, "
                    f"accuracy={correct}/{completed}, "
                    f"errors={errors}"
                )

    elapsed = time.time() - start_time

    # Write JSONL output
    with open(output_file, "w", encoding="utf-8") as f:
        for r in results:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")

    # Quick summary
    correct = sum(1 for r in results if r.get("gt_cwe") == r.get("pred_cwe"))
    total = len(results)

    print(f"\nShard {shard_id} completed:")
    print(f"  Total: {total}, Correct: {correct}, Errors: {errors}")
    print(f"  CWE Accuracy: {correct/total:.2%}" if total > 0 else "  No results")
    print(f"  Elapsed: {elapsed:.1f}s ({total/elapsed:.1f} samples/sec)" if elapsed > 0 else "")
    print(f"  Output: {output_file}")

    return results


def main():
    parser = argparse.ArgumentParser(description="MulVul 130-class CWE evaluation")
    parser.add_argument("--data", required=True, help="Path to test JSONL file")
    parser.add_argument("--kb", required=True, help="Path to knowledge base JSON")
    parser.add_argument("--model", default=None, help="Model name (default: from .env)")
    parser.add_argument("--output", required=True, help="Output JSONL file path")
    parser.add_argument("--shard-id", type=int, default=0, help="Shard index (0-based)")
    parser.add_argument("--num-shards", type=int, default=1, help="Total number of shards")
    parser.add_argument("--max-samples", type=int, default=None, help="Max samples per shard")
    parser.add_argument("--workers", type=int, default=16, help="Thread pool workers")
    parser.add_argument("--k", type=int, default=3, help="Top-k routing")

    args = parser.parse_args()

    if not os.path.exists(args.data):
        print(f"ERROR: Data file not found: {args.data}")
        return 1

    model = args.model or os.getenv("MODEL_NAME", "gpt-4o")

    run_shard(
        data_file=args.data,
        kb_path=args.kb,
        model=model,
        output_file=args.output,
        shard_id=args.shard_id,
        num_shards=args.num_shards,
        max_samples=args.max_samples,
        max_workers=args.workers,
        k=args.k,
    )
    return 0


if __name__ == "__main__":
    sys.exit(main())
