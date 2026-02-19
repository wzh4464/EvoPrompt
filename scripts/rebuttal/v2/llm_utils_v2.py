"""Shared LLM utilities for v2 rebuttal experiments — CWE fine-grained classification."""
import json
import os
import re
import time
from collections import defaultdict
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional

from dotenv import load_dotenv
from openai import OpenAI

load_dotenv(Path(__file__).resolve().parents[3] / ".env")

API_BASE_URL = os.getenv("API_BASE_URL", "https://api.chatanywhere.org/v1")
API_KEY = os.getenv("API_KEY", "")
MODEL_NAME = os.getenv("MODEL_NAME", "gpt-4o")

DATA_PATH = Path(__file__).resolve().parents[3] / "outputs" / "rebuttal" / "sampled_v2_200.jsonl"
OUT_DIR = Path(__file__).resolve().parents[3] / "outputs" / "rebuttal" / "exp2_v2"

# ── Extended CWE-to-major mapping (covers PrimeVul test set) ──
CWE_TO_MAJOR = {
    # Buffer / Bounds
    119: "Buffer Errors", 120: "Buffer Errors", 121: "Buffer Errors",
    122: "Buffer Errors", 125: "Buffer Errors", 131: "Buffer Errors",
    787: "Buffer Errors", 805: "Buffer Errors",
    # Memory Management
    416: "Memory Management", 415: "Memory Management",
    401: "Memory Management", 909: "Memory Management",
    # Pointer / Assertion
    476: "Pointer Dereference", 617: "Pointer Dereference",
    754: "Pointer Dereference",
    # Integer / Type
    190: "Integer Errors", 191: "Integer Errors",
    704: "Integer Errors", 843: "Integer Errors",
    369: "Integer Errors",
    # Input Validation
    20: "Input Validation",
    # Injection
    74: "Injection", 77: "Injection", 78: "Injection",
    79: "Injection", 89: "Injection",
    # Authentication / Authorization
    287: "Authentication", 862: "Authentication",
    269: "Authentication", 284: "Authentication",
    # Cryptography
    295: "Cryptography Issues", 310: "Cryptography Issues",
    311: "Cryptography Issues", 327: "Cryptography Issues",
    354: "Cryptography Issues",
    # Resource Management
    703: "Resource Management", 400: "Resource Management",
    835: "Resource Management", 404: "Resource Management",
    # Information Exposure
    200: "Information Exposure",
    # Concurrency
    362: "Concurrency Issues", 667: "Concurrency Issues",
    # Path Traversal
    22: "Path Traversal",
}

MAJOR_CATEGORIES = [
    "Benign",
    "Buffer Errors",
    "Memory Management",
    "Pointer Dereference",
    "Integer Errors",
    "Input Validation",
    "Injection",
    "Authentication",
    "Cryptography Issues",
    "Resource Management",
    "Information Exposure",
    "Concurrency Issues",
    "Path Traversal",
    "Other",
]

CWE_ID_RE = re.compile(r"CWE-(\d+)")


def sample_ground_truth(sample: dict) -> str:
    """Get ground-truth label for a sample: 'Benign' or major CWE category."""
    if sample["target"] == 0:
        return "Benign"
    cwes = sample.get("cwe", [])
    for c in cwes:
        m = CWE_ID_RE.search(str(c))
        if m:
            cid = int(m.group(1))
            if cid in CWE_TO_MAJOR:
                return CWE_TO_MAJOR[cid]
    return "Other"


def parse_category(response: str) -> str:
    """Parse LLM response into a major CWE category."""
    resp = response.strip()
    resp_lower = resp.lower()

    # 1. Exact match
    for cat in MAJOR_CATEGORIES:
        if cat.lower() in resp_lower:
            return cat

    # 2. CWE ID extraction
    m = CWE_ID_RE.search(resp)
    if m:
        cid = int(m.group(1))
        return CWE_TO_MAJOR.get(cid, "Other")

    # 3. Keyword matching
    kw_map = {
        "buffer": "Buffer Errors", "overflow": "Buffer Errors", "out-of-bounds": "Buffer Errors",
        "oob": "Buffer Errors",
        "use-after-free": "Memory Management", "uaf": "Memory Management",
        "double free": "Memory Management", "double-free": "Memory Management",
        "memory leak": "Memory Management",
        "null pointer": "Pointer Dereference", "null deref": "Pointer Dereference",
        "nullptr": "Pointer Dereference", "assertion": "Pointer Dereference",
        "integer overflow": "Integer Errors", "integer underflow": "Integer Errors",
        "type confusion": "Integer Errors", "divide by zero": "Integer Errors",
        "division by zero": "Integer Errors",
        "input validation": "Input Validation", "improper input": "Input Validation",
        "injection": "Injection", "xss": "Injection", "sql injection": "Injection",
        "command injection": "Injection",
        "authentication": "Authentication", "authorization": "Authentication",
        "missing auth": "Authentication",
        "crypto": "Cryptography Issues", "certificate": "Cryptography Issues",
        "encryption": "Cryptography Issues",
        "race condition": "Concurrency Issues", "deadlock": "Concurrency Issues",
        "path traversal": "Path Traversal", "directory traversal": "Path Traversal",
        "information": "Information Exposure", "info leak": "Information Exposure",
        "resource": "Resource Management", "denial of service": "Resource Management",
        "dos": "Resource Management",
        "benign": "Benign", "safe": "Benign", "no vulnerability": "Benign",
        "not vulnerable": "Benign", "non-vulnerable": "Benign",
    }
    for kw, cat in kw_map.items():
        if kw in resp_lower:
            return cat

    return "Benign"  # default conservative


# ── Token tracking (same as v1) ──

@dataclass
class TokenStats:
    total_prompt_tokens: int = 0
    total_completion_tokens: int = 0
    total_calls: int = 0
    latencies: list = field(default_factory=list)

    @property
    def total_tokens(self) -> int:
        return self.total_prompt_tokens + self.total_completion_tokens

    def record(self, prompt_tokens: int, completion_tokens: int, latency: float):
        self.total_prompt_tokens += prompt_tokens
        self.total_completion_tokens += completion_tokens
        self.total_calls += 1
        self.latencies.append(latency)

    def summary(self, n_samples: int) -> dict:
        return {
            "n_samples": n_samples,
            "total_api_calls": self.total_calls,
            "avg_calls_per_sample": round(self.total_calls / max(n_samples, 1), 2),
            "total_prompt_tokens": self.total_prompt_tokens,
            "total_completion_tokens": self.total_completion_tokens,
            "total_tokens": self.total_tokens,
            "avg_tokens_per_sample": round(self.total_tokens / max(n_samples, 1), 1),
            "avg_prompt_tokens_per_call": round(self.total_prompt_tokens / max(self.total_calls, 1), 1),
            "avg_completion_tokens_per_call": round(self.total_completion_tokens / max(self.total_calls, 1), 1),
            "avg_latency_sec": round(sum(self.latencies) / len(self.latencies) if self.latencies else 0, 3),
            "total_latency_sec": round(sum(self.latencies), 2),
        }


def create_client() -> OpenAI:
    return OpenAI(base_url=API_BASE_URL, api_key=API_KEY)


def call_llm(client, prompt, stats, model=None, temperature=0.1,
             max_tokens=512, system_prompt=None):
    messages = []
    if system_prompt:
        messages.append({"role": "system", "content": system_prompt})
    messages.append({"role": "user", "content": prompt})
    for attempt in range(3):
        try:
            t0 = time.time()
            resp = client.chat.completions.create(
                model=model or MODEL_NAME, messages=messages,
                temperature=temperature, max_tokens=max_tokens)
            latency = time.time() - t0
            u = resp.usage
            stats.record(u.prompt_tokens if u else 0,
                         u.completion_tokens if u else 0, latency)
            return resp.choices[0].message.content.strip()
        except Exception as e:
            if attempt == 2:
                print(f"  [ERROR] {e}")
                return "[ERROR]"
            time.sleep(2 ** (attempt + 1))
    return "[ERROR]"


def call_llm_multi_turn(client, messages, stats, model=None,
                        temperature=0.1, max_tokens=512):
    for attempt in range(3):
        try:
            t0 = time.time()
            resp = client.chat.completions.create(
                model=model or MODEL_NAME, messages=messages,
                temperature=temperature, max_tokens=max_tokens)
            latency = time.time() - t0
            u = resp.usage
            stats.record(u.prompt_tokens if u else 0,
                         u.completion_tokens if u else 0, latency)
            return resp.choices[0].message.content.strip()
        except Exception as e:
            if attempt == 2:
                print(f"  [ERROR] {e}")
                return "[ERROR]"
            time.sleep(2 ** (attempt + 1))
    return "[ERROR]"


def load_samples(path=None):
    path = path or DATA_PATH
    samples = []
    with open(path) as f:
        for line in f:
            samples.append(json.loads(line))
    return samples


def compute_multiclass_metrics(predictions: list, ground_truths: list) -> dict:
    """Compute multi-class Macro-F1 across all CWE categories."""
    all_labels = sorted(set(ground_truths) | set(predictions))
    n = len(predictions)
    correct = sum(1 for p, g in zip(predictions, ground_truths) if p == g)
    accuracy = correct / n if n else 0

    per_class = {}
    for label in all_labels:
        tp = sum(1 for p, g in zip(predictions, ground_truths) if p == label and g == label)
        fp = sum(1 for p, g in zip(predictions, ground_truths) if p == label and g != label)
        fn = sum(1 for p, g in zip(predictions, ground_truths) if p != label and g == label)
        prec = tp / (tp + fp) if (tp + fp) > 0 else 0
        rec = tp / (tp + fn) if (tp + fn) > 0 else 0
        f1 = 2 * prec * rec / (prec + rec) if (prec + rec) > 0 else 0
        support = sum(1 for g in ground_truths if g == label)
        per_class[label] = {
            "precision": round(prec, 4), "recall": round(rec, 4),
            "f1": round(f1, 4), "support": support,
            "tp": tp, "fp": fp, "fn": fn,
        }

    # Macro-F1: average F1 across all ground-truth classes
    gt_labels = sorted(set(ground_truths))
    macro_f1 = sum(per_class[l]["f1"] for l in gt_labels) / len(gt_labels) if gt_labels else 0

    # Also compute binary detection metrics
    binary_preds = ["vulnerable" if p != "Benign" else "benign" for p in predictions]
    binary_gts = ["vulnerable" if g != "Benign" else "benign" for g in ground_truths]
    tp_bin = sum(1 for p, g in zip(binary_preds, binary_gts) if p == "vulnerable" and g == "vulnerable")
    fp_bin = sum(1 for p, g in zip(binary_preds, binary_gts) if p == "vulnerable" and g == "benign")
    fn_bin = sum(1 for p, g in zip(binary_preds, binary_gts) if p == "benign" and g == "vulnerable")
    tn_bin = sum(1 for p, g in zip(binary_preds, binary_gts) if p == "benign" and g == "benign")
    p_v = tp_bin / (tp_bin + fp_bin) if (tp_bin + fp_bin) else 0
    r_v = tp_bin / (tp_bin + fn_bin) if (tp_bin + fn_bin) else 0
    f1_v = 2 * p_v * r_v / (p_v + r_v) if (p_v + r_v) else 0

    return {
        "accuracy": round(accuracy, 4),
        "macro_f1": round(macro_f1, 4),
        "n_classes": len(gt_labels),
        "binary_vuln_precision": round(p_v, 4),
        "binary_vuln_recall": round(r_v, 4),
        "binary_vuln_f1": round(f1_v, 4),
        "binary_confusion": {"tp": tp_bin, "fp": fp_bin, "fn": fn_bin, "tn": tn_bin},
        "per_class": per_class,
    }


# Category list formatted for prompts
CATEGORIES_FOR_PROMPT = """Categories (choose exactly ONE):
- Benign (no vulnerability)
- Buffer Errors (buffer overflow, out-of-bounds read/write, CWE-119/120/125/787)
- Memory Management (use-after-free, double-free, memory leak, CWE-416/415/401)
- Pointer Dereference (null pointer dereference, assertion failure, CWE-476/617)
- Integer Errors (integer overflow, type confusion, divide-by-zero, CWE-190/369/704/843)
- Input Validation (improper input validation, CWE-20)
- Injection (command/SQL/XSS injection, CWE-74/78/79/89)
- Authentication (improper authentication/authorization, CWE-287/862)
- Cryptography Issues (weak crypto, certificate validation, CWE-295/327)
- Resource Management (resource leak, DoS, improper check, CWE-703/400)
- Information Exposure (sensitive data exposure, CWE-200)
- Concurrency Issues (race condition, CWE-362)
- Path Traversal (directory traversal, CWE-22)
- Other (vulnerability not matching above categories)"""
