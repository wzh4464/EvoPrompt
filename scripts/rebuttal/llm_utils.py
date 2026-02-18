"""Shared LLM utilities for rebuttal experiments with token tracking."""
import json
import os
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional

from dotenv import load_dotenv
from openai import OpenAI

load_dotenv(Path(__file__).resolve().parents[2] / ".env")

API_BASE_URL = os.getenv("API_BASE_URL", "https://api.chatanywhere.org/v1")
API_KEY = os.getenv("API_KEY", "")
MODEL_NAME = os.getenv("MODEL_NAME", "gpt-4o")
BACKUP_API_BASE_URL = os.getenv("BACKUP_API_BASE_URL", "")

DATA_PATH = Path(__file__).resolve().parents[2] / "outputs" / "rebuttal" / "sampled_150.jsonl"


@dataclass
class TokenStats:
    """Accumulates token usage across multiple API calls."""
    total_prompt_tokens: int = 0
    total_completion_tokens: int = 0
    total_calls: int = 0
    latencies: list = field(default_factory=list)

    @property
    def total_tokens(self) -> int:
        return self.total_prompt_tokens + self.total_completion_tokens

    @property
    def avg_latency(self) -> float:
        return sum(self.latencies) / len(self.latencies) if self.latencies else 0.0

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
            "avg_latency_sec": round(self.avg_latency, 3),
            "total_latency_sec": round(sum(self.latencies), 2),
        }


def create_client(base_url: Optional[str] = None, api_key: Optional[str] = None) -> OpenAI:
    return OpenAI(
        base_url=base_url or API_BASE_URL,
        api_key=api_key or API_KEY,
    )


def call_llm(
    client: OpenAI,
    prompt: str,
    stats: TokenStats,
    model: Optional[str] = None,
    temperature: float = 0.1,
    max_tokens: int = 512,
    system_prompt: Optional[str] = None,
) -> str:
    """Single LLM call with token tracking and retry."""
    messages = []
    if system_prompt:
        messages.append({"role": "system", "content": system_prompt})
    messages.append({"role": "user", "content": prompt})

    for attempt in range(3):
        try:
            t0 = time.time()
            resp = client.chat.completions.create(
                model=model or MODEL_NAME,
                messages=messages,
                temperature=temperature,
                max_tokens=max_tokens,
            )
            latency = time.time() - t0

            usage = resp.usage
            stats.record(
                prompt_tokens=usage.prompt_tokens if usage else 0,
                completion_tokens=usage.completion_tokens if usage else 0,
                latency=latency,
            )
            return resp.choices[0].message.content.strip()
        except Exception as e:
            if attempt == 2:
                print(f"  [ERROR] API call failed after 3 retries: {e}")
                return "[ERROR]"
            wait = 2 ** (attempt + 1)
            print(f"  [WARN] API error (attempt {attempt+1}): {e}, retrying in {wait}s...")
            time.sleep(wait)
    return "[ERROR]"


def call_llm_multi_turn(
    client: OpenAI,
    messages: list,
    stats: TokenStats,
    model: Optional[str] = None,
    temperature: float = 0.1,
    max_tokens: int = 512,
) -> str:
    """Multi-turn LLM call (for Reflexion/MAD that need conversation history)."""
    for attempt in range(3):
        try:
            t0 = time.time()
            resp = client.chat.completions.create(
                model=model or MODEL_NAME,
                messages=messages,
                temperature=temperature,
                max_tokens=max_tokens,
            )
            latency = time.time() - t0

            usage = resp.usage
            stats.record(
                prompt_tokens=usage.prompt_tokens if usage else 0,
                completion_tokens=usage.completion_tokens if usage else 0,
                latency=latency,
            )
            return resp.choices[0].message.content.strip()
        except Exception as e:
            if attempt == 2:
                print(f"  [ERROR] API call failed after 3 retries: {e}")
                return "[ERROR]"
            wait = 2 ** (attempt + 1)
            print(f"  [WARN] API error (attempt {attempt+1}): {e}, retrying in {wait}s...")
            time.sleep(wait)
    return "[ERROR]"


def load_samples(path: Optional[Path] = None) -> list:
    """Load sampled test data."""
    path = path or DATA_PATH
    samples = []
    with open(path) as f:
        for line in f:
            samples.append(json.loads(line))
    return samples


def parse_vulnerability_label(response: str) -> str:
    """Extract vulnerability prediction from LLM response."""
    resp_lower = response.lower()
    # Check for explicit labels
    if "non-vulnerable" in resp_lower or "not vulnerable" in resp_lower or "benign" in resp_lower:
        return "benign"
    if "vulnerable" in resp_lower:
        return "vulnerable"
    # Check for CWE mentions (indicates vulnerable)
    if "cwe-" in resp_lower:
        return "vulnerable"
    # Default
    return "benign"


def compute_metrics(predictions: list, ground_truths: list) -> dict:
    """Compute macro-F1, accuracy, precision, recall."""
    tp = fp = fn = tn = 0
    for pred, gt in zip(predictions, ground_truths):
        if pred == "vulnerable" and gt == 1:
            tp += 1
        elif pred == "vulnerable" and gt == 0:
            fp += 1
        elif pred == "benign" and gt == 1:
            fn += 1
        else:
            tn += 1

    # Per-class F1
    # Vulnerable class
    p_v = tp / (tp + fp) if (tp + fp) > 0 else 0
    r_v = tp / (tp + fn) if (tp + fn) > 0 else 0
    f1_v = 2 * p_v * r_v / (p_v + r_v) if (p_v + r_v) > 0 else 0

    # Benign class
    p_b = tn / (tn + fn) if (tn + fn) > 0 else 0
    r_b = tn / (tn + fp) if (tn + fp) > 0 else 0
    f1_b = 2 * p_b * r_b / (p_b + r_b) if (p_b + r_b) > 0 else 0

    macro_f1 = (f1_v + f1_b) / 2
    accuracy = (tp + tn) / (tp + fp + fn + tn) if (tp + fp + fn + tn) > 0 else 0

    return {
        "accuracy": round(accuracy, 4),
        "macro_f1": round(macro_f1, 4),
        "precision_vuln": round(p_v, 4),
        "recall_vuln": round(r_v, 4),
        "f1_vuln": round(f1_v, 4),
        "precision_benign": round(p_b, 4),
        "recall_benign": round(r_b, 4),
        "f1_benign": round(f1_b, 4),
        "tp": tp, "fp": fp, "fn": fn, "tn": tn,
    }
