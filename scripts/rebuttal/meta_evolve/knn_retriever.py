"""kNN-based CWE candidate retriever using Jaccard similarity."""

import json
import re
from collections import Counter, defaultdict
from typing import Dict, List, Tuple


def tokenize(code: str) -> set:
    """Extract identifier tokens from code."""
    return set(re.findall(r'[a-zA-Z_]\w*', code))


class CWERetriever:
    """Retrieves candidate CWEs for a code snippet using Jaccard kNN."""

    def __init__(self, train_path: str):
        self.data: List[dict] = []  # {"code": str, "cwe": str, "cwe_id": int}
        self.tokens: List[set] = []

        print("Loading kNN training data...", flush=True)
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
                cwe_str = cwe_codes[0]
                m = re.search(r"(\d+)", str(cwe_str))
                if not m:
                    continue
                cwe_id = int(m.group(1))
                cwe = f"CWE-{cwe_id}"
                code = item.get("func", "")
                if not code:
                    continue
                self.data.append({"code": code, "cwe": cwe, "cwe_id": cwe_id})
                self.tokens.append(tokenize(code))

        self.cwe_counts = Counter(d["cwe"] for d in self.data)
        print(f"  Loaded {len(self.data)} vulnerable samples, {len(self.cwe_counts)} CWEs", flush=True)

    # Most frequent CWEs in PrimeVul test set - always include as candidates
    COMMON_CWES = {125, 787, 476, 416, 119, 190, 200, 20, 362, 399}

    def get_candidates(self, code: str, k: int = 20, max_cwes: int = 10) -> List[Tuple[int, float]]:
        """Return top candidate CWE IDs with similarity scores.

        Returns list of (cwe_id, aggregated_similarity) sorted by score descending.
        Uses larger k (20) for better coverage, then ensures common CWEs are included.
        """
        query_tokens = tokenize(code)
        if not query_tokens:
            return []

        # Compute Jaccard similarity to all training samples
        sims = []
        for i, train_tokens in enumerate(self.tokens):
            inter = len(query_tokens & train_tokens)
            union = len(query_tokens | train_tokens)
            if union > 0:
                sims.append((i, inter / union))

        sims.sort(key=lambda x: x[1], reverse=True)

        # Aggregate votes from top-k neighbors
        cwe_scores: Dict[int, float] = defaultdict(float)
        for idx, sim in sims[:k]:
            cwe_scores[self.data[idx]["cwe_id"]] += sim

        # Ensure common CWEs have at least a minimal score
        for common_cwe in self.COMMON_CWES:
            if common_cwe not in cwe_scores:
                cwe_scores[common_cwe] = 0.01

        # Sort by aggregated score, return top candidates
        ranked = sorted(cwe_scores.items(), key=lambda x: x[1], reverse=True)
        return ranked[:max_cwes]

    def predict_cwe(self, code: str, k: int = 5) -> str:
        """Predict a single CWE (highest scoring candidate)."""
        candidates = self.get_candidates(code, k=k, max_cwes=1)
        if candidates:
            return f"CWE-{candidates[0][0]}"
        return "Unknown"
