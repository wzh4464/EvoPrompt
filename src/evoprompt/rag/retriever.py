"""RAG retriever for code vulnerability examples.

Retrieves similar code examples from knowledge base and formats them for prompt enhancement.
"""

from typing import List, Optional, Tuple
import re
from dataclasses import dataclass

from .knowledge_base import KnowledgeBase, CodeExample
from ..prompts.hierarchical_three_layer import MajorCategory, MiddleCategory


@dataclass
class RetrievalResult:
    """Result of example retrieval.

    Attributes:
        examples: Retrieved code examples
        formatted_text: Formatted text ready for prompt injection
        similarity_scores: Similarity scores for each example
        debug_info: Debug information about retrieval
    """
    examples: List[CodeExample]
    formatted_text: str
    similarity_scores: List[float]
    debug_info: Optional[dict] = None


class CodeSimilarityRetriever:
    """Retrieves similar code examples using simple text similarity.

    For production use, consider using embeddings (e.g., CodeBERT, OpenAI embeddings).
    This implementation uses fast lexical similarity for efficiency.
    """

    def __init__(self, knowledge_base: KnowledgeBase, debug: bool = False):
        """Initialize retriever.

        Args:
            knowledge_base: Knowledge base containing examples
            debug: Enable debug output
        """
        self.kb = knowledge_base
        self.debug = debug

    def retrieve_for_major_category(
        self,
        query_code: str,
        top_k: int = 2
    ) -> RetrievalResult:
        """Retrieve similar examples across all major categories.

        Args:
            query_code: Code to find similar examples for
            top_k: Number of examples to retrieve

        Returns:
            Retrieval result with examples and formatted text
        """
        all_examples = []

        # Gather examples from all major categories
        for examples in self.kb.major_examples.values():
            all_examples.extend(examples)

        return self._retrieve_from_pool(query_code, all_examples, top_k)

    def retrieve_for_middle_category(
        self,
        query_code: str,
        major_category: MajorCategory,
        top_k: int = 2
    ) -> RetrievalResult:
        """Retrieve similar examples for a specific major category.

        Args:
            query_code: Code to find similar examples for
            major_category: Major category to search within
            top_k: Number of examples to retrieve

        Returns:
            Retrieval result with examples and formatted text
        """
        # Get examples for this major category from middle categories
        all_examples = []
        for examples in self.kb.middle_examples.values():
            # Filter to examples matching this major category
            for ex in examples:
                if ex.category in [mc.value for mc in major_category.__class__]:
                    continue
                all_examples.append(ex)

        # Fallback: use major category examples if not enough middle examples
        if len(all_examples) < top_k:
            major_examples = self.kb.major_examples.get(major_category.value, [])
            all_examples.extend(major_examples)

        return self._retrieve_from_pool(query_code, all_examples, top_k)

    def retrieve_for_cwe(
        self,
        query_code: str,
        middle_category: MiddleCategory,
        top_k: int = 2
    ) -> RetrievalResult:
        """Retrieve similar examples for a specific middle category.

        Args:
            query_code: Code to find similar examples for
            middle_category: Middle category to search within
            top_k: Number of examples to retrieve

        Returns:
            Retrieval result with examples and formatted text
        """
        # Get CWE examples for this middle category
        from ..prompts.hierarchical_three_layer import MIDDLE_TO_CWE

        all_examples = []
        cwes = MIDDLE_TO_CWE.get(middle_category, [])

        for cwe in cwes:
            examples = self.kb.cwe_examples.get(cwe, [])
            all_examples.extend(examples)

        # Fallback: use middle category examples
        if len(all_examples) < top_k:
            middle_examples = self.kb.middle_examples.get(middle_category.value, [])
            all_examples.extend(middle_examples)

        return self._retrieve_from_pool(query_code, all_examples, top_k)

    def _retrieve_from_pool(
        self,
        query_code: str,
        example_pool: List[CodeExample],
        top_k: int
    ) -> RetrievalResult:
        """Retrieve top-k similar examples from pool.

        Args:
            query_code: Code to compare against
            example_pool: Pool of examples to search
            top_k: Number to retrieve

        Returns:
            Retrieval result
        """
        if not example_pool:
            return RetrievalResult(
                examples=[],
                formatted_text="",
                similarity_scores=[],
                debug_info={"pool_size": 0, "message": "Empty example pool"}
            )

        # Compute similarities
        scored_examples = []
        for example in example_pool:
            similarity = self._compute_similarity(query_code, example.code)
            scored_examples.append((similarity, example))

        # Sort by similarity (descending)
        scored_examples.sort(key=lambda x: x[0], reverse=True)

        # Take top-k
        top_examples = scored_examples[:top_k]

        # Format for prompt
        formatted = self._format_examples([ex for _, ex in top_examples])

        # Build debug info
        debug_info = None
        if self.debug:
            debug_info = {
                "pool_size": len(example_pool),
                "top_k": top_k,
                "retrieved": [
                    {
                        "category": ex.category,
                        "cwe": ex.cwe,
                        "similarity": round(score, 4),
                        "code_preview": ex.code[:100] + "..." if len(ex.code) > 100 else ex.code
                    }
                    for score, ex in top_examples
                ],
                "query_preview": query_code[:100] + "..." if len(query_code) > 100 else query_code
            }
            self._print_debug(debug_info)

        return RetrievalResult(
            examples=[ex for _, ex in top_examples],
            formatted_text=formatted,
            similarity_scores=[score for score, _ in top_examples],
            debug_info=debug_info
        )

    def _print_debug(self, debug_info: dict) -> None:
        """Print debug information about retrieval."""
        print("\n" + "=" * 60)
        print("RAG Retrieval Debug Info")
        print("=" * 60)
        print(f"Pool size: {debug_info['pool_size']}")
        print(f"Top-k: {debug_info['top_k']}")
        print(f"\nQuery preview: {debug_info['query_preview']}")
        print("\nRetrieved examples:")
        for i, ex in enumerate(debug_info['retrieved'], 1):
            print(f"\n  [{i}] Category: {ex['category']}, CWE: {ex['cwe']}")
            print(f"      Similarity: {ex['similarity']}")
            print(f"      Code: {ex['code_preview']}")
        print("=" * 60 + "\n")

    def _compute_similarity(self, code1: str, code2: str) -> float:
        """Compute simple lexical similarity between two code snippets.

        Uses Jaccard similarity on token sets for efficiency.
        For production, consider using embeddings.

        Args:
            code1: First code snippet
            code2: Second code snippet

        Returns:
            Similarity score in [0, 1]
        """
        # Tokenize (simple approach)
        tokens1 = set(self._tokenize(code1))
        tokens2 = set(self._tokenize(code2))

        if not tokens1 or not tokens2:
            return 0.0

        # Jaccard similarity
        intersection = len(tokens1 & tokens2)
        union = len(tokens1 | tokens2)

        return intersection / union if union > 0 else 0.0

    def _tokenize(self, code: str) -> List[str]:
        """Simple tokenization of code.

        Args:
            code: Source code

        Returns:
            List of tokens
        """
        # Remove comments (simple approach)
        code = re.sub(r'//.*', '', code)
        code = re.sub(r'/\*.*?\*/', '', code, flags=re.DOTALL)

        # Split on non-alphanumeric characters
        tokens = re.findall(r'\w+', code.lower())

        return tokens

    def _format_examples(self, examples: List[CodeExample]) -> str:
        """Format examples for prompt injection.

        Args:
            examples: Code examples to format

        Returns:
            Formatted string for prompt
        """
        if not examples:
            return ""

        formatted_parts = ["Here are similar examples for reference:\n"]

        for i, example in enumerate(examples, 1):
            formatted_parts.append(f"Example {i}:")
            formatted_parts.append(f"Category: {example.category}")
            if example.cwe:
                formatted_parts.append(f"CWE: {example.cwe}")
            formatted_parts.append(f"Description: {example.description}")
            formatted_parts.append(f"Code:\n{example.code}")
            formatted_parts.append("")  # Empty line

        return "\n".join(formatted_parts)


class EmbeddingRetriever(CodeSimilarityRetriever):
    """Retriever using embeddings for semantic similarity.

    This is a placeholder for future implementation using:
    - OpenAI embeddings
    - CodeBERT
    - Other code embedding models

    For now, falls back to lexical similarity.
    """

    def __init__(
        self,
        knowledge_base: KnowledgeBase,
        embedding_model: Optional[str] = None
    ):
        """Initialize embedding retriever.

        Args:
            knowledge_base: Knowledge base
            embedding_model: Name of embedding model (not yet implemented)
        """
        super().__init__(knowledge_base)
        self.embedding_model = embedding_model

        if embedding_model:
            import warnings
            warnings.warn(
                f"Embedding model '{embedding_model}' not yet implemented. "
                "Falling back to lexical similarity."
            )

    def _compute_similarity(self, code1: str, code2: str) -> float:
        """Compute similarity using embeddings (not yet implemented).

        Falls back to lexical similarity.
        """
        # TODO: Implement embedding-based similarity
        # For now, use parent's lexical similarity
        return super()._compute_similarity(code1, code2)


def create_retriever(
    knowledge_base: KnowledgeBase,
    retriever_type: str = "lexical",
    **kwargs
) -> CodeSimilarityRetriever:
    """Factory function to create retriever.

    Args:
        knowledge_base: Knowledge base to use
        retriever_type: Type of retriever ("lexical" or "embedding")
        **kwargs: Additional arguments for retriever

    Returns:
        Configured retriever
    """
    if retriever_type == "lexical":
        return CodeSimilarityRetriever(knowledge_base)
    elif retriever_type == "embedding":
        return EmbeddingRetriever(knowledge_base, **kwargs)
    else:
        raise ValueError(f"Unknown retriever type: {retriever_type}")


class MulVulRetriever(CodeSimilarityRetriever):
    """Retriever for MulVul multi-agent system.

    Provides cross-type contrastive retrieval for Router Agent
    and category-specific retrieval for Detector Agents.
    """

    # Major category mapping
    MAJOR_CATEGORIES = ["Memory", "Injection", "Logic", "Input", "Crypto"]

    def retrieve_contrastive(
        self,
        query_code: str,
        n_per_category: int = 2
    ) -> List[dict]:
        """Retrieve cross-type contrastive evidence for Router Agent.

        Retrieves samples from each category to provide contrastive patterns.

        Args:
            query_code: Code to find similar examples for
            n_per_category: Number of examples per category

        Returns:
            List of dicts with code, category, and similarity
        """
        results = []

        for category in self.MAJOR_CATEGORIES:
            samples = self.retrieve_from_category(query_code, category, top_k=n_per_category)
            results.extend(samples)

        return results

    def retrieve_from_category(
        self,
        query_code: str,
        category: str,
        top_k: int = 3
    ) -> List[dict]:
        """Retrieve examples from a specific category.

        Used by Detector Agents to get category-specific evidence.

        Args:
            query_code: Code to find similar examples for
            category: Category to retrieve from
            top_k: Number of examples to retrieve

        Returns:
            List of dicts with code, category, cwe, and similarity
        """
        # Map category to major category examples
        category_mapping = {
            "Memory": ["Memory", "Buffer Overflow", "Use After Free", "Null Pointer"],
            "Injection": ["Injection", "SQL Injection", "Command Injection", "XSS"],
            "Logic": ["Logic", "Race Condition", "Access Control", "Information Exposure"],
            "Input": ["Input", "Path Traversal", "Input Validation"],
            "Crypto": ["Crypto", "Cryptography", "Weak Crypto"],
        }

        # Gather examples from matching categories
        all_examples = []
        search_categories = category_mapping.get(category, [category])

        for cat in search_categories:
            examples = self.kb.major_examples.get(cat, [])
            all_examples.extend(examples)

            # Also check middle examples
            for key, examples in self.kb.middle_examples.items():
                if cat.lower() in key.lower():
                    all_examples.extend(examples)

        if not all_examples:
            return []

        # Compute similarities and rank
        scored = []
        for example in all_examples:
            similarity = self._compute_similarity(query_code, example.code)
            scored.append((similarity, example))

        scored.sort(key=lambda x: x[0], reverse=True)

        # Return top-k as dicts
        return [
            {
                "code": ex.code,
                "category": category,
                "cwe": ex.cwe,
                "type": ex.category,
                "similarity": round(score, 4),
            }
            for score, ex in scored[:top_k]
        ]
