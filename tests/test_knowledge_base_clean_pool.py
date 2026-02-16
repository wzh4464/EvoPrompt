# tests/test_knowledge_base_clean_pool.py
"""Tests for KnowledgeBase clean_examples field (contrastive retrieval support)."""

import pytest
from evoprompt.rag.knowledge_base import KnowledgeBase, CodeExample


def test_knowledge_base_has_clean_examples_field():
    """KnowledgeBase should have clean_examples field."""
    kb = KnowledgeBase()
    assert hasattr(kb, 'clean_examples')
    assert kb.clean_examples == []


def test_add_clean_example():
    """Should be able to add clean/benign examples."""
    kb = KnowledgeBase()
    example = CodeExample(
        code="int x = 5; return x;",
        category="Benign",
        description="Safe integer assignment"
    )
    kb.add_clean_example(example)
    assert len(kb.clean_examples) == 1
    assert kb.clean_examples[0].category == "Benign"


def test_clean_examples_in_statistics():
    """Statistics should include clean pool size."""
    kb = KnowledgeBase()
    kb.add_clean_example(CodeExample(
        code="safe code",
        category="Benign",
        description="Safe"
    ))
    stats = kb.statistics()
    assert "clean_examples" in stats
    assert stats["clean_examples"] == 1


def test_save_and_load_with_clean_examples():
    """Clean examples should be saved and loaded correctly."""
    import tempfile
    import os

    kb = KnowledgeBase()
    kb.add_clean_example(CodeExample(
        code="safe code here",
        category="Benign",
        description="Safe operation"
    ))

    with tempfile.NamedTemporaryFile(mode='w', suffix='.json', delete=False) as f:
        temp_path = f.name

    try:
        kb.save(temp_path)
        loaded_kb = KnowledgeBase.load(temp_path)
        assert len(loaded_kb.clean_examples) == 1
        assert loaded_kb.clean_examples[0].code == "safe code here"
    finally:
        os.unlink(temp_path)
