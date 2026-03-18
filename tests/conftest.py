"""
tests/conftest.py
─────────────────────────────────────────────────────────────────────────────
Shared pytest fixtures for the entire test suite.
All fixtures that are referenced from more than one test module live here so
pytest can discover them automatically (conftest.py is loaded before any test
file in the same directory or below).
"""
from __future__ import annotations

from unittest.mock import MagicMock, patch

import pytest

from src.core.atom import atom
from src.core.atom_list import atomList
from src.core.cache import ConditionCache
from src.core.knowledge_base import conditionProgram, knowledgeBase
from src.core.predicate_condition import predicate_condition
from src.core.yaml_parser import application_configuration, behaviour_configuration


@pytest.fixture
def atom_list_fixture() -> atomList:
    """A non-empty atomList with two edge atoms."""
    return atomList(atoms=[atom("edge(a,b)."), atom("edge(b,c).")])


@pytest.fixture
def empty_atom_list() -> atomList:
    """An atomList with no atoms."""
    return atomList(atoms=[])

@pytest.fixture
def condition_cache_all() -> ConditionCache:
    """Cache enabled in 'all' mode (caches both monotone and non-monotone)."""
    return ConditionCache(enabled=True, cache_mode="all")


@pytest.fixture
def condition_cache_monotone() -> ConditionCache:
    """Cache enabled in 'monotone' mode (only caches monotone conditions)."""
    return ConditionCache(enabled=True, cache_mode="monotone")


@pytest.fixture
def condition_cache_non_monotone() -> ConditionCache:
    """Cache enabled in 'non_monotone' mode (only caches non-monotone conditions)."""
    return ConditionCache(enabled=True, cache_mode="non_monotone")


@pytest.fixture
def condition_cache_disabled() -> ConditionCache:
    """Cache fully disabled."""
    return ConditionCache(enabled=False, cache_mode="all")

@pytest.fixture
def monotone_condition() -> predicate_condition:
    """A monotone predicate condition."""
    return predicate_condition(condition="node(_, _)", monotone=True)


@pytest.fixture
def non_monotone_condition() -> predicate_condition:
    """A non-monotone (negated) predicate condition."""
    return predicate_condition(condition="not error(_)", monotone=False)


@pytest.fixture
def mock_beh_cfg() -> behaviour_configuration:
    """A minimal behaviour_configuration suitable for unit tests."""
    return behaviour_configuration(
        init="You are a helpful assistant.",
        context="Context: {context}",
        mapping="{input} {instructions} {atom}",
    )


@pytest.fixture
def mock_app_cfg() -> application_configuration:
    """A minimal application_configuration with no predicates or KB."""
    return application_configuration(
        context=None,
        strings=[],
        predicates=[],
        kb=None,
    )


@pytest.fixture
def mock_ollama_client() -> MagicMock:
    """
    A MagicMock standing in for an ollama.Client instance.
    Provides a .chat() method that returns a minimal valid response dict.
    """
    client = MagicMock()
    client.chat.return_value = {
        "message":          {"content": "{}"},
        "prompt_eval_count": 10,
        "eval_count":        10,
        "total_duration":    100,
    }
    return client