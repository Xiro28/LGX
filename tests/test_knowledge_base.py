"""
tests/test_knowledge_base.py
─────────────────────────────────────────────────────────────────────────────
Tests for knowledgeBase and conditionProgram.
dumbo_asp.Model is mocked to keep tests self-contained (no ASP solver needed).
"""
from __future__ import annotations

from unittest.mock import MagicMock, patch

import pytest

from src.core.atom import atom
from src.core.atom_list import atomList
from src.core.knowledge_base import conditionProgram, knowledgeBase


# ── Helpers ───────────────────────────────────────────────────────────────────

def _mock_model(facts: str):
    """Return a patch context that makes Model.of_program produce *facts*."""
    mock_result      = MagicMock()
    mock_result.as_facts = facts
    return patch(
        "src.core.knowledge_base.Model.of_program",
        return_value=mock_result,
    )


# ── knowledgeBase ─────────────────────────────────────────────────────────────

class TestKnowledgeBase:

    def test_program_stored(self):
        kb = knowledgeBase("fact(1).")
        assert kb.program == "fact(1)."

    def test_str_returns_program(self):
        kb = knowledgeBase("a :- b.")
        assert str(kb) == "a :- b."

    def test_execute_parses_output_atoms(self):
        with _mock_model("result(1).\nresult(2)."):
            kb  = knowledgeBase("#show result/1.")
            out = kb.execute()
        assert len(out) == 2
        assert out.atoms[0].atom_str == "result(1)."
        assert out.atoms[1].atom_str == "result(2)."

    def test_execute_with_database_atoms(self, atom_list_fixture):
        with _mock_model("derived(x).") as mock_op:
            kb  = knowledgeBase("derived(X) :- edge(X,_).")
            out = kb.execute(database_atoms=atom_list_fixture)
        # Model.of_program must have been called with the stringified atoms
        call_args = mock_op.call_args
        assert "edge(a,b)." in call_args.args[1]

    def test_execute_with_extracted_atoms(self, atom_list_fixture):
        with _mock_model("out(x).") as mock_op:
            kb = knowledgeBase("out(X) :- edge(X,_).")
            kb.execute(extracted_atoms=atom_list_fixture)
        call_args = mock_op.call_args
        assert "edge(a,b)." in call_args.args[1]

    def test_execute_returns_empty_on_no_output(self):
        with _mock_model(""):
            kb  = knowledgeBase("% comment only")
            out = kb.execute()
        assert out.not_empty() is False

    def test_execute_ignores_blank_lines(self):
        with _mock_model("a.\n\n\nb."):
            kb  = knowledgeBase("a. b.")
            out = kb.execute()
        assert len(out) == 2

    def test_validate_returns_true_on_success(self):
        with _mock_model(""):
            kb = knowledgeBase("fact.")
            assert kb.validate() is True

    def test_validate_returns_false_on_exception(self):
        with patch(
            "src.core.knowledge_base.Model.of_program",
            side_effect=RuntimeError("solver error"),
        ):
            kb = knowledgeBase("bad program :-.")
            assert kb.validate() is False

    def test_add_combines_programs(self):
        kb1 = knowledgeBase("a :- b.")
        kb2 = knowledgeBase("b.")
        combined = kb1 + kb2
        assert "a :- b." in combined.program
        assert "b."       in combined.program

    def test_add_returns_same_type(self):
        kb1 = knowledgeBase("a.")
        kb2 = knowledgeBase("b.")
        assert type(kb1 + kb2) is knowledgeBase

    def test_frozen_immutable(self):
        kb = knowledgeBase("a.")
        with pytest.raises((AttributeError, TypeError)):
            kb.program = "b."  # type: ignore[misc]


# ── conditionProgram ──────────────────────────────────────────────────────────

class TestConditionProgram:

    def test_is_subclass_of_knowledge_base(self):
        cp = conditionProgram("t :- p.")
        assert isinstance(cp, knowledgeBase)

    def test_empty_program(self):
        cp = conditionProgram("")
        assert cp.program == ""

    def test_execute_delegates_to_model(self):
        with _mock_model("target_0."):
            cp  = conditionProgram("target_0 :- p.")
            out = cp.execute()
        assert out.contain_atom_with_suffix("_0.")

    def test_add_preserves_type(self):
        cp1 = conditionProgram("a.")
        cp2 = conditionProgram("b.")
        result = cp1 + cp2
        # __add__ is inherited from knowledgeBase which uses type(self)(...)
        assert isinstance(result, conditionProgram)

    def test_validate_on_empty_program(self):
        with _mock_model(""):
            assert conditionProgram("").validate() is True