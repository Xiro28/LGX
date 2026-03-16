"""
tests/test_predicate.py
"""
from __future__ import annotations

from unittest.mock import MagicMock, patch

import pytest

from src.core.atom import atom
from src.core.atom_list import atomList
from src.core.knowledge_base import conditionProgram, knowledgeBase
from src.core.predicate import predicate
from src.core.predicate_condition import predicate_condition


# ── Fixtures ──────────────────────────────────────────────────────────────────

@pytest.fixture
def grammar_mock() -> MagicMock:
    m = MagicMock()
    m.__info__ = {"list_person": [{"name": "any"}]}
    return m


@pytest.fixture
def patched_json_builder(grammar_mock):
    with patch("src.core.predicate.JSONSchemaBuilder") as mock_cls:
        inst = mock_cls.return_value
        inst.generate.return_value = grammar_mock
        inst.generate_single_grammar.return_value = grammar_mock
        yield inst


@pytest.fixture
def simple_pred_def() -> dict:
    return {
        "person": {
            "prompt": "Extract names from {text}",
            "knowledge_base": "person(N) :- extracted_name(N).",
        }
    }


@pytest.fixture
def conditioned_pred_def() -> dict:
    return {
        "edge": {
            "prompt": "Extract edges.",
            "extraction_condition": [
                {"condition": "node(_, _)", "monotone": True},
                {"condition": "not error(_)", "monotone": False},
            ],
        }
    }


@pytest.fixture
def group_pred_def() -> dict:
    return {
        "predicates": ["person(name)", "city(name)"],
        "prompt": "Extract entities.",
    }


# ── create() factory ──────────────────────────────────────────────────────────

class TestPredicateFactory:

    def test_create_simple_predicate(self, patched_json_builder, simple_pred_def):
        pred = predicate.create(simple_pred_def, [{"text": "Alice and Bob"}])
        assert pred.defined_predicate == "person"
        assert "Alice and Bob" in pred.prompt
        assert pred.advanced_prompt_type is True
        assert isinstance(pred.kb, knowledgeBase)
        assert pred.kb.program == "person(N) :- extracted_name(N)."

    def test_create_group_predicate(self, patched_json_builder, group_pred_def):
        pred = predicate.create(group_pred_def, [])
        assert "person(name)" in pred.defined_predicate
        assert "city(name)"   in pred.defined_predicate
        assert pred.advanced_prompt_type is False

    def test_create_with_conditions(self, patched_json_builder, conditioned_pred_def):
        with patch("src.core.predicate.generate_min_program", return_value="t_0 :- node(_,_)."):
            with patch.object(conditionProgram, "validate", return_value=True):
                pred = predicate.create(conditioned_pred_def, [])
        assert len(pred.conditions) == 2
        assert pred.conditions[0].monotone is True
        assert pred.conditions[1].monotone is False

    def test_create_plain_string_config(self, patched_json_builder):
        pred = predicate.create({"city": "Extract city names."}, [])
        assert pred.defined_predicate == "city"
        assert pred.prompt == "Extract city names."
        assert pred.advanced_prompt_type is False

    def test_create_missing_prompt_uses_empty(self, patched_json_builder):
        pred = predicate.create({"x": {}}, [])
        assert pred.prompt == ""


# ── _interpolate_prompt ───────────────────────────────────────────────────────

class TestInterpolatePrompt:

    @pytest.mark.parametrize(
        "template, strings, expected",
        [
            ("Hello {name}!", [{"name": "Alice"}], "Hello Alice!"),
            ("{a} and {b}", [{"a": "X"}, {"b": "Y"}], "X and Y"),
            ("No tokens here.", [],                 "No tokens here."),
            ("Missing {xyz}.", [{"a": "1"}],        "Missing {xyz}."),
            ("",               [{"a": "1"}],        ""),
            ("{n} {n}",        [{"n": "go"}],       "go go"),
        ],
    )
    def test_various_templates(self, template, strings, expected):
        assert predicate._interpolate_prompt(template, strings) == expected


# ── _process_conditions ───────────────────────────────────────────────────────

class TestProcessConditions:

    def test_empty_conditions_return_empty_program(self):
        prog = predicate._process_conditions([])
        assert isinstance(prog, conditionProgram)
        assert prog.program == ""

    @patch("src.core.predicate.generate_min_program", return_value="target_0 :- test.")
    def test_single_condition_builds_program(self, mock_gen):
        with patch.object(conditionProgram, "validate", return_value=True):
            prog = predicate._process_conditions(
                [predicate_condition("test", False)]
            )
        assert isinstance(prog, conditionProgram)
        assert "target_0" in prog.program

    @patch("src.core.predicate.generate_min_program", side_effect=RuntimeError("boom"))
    def test_fallback_on_generate_error(self, mock_gen):
        with patch.object(conditionProgram, "validate", return_value=True):
            prog = predicate._process_conditions(
                [predicate_condition("raw_condition.", False)]
            )
        assert "raw_condition" in prog.program


# ── has_condition ─────────────────────────────────────────────────────────────

class TestHasCondition:

    def _make(self, conditions):
        return predicate(
            predicate_definition={},
            defined_predicate="p",
            grammar=MagicMock(),
            formatted_predicate="",
            advanced_prompt_type=False,
            prompt="",
            kb=knowledgeBase(""),
            condition_program=conditionProgram(""),
            conditions=conditions,
        )

    def test_with_conditions_returns_true(self, monotone_condition):
        assert self._make([monotone_condition]).has_condition() is True

    def test_without_conditions_returns_false(self):
        assert self._make([]).has_condition() is False


# ── execute_knowledge ─────────────────────────────────────────────────────────

class TestExecuteKnowledge:

    def test_delegates_to_kb_when_advanced(self, atom_list_fixture):
        mock_kb = MagicMock(spec=knowledgeBase)
        mock_kb.execute.return_value = atomList(atoms=[atom("result(1).")])

        pred = predicate(
            predicate_definition={},
            defined_predicate="p",
            grammar=MagicMock(),
            formatted_predicate="",
            advanced_prompt_type=True,
            prompt="",
            kb=mock_kb,
            condition_program=conditionProgram(""),
            conditions=[],
        )
        result = pred.execute_knowledge(atom_list_fixture, atomList(atoms=[]))
        mock_kb.execute.assert_called_once()
        assert result.atoms[0].atom_str == "result(1)."

    def test_returns_extracted_when_not_advanced(self, atom_list_fixture):
        extracted = atomList(atoms=[atom("raw(1).")])
        pred = predicate(
            predicate_definition={},
            defined_predicate="p",
            grammar=MagicMock(),
            formatted_predicate="",
            advanced_prompt_type=False,
            prompt="",
            kb=knowledgeBase(""),
            condition_program=conditionProgram(""),
            conditions=[],
        )
        result = pred.execute_knowledge(atom_list_fixture, extracted)
        assert result is extracted


# ── evaluate_program ──────────────────────────────────────────────────────────

class TestEvaluateProgram:

    def _make_pred_with_conditions(self, conditions):
        return predicate(
            predicate_definition={},
            defined_predicate="p",
            grammar=MagicMock(),
            formatted_predicate="",
            advanced_prompt_type=False,
            prompt="",
            kb=knowledgeBase(""),
            condition_program=conditionProgram(""),
            conditions=conditions,
        )

    def test_list_conditions_all_satisfied(self, monotone_condition):
        atoms = atomList(atoms=[atom("t_0.")])
        # Patch contain_atom_with_suffix to return True for _0
        with patch.object(atomList, "contain_atom_with_suffix", return_value=True):
            pred = self._make_pred_with_conditions([monotone_condition])
            result, details = pred.evaluate_program(atoms)
        assert result is True
        assert details[0][1] is True

    def test_list_conditions_one_fails(self, monotone_condition, non_monotone_condition):
        def side_effect(suffix):
            return suffix == "_0."

        with patch.object(atomList, "contain_atom_with_suffix", side_effect=side_effect):
            pred = self._make_pred_with_conditions([monotone_condition, non_monotone_condition])
            result, details = pred.evaluate_program(atomList(atoms=[]))
        assert result is False


# ── parse_response ────────────────────────────────────────────────────────────

def test_parse_response_splits_lines():
    pred = predicate(
        predicate_definition={},
        defined_predicate="p",
        grammar=MagicMock(),
        formatted_predicate="",
        advanced_prompt_type=False,
        prompt="",
        kb=knowledgeBase(""),
        condition_program=conditionProgram(""),
        conditions=[],
    )
    response = MagicMock()
    response.__str__ = lambda _: "line1.\nline2.\nline3."
    assert pred.parse_response(response) == ["line1.", "line2.", "line3."]