import pytest
import json
from unittest.mock import MagicMock, patch
from src.core.predicate import predicate
from src.core.predicate_condition import predicate_condition
from src.core.knowledge_base import knowledgeBase, conditionProgram
from src.core.atom_list import atomList
from src.core.atom import atom

@pytest.fixture
def mock_json_builder():
    with patch("src.core.predicate.JSONSchemaBuilder") as mock:
        builder_inst = mock.return_value
        # Mock del ritorno di generate()
        grammar_mock = MagicMock()
        grammar_mock.__info__ = {"type": "object", "properties": {}}
        builder_inst.generate.return_value = grammar_mock
        builder_inst.generate_single_grammar.return_value = grammar_mock
        yield builder_inst

@pytest.fixture
def basic_predicate_def():
    return {
        "person": {
            "prompt": "Extract person names from {text}",
            "knowledge_base": "person(N) :- extracted_name(N)."
        }
    }

class TestPredicate:

    def test_create_simple(self, mock_json_builder, basic_predicate_def):
        """Testa la creazione di un predicato semplice (non-group)."""
        strings = [{"text": "John Doe"}]
        
        pred = predicate.create(basic_predicate_def, strings)
        
        assert pred.defined_predicate == "person"
        assert "John Doe" in pred.prompt
        assert isinstance(pred.kb, knowledgeBase)
        assert pred.advanced_prompt_type is True

    def test_interpolate_prompt(self):
        """Testa il metodo statico di interpolazione stringhe."""
        template = "Hello {name}, welcome to {city}."
        strings = [{"name": "Alice"}, {"city": "Rome"}]
        
        result = predicate._interpolate_prompt(template, strings)
        assert result == "Hello Alice, welcome to Rome."
        
        # Test con token mancante
        result_missing = predicate._interpolate_prompt("Hello {age}", strings)
        assert result_missing == "Hello {age}"

    @patch("src.core.predicate.generate_min_program")
    def test_process_conditions(self, mock_gen_min):
        """Testa l'elaborazione delle condizioni ASP."""
        mock_gen_min.return_value = "target(1)."
        cond_info = [predicate_condition(_condition="test", _monotone=False)]
        
        # Mock della validazione per evitare esecuzione ASP reale nel test di unità
        with patch.object(conditionProgram, 'validate', return_value=True):
            prog = predicate._process_conditions(cond_info)
            assert isinstance(prog, conditionProgram)
            assert "target(1)" in prog.program

    def test_has_condition(self, mock_json_builder):
        """Verifica la logica di has_condition."""
        # Caso con condizioni
        pred_with = predicate(
            predicate_definition={}, defined_predicate="p", grammar=None,
            formatted_predicate="", advanced_prompt_type=False, prompt="",
            kb=None, conditions=[MagicMock()], condition_program=None
        )
        assert pred_with.has_condition() is True

        # Caso senza (Nota: il codice attuale usa 'is not []', testiamo il comportamento)
        pred_without = predicate(
            predicate_definition={}, defined_predicate="p", grammar=None,
            formatted_predicate="", advanced_prompt_type=False, prompt="",
            kb=None, conditions=[], condition_program=None
        )
        # Se self.conditions è [], '[] is not []' è True in Python (nuovi oggetti)
        # Questo test serve a monitorare comportamenti inattesi
        assert pred_without.has_condition() is False 

    def test_execute_knowledge(self, mock_json_builder):
        """Testa l'esecuzione della KB associata al predicato."""
        mock_kb = MagicMock(spec=knowledgeBase)
        mock_kb.execute.return_value = atomList(atoms=[atom("result(1)")])
        
        pred = predicate(
            predicate_definition={}, defined_predicate="p", grammar=None,
            formatted_predicate="", advanced_prompt_type=True, prompt="",
            kb=mock_kb, conditions=None, condition_program=None
        )
        
        db = atomList(atoms=[atom("db(1)")])
        ext = atomList(atoms=[atom("ext(1)")])
        
        result = pred.execute_knowledge(db, ext)
        
        assert mock_kb.execute.called
        assert result.atoms[0].atom_str == "result(1)"

    def test_evaluate_program_with_list(self, mock_json_builder):
        """Testa la valutazione del risultato rispetto ai suffissi delle condizioni."""
        conds = [MagicMock(), MagicMock()] # Due condizioni
        pred = predicate(
            predicate_definition={}, defined_predicate="p", grammar=None,
            formatted_predicate="", advanced_prompt_type=False, prompt="",
            kb=None, conditions=conds, condition_program=None
        )
        
        # Mock del risultato che contiene i suffissi _0 e _1
        mock_result = MagicMock(spec=atomList)
        mock_result.contain_atom_with_suffix.side_effect = [True, True]
        
        assert pred.evaluate_program(mock_result) is True
        
        # Se una manca
        mock_result.contain_atom_with_suffix.side_effect = [True, False]
        assert pred.evaluate_program(mock_result) is False

    def test_parse_response(self, mock_json_builder):
        """Verifica che la risposta venga splittata correttamente in righe."""
        pred = predicate(
            predicate_definition={}, defined_predicate="p", grammar=None,
            formatted_predicate="", advanced_prompt_type=False, prompt="",
            kb=None, conditions=None, condition_program=None
        )
        response = "fact(1)\nfact(2)"
        parsed = pred.parse_response(response)
        assert parsed == ["fact(1)", "fact(2)"]