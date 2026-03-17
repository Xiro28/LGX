import pytest
from unittest.mock import MagicMock, patch
from src.core.atom import atom
from src.core.atom_list import atomList
from core.knowledge_base import knowledgeBase

def test_atom_creation():
    a = atom(atom_str="test_predicate(1)")
    assert str(a) == "test_predicate(1)"

def test_atom_list_operations():
    a1 = atom(atom_str="a(1)")
    a2 = atom(atom_str="b(2)")
    al1 = atomList(atoms=[a1])
    al2 = atomList(atoms=[a2])
    
    # Test str representation
    assert str(al1) == "a(1)."
    
    # Test addition (assuming the fix for frozen dataclasses)
    combined = al1 + al2
    assert len(combined.atoms) == 2
    assert combined.atoms[0].atom_str == "a(1)"
    assert combined.atoms[1].atom_str == "b(2)"

@patch("src.core.knowledge_base.Model")
def test_knowledge_base_execute(mock_model):
    # Mocking dumbo_asp Model
    mock_instance = MagicMock()
    mock_instance.as_facts = "result(1).\nresult(2)."
    mock_model.of_program.return_value = mock_instance
    
    kb = knowledgeBase(program="fact(1).")
    result = kb.execute(atomList(atoms=[atom("input(1)")]))
    
    assert len(result.atoms) == 2
    assert result.atoms[0].atom_str == "result(1)"
    mock_model.of_program.assert_called_once()