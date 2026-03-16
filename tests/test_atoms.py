"""
tests/test_atoms.py
─────────────────────────────────────────────────────────────────────────────
Unit tests for atom, atomList, and predicate_condition.
"""
from __future__ import annotations

import pytest

from src.core.atom import atom
from src.core.atom_list import atomList
from src.core.predicate_condition import predicate_condition


# ═══════════════════════════════════════════════════════════════════════════════
# atom
# ═══════════════════════════════════════════════════════════════════════════════

class TestAtom:

    @pytest.mark.parametrize("atom_str", [
        "node(a).",
        "edge(a,b).",
        "fact.",
        "triple(x,y,z).",
    ])
    def test_str_roundtrip(self, atom_str):
        a = atom(atom_str=atom_str)
        assert str(a) == atom_str

    def test_repr_contains_atom_str(self):
        a = atom(atom_str="p(1).")
        assert "p(1)." in repr(a)

    @pytest.mark.parametrize("atom_str, expected_functor", [
        ("edge(a,b).", "edge"),
        ("fact.",      "fact"),
        ("node(x).",   "node"),
        ("x",          "x"),
    ])
    def test_functor(self, atom_str, expected_functor):
        assert atom(atom_str=atom_str).functor() == expected_functor

    @pytest.mark.parametrize("atom_str, expected_arity", [
        ("fact.",        0),
        ("node(x).",     1),
        ("edge(a,b).",   2),
        ("triple(x,y,z).", 3),
    ])
    def test_arity(self, atom_str, expected_arity):
        assert atom(atom_str=atom_str).arity() == expected_arity

    def test_frozen_dataclass_immutable(self):
        a = atom(atom_str="p.")
        with pytest.raises((AttributeError, TypeError)):
            a.atom_str = "q."  # type: ignore[misc]


# ═══════════════════════════════════════════════════════════════════════════════
# atomList
# ═══════════════════════════════════════════════════════════════════════════════

class TestAtomList:

    def test_empty_list_not_empty_returns_false(self, empty_atom_list):
        assert empty_atom_list.not_empty() is False

    def test_non_empty_list_not_empty_returns_true(self, atom_list_fixture):
        assert atom_list_fixture.not_empty() is True

    def test_str_joins_atoms_with_newline(self):
        al = atomList(atoms=[atom("a."), atom("b.")])
        assert str(al) == "a.\nb."

    def test_repr(self, atom_list_fixture):
        assert "2 atoms" in repr(atom_list_fixture)

    def test_len(self, atom_list_fixture):
        assert len(atom_list_fixture) == 2

    def test_iteration(self, atom_list_fixture):
        strs = [str(a) for a in atom_list_fixture]
        assert "edge(a,b)." in strs
        assert "edge(b,c)." in strs

    def test_add_returns_new_list(self):
        al1 = atomList(atoms=[atom("a.")])
        al2 = atomList(atoms=[atom("b.")])
        combined = al1 + al2
        assert len(combined) == 2
        assert combined is not al1
        assert combined is not al2

    def test_add_does_not_mutate_originals(self):
        al1 = atomList(atoms=[atom("a.")])
        al2 = atomList(atoms=[atom("b.")])
        _ = al1 + al2
        assert len(al1) == 1
        assert len(al2) == 1

    @pytest.mark.parametrize("suffix, expected", [
        ("_0.",  True),
        ("_1.",  True),
        ("_99.", False),
        ("x.",   False),
    ])
    def test_contain_atom_with_suffix(self, suffix, expected):
        al = atomList(atoms=[atom("target_0."), atom("target_1.")])
        assert al.contain_atom_with_suffix(suffix) is expected

    def test_filter_by_functor(self):
        al = atomList(atoms=[atom("edge(a,b)."), atom("node(x)."), atom("edge(c,d).")])
        edges = al.filter_by_functor("edge")
        assert len(edges) == 2
        nodes = al.filter_by_functor("node")
        assert len(nodes) == 1

    def test_filter_by_functor_no_match(self, atom_list_fixture):
        result = atom_list_fixture.filter_by_functor("no_such_functor")
        assert result.not_empty() is False

    def test_to_facts_adds_dot_if_missing(self):
        al = atomList(atoms=[atom("node(x)"), atom("edge(a,b).")])
        facts = al.to_facts()
        assert facts.count("node(x).") == 1
        assert facts.count("edge(a,b).") == 1

    def test_empty_atomlist_str(self, empty_atom_list):
        assert str(empty_atom_list) == ""

    def test_empty_atomlist_to_facts(self, empty_atom_list):
        assert empty_atom_list.to_facts() == ""


# ═══════════════════════════════════════════════════════════════════════════════
# predicate_condition
# ═══════════════════════════════════════════════════════════════════════════════

class TestPredicateCondition:

    def test_create_factory_strips_whitespace(self):
        pc = predicate_condition.create("  edge(_, _)  ", monotone=True)
        assert pc.condition == "edge(_, _)"

    def test_create_factory_empty_raises(self):
        with pytest.raises(ValueError, match="must not be empty"):
            predicate_condition.create("   ")

    def test_direct_construction(self):
        pc = predicate_condition(condition="node(x)", monotone=False)
        assert pc.condition == "node(x)"
        assert pc.monotone  is False

    def test_frozen_immutable(self):
        pc = predicate_condition(condition="p", monotone=True)
        with pytest.raises((AttributeError, TypeError)):
            pc.condition = "q"  # type: ignore[misc]

    def test_negated_adds_not(self):
        pc = predicate_condition(condition="edge(_, _)", monotone=True)
        neg = pc.negated()
        assert neg.condition == "not edge(_, _)"
        assert neg.monotone  is True

    def test_negated_removes_not(self):
        pc = predicate_condition(condition="not error(_)", monotone=False)
        neg = pc.negated()
        assert neg.condition == "error(_)"

    def test_str_contains_kind_and_condition(self):
        pc = predicate_condition(condition="p(x)", monotone=True)
        s  = str(pc)
        assert "monotone"  in s
        assert "p(x)"      in s

    def test_repr_roundtrip_info(self):
        pc = predicate_condition(condition="q", monotone=False)
        r  = repr(pc)
        assert "non-monotone" not in r   # repr uses bool, not kind label
        assert "False"         in r
        assert "q"             in r

    @pytest.mark.parametrize("monotone", [True, False])
    def test_equality_based_on_fields(self, monotone):
        pc1 = predicate_condition(condition="p", monotone=monotone)
        pc2 = predicate_condition(condition="p", monotone=monotone)
        assert pc1 == pc2

    def test_inequality_different_condition(self):
        pc1 = predicate_condition(condition="p", monotone=True)
        pc2 = predicate_condition(condition="q", monotone=True)
        assert pc1 != pc2

    def test_inequality_different_monotone(self):
        pc1 = predicate_condition(condition="p", monotone=True)
        pc2 = predicate_condition(condition="p", monotone=False)
        assert pc1 != pc2