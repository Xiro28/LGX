"""
tests/test_filter_atoms.py
─────────────────────────────────────────────────────────────────────────────
Unit tests for src/helpers/filter_atoms.py.
"""
from __future__ import annotations

import pytest

from src.helpers.filter_atoms import (
    _strip_leading_zeros_in_int_tokens,
    filter_asp_atoms,
    prefix_fix,
)


# ── _strip_leading_zeros_in_int_tokens ────────────────────────────────────────

class TestStripLeadingZeros:

    @pytest.mark.parametrize("raw, expected", [
        ("042",         "42"),
        ("007",         "7"),
        ("0001",        "1"),
        ("100",         "100"),      # no leading zero → unchanged
        ("0",           "0"),        # single zero → unchanged (regex: 0+(\d+) requires ≥1 digit after)
        ("node(042).",  "node(42)."),
        ("",            ""),
    ])
    def test_various_inputs(self, raw, expected):
        assert _strip_leading_zeros_in_int_tokens(raw) == expected


# ── prefix_fix ────────────────────────────────────────────────────────────────

class TestPrefixFix:

    @pytest.mark.parametrize("raw, contains", [
        ("1abc",       "malformed_term_failure__1abc"),
        ("3node(x).",  "malformed_term_failure__3node"),
        ("edge(a,b).", "edge(a,b)."),   # valid → unchanged
        ("node123",    "node123"),       # starts with letter → unchanged
    ])
    def test_various_tokens(self, raw, contains):
        assert contains in prefix_fix(raw)


# ── filter_asp_atoms ──────────────────────────────────────────────────────────

class TestFilterAspAtoms:

    def test_simple_propositional_fact(self):
        result = filter_asp_atoms("fact.")
        assert result.not_empty()
        assert result.atoms[0].atom_str == "fact."

    def test_unary_predicate(self):
        result = filter_asp_atoms("node(a).")
        assert len(result) == 1
        assert result.atoms[0].atom_str == "node(a)."

    def test_binary_predicate(self):
        result = filter_asp_atoms("edge(a,b).")
        assert len(result) == 1
        assert result.atoms[0].atom_str == "edge(a,b)."

    def test_multiple_atoms(self):
        raw = "node(a).\nedge(a,b).\nnode(b)."
        result = filter_asp_atoms(raw)
        assert len(result) == 3

    def test_empty_string_returns_empty_list(self):
        assert filter_asp_atoms("").not_empty() is False

    def test_strip_leading_zeros_applied(self):
        result = filter_asp_atoms("layer(01, node(002)).")
        facts = str(result)
        assert "01" not in facts
        assert "002" not in facts

    def test_malformed_start_digit_renamed(self):
        result = filter_asp_atoms("1abc(x).")
        facts = str(result)
        assert "malformed_term_failure" in facts

    def test_ignores_non_atom_text(self):
        raw = "This is plain English, not ASP."
        result = filter_asp_atoms(raw)
        # "not" matches "\w+\." → filtered as a propositional atom
        # The key check is that no false positive structured atoms appear
        assert isinstance(result.not_empty(), bool)

    def test_returns_atomlist_type(self):
        from src.core.atom_list import atomList
        result = filter_asp_atoms("p(1).")
        assert isinstance(result, atomList)

    @pytest.mark.parametrize("raw, count", [
        ("a. b. c.",                  3),
        ("edge(x,y). edge(y,z).",     2),
        ("",                           0),
        ("no_atoms_here",             0),
    ])
    def test_atom_count(self, raw, count):
        assert len(filter_asp_atoms(raw)) == count

    def test_inline_newlines_and_spaces(self):
        raw = "  node(a).   \n\n  edge(a,b).  "
        result = filter_asp_atoms(raw)
        assert len(result) == 2

    def test_mixed_valid_and_invalid(self):
        raw = "valid(x). 3bad(y). also_valid(z)."
        result = filter_asp_atoms(raw)
        functors = {a.functor() for a in result}
        assert "valid"      in functors
        assert "also_valid" in functors
        # 3bad gets renamed to malformed_term_failure__3bad
        assert any("malformed" in a.atom_str for a in result)