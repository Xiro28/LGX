"""
src/core/atom_list.py

Notes
-----
The original ``__add__`` used ``list.extend()`` which returns ``None``, so
``al1 + al2`` silently returned ``None`` instead of the combined list.
Fixed below: returns a *new* ``atomList`` (frozen-dataclass-safe).
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Iterator

from typeguard import typechecked

from src.core.atom import atom


@typechecked
@dataclass(frozen=True)
class atomList:
    atoms: list[atom] = field(default_factory=list)

    # ── String / iteration ────────────────────────────────────────────────────
    def __str__(self) -> str:
        return "\n".join(str(a) for a in self.atoms)

    def __repr__(self) -> str:
        return f"atomList({len(self.atoms)} atoms)"

    def __iter__(self) -> Iterator[atom]:
        return iter(self.atoms)

    def __len__(self) -> int:
        return len(self.atoms)

    # ── Operators ─────────────────────────────────────────────────────────────
    def __add__(self, other: "atomList") -> "atomList":
        """Return a *new* atomList containing atoms from both operands."""
        return atomList(atoms=list(self.atoms) + list(other.atoms))

    # ── Query helpers ─────────────────────────────────────────────────────────
    def not_empty(self) -> bool:
        return len(self.atoms) > 0

    def contain_atom_with_suffix(self, suffix: str) -> bool:
        """Return True if any atom's string representation ends with *suffix*.

        Used to check for condition-program output atoms of the form
        ``target_<idx>.`` produced by ``generate_min_program``.
        """
        return any(a.atom_str.endswith(suffix) for a in self.atoms)

    def filter_by_functor(self, functor: str) -> "atomList":
        """Return a new atomList containing only atoms with *functor* as their functor name."""
        return atomList(atoms=[a for a in self.atoms if a.functor() == functor])

    def to_facts(self) -> str:
        """Render all atoms as a newline-separated block of ASP facts."""
        return "\n".join(
            s if s.endswith(".") else s + "." for s in (str(a) for a in self.atoms)
        )