from __future__ import annotations

from dataclasses import dataclass, field
from typing import Iterator

from typeguard import typechecked

from src.core.atom import atom


@typechecked
@dataclass(frozen=True)
class atomList:
    atoms: list[atom] = field(default_factory=list)

    def __str__(self) -> str:
        return "\n".join(str(a) for a in self.atoms)

    def __repr__(self) -> str:
        return f"atomList({len(self.atoms)} atoms)"

    def __iter__(self) -> Iterator[atom]:
        return iter(self.atoms)

    def __len__(self) -> int:
        return len(self.atoms)

    def __add__(self, other: "atomList") -> "atomList":
        return atomList(atoms=list(self.atoms) + list(other.atoms))

    def not_empty(self) -> bool:
        return len(self.atoms) > 0

    def contain_atom_with_suffix(self, suffix: str) -> bool:
        return any(a.atom_str.endswith(suffix) for a in self.atoms)

    def filter_by_functor(self, functor: str) -> "atomList":
        return atomList(atoms=[a for a in self.atoms if a.functor() == functor])

    def to_facts(self) -> str:
        return "\n".join(
            s if s.endswith(".") else s + "." for s in (str(a) for a in self.atoms)
        )