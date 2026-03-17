from __future__ import annotations

from dataclasses import dataclass

from typeguard import typechecked


@typechecked
@dataclass(frozen=True)
class atom:
    atom_str: str

    def __str__(self) -> str:
        return self.atom_str

    def __repr__(self) -> str:
        return f"atom({self.atom_str!r})"

    def functor(self) -> str:
        for ch in ("(", "."):
            idx = self.atom_str.find(ch)
            if idx != -1:
                return self.atom_str[:idx]
        return self.atom_str

    def arity(self) -> int:
        start = self.atom_str.find("(")
        end   = self.atom_str.rfind(")")
        if start == -1 or end == -1:
            return 0
        inner = self.atom_str[start + 1 : end].strip()
        return len(inner.split(",")) if inner else 0