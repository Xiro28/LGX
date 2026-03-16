"""
src/core/predicate_condition.py
"""
from __future__ import annotations

from dataclasses import dataclass

from typeguard import typechecked


@typechecked
@dataclass(frozen=True)
class predicate_condition:
    condition: str
    monotone:  bool

    # ── Factory ───────────────────────────────────────────────────────────────
    @classmethod
    def create(cls, condition: str, monotone: bool = False) -> "predicate_condition":
        """Canonical factory method — strips whitespace and validates non-empty."""
        condition = condition.strip()
        if not condition:
            raise ValueError("predicate_condition: condition string must not be empty")
        return cls(condition=condition, monotone=monotone)

    # ── Helpers ───────────────────────────────────────────────────────────────
    def negated(self) -> "predicate_condition":
        """Return a new condition with the ASP negation-as-failure wrapper."""
        neg = (
            self.condition.removeprefix("not ").strip()
            if self.condition.startswith("not ")
            else f"not {self.condition}"
        )
        return predicate_condition(condition=neg, monotone=self.monotone)

    def __str__(self) -> str:
        kind = "monotone" if self.monotone else "non-monotone"
        return f"[{kind}] {self.condition}"

    def __repr__(self) -> str:
        return (
            f"predicate_condition(condition={self.condition!r},"
            f" monotone={self.monotone})"
        )
    
    @property
    def is_monotone(self) -> bool:
        return self.monotone