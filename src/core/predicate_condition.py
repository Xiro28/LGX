from __future__ import annotations

from dataclasses import dataclass

from typeguard import typechecked


@typechecked
@dataclass(frozen=True)
class predicate_condition:
    condition: str
    monotone:  bool

    @classmethod
    def create(cls, condition: str, monotone: bool = False) -> "predicate_condition":
        condition = condition.strip()
        if not condition:
            raise ValueError("predicate_condition: condition string must not be empty")
        return cls(condition=condition, monotone=monotone)

    def __str__(self) -> str:
        kind = "monotone" if self.monotone else "non-monotone"
        return f"[{kind}] {self.condition}"

    def __repr__(self) -> str:
        return (
            f"predicate_condition(condition={self.condition!r},"
            f" monotone={self.monotone})"
        )