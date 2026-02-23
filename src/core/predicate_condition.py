from dataclasses import dataclass, field

@dataclass(frozen=True)
class predicate_condition:
    _condition : str 
    _monotone : bool

    @property
    def condition(self) -> str:
        return self._condition
    
    @property
    def is_monotone(self) -> bool:
        return self._monotone