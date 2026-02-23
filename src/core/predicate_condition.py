from dataclasses import dataclass, field

@dataclass(frozen=True)
class predicate_condition:
    condition : str 
    monotone : bool