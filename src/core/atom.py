from dataclasses import dataclass, field
from typing import Optional
from typeguard   import typechecked

@typechecked
@dataclass(frozen=True)
class atom:
    atom_str: str
    
    def __str__(self):
        return self.atom_str