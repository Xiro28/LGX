from dataclasses import dataclass, field
from typeguard   import typechecked

from src.core.atom import atom

@typechecked
@dataclass(frozen=True)
class atomList:
    atoms: list[atom] = field(default_factory=list)

    def __str__(self):
        return '.\n'.join(str(atom) for atom in self.atoms) + '.'

    def __add__(self, other: 'atomList') -> 'atomList':
        return self.atoms.extend(other.atoms) or self
    
    # It's used for checking the presence of the "_{idx}" atoms in the evaluation of conditions.
    def contain_atom_with_suffix(self, item: str) -> bool:
        return any(atom.atom_str.endswith(item) for atom in self.atoms)
    
    def not_empty(self) -> bool:
        return len(self.atoms) > 0

    
    def __iter__(self):
        return iter(self.atoms)