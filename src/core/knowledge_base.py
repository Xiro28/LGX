from dataclasses                 import dataclass, field
from typeguard                   import typechecked
from typing                      import Optional

from dumbo_asp.primitives.models import Model

from src.core.atom import atom
from src.core.atom_list import atomList

@typechecked
@dataclass(frozen=True)
class knowledgeBase:
    program: str

    def execute(self, database_atoms: Optional[atomList] = None, extracted_atoms: Optional[atomList] = None) -> atomList:
        atoms_input = str(database_atoms) if database_atoms is not None else ""

        if extracted_atoms is not None:
            atoms_input += "\n" + str(extracted_atoms)
        
        kb_output: str = Model.of_program(
            self.program, 
            atoms_input, 
            sort=False
        ).as_facts

        parsed_atoms = [
            atom(line) 
            for line in kb_output.splitlines() 
            if line.strip()
        ]
        
        return atomList(atoms=parsed_atoms)

    def validate(self) -> bool:
        try:
            self.execute()
            return True
        except Exception:
            return False

    def __add__(self, other: 'knowledgeBase') -> 'knowledgeBase':
        combined_program = self.program + "\n" + other.program
        return type(self)(combined_program)

    def __str__(self):
        return self.program


@typechecked
@dataclass(frozen=True)
class conditionProgram(knowledgeBase):
    pass