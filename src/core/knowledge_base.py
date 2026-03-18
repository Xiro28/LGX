from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

from dumbo_asp.primitives.models import Model
from typeguard import typechecked

from src.core.atom import atom
from src.core.atom_list import atomList
from src.helpers.console import get_logger

log = get_logger(__name__)


@typechecked
@dataclass(frozen=True)
class knowledgeBase:
    program: str

    def execute(
        self,
        database_atoms: Optional[atomList] = None,
        extracted_atoms: Optional[atomList] = None,
    ) -> atomList:
        atoms_input = str(database_atoms) if database_atoms is not None else ""
        if extracted_atoms is not None:
            atoms_input += "\n" + str(extracted_atoms)

        log.debug(f"[info]KB execute[/] input_atoms={len(atoms_input.splitlines())} lines")

        kb_output: str = Model.of_program(
            self.program, atoms_input, sort=False
        ).as_facts

        parsed = [
            atom(line)
            for line in kb_output.splitlines()
            if line.strip()
        ]

        log.debug(f"[info]KB produced[/] [bold]{len(parsed)}[/] atom(s)")
        return atomList(atoms=parsed)

    def validate(self) -> bool:
        try:
            self.execute()
            return True
        except Exception as exc:
            log.debug(f"[warning]KB validation failed:[/] {exc}")
            return False

    def __add__(self, other: "knowledgeBase") -> "knowledgeBase":
        return type(self)(self.program + "\n" + other.program)

    def __str__(self) -> str:
        return self.program


@typechecked
@dataclass(frozen=True)
class conditionProgram(knowledgeBase):
    pass