from __future__ import annotations

import json
import re
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Tuple, Union

from typeguard import typechecked

from src.core.atom_list import atomList
from src.core.json_schema import JSONSchemaBuilder
from core.knowledge_base import conditionProgram, knowledgeBase
from src.core.predicate_condition import predicate_condition
from src.helpers.console import get_logger
from src.helpers.generate_condition_program import generate_min_program

log = get_logger(__name__)


@typechecked
@dataclass(frozen=True)
class predicate:
    predicate_definition: Dict[str, Any]
    defined_predicate: str
    grammar: Any
    formatted_predicate: str
    advanced_prompt_type: bool
    prompt: str
    kb: knowledgeBase
    condition_program: conditionProgram
    conditions: List[predicate_condition] = field(default_factory=list)

    # ── Factory ───────────────────────────────────────────────────────────────
    @classmethod
    def create(
        cls,
        predicate_definition: Dict[str, Any],
        strings: List[Dict[str, Any]],
    ) -> "predicate":
        # 1. Type: group vs single
        if "predicates" in predicate_definition:
            group            = predicate_definition["predicates"]
            defined_predicate = " ".join(group)
            grammar          = JSONSchemaBuilder().generate_single_grammar(group)
            config_payload   = predicate_definition
            is_group         = True
        else:
            defined_predicate = next(iter(predicate_definition))
            config_payload   = predicate_definition[defined_predicate]
            grammar          = JSONSchemaBuilder().generate(defined_predicate)
            is_group         = False

        formatted_predicate = json.dumps(grammar.__info__)

        # 2. Config resolution
        is_advanced = isinstance(config_payload, dict) and bool(
            config_payload.get("extraction_condition")
            or config_payload.get("knowledge_base")
        )
        raw_prompt, kb, conditions = cls._resolve_config(
            config_payload, defined_predicate, is_advanced, is_group
        )

        # 3. Prompt interpolation
        prompt = cls._interpolate_prompt(raw_prompt, strings)

        # 4. Condition program
        condition_program = cls._process_conditions(conditions)

        log.debug(
            f"[predicate]Created predicate[/] [bold]{defined_predicate}[/]"
            f" advanced={is_advanced} conditions={len(conditions)}"
        )
        return cls(
            predicate_definition=predicate_definition,
            defined_predicate=defined_predicate,
            grammar=grammar,
            formatted_predicate=formatted_predicate,
            advanced_prompt_type=is_advanced,
            prompt=prompt,
            kb=kb,
            condition_program=condition_program,
            conditions=conditions,
        )

    @staticmethod
    def _resolve_config(
        config: Union[Dict, str],
        name: str,
        is_advanced: bool,
        is_group: bool,
    ) -> Tuple[str, knowledgeBase, List[predicate_condition]]:
        if isinstance(config, str):
            return config, knowledgeBase(""), []

        if is_advanced:
            prompt    = config.get("prompt", "")
            kb_source = config.get("knowledge_base", "")
            raw_conds = config.get("extraction_condition")
            conditions: List[predicate_condition] = []

            if isinstance(raw_conds, dict) and "condition" in raw_conds:
                conditions.append(
                    predicate_condition(raw_conds["condition"], raw_conds.get("monotone", False))
                )
            elif isinstance(raw_conds, list):
                conditions = [
                    predicate_condition(c["condition"], c.get("monotone", False))
                    for c in raw_conds
                    if isinstance(c, dict) and "condition" in c
                ]
            return prompt, knowledgeBase(kb_source or ""), conditions

        # Simple dict
        if is_group:
            prompt = config.get("prompt", "")
        else:
            prompt = config.get("prompt", "") if "prompt" in config else config.get(name, "")
        return prompt, knowledgeBase(""), []

    @staticmethod
    def _interpolate_prompt(template: str, strings: List[Dict]) -> str:
        if not template:
            return ""
        replacements = {k: v for d in strings for k, v in d.items()}
        for token in re.findall(r"\{([A-Za-z0-9_]+)\}", template):
            if token in replacements:
                template = template.replace(f"{{{token}}}", str(replacements[token]))
        return template

    @classmethod
    def _process_conditions(
        cls, cond_info: List[predicate_condition]
    ) -> conditionProgram:
        if not cond_info:
            return conditionProgram("")

        lines: List[str] = []
        for cond in cond_info:
            try:
                lines.append(generate_min_program(cond.condition))
            except Exception:
                lines.append(cond.condition)

        full = "\n".join(lines)
        try:
            prog = conditionProgram(full)
            prog.validate()
            return prog
        except Exception as exc:
            log.error(f"[error]Condition program validation failed:[/] {exc}")
            return conditionProgram("")

    def evaluate_program(self, result: atomList) -> Tuple[bool, list]:
        has_model        = True
        condition_results: list = []

        if isinstance(self.conditions, list):
            for idx, cond in enumerate(self.conditions):
                check = result.contain_atom_with_suffix(f"_{idx}.")
                has_model = has_model and check
                condition_results.append((cond, check))
        else:
            has_model = result.not_empty()
            condition_results.append((self.conditions, has_model))

        return has_model, condition_results

    def execute_condition(self, atoms_database: atomList) -> Tuple[bool, List]:
        if not self.condition_program or not self.conditions:
            return False, []
        try:
            log.debug(f"[predicate]Executing condition[/] for [bold]{self.defined_predicate}[/]")
            result_atoms = self.condition_program.execute(atoms_database)
            return self.evaluate_program(result_atoms)
        except Exception as exc:
            log.error(
                f"[error]Condition error[/] for [bold]{self.defined_predicate}[/]: {exc}",
                exc_info=True,
            )
            return False, []

    def execute_knowledge(
        self, atoms_database: atomList, extracted_atoms: atomList
    ) -> atomList:
        if self.advanced_prompt_type and self.kb:
            try:
                return self.kb.execute(atoms_database, extracted_atoms)
            except Exception as exc:
                log.error(
                    f"[error]KB error[/] for [bold]{self.defined_predicate}[/]: {exc}"
                )
        return extracted_atoms

    def get_grammar(self) -> type:
        return self.grammar

    def parse_response(self, response: Any) -> List[str]:
        return str(response).splitlines()

    def has_condition(self) -> bool:
        return len(self.conditions) > 0

    @property
    def prompt_description(self) -> str:
        return self.prompt

    @property
    def predicate_formatted(self) -> str:
        return str(self.formatted_predicate)

    def __str__(self) -> str:
        return self.defined_predicate