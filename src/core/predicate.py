import logging
import re
import json
from typing import Any, Dict, List, Optional, Union, Tuple
from dataclasses import dataclass, field

from typeguard import typechecked

# Assuming these imports exist in your project structure
from src.core.atom_list import atomList
from src.core.knowledge_base import knowledgeBase, conditionProgram
from src.core.json_schema import JSONSchemaBuilder
from src.core.predicate_condition import predicate_condition
from src.helpers.generate_condition_program import generate_min_program


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

    @classmethod
    def create(cls, predicate_definition: Dict[str, Any], strings: List[Dict[str, Any]]) -> "predicate":
        
        # 1. Determine Predicate Type (Group vs Single)
        if "predicates" in predicate_definition:
            # Group Logic
            group = predicate_definition["predicates"]
            defined_predicate = " ".join(group)
            grammar = JSONSchemaBuilder().generate_single_grammar(group)
            config_payload = predicate_definition # The definition itself holds the config for groups
            is_group = True
        else:
            # Single Logic
            defined_predicate = next(iter(predicate_definition))
            config_payload = predicate_definition[defined_predicate]
            grammar = JSONSchemaBuilder().generate(defined_predicate)
            is_group = False

        formatted_predicate = json.dumps(grammar.__info__)

        # 2. Extract Configuration
        # Check if it's an advanced config (dict with KB/Conditions) or simple (just prompt string)
        is_advanced = False
        if isinstance(config_payload, dict):
            is_advanced = bool(config_payload.get("extraction_condition") or config_payload.get("knowledge_base"))

        raw_prompt, kb, conditions = cls._resolve_config(config_payload, defined_predicate, is_advanced, is_group)

        # 3. Interpolate Prompt
        prompt = cls._interpolate_prompt(raw_prompt, strings)

        # 4. Build Condition Program
        condition_program = cls._process_conditions(conditions)

        return cls(
            predicate_definition=predicate_definition,
            defined_predicate=defined_predicate,
            grammar=grammar,
            formatted_predicate=formatted_predicate,
            advanced_prompt_type=is_advanced,
            prompt=prompt,
            kb=kb,
            condition_program=condition_program,
            conditions=conditions
        )

    @staticmethod
    def _resolve_config(config: Union[Dict, str], name: str, is_advanced: bool, is_group: bool) -> Tuple[str, knowledgeBase, List[predicate_condition]]:
        # Case 1: Simple String Config
        if isinstance(config, str):
            return config, knowledgeBase(""), []

        # Case 2: Advanced Config (KB or Conditions present)
        if is_advanced:
            prompt = config.get("prompt", "")
            kb_source = config.get("knowledge_base", "")
            
            raw_conds = config.get("extraction_condition")
            conditions = []
            
            if raw_conds:
                # Handle single condition (dict) vs list of conditions
                if isinstance(raw_conds, dict) and "condition" in raw_conds:
                    conditions.append(predicate_condition(raw_conds["condition"], raw_conds.get("monotone", False)))
                elif isinstance(raw_conds, list):
                    conditions = [
                        predicate_condition(c["condition"], c.get("monotone", False)) 
                        for c in raw_conds if isinstance(c, dict) and "condition" in c
                    ]
            
            return prompt, knowledgeBase(kb_source), conditions
        
        # Case 3: Simple Dict Config (Prompt only, nested or direct)
        if is_group:
            prompt = config.get("prompt", "")
        else:
            # Handle {"predicate": {"prompt": "..."}} vs {"predicate": "prompt"}
            prompt = config.get("prompt", "") if "prompt" in config else config.get(name, "")
        
        return prompt, knowledgeBase(""), []

    @staticmethod
    def _interpolate_prompt(template: str, strings: List[Dict]) -> str:
        if not template:
            return ""
        
        # Flatten list of dicts into single dict
        replacements = {k: v for d in strings for k, v in d.items()}
        
        # Find all tokens {token}
        tokens = re.findall(r"\{([A-Za-z0-9\_]+)\}", template)
        
        result = template
        for token in tokens:
            if token in replacements:
                result = result.replace(f"{{{token}}}", str(replacements[token]))
        return result

    @classmethod
    def _process_conditions(cls, cond_info: List[predicate_condition]) -> conditionProgram:
        if not cond_info:
            return conditionProgram("")
        
        combined_program_lines = []

        for idx, cond in enumerate(cond_info):
            try:
                # Try to generate a minimized program (e.g., "uuid_0 :- condition.")
                # Note: generate_min_program likely needs to handle the index to create unique heads
                # or we assume the condition is complex and validate it as is.
                min_program = generate_min_program(cond.condition)
                combined_program_lines.append(min_program)
            except Exception:
                # Fallback: treat as raw complex condition
                combined_program_lines.append(cond.condition)

        full_program_str = "\n".join(combined_program_lines)

        try:
            prog = conditionProgram(full_program_str)
            prog.validate()
            return prog
        except Exception as e:
            logging.error(f"Failed to validate combined condition program: {e}")
            return conditionProgram("")


    def evaluate_program(self, result: atomList) -> tuple[bool, list]:
        has_model = True
        condition_results = list()

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
            return (False, [])

        try:
            logging.debug(f"Executing condition for {self.defined_predicate}")
            
            result_atoms = self.condition_program.execute(atoms_database)
            evaluated, results = self.evaluate_program(result_atoms)
            
            return evaluated, results
        except Exception as e:
            logging.error(f"Error executing condition for {self.defined_predicate}: {e}", exc_info=True)
        
        return (False, [])
        
    def execute_knowledge(self, atoms_database: atomList, extracted_atoms: atomList) -> atomList:
        if self.advanced_prompt_type and self.kb:
            try:
                # Combine DB atoms with newly extracted atoms for the KB check
                return self.kb.execute(atoms_database + extracted_atoms)
            except Exception as e:
                logging.error(f"Error executing KB for {self.defined_predicate}: {e}")
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
    def predicate(self) -> str:
        return self.defined_predicate
    
    @property
    def predicate_formatted(self) -> str:
        return str(self.formatted_predicate)
    
    def __str__(self) -> str:
        return self.defined_predicate