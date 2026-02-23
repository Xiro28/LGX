import logging
import re
import json
from time import time
from typing import Any, Dict, List, Optional, Union, Tuple
from dataclasses import dataclass

from src.core.atom_list import atomList
from src.core.knowledge_base import knowledgeBase, conditionProgram
from src.core.json_schema import JSONSchemaBuilder
from src.core.predicate_condition import predicate_condition

from src.helpers.generate_condition_program import generate_min_program

from typeguard import typechecked

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

    conditions: List[predicate_condition] | predicate_condition | None
    condition_program: conditionProgram

    @classmethod
    def create(cls, predicate_definition: Dict[str, Any], strings: List[Dict[str, Any]]) -> "predicate":
        
        is_group = "predicates" in predicate_definition
        
        if is_group:
            group = predicate_definition["predicates"]
            defined_predicate = " ".join(group)
            grammar = JSONSchemaBuilder().generate_single_grammar(group)
            config_payload = predicate_definition
        else:
            defined_predicate = next(iter(predicate_definition))
            config_payload = predicate_definition[defined_predicate]
            grammar = JSONSchemaBuilder().generate(defined_predicate)

        formatted_predicate = json.dumps(grammar.__info__)

        is_advanced = any(k in config_payload for k in ["extraction_condition", "knowledge_base"])

        raw_prompt, kb, conditions = cls._resolve_config(config_payload, defined_predicate, is_advanced, is_group)
        
        prompt = cls._interpolate_prompt(raw_prompt, strings)

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
    def _resolve_config(config: Union[Dict, str], name: str, is_advanced: bool, is_group: bool) -> tuple:
        if isinstance(config, str):
            return config, knowledgeBase(""), []

        if is_advanced:
            prompt = config.get("prompt", "")
            kb = config.get("knowledge_base", None)
            cond = config.get("extraction_condition")
            
            conditions = [] 
            if cond and isinstance(cond, list):
                conditions = [
                    predicate_condition(c["condition"], c.get("monotone", False)) 
                    for c in cond if isinstance(c, dict) and "condition" in c
                ]
            return prompt, knowledgeBase(kb), conditions
        
        if is_group:
            prompt = config.get("prompt", "")
        else:
            prompt = config.get("prompt") if "prompt" in config else config.get(name, "")
        
        return prompt, knowledgeBase(""), []

    @staticmethod
    def _interpolate_prompt(template: str, strings: List[Dict]) -> str:
        if not template:
            return ""
        replacements = {k: v for d in strings for k, v in d.items()}
        tokens = re.findall(r"\{([A-Za-z0-9\_]+)\}", template)
        result = template
        for token in tokens:
            if token in replacements:
                result = result.replace(f"{{{token}}}", str(replacements[token]))
        return result

    @classmethod
    def _process_conditions(cls, cond_info: list[predicate_condition] | None) -> conditionProgram:
        # Determine if we have a complex condition (a program) or a simple one (just a condition string)

        if not cond_info:
            return conditionProgram("")
        
        complete_condition_program = conditionProgram("")
        
        for cond in cond_info:

            min_program: str = generate_min_program(cond.condition)

            current_condition = conditionProgram(min_program)
            try:
                current_condition.validate() # If this fails, it will raise an exception and we won't add it to the program
                complete_condition_program = complete_condition_program + current_condition
                return complete_condition_program
            except Exception as e:
                pass

            try:
                # If validation fails, we treat it as a program string directly (complex condition)
                current_condition = conditionProgram(cond.condition)
                current_condition.validate() # If this fails, it will raise an exception and we won't add it to the program
                complete_condition_program = complete_condition_program + current_condition
                return complete_condition_program
            except Exception as e:
                logging.error(f"Invalid condition definition: {cond.condition}: {e}", exc_info=True)
            
            return conditionProgram("")        


    def evaluate_program(self, result: atomList) -> bool:
        has_model = True

        if isinstance(self.conditions, list):
            for idx, cond in enumerate(self.conditions):
                check = result.contain_atom_with_suffix(f"_{idx}")
                has_model = has_model and check
                # Update the cache for each condition
        else:
            has_model = result.not_empty()
            # Update the cache for each condition

        return has_model

    def execute_condition(self) -> bool:
        if not self.condition_program:
            return False

        try:
            result = self.condition_program.execute()

            return self.evaluate_program(result)
        except Exception:
            logging.error(f"Error executing condition program for predicate {self.defined_predicate}", exc_info=True)
            return False

    def has_to_be_extracted(self) -> bool:
        return self.execute_condition()
        
    def execute_knowledge(self, atoms_database: atomList, extracted_atoms: atomList) -> atomList:
        if self.advanced_prompt_type and self.kb:
            try:
                full_program = f"{str(atoms_database)}\n{str(extracted_atoms)}"
                return self.kb.execute(full_program)
            except Exception:
                pass
        return extracted_atoms
    
    def get_grammar(self) -> type:
        return self.grammar
    
    def parse_response(self, response: Any) -> List[str]:
        return str(response).splitlines()
    
    def has_condition(self) -> bool:
        return self.conditions != None and len(self.conditions) > 0

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