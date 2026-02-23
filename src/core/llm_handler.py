import logging
import ollama
import os
from dataclasses import dataclass, field
from typing import Any, Generator, Optional
from typeguard import typechecked

# Assumo che questi siano importati correttamente dal tuo progetto
from src.core.atom_list import atomList
from src.core.prompt_database import promptDatabase
from src.helpers.filter_atoms import filter_asp_atoms
from src.core.yaml_parser import application_configuration, behaviour_configuration
from src.core.cache import ConditionCache
from src.core.predicate import predicate

@typechecked
@dataclass(frozen=True)
class llmHandler:
    llm_model: str
    application_config: application_configuration
    behaviour_config: behaviour_configuration
    prompt_database: Optional[promptDatabase]
    predicate_condition_cache: ConditionCache
    _llm_chat_fn: Any 

    atom_database: atomList = field(init=False, default=atomList(atoms=[]))

    @classmethod
    def create(cls, 
               llm_model: str, 
               application_cfg: application_configuration, 
               behaviour_cfg: behaviour_configuration) -> "llmHandler":
        
        use_cached_prompts = os.getenv("LGX_ENABLE_PROMPT_CACHE", "true").lower() == "true"

        if use_cached_prompts:
            database = os.getenv("LGX_USE_DB_FILENAME", "cached_prompts.db")
            db = promptDatabase.initialize(database)
        else:
            db = None

        use_condition_cache = os.getenv("LGX_ENABLE_CONDITION_CACHE", "true").lower() == "true"
        condition_cache_mode = os.getenv("LGX_CONDITION_CACHE_MODE", "all").lower()
        cache = ConditionCache(use_condition_cache, condition_cache_mode)

        headers = {}
        if api_key := os.getenv("LGX_OLLAMA_API_KEY"):
            headers['Authorization'] = f'Bearer {api_key}'
        
        client = ollama.Client(
            host=os.getenv("LGX_OLLAMA_URL", "http://localhost:1143"), 
            headers=headers
        )

        return cls(
            llm_model=llm_model,
            application_config=application_cfg,
            behaviour_config=behaviour_cfg,
            prompt_database=db,
            predicate_condition_cache=cache,
            _llm_chat_fn=client.chat # Passiamo direttamente il metodo callable
        )

    def __del__(self):
        if self.prompt_database is not None:
            self.prompt_database.close()

    def craft_message_history(self, prompt: str, context: str) -> list:
        return [
            {"role": "system", "content": f"{self.behaviour_config.init}{context}"},
            {"role": "user", "content": prompt}
        ]

    # the return value here is the class dynamic built class so use ANY to cover all of them
    def invoke_llm_constrained(self, prompt: str, class_response: any, context: str) -> Optional[Any]: 
        configuration = {'temperature': 0}
        grammar = class_response.model_json_schema(mode='serialization')
        history = self.craft_message_history(prompt, context)

        if self.prompt_database is not None:
            query_result = self.prompt_database.get_cached_response(history, self.llm_model, configuration)
            if query_result:
                try:
                    return class_response.model_validate_json(query_result[0])
                except Exception as e:
                    print(e)
                    #self.prompt_database.delete_cached_response(history, self.llm_model, configuration)

        try:
            llm_response = self._llm_chat_fn(
                model=self.llm_model,
                messages=history,
                options=configuration,
                format=grammar
            )
            
            content = llm_response["message"]["content"]

            if self.prompt_database is not None:
                self.prompt_database.cache_response(
                    history, self.llm_model, configuration, content, 
                    llm_response["prompt_eval_count"], 
                    llm_response["eval_count"], 
                    llm_response["total_duration"]
                )

            return class_response.model_validate_json(content)
        except Exception as e:
            logging.error(f"Error invoking/validating LLM: {e}")
            return None

    def _structured_output_call(self, prompt: str) -> Generator[tuple[Any, predicate], None, None]:
        behaviour_context = ""
        behaviour_mapping = self.behaviour_config.mapping.replace("{input}", prompt)
        
        if self.application_config.context:
            behaviour_context = self.behaviour_config.context.replace("{context}", f"{self.application_config.context}")
        
        for pred in self.application_config.predicates:
            if pred.has_condition():
                if not self.predicate_condition_cache.skip_logic_solver(pred.conditions): 
                    # Evaluate the program
                    call_oracle, condition_results = pred.execute_condition(self.atom_database)

                    for result in condition_results:
                        self.predicate_condition_cache.update(result[0], result[1])
                    
                    if not call_oracle:
                        continue
                elif not self.predicate_condition_cache.get(pred.conditions):
                    continue


            appl_mapping = behaviour_mapping.replace("{instructions}", f"{pred.prompt_description}")
            appl_mapping = appl_mapping.replace("{atom}", pred.predicate_formatted)

            yield self.invoke_llm_constrained(appl_mapping, pred.get_grammar(), behaviour_context), pred

    def run(self, prompt: str) -> atomList:

        for response, pred in self._structured_output_call(prompt):
            if response is None:
                continue

            atom_strings = pred.parse_response(response)
            extracted_facts = filter_asp_atoms("\n".join(atom_strings))

            if extracted_facts.not_empty():
                kb_execution_result = pred.execute_knowledge(self.atom_database, extracted_facts)
                self.atom_database.atoms.extend(kb_execution_result.atoms)

        return self.atom_database
    
    def get_extracted_atoms(self) -> atomList:
        return self.atom_database
    
    def cleanup(self):
        self.atom_database.atoms.clear()
        self.predicate_condition_cache.clear()