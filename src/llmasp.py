import yaml
import logging

from dumbo_asp.primitives.models    import Model
from typeguard                      import typechecked
from dataclasses                    import dataclass, field


from src.utils.statistics                   import Statistics

from src.core.input.evaluate_prompt         import EvaluatePrompt
from src.core.predicate.condition_cache     import ConditionCache
from src.core.predicate.predicate_container import PredicateContainer


@typechecked
@dataclass
class LLMASP:
    __behaviour_config_filename:   str = field(init=True, default="")
    __application_config_filename: str = field(init=True, default="")
    
    __llm_model:            str = field(init=True, default="")

    __behaviour_config:     dict = field(init=False)
    __application_config:   dict = field(init=False)

    __result_preds:         str = field(init=False, default="")
    __extracted_preds:      str = field(init=False, default="")
    __total_extracted_pred: str = field(init=False, default="")

    
    def __post_init__(self):
        assert self.__llm_model != "", "The LLM extractor cannot be empty"

        self.__application_config = self.__load_config__(self.__application_config_filename)
        self.__behaviour_config = self.__load_config__(self.__behaviour_config_filename)

        self.evaluator = EvaluatePrompt(self.__llm_model,  self.__behaviour_config["preprocessing"], self.__application_config)
        self.reset()

    def __load_config__(self, path: str) -> dict | list:
        return yaml.load(open(path, "r"), Loader=yaml.Loader)
    
    def infer(self, prompt:str, mode:str) -> "LLMASP":
        self.__extracted_preds = self.evaluator.run(prompt, mode)
        self.__total_extracted_pred += self.__extracted_preds
        return self
    
    def run_asp(self) -> "LLMASP":
        self.__result_preds = self.extracted_preds
        if self.__application_config['knowledge_base'] and self.__total_extracted_pred != "":
            try:
                self.__result_preds = Model.of_program(self.__application_config['knowledge_base'], self.__total_extracted_pred, sort=False).as_facts
                Statistics.log_solver_call()
            except Exception as e:
                logging.error(e)

        return self
    
    def clean(self):
        self.__result_preds = ""
        self.__extracted_preds = ""
        self.__total_extracted_pred = ""

    def reset(self):
        Statistics.reset()
        ConditionCache.clear()
        PredicateContainer.reset_container()
        self.clean()
    
    @property
    def extracted_preds(self) -> str:
        return self.__extracted_preds
    
    @property
    def total_extracted_pred(self) -> str:
        return self.__total_extracted_pred
    
    @property
    def inferred_preds(self) -> str:
        return self.__result_preds
