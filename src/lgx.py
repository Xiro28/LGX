from typeguard     import typechecked
from dataclasses   import dataclass, field

from src.core.llm_handler import llmHandler
from src.core.atom_list import atomList
from src.core.knowledge_base import knowledgeBase
from src.core.yaml_parser import applicationParser, behaviourParser

@typechecked
@dataclass(frozen=True)
class lgx:
    llm_instance: llmHandler

    @classmethod
    def create(cls, 
               llm_model: str = "", 
               behaviour_filename: str = "", 
               application_filename: str = "") -> "lgx":

        app_cfg = applicationParser.from_yaml(application_filename)
        beh_cfg = behaviourParser.from_yaml(behaviour_filename)

        llm_inst = llmHandler.create(llm_model, app_cfg, beh_cfg)
        
        return cls(llm_instance=llm_inst)    

    def infer(self, prompt:str) -> "lgx":
        self.llm_instance.run(prompt)
        return self
    
    def execute_knowledge_base(self, kb: knowledgeBase) -> atomList:
        return kb.execute(self.llm_instance.get_extracted_atoms())
    
    def get_extracted_atoms(self) -> atomList:
        return self.llm_instance.get_extracted_atoms()
    
    def explain(self) -> str:
        raise NotImplementedError("NOT IMPLEMENTED YET")
    
    def cleanup(self):
        self.llm_instance.cleanup()
