from pydantic import BaseModel, Field
from typing import List, Optional, Union
import json

class JSONSchemaBuilder:

    def __init__(self):
        self.__classes = []

    def generate(self, predicate: str, enable_types: bool = False) -> type:
        data = predicate.split("(")
        class_name = data[0]
        
        if "(" in predicate:
            args_part = data[1].replace("\"", "").replace(")", "").replace(" ", "")
            terms = args_part.split(",") if ',' in args_part else [args_part]
        else:
            terms = [] # For predicates like type(layered) or just "type"

        class_dict = {}
        annotations = {}

        for term in terms:
            class_dict[term] = Field(default=None)
            annotations[term] = Union[int, str, None]

        def str_method(self):
            values = [v for v in self.dict().values() if v is not None]
            if not values and terms:
                return ""
            args_str = ", ".join(str(v) for v in values)
            return f"{class_name}({args_str}).".lower() if values else f"{class_name}.".lower()

        class_dict['__annotations__'] = annotations
        class_dict['__name__'] = class_name
        class_dict['__str__'] = str_method
        
        new_class = type(class_name, (BaseModel,), class_dict)

        def list_str_method(self):
            atom_list = getattr(self, f"list_{class_name}", [])
            return " ".join([str(atom) for atom in atom_list if atom])

        example_obj = {term: "any" for term in terms} if terms else "null"

        return type(
            f"list_{class_name}",
            (BaseModel,),
            {
                "__name__": f"{class_name}_list",
                "__annotations__": {f"list_{class_name}": List[Optional[new_class]]},
                "__class_params__": terms,
                "__str__": list_str_method,
                "__info__": {f"list_{class_name}": [example_obj] if terms else []}
            },
        )

    def generate_single_grammar(self, predicates: list, enable_types: bool = False) -> type:
        self.__classes = []

        for predicate in predicates:
            self.__classes.append(self.generate(predicate))

        class_dict = {}
        annotations = {}
        

        for cls in self.__classes:
            field_name = f"wrapper_{cls.__name__}"
            class_dict[field_name] = Field(default=None)
            annotations[field_name] = Optional[cls]

        class_dict['__annotations__'] = annotations
        
        def str_method(self):
            asp_facts = []
            for _, wrapper_obj in self:
                if wrapper_obj is not None:
                    fact_str = str(wrapper_obj)
                    if fact_str:
                        asp_facts.append(fact_str)
            return "\n".join(asp_facts)
        
        class_dict['__str__'] = str_method

        schema_map = {}
        for list_cls in self.__classes:
            schema_map.update(list_cls.__info__)
        
        class_info = json.dumps(schema_map, indent=2)

        class_dict['__info__'] = class_info
            
        _type = type(
            "predicate_wrapper",
            (BaseModel,),
            class_dict
        )

        return _type

    def get_classes(self) -> tuple:
        return (self.__classes, None)

