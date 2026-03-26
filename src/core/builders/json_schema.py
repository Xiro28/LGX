from pydantic import BaseModel, Field
from typing import List, Optional, Union, get_origin, get_args
import json

class JSONSchemaBuilder:

    def __init__(self):
        self.__classes = []

    def generate(self, predicate: str, enable_types: bool = False) -> type:
        print(f"Generating class for predicate: {predicate}")

        data = predicate.split("(")
        class_name = data[0]
        
        # 1. Parse terms/arguments safely
        if "(" in predicate:
            args_part = data[1].replace("\"", "").replace(")", "").replace(" ", "")
            terms = args_part.split(",") if ',' in args_part else [args_part]
        else:
            terms = [] # For predicates like type(layered) or just "type"

        class_dict = {}
        annotations = {}

        # 2. Build Pydantic fields and annotations
        for term in terms:
            class_dict[term] = Field(default=None) # Default None handles missing values
            annotations[term] = Union[int, str, None]

        # 3. Method to convert object to ASP fact string: predicate(val1, val2).
        def str_method(self):
            values = [v for v in self.dict().values() if v is not None]
            if not values and terms: # Avoid printing incomplete atoms
                return ""
            args_str = ", ".join(str(v) for v in values)
            return f"{class_name}({args_str}).".lower() if values else f"{class_name}.".lower()

        class_dict['__annotations__'] = annotations
        class_dict['__name__'] = class_name
        class_dict['__str__'] = str_method
        
        # Create the 'Atom' class (e.g., in_layer)
        new_class = type(class_name, (BaseModel,), class_dict)

        # 4. Method for the List wrapper to join all facts with spaces
        def list_str_method(self):
            atom_list = getattr(self, f"list_{class_name}", [])
            return " ".join([str(atom) for atom in atom_list if atom])

        # 5. THE KEY CHANGE: Define the schema entry for this predicate
        # This creates the {"layer": "any", "node": "any"} structure
        example_obj = {term: "any" for term in terms} if terms else "null"

        return type(
            f"list_{class_name}",
            (BaseModel,),
            {
                "__name__": f"{class_name}_list",
                "__annotations__": {f"list_{class_name}": List[Optional[new_class]]},
                "__class_params__": terms,
                "__str__": list_str_method,
                # Store the info as a raw dictionary (generate_single_grammar will JSONify it later)
                "__info__": {f"list_{class_name}": [example_obj] if terms else []}
            },
        )

    def generate_single_grammar(self, predicates: list, enable_types: bool = False) -> type:
        self.__classes = []

        # 1. Generate the individual atom/list classes
        for predicate in predicates:
            self.__classes.append(self.generate(predicate))

        class_dict = {}
        annotations = {}
        
        # 2. Build the wrapper fields (e.g., wrapper_in_layer_list)
        for cls in self.__classes:
            # cls.__name__ is e.g., "in_layer_list"
            field_name = f"wrapper_{cls.__name__}"
            class_dict[field_name] = Field(default=None)
            annotations[field_name] = Optional[cls]

        class_dict['__annotations__'] = annotations
        
        # 3. ASP Fact string representation logic
        def str_method(self):
            asp_facts = []
            for _, wrapper_obj in self:
                if wrapper_obj is not None:
                    # str(wrapper_obj) calls list_str_method defined in generate()
                    fact_str = str(wrapper_obj)
                    if fact_str:
                        asp_facts.append(fact_str)
            return "\n".join(asp_facts)
        
        class_dict['__str__'] = str_method

        # 4. THE FIX: Build the JSON Schema for the LLM Prompt
        # This creates a single dictionary containing all predicate structures
        schema_map = {}
        for list_cls in self.__classes:
            # list_cls.__info__ is a dict: {"list_name": [{"arg": "any"}]}
            schema_map.update(list_cls.__info__)
        
        # json.dumps ensures double quotes and valid JSON syntax
        # Using indent=2 makes the prompt much easier for the AI to read
        class_info = json.dumps(schema_map, indent=2)

        class_dict['__info__'] = class_info
            
        # Create the final root wrapper class
        _type = type(
            "predicate_wrapper",
            (BaseModel,),
            class_dict
        )

        return _type

    def get_classes(self) -> tuple:
        return (self.__classes, None)

