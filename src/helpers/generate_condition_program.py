from random import randint


def generate_min_program(condition: str | list[str]) -> str:
    if not condition:
        return ""

    uuid8 = f"uuid{randint(1000, 9999)}"

    if isinstance(condition, list):
        parts = [f"{uuid8}_{i} :- {c}.\n#show {uuid8}_{i}/0." for i, c in enumerate(condition)]
        return "\n".join(parts)
    
    return f"{uuid8} :- {condition.strip().rstrip('.')}.\n#show {uuid8}/0."