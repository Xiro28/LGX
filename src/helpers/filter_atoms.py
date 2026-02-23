import re

from src.core.atom import atom
from src.core.atom_list import atomList


_LEADING_ZERO_INT = re.compile(r"\b0+(\d+)\b")
def _strip_leading_zeros_in_int_tokens(s: str) -> str:
    def repl(m: re.Match) -> str:
        return str(int(m.group(1)))
    return _LEADING_ZERO_INT.sub(r"\1", s)


_BAD_START = re.compile(r"\b(\d+[A-Za-z_][A-Za-z0-9_]*)\b")
def prefix_fix(s: str) -> str:
    return _BAD_START.sub(r"malformed_term_failure__\1", s)

def filter_asp_atoms(atoms: str) -> atomList:
    req = _strip_leading_zeros_in_int_tokens(atoms)
    req = prefix_fix(req)

    atom_list = list()
    for atom_str in re.findall(r"\w+(?:\([a-zA-Z0-9_]+(?:,\s*[a-zA-Z0-9_]+)*\))?\.", req):
        atom_list.append(atom(atom_str=atom_str))

    return atomList(atoms=atom_list)