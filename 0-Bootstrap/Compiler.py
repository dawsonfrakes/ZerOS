import collections.abc
from dataclasses import dataclass
from enum import IntEnum

class Parse_Error(Exception):
  def __init__(self, message: str, location: int) -> None:
    super().__init__(message)
    self.location = location

class Code_Kind(IntEnum):
  IDENTIFIER = 0
  NUMBER = 1
  STRING = 2
  TUPLE = 3

@dataclass
class Code:
  location: int
  kind: Code_Kind
  data: str | list["Code"]
  @property
  def as_atom(self) -> str: assert isinstance(self.data, str); return self.data
  @property
  def as_tuple(self) -> list["Code"]: assert isinstance(self.data, list); return self.data

def isbasedigit(s: str, base: int) -> bool:
  if base == 2: return s >= "0" and s <= "1"
  if base == 8: return s >= "0" and s <= "7"
  if base == 10: return s >= "0" and s <= "9"
  if base == 16: return s >= "0" and s <= "9" or s >= "A" and s <= "F" or s >= "a" and s <= "f"
  raise NotImplementedError(base)

def parse_code(s: str, p: int) -> tuple[Code | None, int]:
  initial_p = p
  level = 0
  codes: list[Code] = []
  while True:
    while True:
      while p < len(s) and s[p].isspace(): p += 1
      if p < len(s) and s[p] == ";":
        while p < len(s) and s[p] != "\n": p += 1
        continue
      break
    if p >= len(s): break
    start = p
    if s[p] == "(":
      p += 1
      level += 1
      codes.append(Code(start, Code_Kind.TUPLE, []))
    elif s[p] == ")":
      p += 1
      if level == 0: raise Parse_Error("You have an extraneous closing parenthesis.", start)
      level -= 1
      if len(codes) > 1: popped = codes.pop(); codes[-1].as_tuple.append(popped)
    elif s[p] == "'":
      p += 1
      code, next_pos = parse_code(s, p)
      if code is None: raise Parse_Error("You tried to take the $code of nothing.", start)
      p = next_pos
      (codes[-1].as_tuple if len(codes) > 0 else codes).append(Code(start, Code_Kind.TUPLE, [Code(start, Code_Kind.IDENTIFIER, "$code"), code]))
    elif s[p] == ",":
      p += 1
      code, next_pos = parse_code(s, p)
      if code is None: raise Parse_Error("You tried to $insert nothing.", start)
      p = next_pos
      (codes[-1].as_tuple if len(codes) > 0 else codes).append(Code(start, Code_Kind.TUPLE, [Code(start, Code_Kind.IDENTIFIER, "$insert"), code]))
    elif s[p] == "\"":
      p += 1
      while p < len(s) and (s[p - 1] == "\\" or s[p] != "\""): p += 1
      if p >= len(s) or s[p] != "\"": raise Parse_Error("You have an unterminated string literal.", start)
      p += 1
      (codes[-1].as_tuple if len(codes) > 0 else codes).append(Code(start, Code_Kind.STRING, s[start:p]))
    elif s[p].isdigit() or (p + 1 < len(s) and s[p] in "+-" and s[p + 1].isdigit()):
      base = 10
      if s[p] in "+-": p += 1
      if p + 1 < len(s) and s[p] == "0" and s[p + 1] in "box":
        p += 1
        if s[p] == "b": base = 2
        elif s[p] == "o": base = 8
        elif s[p] == "x": base = 16
        else: raise NotImplementedError()
        p += 1
      if p >= len(s) or not isbasedigit(s[p], base): raise Parse_Error(f"You have an invalid number in your numeric literal of base {base}.", start)
      while p < len(s) and isbasedigit(s[p], base): p += 1
      if p < len(s) and s[p] == ".":
        if base != 10: raise Parse_Error("A floating point literal can not contain an integer base prefix.", start)
        p += 1
        while p < len(s) and s[p].isdigit(): p += 1
      if p < len(s) and s[p].lower() == "e":
        if base != 10: raise Parse_Error("A floating point literal can not contain an integer base prefix.", start)
        p += 1
        if p < len(s) and s[p] in "+-": p += 1
        if p < len(s) and not s[p].isdigit(): raise Parse_Error("You are missing a digit after your floating point literal's exponentiation demarcator.", p - 1)
        while p < len(s) and s[p].isdigit(): p += 1
      (codes[-1].as_tuple if len(codes) > 0 else codes).append(Code(start, Code_Kind.NUMBER, s[start:p]))
    else:
      while p < len(s) and not s[p].isspace() and s[p] not in "();": p += 1
      (codes[-1].as_tuple if len(codes) > 0 else codes).append(Code(start, Code_Kind.IDENTIFIER, s[start:p]))
    if p < len(s) and s[p - 1] != "(" and not s[p].isspace() and s[p] not in ");":
      raise Parse_Error("You have two conjoined expressions without whitespace separating them.", p)
    if level == 0: break
  if level != 0: raise Parse_Error("You are missing a closing parenthesis.", initial_p)
  assert len(codes) <= 1
  return codes.pop() if len(codes) > 0 else None, p

def code_as_string(code: Code) -> str:
  if code.kind != Code_Kind.TUPLE: return code.as_atom
  return "(" + " ".join(map(code_as_string, code.as_tuple)) + ")"

class Type_Kind(IntEnum):
  TYPE = 0
  CODE = 1
  ANY = 2
  VOID = 5
  UNTYPED_INTEGER = 8
  UNTYPED_FLOAT = 9
  INTEGER = 10
  FLOAT = 11
  PROCEDURE = 20

@dataclass(eq=True)
class Type:
  kind: Type_Kind

@dataclass(eq=True)
class Type_Procedure(Type):
  return_type: Type
  parameter_types: tuple[Type, ...]
  varargs_type: Type | None
  is_macro: bool
  is_c_varargs: bool

type_type = Type(Type_Kind.TYPE)
type_code = Type(Type_Kind.CODE)
type_any = Type(Type_Kind.ANY)
type_void = Type(Type_Kind.VOID)
type_untyped_integer = Type(Type_Kind.UNTYPED_INTEGER)
type_untyped_float = Type(Type_Kind.UNTYPED_FLOAT)

def type_as_string(type: Type) -> str:
  if type == type_type: return "($type 'TYPE)"
  if type == type_code: return "($type 'CODE)"
  if type == type_any: return "($type 'ANY)"
  if type == type_void: return "($type 'VOID)"
  if type == type_untyped_integer: return "($type 'UNTYPED_INTEGER)"
  if type == type_untyped_float: return "($type 'UNTYPED_FLOAT)"
  raise NotImplementedError(type.kind.name)

@dataclass
class Procedure:
  parameter_names: tuple[str, ...]
  body: collections.abc.Callable[..., "Value"]

@dataclass
class Value:
  type: Type
  data: Type | Code | int | float | Procedure | None
  @property
  def as_type(self) -> Type: assert isinstance(self.data, Type); return self.data
  @property
  def as_code(self) -> Code: assert isinstance(self.data, Code); return self.data
  @property
  def as_integer(self) -> int: assert isinstance(self.data, int); return self.data
  @property
  def as_float(self) -> float: assert isinstance(self.data, float); return self.data
  @property
  def as_procedure(self) -> Procedure: assert isinstance(self.data, Procedure); return self.data

value_void = Value(type_void, None)

@dataclass
class Env_Entry:
  value: Value

@dataclass
class Env:
  parent: "Env | None"
  table: dict[str, Env_Entry]

  def find(self, key: str) -> Env_Entry | None:
    if key in self.table: return self.table[key]
    if self.parent is not None: return self.parent.find(key)
    return None

class Evaluation_Error(Exception):
  def __init__(self, message: str, code: Code) -> None:
    super().__init__(message)
    self.code = code

def coerce(value: Value, to: Type, nearest_code: Code) -> Value:
  if value.type == to: return value
  if to == type_any: return value
  raise Evaluation_Error(f"I failed to coerce `{value_as_string(value)}` to `{type_as_string(to)}`.", nearest_code)

def evaluate_code(code: Code, env: Env) -> Value:
  if code.kind == Code_Kind.IDENTIFIER:
    entry = env.find(code.as_atom)
    if entry is None: raise Evaluation_Error(f"I failed to find `{code.as_atom}` in the environment.", code)
    return entry.value
  if code.kind == Code_Kind.NUMBER:
    try: return Value(type_untyped_integer, int(code.as_atom, base=0))
    except ValueError: return Value(type_untyped_float, float(code.as_atom))
  if code.kind == Code_Kind.TUPLE:
    if len(code.as_tuple) == 0: raise Evaluation_Error("You tried to call a procedure or macro but didn't specify its name.", code)
    op_code, *arg_codes = code.as_tuple
    op = evaluate_code(op_code, env)
    if op.type.kind != Type_Kind.PROCEDURE: raise Evaluation_Error(f"You tried to call `{code_as_string(op_code)}` => `{value_as_string(op)}`, but it is not a procedure or macro.", op_code)
    assert isinstance(op.type, Type_Procedure)
    if len(arg_codes) != len(op.type.parameter_types): raise Evaluation_Error(f"You tried to call `{code_as_string(op_code)}` => `{value_as_string(op)}` with {len(arg_codes)} argument{"s" if len(arg_codes) != 1 else ""}, but it expects {len(op.type.parameter_types)} argument{"s" if len(op.type.parameter_types) != 1 else ""}.", op_code)
    args = [coerce(evaluate_code(arg_code, env), op.type.parameter_types[i], arg_code) if not op.type.is_macro or op.type.parameter_types[i] != type_code else Value(type_code, arg_code) for i, arg_code in enumerate(arg_codes)]
    result = op.as_procedure.body(*args, calling_env=env, nearest_code=op_code)
    if op.type.is_macro and op.type.return_type == type_code and op != default_env.table["$code"].value:
      result = evaluate_code(result.as_code, env)
    return coerce(result, op.type.return_type, op_code)
  raise NotImplementedError(code.kind)

def value_as_string(value: Value) -> str:
  if value.type == type_type: return type_as_string(value.as_type)
  if value.type == type_code: return code_as_string(value.as_code)
  if value.type == type_any: return "($cast ($type 'ANY) 0)"
  if value.type == type_void: return "($cast ($type 'VOID) 0)"
  if value.type == type_untyped_integer: return str(value.as_integer)
  if value.type == type_untyped_float: return str(value.as_float)
  if value.type.kind == Type_Kind.PROCEDURE:
    return "($proc (...) ... ...)"
  raise NotImplementedError(value.type.kind.name)

def builtin_code(code_value: Value, **_: None) -> Value:
  return code_value

def builtin_insert(code_value: Value, **kwargs: Env) -> Value:
  return evaluate_code(code_value.as_code, kwargs["calling_env"])

def builtin_type(kind_value: Value, details_value: Value, **_: None) -> Value:
  if kind_value.as_code.kind != Code_Kind.IDENTIFIER: raise Evaluation_Error("$type expects argument `kind` to be a Code_Kind.IDENTIFIER.", kind_value.as_code)
  kind = kind_value.as_code.as_atom
  if kind == "TYPE": return Value(type_type, type_type)
  if kind == "CODE": return Value(type_type, type_code)
  if kind == "ANY": return Value(type_type, type_any)
  if kind == "VOID": return Value(type_type, type_void)
  raise Evaluation_Error(f"$type does not know how to handle type kind `{kind}`.", kind_value.as_code)

def builtin_cast(type_value: Value, value: Value, **kwargs: Code) -> Value:
  try: return coerce(value, type_value.as_type, kwargs["nearest_code"])
  except:
    raise Evaluation_Error(f"I failed to cast `{value_as_string(value)}` to `{type_as_string(type_value.as_type)}`", kwargs["nearest_code"])

default_env = Env(None, {
  "$code": Env_Entry(Value(Type_Procedure(Type_Kind.PROCEDURE, type_code, (type_code,), None, True, False), Procedure(("code",), builtin_code))),
  "$insert": Env_Entry(Value(Type_Procedure(Type_Kind.PROCEDURE, type_any, (type_code,), None, False, False), Procedure(("code",), builtin_insert))),
  "$type": Env_Entry(Value(Type_Procedure(Type_Kind.PROCEDURE, type_type, (type_code, type_code), None, False, False), Procedure(("kind", "details"), builtin_type))),
  "$cast": Env_Entry(Value(Type_Procedure(Type_Kind.PROCEDURE, type_any, (type_type, type_any), None, False, False), Procedure(("type", "value"), builtin_cast))),
})

def repl() -> None:
  file = "repl"
  while True:
    try: src = input("> ")
    except (KeyboardInterrupt, EOFError): print(""); break
    pos = 0
    env = Env(default_env, {})
    while True:
      try: code, pos = parse_code(src, pos)
      except Parse_Error as e: print(f"{file}[{e.location}] {e}"); break
      if code is None: break
      # print(code_as_string(code)) # NOTE(dfra): for debugging parser.
      try: result = evaluate_code(code, env)
      except Evaluation_Error as e: print(f"{file}[{e.code.location}] {e}"); break
      if result != value_void: print(value_as_string(result))

def compile(file: str) -> None:
  try:
    with open(file) as f: src = f.read()
  except FileNotFoundError: print(f"I failed to read `{file}` from your drive. Maybe you need to quote the entire path?"); exit(1)
  pos = 0
  env = Env(default_env, {})
  while True:
    try: code, pos = parse_code(src, pos)
    except Parse_Error as e: print(f"{file}[{e.location}] {e}"); exit(1)
    if code is None: break
    try: evaluate_code(code, env)
    except Evaluation_Error as e: print(f"{file}[{e.code.location}] {e}"); exit(1)

if __name__ == "__main__":
  import sys
  if len(sys.argv) <= 1: repl()
  else: compile(sys.argv[1])
