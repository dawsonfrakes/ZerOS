import Compiler
import operator
import typing

class BinaryWriter:
  class Label(str): pass
  Expr = int | float | Label | typing.Callable[[], int] | bytes | str

  def __init__(self) -> None:
    self.most_recent_label: BinaryWriter.Label | None = None
    self.origin = 0
    self.cursor = 0
    self.labels: dict[BinaryWriter.Label, int] = {}
    self.fixups: list[tuple[BinaryWriter.Label | typing.Callable[[], int], int, int, bool]] = []
    self.output = bytearray()
    self.ord = ord
    setattr(self, ":", self.label)
    setattr(self, "+", operator.add)
    setattr(self, "-", operator.sub)
    setattr(self, "*", operator.mul)
    setattr(self, "/", operator.truediv)
    setattr(self, "&", operator.and_)
    setattr(self, "|", operator.or_)
    setattr(self, "<<", operator.lshift)
    setattr(self, ">>", operator.rshift)

  def fixup(self) -> None:
    for expr, offset, size, signed in self.fixups:
      self.output[offset:offset + size] = (expr() if callable(expr) else self.labels[expr]).to_bytes(size, byteorder="little", signed=signed)

  def label(self, s: str) -> None: self.labels[self.Label(s)] = self.cursor
  def org(self, to: int) -> None: self.origin, self.cursor = to, to
  def emit(self, size: int, *exprs: Expr, signed: bool = False) -> None:
    for expr in exprs:
      if isinstance(expr, int): self.output += expr.to_bytes(size, byteorder="little", signed=signed); self.cursor += size
      elif isinstance(expr, BinaryWriter.Label) or callable(expr): self.fixups.append((expr, len(self.output), size, signed)); self.output += b'\xAA' * size; self.cursor += size
      elif isinstance(expr, bytes): self.output += b''.join(c.to_bytes(size, byteorder="little", signed=signed) for c in expr); self.cursor += len(expr) * size
      else: assert isinstance(expr, str); self.output += b''.join(ord(c).to_bytes(size, byteorder="little", signed=signed) for c in expr); self.cursor += len(expr) * size
  def db(self, *exprs: Expr, signed: bool = False) -> None: self.emit(1, *exprs, signed=signed)
  def dw(self, *exprs: Expr, signed: bool = False) -> None: self.emit(2, *exprs, signed=signed)
  def dd(self, *exprs: Expr, signed: bool = False) -> None: self.emit(4, *exprs, signed=signed)
  def dq(self, *exprs: Expr, signed: bool = False) -> None: self.emit(8, *exprs, signed=signed)

class X86Writer(BinaryWriter):
  class WRegister(str): pass
  ax = WRegister("ax")
  bx = WRegister("bx")
  cx = WRegister("cx")
  dx = WRegister("dx")
  Expr = BinaryWriter.Expr | WRegister
  wregs = [ax, cx, dx, bx]
  def int3(self) -> None: self.db(0xCC)
  def int(self, n: int) -> None: self.db(0xCD, n)
  def mov_rm16_r16(self, lhs: WRegister, rhs: WRegister) -> None: self.db(0x89, 0b11 << 6 | self.wregs.index(lhs) << 3 | self.wregs.index(rhs))
  def mov(self, lhs: WRegister, rhs: "WRegister | int") -> None:
    if isinstance(rhs, X86Writer.WRegister): self.mov_rm16_r16(lhs, rhs)
    else: self.db(0xB8 + self.wregs.index(lhs)); self.dw(rhs)
  def jmp_rel8(self, expr: "int | BinaryWriter.Label") -> None: base = self.cursor + 2; self.db(0xEB); self.db(expr - base if isinstance(expr, int) else lambda: self.labels[expr] - base, signed=True)
  def jmp(self, expr: "int") -> None:
    self.jmp_rel8(expr)
  def hlt(self) -> None: self.db(0xF4)
  def cli(self) -> None: self.db(0xFA)

def evaluate_assembly(b: BinaryWriter, code: Compiler.Code) -> BinaryWriter.Expr | None:
  if code.kind == Compiler.Code_Kind.IDENTIFIER:
    if hasattr(b, code.as_atom): return getattr(b, code.as_atom)
    elif code.as_atom == "$$": return b.origin
    elif code.as_atom == "$": return b.cursor
    else:
      s = code.as_atom
      assert s[0] != "." or b.most_recent_label is not None
      l = b.Label(b.most_recent_label + s if s[0] == "." else s) #type:ignore
      if s[0] != ".": b.most_recent_label = l
      return l
  if code.kind == Compiler.Code_Kind.NUMBER:
    try: return int(code.as_atom, base=0)
    except ValueError: return float(code.as_atom)
  if code.kind == Compiler.Code_Kind.STRING:
    return code.as_atom[1:-1]
  if code.kind == Compiler.Code_Kind.TUPLE:
    if len(code.as_tuple) == 0: return None
    op_code, *arg_codes = code.as_tuple
    op = evaluate_assembly(b, op_code)
    assert callable(op), f"Operator `{Compiler.code_as_string(op_code)}` not found."
    args: list[BinaryWriter.Expr] = []
    for arg_code in arg_codes:
      if arg_code.kind == Compiler.Code_Kind.TUPLE and arg_code.as_tuple[0].kind == Compiler.Code_Kind.IDENTIFIER and arg_code.as_tuple[0].as_atom == "dup":
        count = evaluate_assembly(b, arg_code.as_tuple[1])
        assert isinstance(count, int)
        expr = evaluate_assembly(b, arg_code.as_tuple[2])
        assert isinstance(expr, int)
        args.extend([expr] * count)
      else:
        expr = evaluate_assembly(b, arg_code)
        assert isinstance(expr, int | X86Writer.WRegister | str), type(expr)
        args.append(expr)
    return op(*args)
  raise NotImplementedError(code.kind.name)

if __name__ == "__main__":
  import sys
  from pathlib import Path
  with open(sys.argv[1]) as f: src = f.read()
  b = X86Writer()
  pos = 0
  while True:
    code, next_pos = Compiler.parse_code(src, pos)
    if code is None: break
    pos = next_pos
    evaluate_assembly(b, code)
  b.fixup()
  with open(Path(sys.argv[1]).with_suffix(".bin"), "wb") as f: f.write(b.output)
