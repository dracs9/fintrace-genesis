"""Safe Decimal calculator (SPEC §8.3).

Only ``Expression``, ``BinOp(Add|Sub|Mult|Div)``, ``UnaryOp(UAdd|USub)`` and numeric
``Constant`` (int / float literal → ``Decimal(str(value))``) are accepted. Nothing is ever
evaluated by Python itself. Limits: length ≤ 200, depth ≤ 20, every literal, intermediate
and the result satisfy ``|x| ≤ 1e18``; precision 28 in a local context.
Any violation → ``EnvError("CALC_ERROR")``. Pure and deterministic.
"""

from __future__ import annotations

import ast
from decimal import (
    Context, Decimal, DivisionByZero, InvalidOperation, Overflow, localcontext,
)

from .errors import EnvError

MAX_LENGTH = 200
MAX_DEPTH = 20
MAX_ABS = Decimal("1e18")
PRECISION = 28

_BINOPS = (ast.Add, ast.Sub, ast.Mult, ast.Div)
_UNARY = (ast.UAdd, ast.USub)


def _bounded(value: Decimal) -> Decimal:
    if not value.is_finite() or abs(value) > MAX_ABS:
        raise EnvError("CALC_ERROR")
    return value


def _eval(node: ast.AST, depth: int) -> Decimal:
    if depth > MAX_DEPTH:
        raise EnvError("CALC_ERROR")
    if isinstance(node, ast.Constant):
        value = node.value
        if type(value) not in (int, float):  # excludes bool, str, bytes, complex, None, Ellipsis
            raise EnvError("CALC_ERROR")
        return _bounded(Decimal(str(value)))
    if isinstance(node, ast.UnaryOp) and isinstance(node.op, _UNARY):
        operand = _eval(node.operand, depth + 1)
        return _bounded(-operand if isinstance(node.op, ast.USub) else +operand)
    if isinstance(node, ast.BinOp) and isinstance(node.op, _BINOPS):
        left = _eval(node.left, depth + 1)
        right = _eval(node.right, depth + 1)
        if isinstance(node.op, ast.Add):
            return _bounded(left + right)
        if isinstance(node.op, ast.Sub):
            return _bounded(left - right)
        if isinstance(node.op, ast.Mult):
            return _bounded(left * right)
        if right == 0:
            raise EnvError("CALC_ERROR")
        return _bounded(left / right)
    raise EnvError("CALC_ERROR")


def _format(value: Decimal) -> str:
    if value == 0:
        return "0"
    return format(value.normalize(), "f")


def evaluate(expression: object) -> str:
    """Evaluate ``expression`` and return the result as a plain Decimal string (no exponent)."""
    if not isinstance(expression, str) or len(expression) > MAX_LENGTH:
        raise EnvError("CALC_ERROR")
    try:
        tree = ast.parse(expression.strip(), mode="eval")
    except (SyntaxError, ValueError, RecursionError, MemoryError):
        raise EnvError("CALC_ERROR") from None
    if not isinstance(tree, ast.Expression):
        raise EnvError("CALC_ERROR")
    context = Context(prec=PRECISION, traps=[InvalidOperation, DivisionByZero, Overflow])
    try:
        with localcontext(context):
            return _format(_eval(tree.body, 1))
    except EnvError:
        raise
    except (ArithmeticError, ValueError, RecursionError):
        raise EnvError("CALC_ERROR") from None
