"""Safe arithmetic expression evaluation."""

from __future__ import annotations

import ast
import operator
from typing import Any

from minicpm_container.tools.limits import MAX_EXPRESSION_CHARS

_OPERATORS: dict[type[ast.AST], Any] = {
    ast.Add: operator.add,
    ast.Sub: operator.sub,
    ast.Mult: operator.mul,
    ast.Div: operator.truediv,
    ast.FloorDiv: operator.floordiv,
    ast.Mod: operator.mod,
    ast.Pow: operator.pow,
    ast.USub: operator.neg,
    ast.UAdd: operator.pos,
}


class CalculateError(ValueError):
    """Raised when an expression is invalid or unsafe."""


def _eval_node(node: ast.AST) -> float | int:
    if isinstance(node, ast.Expression):
        return _eval_node(node.body)
    if isinstance(node, ast.Constant):
        if isinstance(node.value, bool):
            raise CalculateError("boolean literals are not allowed")
        if isinstance(node.value, int | float):
            return node.value
        raise CalculateError("only numeric literals are allowed")
    if isinstance(node, ast.UnaryOp):
        operator_fn = _OPERATORS.get(type(node.op))
        if operator_fn is None:
            raise CalculateError("unsupported unary operator")
        return operator_fn(_eval_node(node.operand))
    if isinstance(node, ast.BinOp):
        operator_fn = _OPERATORS.get(type(node.op))
        if operator_fn is None:
            raise CalculateError("unsupported binary operator")
        left = _eval_node(node.left)
        right = _eval_node(node.right)
        return operator_fn(left, right)
    raise CalculateError("unsupported expression")


def safe_calculate(expression: str) -> str:
    text = expression.strip()
    if not text:
        raise CalculateError("expression must not be empty")
    if len(text) > MAX_EXPRESSION_CHARS:
        raise CalculateError(f"expression exceeds {MAX_EXPRESSION_CHARS} characters")

    try:
        tree = ast.parse(text, mode="eval")
    except SyntaxError as exc:
        raise CalculateError("invalid expression syntax") from exc

    allowed_nodes = (
        ast.Expression,
        ast.Constant,
        ast.BinOp,
        ast.UnaryOp,
        ast.Add,
        ast.Sub,
        ast.Mult,
        ast.Div,
        ast.FloorDiv,
        ast.Mod,
        ast.Pow,
        ast.USub,
        ast.UAdd,
    )
    for node in ast.walk(tree):
        if not isinstance(node, allowed_nodes):
            raise CalculateError("expression contains disallowed syntax")

    result = _eval_node(tree)
    if isinstance(result, float) and result.is_integer():
        return str(int(result))
    return str(result)
