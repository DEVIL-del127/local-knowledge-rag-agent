from __future__ import annotations

import ast
import hashlib
import json
import math
import re
from dataclasses import asdict, dataclass
from decimal import Decimal, localcontext
from fractions import Fraction
from typing import Any


class ExpressionError(ValueError):
    """Typed, user-safe expression failure."""

    def __init__(self, code: str, message: str) -> None:
        super().__init__(message)
        self.code = code


@dataclass(frozen=True, slots=True)
class ExpressionPolicy:
    policy_id: str = "expression-policy-v1"
    max_input_bytes: int = 512
    max_tokens: int = 128
    max_parenthesis_depth: int = 16
    max_literal_digits: int = 64
    max_ast_nodes: int = 128
    max_ast_depth: int = 16
    max_operations: int = 64
    max_abs_exponent: int = 1000
    max_intermediate_bits: int = 8192
    max_result_chars: int = 4096

    def digest(self) -> str:
        payload = json.dumps(
            asdict(self), ensure_ascii=False, sort_keys=True, separators=(",", ":")
        ).encode("utf-8")
        return hashlib.sha256(payload).hexdigest()


@dataclass(frozen=True, slots=True)
class ExpressionResult:
    expression: str
    value: str
    policy_digest: str


_TOKEN = re.compile(r"\d+(?:\.\d+)?|\*\*|//|[+\-*/%()]")
_ALLOWED = re.compile(r"^[\d\s.+\-*/%()]+$")
_PREFIX = re.compile(r"^\s*(?:请问|请|帮我)?\s*(?:计算|算一下|算算|求)?\s*", re.I)
_SUFFIX = re.compile(
    r"\s*(?:等于几|等于多少|是多少|结果是什么|结果为多少|的结果)?\s*[？?。.]?\s*$",
    re.I,
)


def extract_expression_candidate(raw: str) -> str | None:
    value = str(raw or "").translate(str.maketrans({"×": "*", "÷": "/", "（": "(", "）": ")"}))
    value = _PREFIX.sub("", value)
    value = _SUFFIX.sub("", value).strip()
    if not value or not _ALLOWED.fullmatch(value) or not re.search(r"\d", value):
        return None
    if not re.search(r"[+\-*/%]", value):
        return None
    return value


class ExpressionService:
    """A bounded arithmetic evaluator that never uses eval/compile."""

    _BINARY = (ast.Add, ast.Sub, ast.Mult, ast.Div, ast.FloorDiv, ast.Mod, ast.Pow)
    _UNARY = (ast.UAdd, ast.USub)

    def __init__(self, policy: ExpressionPolicy | None = None) -> None:
        self.policy = policy or ExpressionPolicy()

    def can_handle(self, raw: str) -> bool:
        return extract_expression_candidate(raw) is not None

    def evaluate(self, raw: str) -> ExpressionResult:
        expression = extract_expression_candidate(raw)
        if expression is None:
            raise ExpressionError("expression_invalid", "没有识别出受支持的算术表达式。")
        self._preflight(expression)
        try:
            tree = ast.parse(expression, mode="eval")
        except (SyntaxError, RecursionError) as exc:
            raise ExpressionError("expression_invalid", "算术表达式格式不正确。") from exc
        nodes = list(ast.walk(tree))
        if len(nodes) > self.policy.max_ast_nodes:
            raise ExpressionError("expression_limit_exceeded", "表达式节点数量超过限制。")
        if self._depth(tree) > self.policy.max_ast_depth:
            raise ExpressionError("expression_limit_exceeded", "表达式结构深度超过限制。")
        value = self._evaluate_tree(tree.body)
        rendered = self._render(value)
        if len(rendered) > self.policy.max_result_chars:
            raise ExpressionError("expression_limit_exceeded", "计算结果长度超过限制。")
        return ExpressionResult(expression, rendered, self.policy.digest())

    def _preflight(self, expression: str) -> None:
        if len(expression.encode("utf-8")) > self.policy.max_input_bytes:
            raise ExpressionError("expression_limit_exceeded", "表达式长度超过限制。")
        compact = re.sub(r"\s+", "", expression)
        tokens = _TOKEN.findall(expression)
        if "".join(tokens) != compact:
            raise ExpressionError("expression_invalid", "表达式包含不受支持的字符。")
        if len(tokens) > self.policy.max_tokens:
            raise ExpressionError("expression_limit_exceeded", "表达式词元数量超过限制。")
        depth = 0
        for token in tokens:
            if token == "(":
                depth += 1
                if depth > self.policy.max_parenthesis_depth:
                    raise ExpressionError("expression_limit_exceeded", "括号层级超过限制。")
            elif token == ")":
                depth -= 1
                if depth < 0:
                    raise ExpressionError("expression_invalid", "括号不匹配。")
            elif token[0].isdigit():
                digits = sum(character.isdigit() for character in token)
                if digits > self.policy.max_literal_digits:
                    raise ExpressionError("expression_limit_exceeded", "数字字面量位数超过限制。")
        if depth:
            raise ExpressionError("expression_invalid", "括号不匹配。")

    @staticmethod
    def _depth(root: ast.AST) -> int:
        maximum = 0
        stack: list[tuple[ast.AST, int]] = [(root, 1)]
        while stack:
            node, depth = stack.pop()
            maximum = max(maximum, depth)
            stack.extend((child, depth + 1) for child in ast.iter_child_nodes(node))
        return maximum

    def _evaluate_tree(self, root: ast.AST) -> Fraction:
        values: dict[int, Fraction] = {}
        stack: list[tuple[ast.AST, bool]] = [(root, False)]
        operations = 0
        while stack:
            node, visited = stack.pop()
            if not visited:
                if isinstance(node, ast.Constant):
                    values[id(node)] = self._constant(node.value)
                    continue
                if isinstance(node, ast.UnaryOp) and isinstance(node.op, self._UNARY):
                    stack.append((node, True))
                    stack.append((node.operand, False))
                    continue
                if isinstance(node, ast.BinOp) and isinstance(node.op, self._BINARY):
                    stack.append((node, True))
                    stack.append((node.right, False))
                    stack.append((node.left, False))
                    continue
                raise ExpressionError("expression_invalid", "表达式包含不受支持的语法。")

            operations += 1
            if operations > self.policy.max_operations:
                raise ExpressionError("expression_limit_exceeded", "表达式运算次数超过限制。")
            if isinstance(node, ast.UnaryOp):
                operand = values[id(node.operand)]
                value = operand if isinstance(node.op, ast.UAdd) else -operand
            else:
                assert isinstance(node, ast.BinOp)
                left, right = values[id(node.left)], values[id(node.right)]
                value = self._binary(node.op, left, right)
            self._check_size(value)
            values[id(node)] = value
        return values[id(root)]

    def _constant(self, value: Any) -> Fraction:
        if isinstance(value, bool) or not isinstance(value, (int, float)):
            raise ExpressionError("expression_invalid", "只允许有限的十进制数字。")
        if isinstance(value, float) and not math.isfinite(value):
            raise ExpressionError("expression_invalid", "只允许有限数值。")
        result = Fraction(value) if isinstance(value, int) else Fraction(str(value))
        self._check_size(result)
        return result

    def _binary(self, operator: ast.operator, left: Fraction, right: Fraction) -> Fraction:
        if isinstance(operator, (ast.Div, ast.FloorDiv, ast.Mod)) and right == 0:
            raise ExpressionError("expression_division_by_zero", "除数不能为零。")
        if isinstance(operator, ast.Add):
            return left + right
        if isinstance(operator, ast.Sub):
            return left - right
        if isinstance(operator, ast.Mult):
            self._precheck_product(left, right)
            return left * right
        if isinstance(operator, ast.Div):
            return left / right
        if isinstance(operator, ast.FloorDiv):
            return Fraction(left // right)
        if isinstance(operator, ast.Mod):
            return left % right
        if isinstance(operator, ast.Pow):
            if right.denominator != 1:
                raise ExpressionError("expression_invalid", "指数必须是整数。")
            exponent = right.numerator
            if abs(exponent) > self.policy.max_abs_exponent:
                raise ExpressionError("expression_limit_exceeded", "指数绝对值超过限制。")
            if exponent < 0 and left == 0:
                raise ExpressionError("expression_division_by_zero", "零不能取负指数。")
            estimate = max(left.numerator.bit_length(), left.denominator.bit_length()) * max(1, abs(exponent))
            if estimate > self.policy.max_intermediate_bits:
                raise ExpressionError("expression_limit_exceeded", "幂运算中间结果超过限制。")
            return left ** exponent
        raise ExpressionError("expression_invalid", "不支持该运算符。")

    def _precheck_product(self, left: Fraction, right: Fraction) -> None:
        estimate = max(
            left.numerator.bit_length() + right.numerator.bit_length(),
            left.denominator.bit_length() + right.denominator.bit_length(),
        )
        if estimate > self.policy.max_intermediate_bits:
            raise ExpressionError("expression_limit_exceeded", "乘法中间结果超过限制。")

    def _check_size(self, value: Fraction) -> None:
        if max(value.numerator.bit_length(), value.denominator.bit_length()) > self.policy.max_intermediate_bits:
            raise ExpressionError("expression_limit_exceeded", "中间结果大小超过限制。")

    def _render(self, value: Fraction) -> str:
        if value.denominator == 1:
            return str(value.numerator)
        denominator = value.denominator
        reduced = denominator
        for factor in (2, 5):
            while reduced % factor == 0:
                reduced //= factor
        if reduced != 1:
            return f"{value.numerator}/{denominator}"
        precision = min(self.policy.max_result_chars, max(64, denominator.bit_length() * 2))
        with localcontext() as context:
            context.prec = precision
            rendered = format(Decimal(value.numerator) / Decimal(denominator), "f")
        if "." in rendered:
            rendered = rendered.rstrip("0").rstrip(".")
        return rendered
