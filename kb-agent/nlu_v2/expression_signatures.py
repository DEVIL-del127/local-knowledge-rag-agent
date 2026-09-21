"""Shared finite expression signatures for semantic compilation."""
from __future__ import annotations

from dataclasses import dataclass


NUMERIC_TYPES = frozenset({"number", "integer", "long", "float", "double"})


@dataclass(frozen=True, slots=True)
class ValueSignature:
    result_type: str
    unit: str | None = None
    shape: str = "scalar"
    semantic_role: str = "measure"


@dataclass(frozen=True, slots=True)
class ExpressionSignature:
    function_id: str
    arity: int
    accepted_input_types: tuple[str, ...]
    accepted_shapes: tuple[str, ...]
    unit_relation: str
    result_type: str
    result_shape: str
    allowed_semantic_roles: tuple[str, ...]


@dataclass(frozen=True, slots=True)
class SignatureResult:
    valid: bool
    result_type: str = "unknown"
    unit: str | None = None
    shape: str = "scalar"
    diagnostic_code: str = ""
    message: str = ""


class ExpressionSignatureRegistry:
    """Validate arity, type, shape, unit, and semantic input roles."""

    VERSION = "m14-signatures-v1"

    def __init__(self) -> None:
        numeric = tuple(sorted(NUMERIC_TYPES))
        scalar_or_relation = ("scalar", "relation")
        self._signatures = {
            "identity": ExpressionSignature(
                "identity", 1, ("any",), scalar_or_relation,
                "preserve", "input", "input", ("measure", "operand"),
            ),
            "avg": ExpressionSignature(
                "avg", 1, numeric, scalar_or_relation,
                "preserve", "number", "scalar", ("measure",),
            ),
            "sum": ExpressionSignature(
                "sum", 1, numeric, scalar_or_relation,
                "preserve", "number", "scalar", ("measure",),
            ),
            "count": ExpressionSignature(
                "count", 1, ("any",), ("scalar", "relation", "event_set"),
                "count", "number", "scalar", ("measure", "row_set", "event_set"),
            ),
            "min": ExpressionSignature(
                "min", 1, numeric, scalar_or_relation,
                "preserve", "number", "scalar", ("measure",),
            ),
            "max": ExpressionSignature(
                "max", 1, numeric, scalar_or_relation,
                "preserve", "number", "scalar", ("measure",),
            ),
            "stddev": ExpressionSignature(
                "stddev", 1, numeric, scalar_or_relation,
                "preserve", "number", "scalar", ("measure",),
            ),
            "ratio": ExpressionSignature(
                "ratio", 2, numeric, scalar_or_relation,
                "ratio", "number", "scalar", ("operand",),
            ),
            "correlation": ExpressionSignature(
                "correlation", 2, numeric, scalar_or_relation,
                "unitless", "number", "scalar", ("operand",),
            ),
            "max_of_product": ExpressionSignature(
                "max_of_product", 2, numeric, scalar_or_relation,
                "product", "number", "scalar", ("operand",),
            ),
        }

    def get(self, function_id: str) -> ExpressionSignature | None:
        return self._signatures.get(function_id)

    def validate(self, function_id: str, inputs: list[ValueSignature], *,
                 semantic_role: str = "measure") -> SignatureResult:
        signature = self.get(function_id)
        if signature is None:
            return SignatureResult(
                False, diagnostic_code="expression_function_unsupported",
                message=f"不支持表达式函数：{function_id}",
            )
        if len(inputs) != signature.arity:
            return SignatureResult(
                False, diagnostic_code="expression_arity_invalid",
                message=f"{function_id} 需要 {signature.arity} 个输入，实际得到 {len(inputs)} 个",
            )
        if semantic_role not in signature.allowed_semantic_roles:
            return SignatureResult(
                False, diagnostic_code="expression_role_invalid",
                message=f"{function_id} 不允许将 {semantic_role} 作为输入角色",
            )
        for value in inputs:
            if ("any" not in signature.accepted_input_types
                    and value.result_type not in signature.accepted_input_types):
                return SignatureResult(
                    False, diagnostic_code="expression_input_type_invalid",
                    message=(f"{function_id} 需要数值输入，不能使用 "
                             f"{value.result_type or 'unknown'} 作为 {semantic_role}"),
                )
            if value.shape not in signature.accepted_shapes:
                return SignatureResult(
                    False, diagnostic_code="expression_input_shape_invalid",
                    message=f"{function_id} 不支持输入形状：{value.shape}",
                )

        first = inputs[0]
        if signature.unit_relation == "preserve":
            result_type = first.result_type if signature.result_type == "input" else signature.result_type
            return SignatureResult(True, result_type, first.unit, first.shape)
        if signature.unit_relation == "count":
            return SignatureResult(True, "number", None, signature.result_shape)
        if signature.unit_relation == "unitless":
            return SignatureResult(True, signature.result_type, "ratio", signature.result_shape)
        if signature.unit_relation == "ratio":
            if inputs[0].unit != inputs[1].unit:
                return SignatureResult(
                    False, diagnostic_code="expression_unit_mismatch",
                    message=f"{function_id} 的两个输入必须具有相同单位",
                )
            return SignatureResult(True, signature.result_type, "ratio", signature.result_shape)
        if signature.unit_relation == "product":
            left, right = inputs
            if left.unit is None:
                return SignatureResult(True, signature.result_type, right.unit, signature.result_shape)
            if right.unit is None:
                return SignatureResult(True, signature.result_type, left.unit, signature.result_shape)
            return SignatureResult(
                False, diagnostic_code="compound_unit_unsupported",
                message="max_of_product 暂不支持两个有量纲输入的复合单位",
            )
        return SignatureResult(
            False, diagnostic_code="expression_signature_invalid",
            message=f"函数签名配置无效：{function_id}",
        )


DEFAULT_EXPRESSION_SIGNATURES = ExpressionSignatureRegistry()
