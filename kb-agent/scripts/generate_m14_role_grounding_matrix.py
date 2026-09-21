"""Generate the frozen pre-M1.4 role-grounding diagnostic matrix.

This script reads recorded batch artifacts only.  It deliberately does not
import the candidate compiler, CoverageMatcher, or validator, so the frozen
baseline cannot be rewritten by the implementation it is intended to assess.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import re
from pathlib import Path


REPAIRABLE = {
    "sequence_event",
    "set_operation",
    "aggregate",
    "scoped_aggregate",
    "calculation",
}


def _case_id(path: Path) -> int:
    match = re.match(r"case_(\d+)_", path.name)
    if not match:
        raise ValueError(f"unexpected case filename: {path.name}")
    return int(match.group(1))


def _role(status: str, *, requirement_ids: list[str] | None = None,
          demand_ids: list[str] | None = None, spans: list[dict] | None = None,
          values: list[str] | None = None, reason: str = "") -> dict:
    return {
        "status": status,
        "requirement_ids": sorted(set(requirement_ids or [])),
        "demand_ids": sorted(set(demand_ids or [])),
        "spans": spans or [],
        "values": sorted(set(values or [])),
        "reason": reason,
    }


def _mapped_demands(requirement: dict, demand_by_id: dict[str, dict]) -> list[dict]:
    identifiers = requirement.get("expected_attributes", {}).get("source_demand_ids", [])
    return [demand_by_id[item] for item in identifiers if item in demand_by_id]


def _field_demands(requirement: dict, demand_by_id: dict[str, dict]) -> list[dict]:
    attrs = requirement.get("expected_attributes", {})
    identifiers = set(attrs.get("input_demand_ids", []))
    for item in _mapped_demands(requirement, demand_by_id):
        identifiers.update(item.get("dependencies", []))
    return [
        demand_by_id[item]
        for item in identifiers
        if item in demand_by_id and demand_by_id[item].get("demand_type") == "field"
    ]


def _dependency_requirements(requirement: dict, requirement_by_id: dict[str, dict]) -> list[dict]:
    return [
        requirement_by_id[item]
        for item in requirement.get("dependencies", [])
        if item in requirement_by_id
    ]


def _spans(items: list[dict]) -> list[dict]:
    return [item["span"] for item in items if item.get("span")]


def _values(items: list[dict]) -> list[str]:
    values = []
    for item in items:
        attrs = item.get("attributes", {})
        values.append(str(attrs.get("field_text") or attrs.get("operator_family") or item.get("text", "")))
    return values


def _requirement_row(requirement: dict, demand_by_id: dict[str, dict],
                     requirement_by_id: dict[str, dict]) -> dict:
    requirement_id = requirement["requirement_id"]
    mapped = _mapped_demands(requirement, demand_by_id)
    operation_demands = [item for item in mapped if item.get("demand_type") == "operator"]
    fields = _field_demands(requirement, demand_by_id)
    dependencies = _dependency_requirements(requirement, requirement_by_id)
    kind = requirement.get("requirement_type", "")
    text = requirement.get("text", "")

    roles = {
        "operation": _role(
            "proven" if operation_demands else "unproven",
            requirement_ids=[requirement_id],
            demand_ids=[item["demand_id"] for item in operation_demands],
            spans=_spans(operation_demands),
            values=_values(operation_demands),
            reason="" if operation_demands else "no exact operator SourceDemand",
        )
    }

    if kind in {"aggregate", "scoped_aggregate"}:
        roles["measure"] = _role(
            "proven" if len(fields) == 1 else "unproven",
            requirement_ids=[requirement_id],
            demand_ids=[item["demand_id"] for item in fields],
            spans=_spans(fields),
            values=_values(fields),
            reason="" if len(fields) == 1 else "aggregate measure is not grounded to exactly one field demand",
        )
        scope_required = bool(
            kind == "scoped_aggregate"
            or re.search(r"这些|上述|筛选|交集|并集|重合|全站|总体", text)
            or requirement.get("expected_scope") in {"filtered", "relation"}
        )
        roles["scope"] = _role(
            "not_required" if not scope_required else "proven" if dependencies else "unproven",
            requirement_ids=[item["requirement_id"] for item in dependencies],
            spans=[item["span"] for item in dependencies if item.get("span")],
            values=[item.get("requirement_type", "") for item in dependencies],
            reason="" if not scope_required or dependencies else "scoped aggregate has no Requirement dependency",
        )
        grain_required = bool(re.search(r"(?:日|天|月|周|年)(?:平均|均值)", text))
        roles["grain"] = _role(
            "not_required" if not grain_required else "proven" if requirement.get("expected_output_grain") not in {"", "scalar"} else "unproven",
            requirement_ids=[requirement_id],
            values=[requirement.get("expected_output_grain", "")],
            reason="" if not grain_required or requirement.get("expected_output_grain") not in {"", "scalar"} else "explicit aggregate grain is not represented",
        )
    elif kind == "set_operation":
        roles["inputs"] = _role(
            "proven" if len(dependencies) >= 2 else "unproven",
            requirement_ids=[item["requirement_id"] for item in dependencies],
            spans=[item["span"] for item in dependencies if item.get("span")],
            values=[item.get("requirement_type", "") for item in dependencies],
            reason="" if len(dependencies) >= 2 else "set operation lacks two distinct producer Requirements",
        )
    elif kind == "sequence_event":
        roles["inputs"] = _role(
            "proven" if dependencies or fields else "unproven",
            requirement_ids=[item["requirement_id"] for item in dependencies],
            demand_ids=[item["demand_id"] for item in fields],
            spans=_spans(fields),
            values=_values(fields),
            reason="" if dependencies or fields else "event predicate/duration inputs are not grounded",
        )
    elif kind == "calculation":
        roles["inputs"] = _role(
            "proven" if dependencies or len(fields) >= 1 else "unproven",
            requirement_ids=[item["requirement_id"] for item in dependencies],
            demand_ids=[item["demand_id"] for item in fields],
            spans=_spans(fields),
            values=_values(fields),
            reason="" if dependencies or fields else "calculation operands are not grounded",
        )

    return {
        "requirement_id": requirement_id,
        "requirement_type": kind,
        "operator_family": requirement.get("operator_family", ""),
        "clause_id": requirement.get("clause_id", ""),
        "span": requirement.get("span"),
        "roles": roles,
    }


def build_matrix(test_dir: Path) -> dict:
    cases = []
    paths = sorted(test_dir.glob("case_*.json"), key=_case_id)
    for path in paths:
        payload = json.loads(path.read_text("utf-8"))
        ir = payload["understanding"]
        demands = ir.get("source_demands", [])
        requirements = ir.get("requirements", [])
        demand_by_id = {item["demand_id"]: item for item in demands}
        requirement_by_id = {item["requirement_id"]: item for item in requirements}
        rows = [
            _requirement_row(item, demand_by_id, requirement_by_id)
            for item in requirements
            if item.get("requirement_type") in REPAIRABLE
        ]
        cases.append({
            "case_id": _case_id(path),
            "artifact": path.name,
            "artifact_sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
            "rows": rows,
            "proven_roles": sum(
                role["status"] == "proven"
                for row in rows for role in row["roles"].values()
            ),
            "unproven_roles": sum(
                role["status"] == "unproven"
                for row in rows for role in row["roles"].values()
            ),
        })
    return {
        "schema_version": "m14-role-grounding-baseline-v1",
        "source": "recorded batch artifacts; independent of production coverage/candidate code",
        "cases": cases,
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--test-dir", type=Path, default=Path("test"))
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("tests/fixtures/m14_role_grounding_baseline.json"),
    )
    args = parser.parse_args()
    payload = build_matrix(args.test_dir)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n", "utf-8")


if __name__ == "__main__":
    main()
