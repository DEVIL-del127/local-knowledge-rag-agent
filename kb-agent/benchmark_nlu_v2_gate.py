"""Evaluate the frozen 23-case M1.2 semantic gate."""
from __future__ import annotations

import argparse
from collections import Counter
import json
from pathlib import Path

from benchmark_nlu_v2_100 import DEFAULT_BENCHMARK_INPUT, parse_questions
from nlu_v2_validate import build_engine


def evaluate(case, result) -> list[str]:
    failures = []
    if result.validation.status != case["expected_status"]:
        failures.append(f"status={result.validation.status}, expected={case['expected_status']}")
    operators = Counter(item.operator_id for item in result.understanding.operator_invocations)
    for operator, count in case.get("required_operators", {}).items():
        if operators[operator] < count:
            failures.append(f"operator {operator}={operators[operator]}, expected>={count}")
    ambiguity_kinds = {item.kind for item in result.understanding.ambiguities}
    for kind in case.get("required_ambiguities", []):
        if kind not in ambiguity_kinds:
            failures.append(f"missing ambiguity {kind}")
    error_codes = {item.code for item in result.validation.errors}
    for code in case.get("required_errors", []):
        if code not in error_codes:
            failures.append(f"missing error {code}")
    for code in case.get("forbidden_errors", []):
        if code in error_codes:
            failures.append(f"forbidden error {code}")
    coverage_by_id = {item.requirement_id: item for item in result.understanding.coverage}
    uncovered = {item.requirement_type for item in result.understanding.requirements
                 if coverage_by_id.get(item.requirement_id) is None
                 or coverage_by_id[item.requirement_id].status != "satisfied"}
    for kind in case.get("required_uncovered_requirements", []):
        if kind not in uncovered:
            failures.append(f"missing uncovered requirement {kind}")
    annotations = {item.kind for item in result.understanding.content_annotations}
    for kind in case.get("required_annotations", []):
        if kind not in annotations:
            failures.append(f"missing annotation {kind}")
    if result.model_calls > case.get("max_model_calls", 2):
        failures.append(f"model_calls={result.model_calls} over budget")
    return failures


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", type=Path, action="append", default=None,
                        help="project-local Markdown or JSON question source; repeatable")
    parser.add_argument("--first", type=Path, default=DEFAULT_BENCHMARK_INPUT,
                        help="compatibility input; defaults to the checked-in project fixture")
    parser.add_argument("--second", type=Path, default=None,
                        help="optional additional Markdown/JSON question source")
    parser.add_argument("--fixture", type=Path, default=Path("data/nlu_v2_semantic_gate_23_v1.json"))
    parser.add_argument("--llm", action="store_true")
    parser.add_argument("--model", default="qwen2.5:7b")
    args = parser.parse_args()
    input_paths = args.input or [item for item in (args.first, args.second) if item is not None]
    if not all(item.is_file() for item in input_paths):
        missing = [str(item) for item in input_paths if not item.is_file()]
        raise FileNotFoundError(f"semantic-gate input does not exist: {missing}")
    questions: dict[int, dict] = {}
    for input_path in input_paths:
        for item in parse_questions(input_path):
            previous = questions.get(item["id"])
            if previous is not None and previous["query"] != item["query"]:
                raise ValueError(f"duplicate benchmark ID with different query: {item['id']}")
            questions[item["id"]] = item
    fixture = json.loads(args.fixture.read_text(encoding="utf-8"))
    failed = {}
    for expected in fixture["cases"]:
        result = build_engine(args.llm, model=args.model).analyze(questions[expected["id"]]["query"])
        failures = evaluate(expected, result)
        print(f"[{expected['id']:03d}] {'PASS' if not failures else 'FAIL'} "
              f"status={result.validation.status} calls={result.model_calls}")
        if failures:
            failed[expected["id"]] = failures
            for item in failures:
                print(f"  - {item}")
    print(json.dumps({"passed": len(fixture["cases"]) - len(failed),
                      "total": len(fixture["cases"]), "failures": failed},
                     ensure_ascii=False, indent=2))
    return 1 if failed else 0


if __name__ == "__main__":
    raise SystemExit(main())
