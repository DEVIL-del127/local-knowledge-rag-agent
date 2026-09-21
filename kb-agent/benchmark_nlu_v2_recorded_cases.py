"""Read-only structural audit for the recorded complex-query cases."""
from __future__ import annotations

import argparse
import json
from collections import Counter
from pathlib import Path

from nlu_v2 import QueryUnderstandingEngine


HARD_FAILURE_CODES = {
    "source_demand_slot_anchor_missing",
    "source_demand_slot_anchor_invalid",
    "repair_contract_input_out_of_scope",
}
STRUCTURE_CODES = {
    "event_structure_incomplete",
    "set_inputs_unbound",
    "formula_input_unbound",
    "output_ref_unbound",
    "arithmetic_output_mismatch",
    "relation_filter_predicate_unbound",
}


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--cases", type=Path, default=Path("test"))
    args = parser.parse_args()
    paths = sorted(args.cases.glob("case_*.json"))
    if not paths:
        raise FileNotFoundError(f"no recorded cases found in {args.cases}")

    engine = QueryUnderstandingEngine()
    crashes = []
    error_counts: Counter[str] = Counter()
    case_summaries = []
    for path in paths:
        try:
            recorded = json.loads(path.read_text(encoding="utf-8"))
            query = recorded["understanding"]["query"]["raw"]
            result = engine.analyze(query)
            codes = [item.code for item in result.validation.errors]
            if result.patch_report and result.patch_report.transaction_error:
                codes.append(result.patch_report.transaction_error)
            error_counts.update(codes)
            case_summaries.append({
                "case": path.name,
                "events": len(result.understanding.events),
                "sets": len(result.understanding.set_operations),
                "aggregates": len(result.understanding.aggregates),
                "calculations": len(result.understanding.calculations),
                "hard_failures": sorted(set(codes) & HARD_FAILURE_CODES),
                "structure_failures": sorted(set(codes) & STRUCTURE_CODES),
            })
        except Exception as exc:  # The audit must report every runtime failure.
            crashes.append({"case": path.name, "error": f"{type(exc).__name__}: {exc}"})

    report = {
        "cases": len(paths),
        "crashes": crashes,
        "hard_failures": {code: error_counts[code] for code in sorted(HARD_FAILURE_CODES)},
        "structure_failures": {code: error_counts[code] for code in sorted(STRUCTURE_CODES)},
        "case_summaries": case_summaries,
    }
    print(json.dumps(report, ensure_ascii=False, indent=2))
    return 1 if crashes or any(error_counts[code] for code in HARD_FAILURE_CODES) else 0


if __name__ == "__main__":
    raise SystemExit(main())
