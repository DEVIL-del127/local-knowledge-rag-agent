"""Generate a quantitative M1.5 acceptance report from a new batch directory."""
from __future__ import annotations

import argparse
import hashlib
import json
from datetime import datetime
from pathlib import Path


def main(argv=None):
    parser = argparse.ArgumentParser()
    parser.add_argument("--batch-dir", type=Path, required=True)
    parser.add_argument("--llm-report", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument(
        "--manifest", type=Path,
        default=Path(__file__).resolve().parents[1] / "tests" / "fixtures" / "m15_obligation_manifest.json",
    )
    args = parser.parse_args(argv)
    cases = []
    for path in sorted(args.batch_dir.glob("case_*.json")):
        try:
            payload = json.loads(path.read_text(encoding="utf-8"))
            readiness = payload.get("event_readiness_report") or {}
            lineage = payload.get("typed_lineage_report") or {}
            m15_plan = payload.get("m15_logical_plan") or {}
            cases.append({
                "file": path.name, "runtime_error": payload.get("status") == "error",
                "event_contracts": len(readiness.get("consecutive_events", [])),
                "duration_contracts": len(readiness.get("duration_accumulations", [])),
                "value_contracts": len(readiness.get("cumulative_values", [])),
                "typed_lineage_ready": bool(lineage.get("ready")),
                "false_ready_count": readiness.get("false_ready_count", 0),
                "m15_plan_status": m15_plan.get("status", "not_evaluated"),
            })
        except Exception as exc:
            cases.append({"file": path.name, "runtime_error": True, "error": str(exc)})
    llm = json.loads(args.llm_report.read_text(encoding="utf-8"))
    manifest = json.loads(args.manifest.read_text(encoding="utf-8"))
    obligation_metrics = _obligation_metrics(manifest, args.batch_dir)
    report = {
        "schema_version": "m15-acceptance-report-v1",
        "generated_at": datetime.now().astimezone().isoformat(),
        "batch_dir": str(args.batch_dir.resolve()),
        "batch_digest": _directory_digest(args.batch_dir),
        "llm_report": str(args.llm_report.resolve()),
        "llm_report_sha256": hashlib.sha256(args.llm_report.read_bytes()).hexdigest(),
        "summary": {
            "case_count": len(cases),
            "runtime_errors": sum(item.get("runtime_error", False) for item in cases),
            "event_contracts": sum(item.get("event_contracts", 0) for item in cases),
            "duration_contracts": sum(item.get("duration_contracts", 0) for item in cases),
            "value_contracts": sum(item.get("value_contracts", 0) for item in cases),
            "typed_lineage_ready_cases": sum(item.get("typed_lineage_ready", False) for item in cases),
            "false_ready_count": sum(item.get("false_ready_count", 0) for item in cases),
            "m15_logical_plan_ready_cases": sum(item.get("m15_plan_status") == "ready" for item in cases),
            "llm_gate": llm.get("gate_result", "not_evaluated"),
            "llm_metrics": llm.get("metrics", {}),
            "obligation_metrics": obligation_metrics,
        },
        "release_authorized": False,
        "default_flags": {
            "enable_candidate_choice_v3": False,
            "enable_m15_requirement_graph_shadow": False,
            "enable_m15_field_binding_shadow": False,
            "enable_m15_quantity_contract": False,
            "enable_m15_temporal_contracts": False,
            "enable_m15_typed_lineage": False,
            "enable_m15_logical_plan": False,
        },
        "cases": cases,
    }
    report["release_authorized"] = (
        len(cases) == 50 and report["summary"]["runtime_errors"] == 0
        and report["summary"]["false_ready_count"] == 0
        and report["summary"]["llm_gate"] == "pass"
        and _semantic_gates_pass(obligation_metrics)
    )
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(report["summary"], ensure_ascii=False, indent=2))
    return 0 if report["release_authorized"] else 1


def _directory_digest(path):
    digest = hashlib.sha256()
    for item in sorted(path.glob("case_*.json")):
        digest.update(item.name.encode("utf-8"))
        digest.update(hashlib.sha256(item.read_bytes()).digest())
    return digest.hexdigest()


def _obligation_metrics(manifest, batch_dir):
    case_payloads = {}
    for path in batch_dir.glob("case_*.json"):
        match = __import__("re").search(r"case_(\d+)", path.name)
        if match:
            case_payloads[int(match.group(1))] = json.loads(path.read_text(encoding="utf-8"))
    names = ["event_structure", "duration_accumulation", "cumulative_value",
             "event_projection_set", "full_combined"]
    metrics = {}
    for name in names:
        obligations = [item for item in manifest["obligations"]
                       if item.get("in_scope") and name in item.get("metric_membership", [])]
        completed = []
        failed = []
        for obligation in obligations:
            payload = case_payloads.get(int(obligation["case_id"]), {})
            coverage = {item["requirement_id"]: item["status"]
                        for item in payload.get("understanding", {}).get("coverage", [])}
            ids = obligation.get("raw_requirement_ids", [])
            if ids and all(coverage.get(item) == "satisfied" for item in ids):
                completed.append(obligation["obligation_id"])
            else:
                failed.append(obligation["obligation_id"])
        denominator = len(obligations)
        metrics[name] = {
            "completed": len(completed), "denominator": denominator,
            "completion_rate": len(completed) / denominator if denominator else None,
            "completed_obligation_ids": completed, "failed_obligation_ids": failed,
        }
    diagnostics = {}
    memberships = sorted({membership for item in manifest["obligations"]
                          for membership in item.get("metric_membership", [])
                          if membership.startswith("diagnostic:")})
    for membership in memberships:
        obligations = [item for item in manifest["obligations"]
                       if item.get("in_scope") and membership in item.get("metric_membership", [])]
        baseline_failures = sum(item.get("baseline_coverage_status") != "satisfied" for item in obligations)
        candidate_failures = 0
        for obligation in obligations:
            payload = case_payloads.get(int(obligation["case_id"]), {})
            coverage = {item["requirement_id"]: item["status"]
                        for item in payload.get("understanding", {}).get("coverage", [])}
            ids = obligation.get("raw_requirement_ids", [])
            if not ids or not all(coverage.get(item) == "satisfied" for item in ids):
                candidate_failures += 1
        diagnostics[membership] = {
            "baseline_failures": baseline_failures,
            "candidate_failures": candidate_failures,
            "reduction_rate": ((baseline_failures - candidate_failures) / baseline_failures
                               if baseline_failures else None),
        }
    metrics["diagnostic_reduction"] = diagnostics
    return metrics


def _semantic_gates_pass(metrics):
    thresholds = {
        "event_structure": .90, "duration_accumulation": 1.0,
        "cumulative_value": 1.0, "event_projection_set": .85,
        "full_combined": .80,
    }
    return all((metrics.get(name, {}).get("completion_rate") or 0) >= threshold
               for name, threshold in thresholds.items())


if __name__ == "__main__":
    raise SystemExit(main())
