"""Verify local test evidence without trusting a claimed pass flag."""
from pathlib import Path
import hashlib
import xml.etree.ElementTree as ET


def verify_gate_evidence(root: Path, entry: dict | None, *, code_digest: str,
                         required_test_ids: set[str] | None = None) -> dict:
    if not entry:
        return {"status": "not_run", "reason": "evidence_missing"}
    try:
        if entry["code_digest"] != code_digest:
            return {"status": "fail", "reason": "stale_code_evidence"}
        report = (root / entry["report"]).resolve()
        if root.resolve() not in report.parents or not report.is_file():
            return {"status": "fail", "reason": "report_unavailable"}
        raw = report.read_bytes()
        if hashlib.sha256(raw).hexdigest() != entry["sha256"]:
            return {"status": "fail", "reason": "report_digest_mismatch"}
        cases = ET.fromstring(raw).findall(".//testcase")
        required = set(entry["test_ids"])
        if required_test_ids is not None and required != required_test_ids:
            return {"status": "fail", "reason": "gate_test_contract_mismatch"}
        if not required:
            return {"status": "fail", "reason": "test_ids_missing"}
        found = {}
        for case in cases:
            identity = f"{case.get('classname', '')}::{case.get('name', '')}"
            if identity in required:
                passed = not any(case.find(tag) is not None for tag in ("failure", "error", "skipped"))
                found[identity] = found.get(identity, True) and passed
        if set(found) != required or not all(found.values()):
            return {"status": "fail", "reason": "required_test_not_passed"}
        return {"status": "pass", "reason": "verified_test_evidence"}
    except (KeyError, TypeError, ValueError, OSError, ET.ParseError):
        return {"status": "fail", "reason": "invalid_evidence_contract"}


def code_tree_digest(root: Path) -> str:
    digest = hashlib.sha256()
    files = []
    for folder in ("agent", "core", "ingestion", "memory", "kb-agent/nlu_v2"):
        files.extend((root / folder).rglob("*.py"))
    files.extend(root / name for name in ("agent_main.py", "main.py", "pyproject.toml", "uv.lock"))
    for path in sorted(files, key=lambda p: p.relative_to(root).as_posix()):
        digest.update(path.relative_to(root).as_posix().encode())
        digest.update(b"\0")
        digest.update(hashlib.sha256(path.read_bytes()).digest())
    return digest.hexdigest()
