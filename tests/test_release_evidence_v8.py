import hashlib

from agent.release_evidence import verify_gate_evidence


def test_missing_or_claimed_pass_is_not_evidence(tmp_path):
    assert verify_gate_evidence(tmp_path, None, code_digest="code")["status"] == "not_run"
    assert verify_gate_evidence(tmp_path, {"passed": True}, code_digest="code")["status"] == "fail"


def test_hash_code_and_required_tests_are_verified(tmp_path):
    raw = b'<testsuites><testsuite><testcase classname="contract" name="required"/></testsuite></testsuites>'
    (tmp_path / "report.xml").write_bytes(raw)
    entry = dict(code_digest="code", report="report.xml", sha256=hashlib.sha256(raw).hexdigest(),
                 test_ids=["contract::required"])
    assert verify_gate_evidence(tmp_path, entry, code_digest="code")["status"] == "pass"
    assert verify_gate_evidence(tmp_path, entry, code_digest="new-code")["status"] == "fail"
    (tmp_path / "report.xml").write_bytes(raw.replace(b'/></testsuite>', b'><skipped/></testcase></testsuite>'))
    assert verify_gate_evidence(tmp_path, entry, code_digest="code")["status"] == "fail"
    entry["sha256"] = hashlib.sha256((tmp_path / "report.xml").read_bytes()).hexdigest()
    assert verify_gate_evidence(tmp_path, entry, code_digest="code")["status"] == "fail"
