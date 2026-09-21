from pathlib import Path
import sys

from agent_main import _bootstrap_source_paths


def test_source_checkout_bootstraps_nlu_v2_without_reinstall(monkeypatch):
    root = Path(__file__).resolve().parents[1]
    nlu_root = str((root / "kb-agent").resolve())
    monkeypatch.setattr(sys, "path", [item for item in sys.path if item != nlu_root])

    _bootstrap_source_paths(root)

    assert sys.path[0] == nlu_root
    from nlu_v2.literature_semantics import analyze_literature
    assert analyze_literature("ESN文献").request.canonical_topic == "Echo State Network"
