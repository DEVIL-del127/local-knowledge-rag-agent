from types import SimpleNamespace
import sys

import pytest

from agent.semantic_compiler_adapter import SemanticCompilerAdapter


def test_wrong_cached_nlu_package_is_rejected(tmp_path, monkeypatch):
    adapter = SemanticCompilerAdapter(search_backend=object(), enable_llm=False)
    monkeypatch.setitem(sys.modules, "nlu_v2.injected", SimpleNamespace(__file__=str(tmp_path / "foreign.py")))
    with pytest.raises(RuntimeError, match="source locator mismatch"):
        adapter._verify_loaded_source()


def test_source_changes_cannot_keep_old_digest(monkeypatch):
    adapter = SemanticCompilerAdapter(search_backend=object(), enable_llm=False)
    monkeypatch.setattr("agent.semantic_compiler_adapter._tree_digest", lambda root: "changed")
    with pytest.raises(RuntimeError, match="tree changed"):
        adapter._verify_loaded_source()
