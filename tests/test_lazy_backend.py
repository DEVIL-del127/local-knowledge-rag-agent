from types import SimpleNamespace

from agent.lazy_backend import LazyDependency, LazySearchBackend


def test_lazy_dependency_delegates_parameterized_get():
    calls = []

    class Registry:
        def get(self, generation_id):
            calls.append(generation_id)
            return f"record:{generation_id}"

    lazy = LazyDependency(Registry)

    assert lazy.get("g1") == "record:g1"
    assert calls == ["g1"]
    assert isinstance(lazy.resolve(), Registry)


def test_lazy_search_backend_resolves_nested_dependencies_once():
    created = []
    es = object()
    embedder = object()
    manager = SimpleNamespace(es=es, inspect_status=lambda: "ready")

    def factory():
        created.append(True)
        return SimpleNamespace(es_manager=manager, embedder=embedder)

    backend = LazySearchBackend(factory, label="formal", pdf_dir="pdfs", index_name="docs")

    assert backend.es_manager.inspect_status() == "ready"
    assert backend.es_manager.es.resolve() is es
    assert backend.embedder.resolve() is embedder
    assert len(created) == 1
