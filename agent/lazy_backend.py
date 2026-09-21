"""Delay knowledge-base dependencies until an actual data operation."""
from threading import RLock


class LazyDependency:
    def __init__(self, factory):
        self._factory = factory
        self._value = None
        self._lock = RLock()

    def resolve(self):
        """Return the materialized dependency without shadowing its public API.

        ``get`` is deliberately not used here: several wrapped dependencies
        (notably GenerationRegistry) expose their own parameterized ``get``
        method, which must be delegated by ``__getattr__``.
        """
        with self._lock:
            if self._value is None:
                value = self._factory()
                if value is None:
                    raise RuntimeError("dependency_factory_returned_none")
                self._value = value
            return self._value

    def __getattr__(self, name):
        return getattr(self.resolve(), name)


class LazySearchBackend(LazyDependency):
    def __init__(self, factory, *, label, pdf_dir, index_name):
        super().__init__(factory)
        self.label = label
        self.pdf_dir = pdf_dir
        self.es_manager = _LazyManager(self, index_name)
        self.embedder = LazyDependency(lambda: self.resolve().embedder)


class _LazyManager:
    def __init__(self, backend, index_name):
        self._backend = backend
        self.index_name = index_name
        self.es = LazyDependency(lambda: backend.resolve().es_manager.es)

    def inspect_status(self, *args, **kwargs):
        return self._backend.resolve().es_manager.inspect_status(*args, **kwargs)

    def __getattr__(self, name):
        return getattr(self._backend.resolve().es_manager, name)
