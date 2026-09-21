"""Narrow application capabilities for Chroma reads, not an OS sandbox."""
from __future__ import annotations


class ReadOnlyCollection:
    __slots__ = ("__collection",)

    def __init__(self, collection):
        self.__collection = collection

    @property
    def id(self):
        return self.__collection.id

    @property
    def name(self):
        return self.__collection.name

    @property
    def metadata(self):
        return dict(self.__collection.metadata or {})

    def count(self):
        return self.__collection.count()

    def get(self, *args, **kwargs):
        return self.__collection.get(*args, **kwargs)

    def query(self, *args, **kwargs):
        return self.__collection.query(*args, **kwargs)

    def __getattr__(self, name):
        raise PermissionError(f"read-only collection does not expose {name}")


class ReadOnlyChromaClient:
    __slots__ = ("__client",)

    def __init__(self, client):
        self.__client = client

    def get_collection(self, name, **kwargs):
        # No remote default embedding function may be installed by the reader.
        kwargs["embedding_function"] = None
        return ReadOnlyCollection(self.__client.get_collection(name=name, **kwargs))

    def list_collections(self, **kwargs):
        return self.__client.list_collections(**kwargs)

    def heartbeat(self):
        return self.__client.heartbeat()

    @property
    def storage_client_type(self):
        return type(self.__client).__qualname__

    def __getattr__(self, name):
        raise PermissionError(f"read-only client does not expose {name}")
