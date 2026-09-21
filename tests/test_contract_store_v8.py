from dataclasses import asdict, replace
import hashlib

import pytest

from core.contract_store import ContractStore, ContractRef, ContractReferenceError
from core.catalog_provider import CatalogContractSnapshot, CatalogSourceContract
from core.retrieval_gateway import GenerationSnapshot
from agent.semantic_compiler_adapter import SemanticCompilerAdapter


def test_content_reference_roundtrip_and_tamper(tmp_path):
    store = ContractStore(tmp_path, store_id="catalogs")
    value = {"schema_version": "synthetic-v1", "value": "数据"}
    ref = store.publish(value)
    assert store.resolve(ref) == value
    assert store.publish(value) == ref
    with pytest.raises(ContractReferenceError):
        store.resolve(replace(ref, locator="../escape"))
    with pytest.raises(ContractReferenceError):
        store.resolve(replace(ref, store_id="other"))
    (tmp_path / ref.locator).write_bytes(b"changed")
    with pytest.raises(ContractReferenceError):
        store.publish(value)


def test_store_construction_never_creates_directory(tmp_path):
    root = tmp_path / "absent"
    store = ContractStore(root, store_id="catalogs")
    assert not root.exists()
    with pytest.raises(ContractReferenceError):
        store.publish({"schema_version": "v1"})
    assert not root.exists()


def test_nlu_loads_pinned_catalog_object_without_live_mapping(tmp_path):
    catalog = CatalogContractSnapshot("test", sources=(CatalogSourceContract(
        "papers", ("论文",), ("retrieve",), (), "1"),))
    store = ContractStore(tmp_path, store_id="catalogs")
    ref = store.publish(catalog.identity_payload())
    snapshot = GenerationSnapshot("g", 1, "papers", "chunks", "model", 3,
                                  catalog_version="test", catalog_digest=catalog.digest(),
                                  catalog_schema_version="catalog-contract-v2", catalog_ref=asdict(ref))
    snapshot = replace(snapshot, snapshot_digest=snapshot.canonical_digest())
    adapter = SemanticCompilerAdapter(search_backend=object(), enable_llm=False, contract_store=store)
    assert adapter._get_engine(snapshot) is not None
    assert adapter._engine_catalog_digest == catalog.digest()
    (tmp_path / ref.locator).write_bytes(b"tampered")
    with pytest.raises(ContractReferenceError):
        adapter._get_engine(snapshot)
