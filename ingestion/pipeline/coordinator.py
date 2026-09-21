from __future__ import annotations

import hashlib
import time
import uuid
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from ingestion.chunking.structural import structural_chunks
from ingestion.quality.scorer import score_document

from .generation_registry import GenerationRecord, GenerationRegistry, GenerationState
from .manifest import (
    IngestionManifest, ManifestDocument, ManifestRef, verify_manifest_ref,
    vector_store_identity, is_legacy_vector_store_identity,
)


class IngestionRejected(RuntimeError):
    pass


class IngestionCoordinator:
    """Parse, validate and stage a generation without activating it."""

    def __init__(
        self,
        *,
        parser: Any,
        registry: GenerationRegistry,
        es_sink_factory,
        vector_sink_factory,
        manifest_dir: str | Path,
        embedding_model: str,
        embedding_dimension: int = 1024,
    ) -> None:
        self.parser = parser
        self.registry = registry
        self.es_sink_factory = es_sink_factory
        self.vector_sink_factory = vector_sink_factory
        self.manifest_dir = Path(manifest_dir)
        self.registry.manifest_root = self.manifest_dir
        self.embedding_model = embedding_model
        self.embedding_dimension = embedding_dimension

    def stage(self, pdf_paths: list[str | Path]) -> IngestionManifest:
        # Generation identifiers are reused in Elasticsearch physical index names.
        # Elasticsearch rejects uppercase characters, so keep this identifier safe
        # for every backend instead of repairing individual sink names later.
        generation = f"{time.strftime('g%Y%m%dt%H%M%S')}-{uuid.uuid4().hex[:8]}"
        es_name = f"pdf_documents_staging_{generation}"
        vector_name = f"pdf_chunks_staging_{generation}"
        record = GenerationRecord(
            generation_id=generation,
            state=GenerationState.BUILDING,
            es_physical_index=es_name,
            vector_collection=vector_name,
            embedding_model=self.embedding_model,
            embedding_dimension=self.embedding_dimension,
        )
        self.registry.put(record)
        build_owner = self.registry.start_build(generation)
        build_resources = [es_name, vector_name]
        manifest = IngestionManifest(
            generation_id=generation,
            embedding_model=self.embedding_model,
            embedding_dimension=self.embedding_dimension,
            created_at_utc=datetime.now(timezone.utc).isoformat(),
            es_physical_index=es_name,
            vector_collection=vector_name,
            source_inventory_root_identity=_canonical_digest(sorted({
                str(Path(item).resolve().parent) for item in pdf_paths
            })),
            discovered_file_count=len(pdf_paths),
        )
        documents: list[dict[str, Any]] = []
        chunks: list[dict[str, Any]] = []
        self.registry.update_build(
            generation, build_owner, stage="parsing", resources=build_resources,
            compensations=["retain_for_manual_gc"],
        )
        for raw_path in pdf_paths:
            path = Path(raw_path)
            try:
                parsed = self.parser.convert(path)
            except Exception as exc:
                manifest.documents.append(ManifestDocument(
                    document_id="", filename=path.name, source_hash="",
                    parser=type(self.parser).__name__, parser_version="",
                    quality={}, chunk_count=0, status="quarantined",
                    relative_locator=path.name,
                    diagnostics=[f"{type(exc).__name__}: {exc}"],
                ))
                continue
            quality = score_document(parsed)
            bibliographic = _extract_bibliographic(path, parsed["markdown"])
            document_id = hashlib.sha256(
                ("pdf-v2\0" + path.name.casefold() + "\0" + parsed["source_hash"]).encode("utf-8")
            ).hexdigest()[:24]
            item_chunks = structural_chunks(
                document_id=document_id,
                content_hash=parsed["source_hash"],
                markdown=parsed["markdown"],
                generation=generation,
                parser_version=parsed["parser_version"],
                embedding_version=self.embedding_model,
                page_blocks=list(parsed.get("page_blocks") or []),
            )
            accepted = quality.status == "pass" and bool(item_chunks)
            manifest.documents.append(ManifestDocument(
                document_id=document_id,
                filename=path.name,
                source_hash=parsed["source_hash"],
                parser=parsed["parser"],
                parser_version=parsed["parser_version"],
                quality=quality.to_dict(),
                chunk_count=len(item_chunks),
                status="accepted" if accepted else "quarantined",
                diagnostics=list(quality.reasons),
                relative_locator=path.name,
                chunk_set_digest=_chunk_set_digest(item_chunks),
                metadata_digest=_canonical_digest(bibliographic),
                bibliographic_metadata=bibliographic,
            ))
            if not accepted:
                continue
            documents.append({
                "document_id": document_id,
                "filename": path.name,
                "content": parsed["markdown"],
                "content_hash": parsed["source_hash"],
                "page_count": quality.page_count,
                "ingestion_generation": generation,
                "parser_version": parsed["parser_version"],
                **bibliographic,
            })
            for item in item_chunks:
                chunk_payload = item.to_dict()
                chunk_payload.update({
                    key: value for key, value in bibliographic.items()
                    if isinstance(value, (str, int, float, bool))
                })
                chunk_payload["filename"] = path.name
                chunks.append(chunk_payload)
        manifest_path = self.manifest_dir / f"{generation}.json"
        if not documents:
            record.state = GenerationState.FAILED
            record.failure_reason = "no document passed the parse quality gate"
            self.registry.put(record)
            manifest.status = "failed"
            manifest.write(manifest_path)
            self.registry.update_build(
                generation, build_owner, stage="failed_quality_gate",
                resources=build_resources, compensations=["retain_for_manual_gc"],
                status="failed",
            )
            raise IngestionRejected("no document passed the parse quality gate")

        es_sink = self.es_sink_factory(es_name)
        vector_sink = self.vector_sink_factory(vector_name)
        try:
            self.registry.update_build(
                generation, build_owner, stage="creating_stores",
                resources=build_resources, compensations=["retain_for_manual_gc"],
            )
            es_sink.create()
            vector_sink.create()
            es_written = es_sink.write(documents)
            vector_written = vector_sink.write(chunks)
        except Exception as exc:
            self._reject(
                record, manifest, manifest_path,
                f"staging backend failure: {type(exc).__name__}: {exc}",
            )
        if es_written != len(documents) or vector_written != len(chunks):
            reason = (
                f"staging count mismatch: es={es_written}/{len(documents)}, "
                f"vector={vector_written}/{len(chunks)}"
            )
            self._reject(record, manifest, manifest_path, reason)
        if es_sink.count() != len(documents) or vector_sink.count() != len(chunks):
            self._reject(record, manifest, manifest_path, "post-write count validation failed")
        if callable(getattr(es_sink, "contract_records", None)):
            es_records = es_sink.contract_records()
            expected_docs = {
                item["document_id"]: (
                    item["content_hash"], item["ingestion_generation"]
                )
                for item in documents
            }
            actual_docs = {
                str(item.get("document_id", "")): (
                    str(item.get("content_hash", "")),
                    str(item.get("ingestion_generation", "")),
                )
                for item in es_records
            }
            if actual_docs != expected_docs:
                self._reject(record, manifest, manifest_path, "ES staging contract mismatch")
        if callable(getattr(vector_sink, "contract_records", None)):
            vector_records, vector_dimension = vector_sink.contract_records()
            expected_chunk_contract = {
                (
                    item["document_id"], item["content_hash"],
                    item["ingestion_generation"], int(item["chunk_index"]),
                )
                for item in chunks
            }
            actual_chunk_contract = {
                (
                    str(item.get("document_id", "")),
                    str(item.get("content_hash", "")),
                    str(item.get("ingestion_generation", "")),
                    int(item.get("chunk_index", -1)),
                )
                for item in vector_records
            }
            if actual_chunk_contract != expected_chunk_contract:
                self._reject(record, manifest, manifest_path, "vector staging contract mismatch")
            if vector_dimension != self.embedding_dimension:
                self._reject(
                    record, manifest, manifest_path,
                    f"embedding dimension mismatch: {vector_dimension}/{self.embedding_dimension}",
                )
        record.document_count = len(documents)
        record.chunk_count = len(chunks)
        record.parser_version = documents[0]["parser_version"]
        manifest.status = "prepared"
        manifest.prepared_at_utc = datetime.now(timezone.utc).isoformat()
        manifest.parser_name = next(
            (item.parser for item in manifest.documents if item.status == "accepted"),
            type(self.parser).__name__,
        )
        manifest.parser_version = documents[0]["parser_version"]
        manifest.parser_config_digest = hashlib.sha256(
            documents[0]["parser_version"].encode("utf-8")
        ).hexdigest()
        manifest.chunker_config_digest = hashlib.sha256(b"max_chars=1600").hexdigest()
        manifest.metadata_extractor_version = BIBLIOGRAPHIC_METADATA_VERSION
        manifest.metadata_extractor_config_digest = _bibliographic_metadata_config_digest()
        manifest.embedding_provider = "ollama"
        manifest.embedding_model_revision = self.embedding_model
        manifest.embedding_normalization = "provider-default"
        manifest.vector_distance_metric = "cosine"
        manifest.chunker_name = "structural"
        manifest.chunker_version = "structural-v2"
        manifest.accepted_document_count = len(documents)
        manifest.quarantined_document_count = len(manifest.documents) - len(documents)
        if hasattr(es_sink, "es"):
            from core.catalog_provider import MainProjectCatalogReader

            mapping = es_sink.es.indices.get_mapping(index=es_name)
            settings = es_sink.es.indices.get_settings(index=es_name)
            index_settings = settings.get(es_name, {}).get("settings", {}).get("index", {})
            manifest.es_index_uuid = str(index_settings.get("uuid", ""))
            manifest.es_mapping_digest = _canonical_digest(mapping)
            manifest.es_settings_digest = _canonical_digest(settings)
            manifest.es_cluster_id = str(es_sink.es.info().get("cluster_uuid", ""))
            manifest.es_document_count = int(es_sink.count())
            catalog = MainProjectCatalogReader(es_sink.es, es_name).snapshot()
            manifest.catalog_schema_version = catalog.schema_version
            manifest.catalog_version = catalog.version
            manifest.catalog_digest = catalog.digest()
        collection = getattr(vector_sink, "collection", None)
        manifest.vector_collection_id = str(getattr(collection, "id", ""))
        manifest.vector_backend = "chroma"
        manifest.vector_store_id = vector_store_identity()
        manifest.vector_metadata_digest = _canonical_digest(
            getattr(collection, "metadata", {}) or {}
        )
        manifest.vector_document_count = len(documents)
        manifest.vector_chunk_count = len(chunks)
        manifest.accepted_document_set_digest = _document_set_digest(documents)
        manifest.es_document_set_digest = manifest.accepted_document_set_digest
        manifest.vector_document_set_digest = manifest.accepted_document_set_digest
        manifest.chunk_set_digest = _chunk_dict_set_digest(chunks)
        record.document_set_digest = manifest.accepted_document_set_digest
        record.chunk_set_digest = manifest.chunk_set_digest
        record.embedding_provider = manifest.embedding_provider
        record.embedding_model_revision = manifest.embedding_model_revision
        record.catalog_version = manifest.catalog_version
        record.catalog_digest = manifest.catalog_digest
        record.state = GenerationState.VALIDATED
        self.registry.put(record)
        reference = manifest.write_content_addressed(self.manifest_dir)
        self.registry.prepare(
            generation, manifest_hash=reference.digest,
            manifest_locator=reference.locator,
            manifest_byte_length=reference.byte_length,
            manifest_schema_version=reference.schema_version,
            manifest_store_id=reference.store_id,
            manifest_locator_scheme=reference.locator_scheme,
            manifest_digest_algorithm=reference.digest_algorithm,
        )
        self.registry.update_build(
            generation, build_owner, stage="prepared", resources=build_resources,
            compensations=["retain_for_manual_gc"], status="complete",
        )
        return manifest

    def _reject(
        self,
        record: GenerationRecord,
        manifest: IngestionManifest,
        manifest_path: Path,
        reason: str,
    ) -> None:
        record.state = GenerationState.FAILED
        record.failure_reason = reason
        self.registry.put(record)
        manifest.status = "failed"
        manifest.write(manifest_path)
        journal = self.registry.get_build(record.generation_id)
        if journal is not None and journal.get("status") == "running":
            self.registry.update_build(
                record.generation_id, str(journal["owner"]), stage="failed",
                resources=list(journal.get("resources") or []),
                compensations=list(journal.get("compensations") or []),
                status="failed",
            )
        raise IngestionRejected(reason)

    def activate(self, generation_id: str) -> int:
        record = self.registry.get(generation_id)
        if record is None or record.state not in {
            GenerationState.PREPARED,
            GenerationState.RETIRED,
        }:
            raise IngestionRejected("generation is not prepared or retired")
        payload = verify_manifest_ref(self.manifest_dir, ManifestRef(
            locator=record.manifest_locator,
            digest=record.manifest_hash,
            byte_length=record.manifest_byte_length,
            store_id=record.manifest_store_id,
            locator_scheme=record.manifest_locator_scheme,
            digest_algorithm=record.manifest_digest_algorithm,
            schema_version=record.manifest_schema_version or "ingestion-manifest-v2",
        ))
        accepted = [item for item in payload.get("documents", []) if item.get("status") == "accepted"]
        if payload.get("generation_id") != generation_id:
            raise IngestionRejected("manifest generation mismatch")
        if len(accepted) != record.document_count:
            raise IngestionRejected("manifest document count mismatch")
        if payload.get("accepted_document_set_digest") != record.document_set_digest:
            raise IngestionRejected("manifest document-set digest mismatch")
        if payload.get("chunk_set_digest") != record.chunk_set_digest:
            raise IngestionRejected("manifest chunk-set digest mismatch")
        record_contract = {
            "embedding_model": record.embedding_model,
            "embedding_dimension": record.embedding_dimension,
            "embedding_provider": record.embedding_provider,
            "embedding_model_revision": record.embedding_model_revision,
            "catalog_version": record.catalog_version,
            "catalog_digest": record.catalog_digest,
        }
        record_mismatches = [
            key for key, expected in record_contract.items()
            if expected not in {None, "", 0} and payload.get(key) != expected
        ]
        if record_mismatches:
            raise IngestionRejected(
                "manifest/registry contract mismatch: " + ", ".join(record_mismatches)
            )
        expected_docs = {
            (str(item.get("document_id", "")), str(item.get("source_hash", "")), generation_id)
            for item in accepted
        }
        es_sink = self.es_sink_factory(record.es_physical_index)
        if hasattr(es_sink, "es"):
            from core.catalog_provider import MainProjectCatalogReader

            mapping = es_sink.es.indices.get_mapping(index=record.es_physical_index)
            settings = es_sink.es.indices.get_settings(index=record.es_physical_index)
            index_settings = settings.get(
                record.es_physical_index, {}
            ).get("settings", {}).get("index", {})
            actual_es = {
                "es_cluster_id": str(es_sink.es.info().get("cluster_uuid", "")),
                "es_index_uuid": str(index_settings.get("uuid", "")),
                "es_mapping_digest": _canonical_digest(mapping),
                "es_settings_digest": _canonical_digest(settings),
            }
            changed = [key for key, value in actual_es.items() if payload.get(key) != value]
            if changed:
                raise IngestionRejected("ES physical identity mismatch: " + ", ".join(changed))
            if payload.get("catalog_schema_version") == "catalog-contract-v2":
                catalog = MainProjectCatalogReader(
                    es_sink.es, record.es_physical_index
                ).snapshot()
                if (catalog.version != payload.get("catalog_version")
                        or catalog.digest() != payload.get("catalog_digest")):
                    raise IngestionRejected("Catalog identity mismatch")
        if callable(getattr(es_sink, "contract_records", None)):
            actual_docs = {
                (str(item.get("document_id", "")), str(item.get("content_hash", "")),
                 str(item.get("ingestion_generation", "")))
                for item in es_sink.contract_records()
            }
            if actual_docs != expected_docs:
                raise IngestionRejected("ES physical document-set mismatch")
        vector_sink = self.vector_sink_factory(record.vector_collection)
        collection = getattr(vector_sink, "collection", None)
        if collection is not None:
            actual_vector = {
                "vector_collection_id": str(getattr(collection, "id", "")),
                "vector_metadata_digest": _canonical_digest(
                    getattr(collection, "metadata", {}) or {}
                ),
                "vector_store_id": vector_store_identity(),
            }
            changed = [
                key for key, value in actual_vector.items()
                if payload.get(key) != value
                and not (key == "vector_store_id" and
                         is_legacy_vector_store_identity(payload.get(key)))
            ]
            if changed:
                raise IngestionRejected(
                    "vector physical identity mismatch: " + ", ".join(changed)
                )
        if callable(getattr(vector_sink, "contract_records", None)):
            vector_records, dimension = vector_sink.contract_records()
            vector_docs = {
                (str(item.get("document_id", "")), str(item.get("content_hash", "")),
                 str(item.get("ingestion_generation", "")))
                for item in vector_records
            }
            if vector_docs != expected_docs:
                raise IngestionRejected("vector physical document-set mismatch")
            if int(dimension) != int(record.embedding_dimension):
                raise IngestionRejected("vector embedding dimension mismatch")
        active, revision = self.registry.get_active()
        switch = (
            self.registry.rollback
            if record.state == GenerationState.RETIRED
            else self.registry.activate
        )
        return switch(
            generation_id,
            expected_active=active.generation_id if active else None,
            expected_revision=revision,
            expected_record_revision=record.record_revision,
        )


def _document_set_digest(documents: list[dict[str, Any]]) -> str:
    values = sorted(
        f"{item['document_id']}:{item['content_hash']}:{item['ingestion_generation']}"
        for item in documents
    )
    return hashlib.sha256("\n".join(values).encode("utf-8")).hexdigest()


def _chunk_set_digest(chunks: list[Any]) -> str:
    return _chunk_dict_set_digest([item.to_dict() for item in chunks])


def _chunk_dict_set_digest(chunks: list[dict[str, Any]]) -> str:
    values = sorted(
        f"{item['document_id']}:{item['content_hash']}:{item['chunk_index']}:"
        f"{hashlib.sha256(str(item.get('text', '')).encode('utf-8')).hexdigest()}"
        for item in chunks
    )
    return hashlib.sha256("\n".join(values).encode("utf-8")).hexdigest()


def _canonical_digest(value: Any) -> str:
    import json
    return hashlib.sha256(json.dumps(
        value, ensure_ascii=False, sort_keys=True, separators=(",", ":"), default=str
    ).encode("utf-8")).hexdigest()


BIBLIOGRAPHIC_METADATA_VERSION = "bibliographic-v5"


def _bibliographic_metadata_config() -> dict[str, Any]:
    """Return the exact, versioned contract executed by `_extract_bibliographic`."""
    return {
        "version": BIBLIOGRAPHIC_METADATA_VERSION,
        "year_priority": "elsevier-pii>filename-explicit>arxiv>frontmatter",
        "document_types": [
            "journal_article", "conference_paper", "preprint", "thesis", "paper",
        ],
        "primary_topic_collision_rejection": True,
        "title_author_artifact_cleanup": True,
    }


def _bibliographic_metadata_config_digest() -> str:
    return _canonical_digest(_bibliographic_metadata_config())


def _clean_bibliographic_title(candidate: str) -> str:
    """Recover a title appended after OCR-flattened author/affiliation markers."""
    import re
    author_marker = re.compile(
        r"\b[A-Z][A-Za-z'’-]+(?:\s+[A-Z][A-Za-z'’-]+){1,3}\s+[a-z]\s*,\s*"
        r"(?:[∗*]\s*,\s*)?"
    )
    matches = list(author_marker.finditer(candidate))
    if len(matches) >= 2:
        suffix = candidate[matches[-1].end():].lstrip(" ,;:–—-").strip()
        if 20 <= len(suffix) <= 300 and len(suffix.split()) >= 4:
            return suffix
    return candidate


def _extract_bibliographic(path: Path, markdown: str) -> dict[str, Any]:
    import re
    stem = re.sub(r"(?:的全文翻译|_[^_]{2,12})$", "", path.stem).strip(" ._-")
    coded_stem = bool(re.match(r"^(?:\d+-s2\.0-|\d{4}\.\d+v\d+|information-\d|CCE-\d|Metaheuristic_)", path.stem, re.I))
    generic = re.compile(
        r"^(?:article|research paper|original paper|review|contents lists available|"
        r"available online|science\s*direct|doi\b|citation\b|received\b|"
        r"\d{4}\s*年|控制工程|上\s*海\s*交\s*通\s*大\s*学\s*学\s*报|分\s*类\s*号)", re.I
    )
    candidates: list[str] = []
    for line in markdown.splitlines()[:100]:
        if not re.match(r"^#{1,6}\s+", line):
            continue
        candidate = re.sub(r"^#{1,6}\s+", "", line).strip()
        candidate = re.sub(r"\s+", " ", candidate)
        if (4 <= len(candidate) <= 300 and not generic.match(candidate)
                and not candidate.startswith(("http", "<!--"))):
            candidates.append(_clean_bibliographic_title(candidate))
    title = stem
    if coded_stem and candidates:
        topic = re.compile(r"(?:ESN|echo|network|GAN|forecast|predict|imputation|"
                           r"Bayesian|贝叶斯|神经网络|降维|状态网络)", re.I)
        title = max(enumerate(candidates), key=lambda pair: (
            3 if topic.search(pair[1]) else 0,
            -5 if re.search(r"(?:\bAuthors?\b|Corresponding|Dept\.|Institute|[∗*])", pair[1], re.I) else 0,
            -3 if re.match(r"^(?:\d+|[IVX]+)[.、]\s*", pair[1], re.I) else 0,
            2 if 20 <= len(pair[1]) <= 180 else 0,
            1 if len(pair[1].split()) >= 4 or re.search(r"[\u4e00-\u9fff]{6}", pair[1]) else 0,
            -len(pair[1]),
            -pair[0],
        ))[1][:300]
    elif not coded_stem:
        title = stem[:300]
    raw_biblio = path.stem + "\n" + markdown[:6000]
    normalized = raw_biblio.translate(str.maketrans("０１２３４５６７８９", "0123456789"))
    normalized = re.sub(r"(?<=\d)[ \u3000]+(?=\d)", "", normalized)
    filename_years = re.findall(r"(?<!\d)((?:19|20)\d{2})(?!\d)", path.stem)
    # Elsevier PII filenames encode the allocation year immediately after the
    # eight-digit ISSN (for example S09521976-25-001290).  This is more reliable
    # than the first arbitrary year in OCR order, which may come from a cited
    # reference or an obsolete conference template.
    elsevier_pii = re.match(
        r"^1-s2\.0-S\d{8}(\d{2})\d+-main(?:$|\b)", path.stem, re.I,
    )
    arxiv_year = re.match(r"^(\d{2})(?:0[1-9]|1[0-2])\.\d+", path.stem)
    priority_years = re.findall(
        r"(?:published|publication|online|citation|©|版权所有|发表|出版)[^\n]{0,80}?"
        r"(?<!\d)((?:19|20)\d{2})(?!\d)", normalized, re.I,
    )
    all_years = [int(value) for value in re.findall(
        r"(?<!\d)((?:19|20)\d{2})(?!\d)", normalized
    ) if 1900 <= int(value) <= 2026]
    if elsevier_pii:
        year = 2000 + int(elsevier_pii.group(1))
    elif filename_years:
        year = int(filename_years[0])
    elif arxiv_year:
        year = 2000 + int(arxiv_year.group(1))
    elif priority_years:
        plausible_priority = [int(value) for value in priority_years if 1900 <= int(value) <= 2026]
        year = plausible_priority[0] if plausible_priority else (all_years[0] if all_years else None)
    else:
        year = all_years[0] if all_years else None
    lowered = (path.stem + " " + title + " " + markdown[:3000]).casefold()
    aliases = []
    esn_collision = re.search(
        r"回声信念网络|enterprise social network|edible swiftlet(?:'s)? nest|"
        r"sialylated mucin",
        (path.stem + " " + title).casefold(),
    )
    if (not esn_collision and (
        "echo state network" in lowered or "echo-state network" in lowered
        or "回声状态网络" in lowered
    )):
        aliases.extend(["Echo State Network", "ESN", "回声状态网络"])
    if "markov chain monte carlo" in lowered or "马尔科夫链蒙特卡" in lowered or re.search(r"\bmcmc\b", lowered):
        aliases.extend(["MCMC", "Markov Chain Monte Carlo", "马尔科夫链蒙特卡洛"])
    frontmatter = markdown[:8000]
    if "学位论文" in frontmatter or path.stem.startswith("论文"):
        document_type = "thesis"
    elif re.match(r"^\d{4}\.\d+v\d+$", path.stem, re.I) or "arxiv" in lowered:
        document_type = "preprint"
    elif re.search(
        r"\b(?:conference|proceedings|procedia|neurips|icml|ijcai|aaai)\b|会议论文|会议录",
        (path.stem + " " + frontmatter[:4000]), re.I,
    ):
        document_type = "conference_paper"
    elif re.search(
        r"\b(?:journal|volume|vol\.|issn)\b|\[[Jj]\]|学报|期刊|杂志|控制工程",
        frontmatter, re.I,
    ):
        document_type = "journal_article"
    else:
        document_type = "paper"
    return {
        "title": title,
        "publication_year": year,
        "venue": "",
        "doi": "",
        "language": "zh" if re.search(r"[\u4e00-\u9fff]", path.stem + title) else "en",
        "document_type": document_type,
        "topic_aliases_text": " ".join(dict.fromkeys(aliases)),
    }
