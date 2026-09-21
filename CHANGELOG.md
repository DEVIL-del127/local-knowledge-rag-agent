# Changelog

All notable changes to this project are documented in this file.

## [2.0.0] - 2026-09-21

### Added

- Backend-authoritative conversation/request identity and independent-session isolation.
- Unified V1-A routing contracts, context applicability and single semantic-encoding budget.
- Immutable bge-m3 semantic matrix package with category thresholds, margins and circuit state.
- Accepted-task dependencies, revalidation state, SourceSpan audit and centralized business outcomes.
- Production-path acceptance tooling, sealed holdout, real-answer smoke and ES/Chroma evidence reports.

### Changed

- Literature semantics now has one authoritative AcceptedIR path into planning and execution.
- Retrieval comparison fails closed when per-document/per-dimension evidence is incomplete.
- Runtime context no longer reuses stale objects after empty results or invalidation.

### Verification

- Untouched holdout: 12/12.
- Production real-answer smoke: 8/8.
- Real ES/Chroma corpus: 74/80; the remaining six are truthful `insufficient_evidence` outcomes.
- Measured end-to-end P95: 716.488 ms; source/page accuracy: 100%; fabricated pages: 0.
- Focused V1-A regression: 119 passed.

### Scope

- V1-B dynamic learning and hot release are intentionally excluded from this stable release.

## [1.0.0] - 2026-09-11

### Added

- Contract-first Agent runtime with explicit request stages and recovery support.
- Versioned PDF ingestion, catalog attestation and active-generation pinning.
- Hybrid Elasticsearch and Chroma retrieval with evidence validation.
- Structured `nlu_v2` query understanding and literature-search semantics.
- Memory, telemetry, privacy, token-budget and protected-backup components.

### Changed

- Established `agent_main.py` as the production question-answering entry point.
- Organized operational scripts and project documentation into dedicated directories.
- Consolidated the local Python environment convention on `venv/`.

### Removed

- Historical generated acceptance outputs and duplicate benchmark result directories.

### Security

- Kept credentials, private PDFs, local databases, model weights and runtime state out of Git.
