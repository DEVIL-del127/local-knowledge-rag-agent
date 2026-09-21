class RuntimeContractError(RuntimeError):
    """Base class for typed runtime contract failures."""


class ContextSnapshotMismatch(RuntimeContractError):
    """Persisted semantic context is missing, stale, or cannot be verified."""


class GenerationSnapshotMismatch(RuntimeContractError):
    """Pinned generation identity changed or failed verification."""
