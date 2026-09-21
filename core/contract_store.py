"""Versioned canonical JSON objects and resolvable content-addressed references."""
from __future__ import annotations

from dataclasses import dataclass
import hashlib
import json
import os
from pathlib import Path
import re
import tempfile


class ContractReferenceError(ValueError):
    pass


def canonical_json(value) -> bytes:
    return json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":"),
                      allow_nan=False).encode("utf-8")


@dataclass(frozen=True, slots=True)
class ContractRef:
    store_id: str
    locator_scheme: str
    locator: str
    digest_algorithm: str
    digest: str
    byte_length: int
    schema_version: str

    def validate(self):
        if (not isinstance(self.store_id, str) or not self.store_id
                or self.locator_scheme != "sha256-json-v1" or self.digest_algorithm != "sha256"
                or not isinstance(self.digest, str) or not re.fullmatch(r"[0-9a-f]{64}", self.digest)
                or self.locator != f"{self.digest}.json"
                or type(self.byte_length) is not int or not 0 < self.byte_length <= 16 * 1024 * 1024
                or not isinstance(self.schema_version, str) or not self.schema_version):
            raise ContractReferenceError("invalid_contract_reference")


class ContractStore:
    """Explicit object publication; construction and resolution never mkdir.

    Objects are not mutable records. The caller owns ACL validation and registry
    transactions. Failed publication may leave an unreferenced object, never a
    replacement of an existing object.
    """

    def __init__(self, root: Path, *, store_id: str):
        self.root = Path(root).absolute()
        self.store_id = store_id

    def _root(self):
        if any(path.is_symlink() for path in (self.root, *self.root.parents)):
            raise ContractReferenceError("contract_store_symlink")
        if not self.root.is_dir():
            raise ContractReferenceError("contract_store_unavailable")
        return self.root.resolve()

    def resolve(self, ref: ContractRef):
        ref.validate()
        if ref.store_id != self.store_id:
            raise ContractReferenceError("contract_store_mismatch")
        target = self._root() / ref.locator
        if target.is_symlink():
            raise ContractReferenceError("contract_object_symlink")
        try:
            with target.open("rb") as handle:
                raw = handle.read(ref.byte_length + 1)
            if len(raw) != ref.byte_length or hashlib.sha256(raw).hexdigest() != ref.digest:
                raise ContractReferenceError("contract_object_digest_mismatch")
            def unique(pairs):
                value = {}
                for key, item in pairs:
                    if key in value:
                        raise ValueError()
                    value[key] = item
                return value
            value = json.loads(raw, object_pairs_hook=unique)
            if not isinstance(value, dict) or value.get("schema_version") != ref.schema_version:
                raise ContractReferenceError("contract_schema_mismatch")
            if canonical_json(value) != raw:
                raise ContractReferenceError("contract_not_canonical")
            return value
        except (OSError, UnicodeError, ValueError, RecursionError):
            raise ContractReferenceError("contract_resolution_failed") from None

    def publish(self, value: dict) -> ContractRef:
        raw = canonical_json(value)
        digest = hashlib.sha256(raw).hexdigest()
        ref = ContractRef(self.store_id, "sha256-json-v1", f"{digest}.json", "sha256", digest,
                          len(raw), value.get("schema_version", ""))
        ref.validate()
        root = self._root()
        target = root / ref.locator
        if target.exists() or target.is_symlink():
            self.resolve(ref)
            return ref
        descriptor, temporary = tempfile.mkstemp(prefix=".contract-", dir=root)
        try:
            with os.fdopen(descriptor, "wb") as handle:
                handle.write(raw)
                handle.flush()
                os.fsync(handle.fileno())
            # Atomic create-only publication; never replace another writer.
            try:
                os.link(temporary, target)
            except FileExistsError:
                pass
            if os.name != "nt":
                directory = os.open(root, os.O_RDONLY | os.O_DIRECTORY)
                try:
                    os.fsync(directory)
                finally:
                    os.close(directory)
            self.resolve(ref)
            return ref
        finally:
            os.unlink(temporary)
