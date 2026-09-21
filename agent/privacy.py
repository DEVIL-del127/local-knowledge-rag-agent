from __future__ import annotations

import base64
import json
import os
import re
import time
import tempfile
import hashlib
import hmac
from pathlib import Path
from typing import Any


_SECRET_PATTERNS = (
    re.compile(r"(?i)\b(?:authorization|api[_-]?key|access[_-]?token|credential)\b\s*[:=]\s*\S+"),
    re.compile(r"\bsk-[A-Za-z0-9_-]{12,}\b"),
)
_FORBIDDEN_KEYS = {"reasoning_content", "chain_of_thought", "authorization", "api_key"}


class DataSanitizer:
    """One recursive sanitizer shared by persistence and release boundaries."""

    @classmethod
    def sanitize(cls, value: Any) -> Any:
        if isinstance(value, dict):
            return {
                str(key): ("[REDACTED]" if str(key).lower() in _FORBIDDEN_KEYS
                           else cls.sanitize(item))
                for key, item in value.items()
            }
        if isinstance(value, list):
            return [cls.sanitize(item) for item in value]
        if isinstance(value, tuple):
            return [cls.sanitize(item) for item in value]
        if isinstance(value, str):
            text = value
            for pattern in _SECRET_PATTERNS:
                text = pattern.sub("[REDACTED]", text)
            return text
        return value


class PersistencePolicyError(ValueError):
    pass


class PersistencePolicy:
    """Fail-closed field policy for operational persistence boundaries."""

    _ALLOWLISTS = {
        "release-audit-v1": frozenset({
            "schema_version", "created_at_utc", "gate_pass", "generation",
            "metrics", "tests", "privacy", "model_calls", "unresolved",
        }),
        "security-event-v1": frozenset({
            "schema_version", "created_at_utc", "boundary", "reason_code",
            "payload_digest",
        }),
    }

    @classmethod
    def prepare(cls, boundary: str, payload: dict[str, Any]) -> dict[str, Any]:
        allowed = cls._ALLOWLISTS.get(boundary)
        if allowed is None:
            raise PersistencePolicyError(f"unknown persistence boundary: {boundary}")
        unknown = sorted(set(map(str, payload)) - allowed)
        if unknown:
            raise PersistencePolicyError(
                f"persistence boundary {boundary} contains unknown fields: {unknown}"
            )
        sanitized = DataSanitizer.sanitize(payload)
        if not isinstance(sanitized, dict):  # defensive, payload is typed above
            raise PersistencePolicyError("sanitized persistence payload must be an object")
        return sanitized


class PersistenceCipher:
    """AEAD envelope with a user-only local key; never stores plaintext fallback."""

    prefix = "enc-v1:"
    key_prefix = b"dpapi-key-v1:"

    @staticmethod
    def _key_protection():
        if os.name == "nt":
            from agent.os_secret_protection import WindowsUserProtection
            return WindowsUserProtection()
        executable = os.environ.get("STUDY_PROTECTION_WINDOWS_PYTHON", "")
        script = os.environ.get("STUDY_PROTECTION_WINDOWS_SCRIPT", "")
        if executable or script:
            from agent.windows_protection_bridge import WindowsProtectionBridge
            return WindowsProtectionBridge(windows_python=executable, windows_script=script)
        return None

    def __init__(self, key_path: str | Path) -> None:
        self.key_path = Path(key_path)
        self.key_path.parent.mkdir(parents=True, exist_ok=True)
        protector = self._key_protection()
        context = ("study-persistence-key-v1:" + self.key_path.name).encode("utf-8")
        candidate = os.urandom(32)
        candidate_envelope = (self.key_prefix + base64.b64encode(protector.protect(candidate, context=context))
                              if protector is not None else candidate)
        descriptor, temporary = tempfile.mkstemp(prefix=".key-", dir=self.key_path.parent)
        created = False
        try:
            with os.fdopen(descriptor, "wb") as handle:
                handle.write(candidate_envelope)
                handle.flush()
                os.fsync(handle.fileno())
            try:
                # Publish complete bytes without replacing an existing key.
                os.link(temporary, self.key_path)
                created = True
            except FileExistsError:
                pass
        finally:
            os.unlink(temporary)
        if not created:
            # Another process owns key creation. The directory entry may be
            # visible before its fixed-size write reaches NTFS through WSL.
            key = b""
            protection_attempts = 0
            for _ in range(200):
                try:
                    key = self.key_path.read_bytes()
                except FileNotFoundError:
                    key = b""
                if len(key) == 32 and not key.startswith(self.key_prefix):
                    break
                if key.startswith(self.key_prefix) and protector is not None:
                    protection_attempts += 1
                    try:
                        decoded = protector.unprotect(
                            base64.b64decode(key[len(self.key_prefix):], validate=True), context=context,
                        )
                        if len(decoded) == 32:
                            key = decoded
                            break
                    except Exception:
                        # Another process may still be finishing the exclusive
                        # key creation. Never interpret a partial envelope as raw.
                        if protection_attempts >= 3:
                            raise PersistencePolicyError("protected persistence key unavailable") from None
                time.sleep(0.01)
        else:
            key = candidate
        if key.startswith(self.key_prefix):
            if protector is None:
                raise PersistencePolicyError("protected key requires configured OS protection provider")
            key = protector.unprotect(base64.b64decode(key[len(self.key_prefix):], validate=True), context=context)
        if len(key) != 32:
            raise RuntimeError("invalid persistence encryption key")
        from cryptography.hazmat.primitives.ciphers.aead import AESGCM

        self._aead = AESGCM(key)
        self._identity_key = key

    def identity_hmac(self, value: bytes) -> str:
        import hashlib
        import hmac
        return hmac.new(self._identity_key, value, hashlib.sha256).hexdigest()

    def encrypt_json(self, value: Any, *, aad: str) -> str:
        sanitized = DataSanitizer.sanitize(value)
        plaintext = json.dumps(
            sanitized, ensure_ascii=False, sort_keys=True, separators=(",", ":")
        ).encode("utf-8")
        nonce = os.urandom(12)
        ciphertext = self._aead.encrypt(nonce, plaintext, aad.encode("utf-8"))
        return self.prefix + base64.urlsafe_b64encode(nonce + ciphertext).decode("ascii")

    def decrypt_json(self, envelope: str, *, aad: str) -> Any:
        if not envelope.startswith(self.prefix):
            raise PersistencePolicyError("plaintext persistence requires explicit migration")
        packed = base64.urlsafe_b64decode(envelope[len(self.prefix):].encode("ascii"))
        plaintext = self._aead.decrypt(
            packed[:12], packed[12:], aad.encode("utf-8")
        )
        return json.loads(plaintext.decode("utf-8"))


def protect_legacy_key(key_path: str | Path, backup_path: str | Path, *, expected_sha256: str) -> dict:
    """Atomically replace one verified raw key with an OS-protected envelope.

    The recovery copy is itself OS protected and create-only. Existing protected
    keys are verified and treated as an idempotent no-op.
    """
    from agent.protected_backup import create_windows_backup

    path = Path(key_path).resolve()
    backup = Path(backup_path).absolute()
    if not path.is_file() or backup.exists() or path == backup:
        raise ValueError("invalid key migration targets")
    raw = path.read_bytes()
    protector = PersistenceCipher._key_protection()
    if protector is None:
        raise PersistencePolicyError("protected key requires configured OS protection provider")
    context = ("study-persistence-key-v1:" + path.name).encode("utf-8")
    if raw.startswith(PersistenceCipher.key_prefix):
        decoded = protector.unprotect(
            base64.b64decode(raw[len(PersistenceCipher.key_prefix):], validate=True), context=context,
        )
        if len(decoded) != 32:
            raise RuntimeError("invalid protected persistence key")
        return {"migrated": False, "already_protected": True}
    if len(raw) != 32 or not hmac.compare_digest(hashlib.sha256(raw).hexdigest(), expected_sha256):
        raise ValueError("legacy key changed since migration inventory")
    backup_digest = create_windows_backup(raw, backup, context=b"study-legacy-key-backup-v1:" + path.name.encode())
    envelope = PersistenceCipher.key_prefix + base64.b64encode(protector.protect(raw, context=context))
    if not hmac.compare_digest(protector.unprotect(
        base64.b64decode(envelope[len(PersistenceCipher.key_prefix):], validate=True), context=context,
    ), raw):
        raise PersistencePolicyError("protected key roundtrip failed")
    descriptor, temporary = tempfile.mkstemp(prefix=".key-migrate-", dir=path.parent)
    try:
        with os.fdopen(descriptor, "wb") as handle:
            handle.write(envelope)
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary, path)
    finally:
        if os.path.exists(temporary):
            os.unlink(temporary)
    # Verify through the normal runtime loader after publication.
    PersistenceCipher(path)
    return {"migrated": True, "backup_sha256": backup_digest}
