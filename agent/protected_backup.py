"""Create-only DPAPI backup primitive; production ACL/maintenance gate is external."""
from __future__ import annotations

import hashlib
import hmac
import os
from pathlib import Path

from agent.os_secret_protection import WindowsUserProtection, SecretProtectionUnavailable


def _protection_provider():
    try:
        return WindowsUserProtection()
    except SecretProtectionUnavailable:
        pass
    executable = os.environ.get("STUDY_PROTECTION_WINDOWS_PYTHON", "")
    script = os.environ.get("STUDY_PROTECTION_WINDOWS_SCRIPT", "")
    if executable and script:
        from agent.windows_protection_bridge import WindowsProtectionBridge
        return WindowsProtectionBridge(windows_python=executable, windows_script=script)
    raise SecretProtectionUnavailable("protected_backup_provider_unavailable")


def create_windows_backup(raw: bytes, destination: str | Path, *, context: bytes) -> str:
    """Persist only ciphertext, without overwriting or plaintext temporary files.

    Caller must verify directory ACLs and freeze the source before invoking this
    primitive on production data. On failure a partial ciphertext file may remain;
    it is never deleted automatically and must not be treated as a valid backup.
    """
    provider = _protection_provider()
    protected = provider.protect(raw, context=context)
    if not hmac.compare_digest(provider.unprotect(protected, context=context), raw):
        raise SecretProtectionUnavailable("backup_roundtrip_failed")
    target = Path(destination)
    # O_EXCL also refuses an existing symlink; parent directory is caller-owned.
    descriptor = os.open(target, os.O_WRONLY | os.O_CREAT | os.O_EXCL | getattr(os, "O_BINARY", 0), 0o600)
    with os.fdopen(descriptor, "wb") as handle:
        handle.write(protected)
        handle.flush()
        os.fsync(handle.fileno())
    persisted = target.read_bytes()
    if not hmac.compare_digest(persisted, protected) or not hmac.compare_digest(
        provider.unprotect(persisted, context=context), raw
    ):
        raise SecretProtectionUnavailable("persisted_backup_verification_failed")
    return hashlib.sha256(persisted).hexdigest()
