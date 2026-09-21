import os

import pytest

from agent.os_secret_protection import SecretProtectionUnavailable
from agent.protected_backup import create_windows_backup


def test_unavailable_provider_does_not_create_backup(tmp_path, monkeypatch):
    def unavailable():
        raise SecretProtectionUnavailable("synthetic")
    monkeypatch.setattr("agent.protected_backup.WindowsUserProtection", unavailable)
    with pytest.raises(SecretProtectionUnavailable):
        create_windows_backup(b"synthetic", tmp_path / "backup", context=b"test")
    assert list(tmp_path.iterdir()) == []


def test_failed_roundtrip_does_not_create_backup(tmp_path, monkeypatch):
    class BrokenProvider:
        def protect(self, raw, *, context):
            return b"ciphertext"
        def unprotect(self, data, *, context):
            return b"wrong"
    monkeypatch.setattr("agent.protected_backup.WindowsUserProtection", BrokenProvider)
    with pytest.raises(SecretProtectionUnavailable):
        create_windows_backup(b"synthetic", tmp_path / "backup", context=b"test")
    assert list(tmp_path.iterdir()) == []
