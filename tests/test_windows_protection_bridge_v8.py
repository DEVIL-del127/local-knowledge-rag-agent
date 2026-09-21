import base64
import json
import subprocess
from types import SimpleNamespace

import pytest

from agent.windows_protection_bridge import WindowsProtectionBridge
from agent.os_secret_protection import SecretProtectionUnavailable


def test_bridge_payload_only_in_pipe(monkeypatch):
    def run(command, **kwargs):
        assert command == ["explicit-python", "-I", "explicit-script"]
        assert kwargs["env"] == {}
        assert kwargs["shell"] is False
        data = json.loads(kwargs["input"])
        assert base64.b64decode(data["payload"]) == b"synthetic"
        return SimpleNamespace(stdout=base64.b64encode(b"protected"))
    monkeypatch.setattr(subprocess, "run", run)
    assert WindowsProtectionBridge(windows_python="explicit-python", windows_script="explicit-script").protect(
        b"synthetic", context=b"test") == b"protected"


def test_bridge_error_does_not_echo_payload(monkeypatch):
    def run(*args, **kwargs):
        raise subprocess.CalledProcessError(1, ["synthetic-private-command"])
    monkeypatch.setattr(subprocess, "run", run)
    with pytest.raises(SecretProtectionUnavailable, match="^protection_bridge_failed$"):
        WindowsProtectionBridge(windows_python="explicit", windows_script="explicit").protect(b"synthetic", context=b"test")
