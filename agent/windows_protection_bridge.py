"""Explicit Windows Python/bridge locators; no shell or environment secrets."""
import base64
import json
import subprocess

from agent.os_secret_protection import SecretProtectionUnavailable


class WindowsProtectionBridge:
    def __init__(self, *, windows_python: str, windows_script: str):
        if not windows_python or not windows_script:
            raise SecretProtectionUnavailable("protection_bridge_locator_missing")
        self._command = [windows_python, "-I", windows_script]

    def protect(self, plaintext: bytes, *, context: bytes) -> bytes:
        return self._invoke("protect", plaintext, context)

    def unprotect(self, ciphertext: bytes, *, context: bytes) -> bytes:
        return self._invoke("unprotect", ciphertext, context)

    def _invoke(self, action: str, payload: bytes, context: bytes) -> bytes:
        if not isinstance(payload, bytes) or not 0 < len(payload) <= 16 * 1024 * 1024:
            raise ValueError("invalid protected payload size")
        if not isinstance(context, bytes) or not 0 < len(context) <= 4096:
            raise ValueError("invalid protection context")
        request = json.dumps({"action": action, "payload": base64.b64encode(payload).decode("ascii"),
                              "context": base64.b64encode(context).decode("ascii")}).encode("ascii")
        try:
            result = subprocess.run(self._command, input=request, stdout=subprocess.PIPE,
                                    stderr=subprocess.DEVNULL, timeout=30, check=True,
                                    shell=False, env={})
            output = base64.b64decode(result.stdout, validate=True)
            if not output or len(output) > 16 * 1024 * 1024:
                raise ValueError()
            return output
        except (OSError, subprocess.SubprocessError, ValueError):
            raise SecretProtectionUnavailable("protection_bridge_failed") from None
