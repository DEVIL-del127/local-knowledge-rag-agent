import os

import pytest

from agent.os_secret_protection import SecretProtectionUnavailable, WindowsUserProtection


def test_platform_is_explicit_and_no_plaintext_fallback():
    if os.name != "nt":
        with pytest.raises(SecretProtectionUnavailable, match="unavailable"):
            WindowsUserProtection()
        return
    provider = WindowsUserProtection()
    secret = os.urandom(32)
    protected = provider.protect(secret, context=b"migration-test-v1")
    assert protected != secret
    assert provider.unprotect(protected, context=b"migration-test-v1") == secret
    with pytest.raises(SecretProtectionUnavailable):
        provider.unprotect(protected, context=b"wrong-context")
