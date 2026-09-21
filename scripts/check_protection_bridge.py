"""Explicit synthetic-data self-test; never loads application settings/secrets."""
import argparse
import os
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from agent.windows_protection_bridge import WindowsProtectionBridge
from agent.os_secret_protection import SecretProtectionUnavailable


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--python", required=True)
    parser.add_argument("--script", required=True)
    args = parser.parse_args()
    provider = WindowsProtectionBridge(windows_python=args.python, windows_script=args.script)
    secret = os.urandom(32)
    protected = provider.protect(secret, context=b"bridge-selftest-v1")
    assert protected != secret
    assert provider.unprotect(protected, context=b"bridge-selftest-v1") == secret
    try:
        provider.unprotect(protected, context=b"wrong")
    except SecretProtectionUnavailable:
        print("DPAPI bridge roundtrip and context isolation passed")
        return
    raise RuntimeError("bridge context isolation failed")


if __name__ == "__main__":
    main()
