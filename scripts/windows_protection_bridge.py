"""Pipe-only DPAPI helper. Never print payloads or traceback on failure."""
from pathlib import Path
import sys
import json
import base64

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from agent.os_secret_protection import WindowsUserProtection


def main():
    try:
        request_bytes = sys.stdin.buffer.read(24 * 1024 * 1024 + 1)
        if len(request_bytes) > 24 * 1024 * 1024:
            raise ValueError()
        request = json.loads(request_bytes)
        if not isinstance(request, dict) or set(request) != {"action", "payload", "context"}:
            raise ValueError()
        if request["action"] not in {"protect", "unprotect"}:
            raise ValueError()
        payload = base64.b64decode(request["payload"], validate=True)
        context = base64.b64decode(request["context"], validate=True)
        provider = WindowsUserProtection()
        result = getattr(provider, request["action"])(payload, context=context)
        sys.stdout.buffer.write(base64.b64encode(result))
        sys.stdout.buffer.flush()
        return 0
    except Exception:
        sys.stderr.write("protection_bridge_failed\n")
        return 1


if __name__ == "__main__":
    sys.exit(main())
