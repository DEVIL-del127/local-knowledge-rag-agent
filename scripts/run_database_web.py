from __future__ import annotations

import os
import shutil
import socket
import subprocess
import sys
import threading
import time
import urllib.parse
import webbrowser
from pathlib import Path

import uvicorn
from dotenv import load_dotenv

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

# Keep the web entrypoint's source-package resolution identical to the formal
# CLI entrypoint.  The nlu_v2 package lives under kb-agent rather than ROOT.
from agent_main import _bootstrap_source_paths

_bootstrap_source_paths(ROOT)


def _endpoint_open(url: str, timeout: float = 0.5) -> bool:
    parsed = urllib.parse.urlparse(url)
    host = parsed.hostname or "localhost"
    port = parsed.port or (443 if parsed.scheme == "https" else 80)
    try:
        with socket.create_connection((host, port), timeout=timeout):
            return True
    except OSError:
        return False


def _wsl_gateway() -> str:
    if not (os.environ.get("WSL_DISTRO_NAME") or os.environ.get("WSL_INTEROP")):
        return ""
    try:
        result = subprocess.run(
            ["ip", "route", "show", "default"], capture_output=True, text=True,
            timeout=2, check=False,
        )
        parts = result.stdout.split()
        return parts[parts.index("via") + 1] if "via" in parts else ""
    except (OSError, subprocess.SubprocessError, ValueError, IndexError):
        return ""


def _ensure_elasticsearch() -> None:
    host = os.environ.get("ES_HOST", "localhost")
    port = int(os.environ.get("ES_PORT", "9200"))
    if _endpoint_open(f"http://{host}:{port}"):
        print("[ready] Elasticsearch")
        return
    if not shutil.which("docker"):
        print("[warning] Elasticsearch 未运行，且找不到 Docker。")
        return
    print("[start] Elasticsearch")
    result = subprocess.run(
        ["docker", "compose", "up", "-d", "elasticsearch"], cwd=ROOT,
        check=False,
    )
    if result.returncode != 0:
        print("[warning] Elasticsearch 启动失败，请检查 Docker Desktop。")
        return
    for _ in range(30):
        if _endpoint_open(f"http://{host}:{port}"):
            print("[ready] Elasticsearch")
            return
        time.sleep(1)
    print("[warning] Elasticsearch 启动超时。")


def _ensure_ollama() -> None:
    configured = os.environ.get("OLLAMA_URL", "http://localhost:11434").rstrip("/")
    if _endpoint_open(configured):
        print(f"[ready] Ollama ({configured})")
        return
    gateway = _wsl_gateway()
    gateway_url = f"http://{gateway}:11434" if gateway else ""
    if gateway_url and _endpoint_open(gateway_url):
        os.environ["OLLAMA_URL"] = gateway_url
        print(f"[ready] Ollama ({gateway_url})")
        return
    if not shutil.which("ollama"):
        print("[warning] Ollama 未连接；请在 Windows 启动 Ollama，或修正 OLLAMA_URL。")
        return
    print("[start] Ollama")
    kwargs: dict[str, object] = {"stdout": subprocess.DEVNULL, "stderr": subprocess.DEVNULL}
    if os.name == "nt":
        kwargs["creationflags"] = subprocess.CREATE_NO_WINDOW
    subprocess.Popen(["ollama", "serve"], **kwargs)
    for _ in range(15):
        if _endpoint_open(configured):
            print("[ready] Ollama")
            return
        time.sleep(1)
    print("[warning] Ollama 启动超时。")


def _open_workspace() -> None:
    time.sleep(1.2)
    webbrowser.open("http://127.0.0.1:8088/")


if __name__ == "__main__":
    load_dotenv(ROOT / ".env")
    _ensure_elasticsearch()
    _ensure_ollama()
    if os.environ.get("DEEPSEEK_API_KEY", "").strip():
        print("[ready] DeepSeek 配置")
    else:
        print("[warning] .env 中未配置 DEEPSEEK_API_KEY，问答生成不可用。")
    threading.Thread(target=_open_workspace, daemon=True).start()
    uvicorn.run("web.database_app:app", host="127.0.0.1", port=8088, reload=False)
