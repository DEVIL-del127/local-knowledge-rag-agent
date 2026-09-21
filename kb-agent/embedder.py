# -*- coding: utf-8 -*-
"""Ollama embedding client for kb-agent's routing and retrieval vectors."""
import logging
import re
import subprocess
import sys
from typing import Sequence

import numpy as np
import requests

from config import EMBED_DIM, MODEL_NAME, OLLAMA_URL, QUERY_PREFIX

logger = logging.getLogger(__name__)


class Embedder:
    """Keep the previous NumPy interface while delegating embeddings to Ollama."""

    def __init__(self, model_name: str = MODEL_NAME, base_url: str = OLLAMA_URL,
                 timeout: float = 180.0):
        self.model_name = model_name
        self.base_url = base_url.rstrip("/")
        self.timeout = timeout
        self._probed = False

    def encode(self, texts: str | Sequence[str], query_mode: bool = False,
               batch_size: int = 32) -> np.ndarray:
        """Return float32 vectors from Ollama's /api/embed endpoint.

        query_mode retains the routing task prefix used by the existing seed and
        query pipeline, so callers do not need to change when moving to bge-m3.
        """
        if isinstance(texts, str):
            texts = [texts]
        items = [str(text).strip() for text in texts]
        if not items:
            return np.empty((0, EMBED_DIM), dtype=np.float32)
        if any(not text for text in items):
            raise ValueError("不能对空文本生成向量")
        self._ensure_available()
        if query_mode:
            items = [QUERY_PREFIX + text for text in items]

        vectors: list[list[float]] = []
        for start in range(0, len(items), batch_size):
            batch = items[start:start + batch_size]
            vectors.extend(self._embed_batch(batch))

        result = np.asarray(vectors, dtype=np.float32)
        if result.shape != (len(items), EMBED_DIM):
            raise RuntimeError(
                f"Ollama 模型 {self.model_name} 返回向量维度 {result.shape}; "
                f"kb-agent 期望 ({len(items)}, {EMBED_DIM})。"
            )
        return result

    @staticmethod
    def _wsl_host_ip() -> str | None:
        """Return the Windows gateway address when the client runs inside WSL."""
        try:
            output = subprocess.run(
                ["ip", "route", "show", "default"],
                capture_output=True, text=True, timeout=3,
            ).stdout
            match = re.search(r"default via (\S+)", output)
            return match.group(1) if match else None
        except (OSError, subprocess.SubprocessError):
            return None

    def _ensure_available(self) -> None:
        """Probe Ollama once and fall back from WSL localhost to the host gateway."""
        if self._probed:
            return
        try:
            response = requests.get(f"{self.base_url}/api/tags", timeout=5)
            response.raise_for_status()
            self._probed = True
            return
        except requests.RequestException as primary_error:
            if not re.search(r"//(?:localhost|127\.0\.0\.1)(?::|/)", self.base_url):
                raise RuntimeError(f"无法连接 Ollama: {primary_error}") from primary_error

        host_ip = self._wsl_host_ip()
        if not host_ip:
            raise RuntimeError(
                "无法连接 Ollama localhost，且未找到 WSL Windows 宿主地址；"
                "请设置 KB_OLLAMA_URL=http://<宿主IP>:11434"
            ) from primary_error
        fallback_url = f"http://{host_ip}:11434"
        try:
            response = requests.get(f"{fallback_url}/api/tags", timeout=5)
            response.raise_for_status()
        except requests.RequestException as fallback_error:
            raise RuntimeError(
                f"无法连接 Ollama ({self.base_url} 或 {fallback_url}): {fallback_error}"
            ) from fallback_error
        logger.info("WSL localhost 不可达，已切换为 Windows Ollama 地址: %s", fallback_url)
        self.base_url = fallback_url
        self._probed = True

    def _embed_batch(self, texts: list[str]) -> list[list[float]]:
        try:
            response = requests.post(
                f"{self.base_url}/api/embed",
                json={"model": self.model_name, "input": texts},
                timeout=self.timeout,
            )
            response.raise_for_status()
        except requests.RequestException as exc:
            raise RuntimeError(
                f"Ollama embedding 调用失败({self.base_url}, 模型 {self.model_name}): {exc}"
            ) from exc

        payload = response.json()
        embeddings = payload.get("embeddings")
        if not isinstance(embeddings, list) or len(embeddings) != len(texts):
            raise RuntimeError("Ollama /api/embed 返回格式异常：未得到与输入等长的 embeddings")
        return embeddings

    def health_check(self) -> bool:
        """Verify that Ollama is reachable and the configured embedding model exists."""
        try:
            self._ensure_available()
            response = requests.get(f"{self.base_url}/api/tags", timeout=5)
            response.raise_for_status()
            names = [item.get("name", "") for item in response.json().get("models", [])]
            return any(name == self.model_name or name.startswith(f"{self.model_name}:") for name in names)
        except (requests.RequestException, RuntimeError) as exc:
            logger.warning("Ollama 健康检查失败: %s", exc)
            return False

    @staticmethod
    def cosine(a, b) -> float:
        a = np.asarray(a, dtype=np.float32)
        b = np.asarray(b, dtype=np.float32)
        a = a / (np.linalg.norm(a) + 1e-9)
        b = b / (np.linalg.norm(b) + 1e-9)
        return float(np.dot(a, b))


def main():
    sys.stdout.reconfigure(encoding="utf-8")
    emb = Embedder()
    print(f"检查 Ollama: {emb.base_url} | 模型: {emb.model_name}", flush=True)
    if not emb.health_check():
        raise SystemExit(f"未发现可用模型 {emb.model_name}，请执行: ollama pull {emb.model_name}")
    vector = emb.encode(["测试句子"], query_mode=True)
    print("OK dim:", vector.shape, "norm:", float(np.linalg.norm(vector[0])))


if __name__ == "__main__":
    main()
