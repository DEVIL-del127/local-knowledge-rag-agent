# -*- coding: utf-8 -*-
"""Ollama bge-m3 embedding client tests. No Ollama or ES service is required."""
import sys
from unittest.mock import Mock, patch

import numpy as np

from embedder import Embedder


def _response(payload):
    response = Mock()
    response.json.return_value = payload
    response.raise_for_status.return_value = None
    return response


def main():
    with patch("embedder.requests.get", return_value=_response({"models": []})), \
            patch("embedder.requests.post") as post:
        post.side_effect = [
            _response({"embeddings": [[0.1] * 1024]}),
            _response({"embeddings": [[0.2] * 1024]}),
        ]
        embedder = Embedder(base_url="http://ollama.test:11434")
        vectors = embedder.encode(["问题一", "问题二"], query_mode=True, batch_size=1)

        assert vectors.shape == (2, 1024)
        assert vectors.dtype == np.float32
        assert post.call_count == 2
        assert post.call_args_list[0].kwargs["json"]["input"][0].endswith("问题一")

    with patch("embedder.requests.get", return_value=_response({"models": []})), \
            patch("embedder.requests.post", return_value=_response({"embeddings": [[0.1] * 512]})):
        try:
            Embedder(base_url="http://ollama.test:11434").encode("维度检查")
        except RuntimeError as exc:
            assert "维度" in str(exc)
        else:
            raise AssertionError("应拒绝非 1024 维向量")

    print("Ollama Embedder 测试: 2/2 通过")


if __name__ == "__main__":
    main()
