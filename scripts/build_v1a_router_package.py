from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import numpy as np

from core.embedder import OllamaEmbedder, clean_embed_text


def main() -> None:
    parser = argparse.ArgumentParser(description="Build immutable V1-A bge-m3 routing matrix")
    parser.add_argument("--seed", default="config/routing/v1a_seed.json")
    parser.add_argument("--output", default="data/routing/v1a")
    parser.add_argument("--ollama-url", default="http://localhost:11434")
    args = parser.parse_args()
    seed = json.loads(Path(args.seed).read_text(encoding="utf-8"))
    examples = seed["examples"]
    if any(item.get("context_dependent") for item in examples):
        raise ValueError("context-dependent expressions cannot enter the independent matrix")
    embedder = OllamaEmbedder(base_url=args.ollama_url, model=seed["encoder"]["model"], timeout=30)
    vectors = np.asarray(embedder.embed([clean_embed_text(item["text"]) for item in examples]), dtype=np.float32)
    if vectors.ndim != 2 or vectors.shape[0] != len(examples) or not np.isfinite(vectors).all():
        raise RuntimeError("encoder returned an invalid fixed matrix")
    vectors /= np.linalg.norm(vectors, axis=1, keepdims=True) + 1e-12
    output = Path(args.output)
    output.mkdir(parents=True, exist_ok=True)
    np.save(output / "matrix.npy", vectors.astype(np.float32), allow_pickle=False)
    package = {**seed, "matrix_file": "matrix.npy", "dimension": int(vectors.shape[1]),
               "matrix_digest": hashlib.sha256(vectors.astype(np.float32).tobytes()).hexdigest(),
               "all_categories_default_disabled": True}
    (output / "package.json").write_text(
        json.dumps(package, ensure_ascii=False, indent=2, sort_keys=True), encoding="utf-8")


if __name__ == "__main__":
    main()
