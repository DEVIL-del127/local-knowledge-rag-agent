from __future__ import annotations

import dataclasses
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from agent.semantic_compiler_adapter import SemanticCompilerAdapter
from ingestion.pipeline.generation_registry import GenerationRegistry
from main import PDFSearchSystem


def main() -> int:
    args = sys.argv[1:]
    previous_query = ""
    if "--previous" in args:
        marker = args.index("--previous")
        previous_query = " ".join(args[marker + 1:]).strip()
        args = args[:marker]
    query = " ".join(args).strip()
    if not query:
        raise SystemExit("usage: diagnose_semantic_compile.py QUERY")
    root = Path(__file__).resolve().parents[1]
    system = PDFSearchSystem()
    adapter = SemanticCompilerAdapter(
        search_backend=system,
        generation_registry=GenerationRegistry(
            root / "data" / "ingestion" / "generation_registry.sqlite3"
        ),
        enable_llm=True,
        require_active_generation=True,
    )
    compiled = adapter.compile(query)
    if previous_query:
        compiled = adapter.compile_with_context(
            previous_query, dataclasses.asdict(compiled), previous_turn_id="diagnostic",
        )
    print(json.dumps(
        dataclasses.asdict(compiled),
        ensure_ascii=False,
        indent=2,
        default=str,
    ))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
