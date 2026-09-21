from __future__ import annotations

import json
import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from main import PDFSearchSystem
from ingestion.pipeline.generation_registry import GenerationRegistry, GenerationState


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--mark-interrupted", default="")
    args = parser.parse_args()
    if args.mark_interrupted:
        registry = GenerationRegistry(ROOT / "data/ingestion/generation_registry.sqlite3")
        record = registry.get(args.mark_interrupted)
        if record is None or record.state != GenerationState.BUILDING:
            raise SystemExit("interrupted generation is missing or not BUILDING")
        record.state = GenerationState.FAILED
        record.failure_reason = "page-provenance staging process interrupted before manifest creation"
        registry.put(record)
        print(json.dumps(record.to_dict(), ensure_ascii=False, indent=2))
        return 0
    result = PDFSearchSystem(db="main")._process_pdfs_staged()
    print(json.dumps(result, ensure_ascii=False, indent=2))
    return 0 if result else 2


if __name__ == "__main__":
    raise SystemExit(main())
