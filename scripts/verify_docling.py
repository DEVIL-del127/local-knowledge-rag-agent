"""Run a real Docling conversion without publishing any ingestion generation."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from ingestion.parsers.docling_parser import DoclingParser
from ingestion.quality.scorer import score_document


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("pdf", type=Path)
    args = parser.parse_args()

    result = DoclingParser().convert(args.pdf)
    quality = score_document(result)
    print(
        json.dumps(
            {
                "parser_version": result["parser_version"],
                "markdown_chars": len(result["markdown"]),
                "quality_report": quality.to_dict(),
            },
            ensure_ascii=False,
            default=str,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
